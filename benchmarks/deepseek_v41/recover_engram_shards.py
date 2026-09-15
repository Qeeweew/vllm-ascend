#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Resume two audited public shards without changing their original prefixes.

Partial suffixes are append-only recovery artifacts. A fixed ModelScope
revision, exact HTTP Content-Range, source identity checks and whole-file
SHA256 guard publication. Existing destination paths are never overwritten.
"""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import signal
import threading
import time
from pathlib import Path

import requests

REVISION = "3bd368ab0f3da472b1adc6e19d37717a6cd0967f"
SHARDS = (
    (47, 84174438400, 101535150936, "824db4881320407ac340736d14dcee5ecd748c27d0f5836b8127ecc2e3781b0f"),
    (48, 83363889152, 101537926640, "976330f4954338e1ad8b508c32aa912032c7ad908959fd53c8307650fe4520ed"),
)
IO_BYTES = 1024 * 1024
CHECKPOINT_BYTES = 16 * IO_BYTES
RANGE_BYTES = 64 * IO_BYTES
NETWORK_BYTES = 64 * 1024
CHECKPOINT_SECONDS = 10
RESERVE_BYTES = 8 * 1024**3


def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def identity(path):
    stat = path.stat()
    return {key: getattr(stat, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".new")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


class Recovery:
    def __init__(self, root, directory, proxy):
        self.root = root
        self.directory = directory
        self.proxy = proxy
        self.stop = threading.Event()
        self.log_lock = threading.Lock()

    def log(self, number, event, **fields):
        record = {"time": timestamp(), "shard": number, "event": event, **fields}
        with self.log_lock:
            print(json.dumps(record), flush=True)

    def status(self, number, event, **fields):
        atomic_json(
            self.directory / f"{number}.status.json",
            {"time": timestamp(), "shard": number, "event": event, **fields},
        )
        self.log(number, event, **fields)

    def check_source(self, info):
        if identity(Path(info["prefix_path"])) != info["prefix_identity"]:
            raise RuntimeError(f"shard {info['number']}: original prefix changed; preserving all files and stopping")
        if self.stop.is_set():
            raise InterruptedError("recovery stop requested")

    def prepare(self):
        manifest_path = self.directory / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["revision"] != REVISION or manifest["source_root"] != str(self.root):
                raise RuntimeError("recovery manifest belongs to another source or revision")
            records = manifest["shards"]
            for info, (number, prefix_bytes, total_bytes, digest) in zip(records, SHARDS, strict=True):
                if (info["number"], info["prefix_bytes"], info["total_bytes"], info["sha256"]) != (
                    number,
                    prefix_bytes,
                    total_bytes,
                    digest,
                ):
                    raise RuntimeError("recovery manifest differs from audited shard constants")
        else:
            records = []
            for number, prefix_bytes, total_bytes, digest in SHARDS:
                filename = f"model-{number:05d}-of-00048.safetensors"
                prefix = self.root / "._____temp" / filename
                snapshot = identity(prefix)
                if snapshot["st_size"] != prefix_bytes:
                    raise RuntimeError(f"shard {number}: prefix length changed since audit; re-audit before recovery")
                records.append(
                    {
                        "number": number,
                        "filename": filename,
                        "prefix_path": str(prefix),
                        "prefix_identity": snapshot,
                        "prefix_bytes": prefix_bytes,
                        "total_bytes": total_bytes,
                        "sha256": digest,
                    }
                )
            atomic_json(
                manifest_path,
                {"revision": REVISION, "source_root": str(self.root), "created": timestamp(), "shards": records},
            )
        remaining = 0
        for info in records:
            self.check_source(info)
            suffix = self.directory / (info["filename"] + ".suffix")
            existing = suffix.stat().st_size if suffix.exists() else 0
            missing = info["total_bytes"] - info["prefix_bytes"]
            if existing > missing:
                raise RuntimeError("suffix exceeds expected size; refusing to truncate")
            complete = self.directory / info["filename"]
            remaining += missing - existing
            if not complete.exists():
                remaining += info["total_bytes"]
        free = shutil.disk_usage(self.directory).free
        if free < remaining + RESERVE_BYTES:
            raise RuntimeError(f"insufficient space: free={free}, needed={remaining + RESERVE_BYTES}")
        self.log(None, "preflight", free_bytes=free, additional_budget_bytes=remaining, reserve_bytes=RESERVE_BYTES)
        return records

    def download(self, info):
        number = info["number"]
        suffix = self.directory / (info["filename"] + ".suffix")
        expected = info["total_bytes"] - info["prefix_bytes"]
        url = f"https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash/resolve/{REVISION}/{info['filename']}"
        session = requests.Session()
        session.trust_env = False
        proxies = {"http": self.proxy, "https": self.proxy}
        retry = 0
        while True:
            self.check_source(info)
            have = suffix.stat().st_size if suffix.exists() else 0
            if have == expected:
                self.status(number, "suffix_downloaded", suffix_bytes=have, expected_suffix_bytes=expected)
                return
            if have > expected:
                raise RuntimeError("suffix exceeds expected size")
            offset = info["prefix_bytes"] + have
            end = min(info["total_bytes"] - 1, offset + RANGE_BYTES - 1)
            response_end = have + end - offset + 1
            self.status(
                number,
                "request",
                offset=offset,
                end=end,
                suffix_bytes=have,
                expected_suffix_bytes=expected,
                retry=retry,
            )
            try:
                with session.get(
                    url,
                    headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"},
                    proxies=proxies,
                    stream=True,
                    timeout=(15, 60),
                ) as response:
                    if response.status_code in (408, 429, 500, 502, 503, 504):
                        raise requests.ConnectionError("retryable server status")
                    if response.status_code != 206:
                        raise RuntimeError(f"server did not honor Range: HTTP {response.status_code}")
                    expected_range = f"bytes {offset}-{end}/{info['total_bytes']}"
                    if response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError("Content-Range does not match requested immutable shard range")
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != end - offset + 1:
                        raise RuntimeError("Content-Length does not match requested range")
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise RuntimeError("unexpected encoded range response")
                    initial = have
                    checkpoint = have
                    started = time.monotonic()
                    last_checkpoint = started
                    with suffix.open("ab") as stream:
                        try:
                            for data in response.iter_content(NETWORK_BYTES):
                                if not data:
                                    continue
                                self.check_source(info)
                                if have + len(data) > response_end:
                                    raise RuntimeError("response exceeds audited shard length")
                                stream.write(data)
                                have += len(data)
                                if (
                                    have - checkpoint >= CHECKPOINT_BYTES
                                    or time.monotonic() - last_checkpoint >= CHECKPOINT_SECONDS
                                ):
                                    stream.flush()
                                    os.fsync(stream.fileno())
                                    self.check_source(info)
                                    self.status(
                                        number,
                                        "downloading",
                                        suffix_bytes=have,
                                        expected_suffix_bytes=expected,
                                        percent=round(have / expected * 100, 3),
                                        bytes_per_second=round(
                                            (have - initial) / max(time.monotonic() - started, 0.001)
                                        ),
                                    )
                                    checkpoint = have
                                    last_checkpoint = time.monotonic()
                        finally:
                            stream.flush()
                            os.fsync(stream.fileno())
                    if have != response_end:
                        raise requests.ConnectionError("response ended before expected suffix length")
                retry = 0
            except requests.RequestException as error:
                retry += 1
                self.status(number, "retry", error_type=type(error).__name__, retry=retry)
                if retry >= 40:
                    raise RuntimeError("download failed 40 times; partial suffix preserved") from None
                if self.stop.wait(min(60, 2 ** min(retry, 6))):
                    raise InterruptedError("recovery stop requested") from None

    def verify_file(self, path, info):
        if path.stat().st_size != info["total_bytes"]:
            raise RuntimeError(f"shard {info['number']}: complete file has wrong size")
        digest = hashlib.sha256()
        done = 0
        with path.open("rb") as stream:
            while data := stream.read(IO_BYTES):
                digest.update(data)
                done += len(data)
                if done % (1024 * IO_BYTES) == 0:
                    self.check_source(info)
                    self.status(info["number"], "verifying_existing", verified_bytes=done)
        if digest.hexdigest() != info["sha256"]:
            raise RuntimeError(f"shard {info['number']}: complete SHA256 mismatch; preserving files")

    def assemble(self, info):
        number = info["number"]
        complete = self.directory / info["filename"]
        if complete.exists():
            self.verify_file(complete, info)
        else:
            self.check_source(info)
            if shutil.disk_usage(self.directory).free < info["total_bytes"] + RESERVE_BYTES:
                raise RuntimeError("insufficient space before assembly")
            assembling = self.directory / (info["filename"] + ".assembling")
            # Restart only our incomplete assembly; suffix and source stay intact.
            digest = hashlib.sha256()
            copied = 0
            self.status(number, "assembling", copied_bytes=0, total_bytes=info["total_bytes"])
            with assembling.open("wb") as output:
                for part in (Path(info["prefix_path"]), self.directory / (info["filename"] + ".suffix")):
                    with part.open("rb") as stream:
                        while data := stream.read(IO_BYTES):
                            output.write(data)
                            digest.update(data)
                            copied += len(data)
                            if copied % (1024 * IO_BYTES) == 0:
                                self.check_source(info)
                                self.status(number, "assembling", copied_bytes=copied, total_bytes=info["total_bytes"])
                output.flush()
                os.fsync(output.fileno())
            self.check_source(info)
            if copied != info["total_bytes"] or digest.hexdigest() != info["sha256"]:
                raise RuntimeError(f"shard {number}: assembled SHA256/size mismatch; preserving all recovery files")
            os.link(assembling, complete)
            assembling.unlink()
            sync_directory(self.directory)
        self.check_source(info)
        self.status(number, "verified", sha256=info["sha256"], complete_path=str(complete))
        target = self.root / info["filename"]
        try:
            # Hard-link publication is atomic, adds no second disk copy and
            # fails if the user's downloader already completed the same path.
            os.link(complete, target)
            sync_directory(self.root)
            self.status(number, "published", target=str(target), sha256=info["sha256"])
        except FileExistsError:
            self.status(number, "target_already_exists_preserved", target=str(target), verified_copy=str(complete))

    def run(self):
        records = self.prepare()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.download, info) for info in records]
            try:
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            except BaseException:
                self.stop.set()
                raise
        for info in records:
            self.assemble(info)
        self.log(None, "all_complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--recovery-dir", type=Path)
    parser.add_argument("--proxy", default="http://127.0.0.1:7897")
    args = parser.parse_args()
    root = args.source_root.resolve()
    directory = args.recovery_dir or root / f".v41-recovery-{REVISION[:12]}"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    recovery = Recovery(root, directory.resolve(), args.proxy)
    with (directory / "recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(directory / "process.json", {"pid": os.getpid(), "started": timestamp()})
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: recovery.stop.set())
        try:
            recovery.run()
        except BaseException as error:
            # Network exception strings may contain redirected signed URLs.
            message = str(error) if isinstance(error, RuntimeError) else None
            recovery.status(None, "failed", error_type=type(error).__name__, reason=message)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
