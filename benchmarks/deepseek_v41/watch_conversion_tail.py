# SPDX-License-Identifier: Apache-2.0
"""Resume the existing converter as the final two published shards arrive.

This process only reads exact final source paths and the conversion manifest.
It never scans download directories, opens partial files, or changes download
processes. Run once; successful completion or any conversion error ends it.
"""

import argparse
import json
import stat
import subprocess
import sys
import time
from pathlib import Path

# Sizes agree in the saved Hugging Face and ModelScope public file metadata.
EXPECTED_TAIL_FILES = (
    ("model-00047-of-00048.safetensors", 101_535_150_936),
    ("model-00048-of-00048.safetensors", 101_537_926_640),
)
TOTAL_SHARDS = 48


class ConversionTailWatcher:
    def __init__(self, source, output, python, converter, *, expected_files=EXPECTED_TAIL_FILES, run=subprocess.run):
        self.source, self.output = Path(source), Path(output)
        self.command = [
            str(python),
            str(converter),
            "--source",
            str(self.source),
            "--output",
            str(self.output),
            "--allow-incomplete",
            "--threads",
            "8",
        ]
        self.expected_files = dict(expected_files)
        self.run = run
        self.attempted = set()
        index = json.loads((self.source / "model.safetensors.index.json").read_text())
        self.all_shards = set(index["weight_map"].values())
        canonical = {f"model-{number:05d}-of-00048.safetensors" for number in range(1, TOTAL_SHARDS + 1)}
        if self.all_shards != canonical or set(self.expected_files) != {name for name, _ in EXPECTED_TAIL_FILES}:
            raise ValueError("Watcher requires the indexed 48-shard V4.1 checkpoint and tail shards 47/48")
        if self.source.resolve() == self.output.resolve() or self.source.resolve() in self.output.resolve().parents:
            raise ValueError("Conversion output must be separate from the source directory")
        self.manifest_path = self.output / "conversion_manifest.json"

    def manifest(self):
        manifest = json.loads(self.manifest_path.read_text())
        shards = set(manifest["shards"])
        if not shards <= self.all_shards or self.all_shards - shards - self.expected_files.keys():
            raise ValueError("Only tail shards 47/48 may be missing from the existing conversion manifest")
        if manifest.get("complete") and shards != self.all_shards:
            raise ValueError("Manifest claims completion without all 48 shards")
        return manifest

    def published_tail(self):
        ready = set()
        for name, expected in self.expected_files.items():
            path = self.source / name
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_size != expected:
                raise ValueError(f"Published source shard is not a regular file of {expected} bytes: {path}")
            ready.add(name)
        return ready

    def complete(self, manifest):
        if not manifest.get("complete"):
            return False
        if not all((self.output / name).is_file() for name in ("config.json", "model.safetensors.index.json")):
            raise ValueError("Complete manifest is missing published output config/index; inspect before resuming")
        return True

    def step(self):
        """Return True on completion; otherwise wait for a new published tail."""
        manifest = self.manifest()
        ready = self.published_tail()
        if self.complete(manifest):
            if ready != set(self.expected_files):
                raise ValueError("Complete conversion has missing final source tail files")
            return True
        pending = ready - manifest["shards"].keys()
        if not pending:
            if set(manifest["shards"]) == self.all_shards:
                raise ValueError("All shards exist but conversion is not finalized; inspect before resuming")
            return False
        if pending & self.attempted:
            raise ValueError("A previously attempted shard disappeared from the conversion manifest")
        self.attempted.update(pending)
        print(json.dumps({"event": "resume_conversion", "new_shards": sorted(pending)}), flush=True)
        # Synchronous execution preserves serialization and the converter's
        # own nonblocking .conversion.lock. A nonzero exit is never retried.
        self.run(self.command, check=True)
        updated = self.manifest()
        if not pending <= updated["shards"].keys():
            raise ValueError("Converter exited successfully without recording every newly published shard")
        if self.published_tail() != set(self.expected_files) and updated.get("complete"):
            raise ValueError("Complete conversion has missing final source tail files")
        return self.complete(updated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if not 0 < args.poll_seconds <= 60:
        parser.error("--poll-seconds must be in (0, 60]")
    converter = Path(__file__).resolve().parents[2] / "examples/quantization/convert_deepseek_v41.py"
    try:
        watcher = ConversionTailWatcher(args.source, args.output, args.python, converter)
        print(json.dumps({"event": "waiting_for_published_tail", "source": str(args.source)}), flush=True)
        while not watcher.step():
            time.sleep(args.poll_seconds)
        print(json.dumps({"event": "conversion_complete", "shards": TOTAL_SHARDS}), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"event": "conversion_stopped", "error": str(error)}), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
