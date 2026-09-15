# SPDX-License-Identifier: Apache-2.0
"""Small-file recovery tests: publication must require whole-file integrity."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def recovery_module():
    path = Path(__file__).resolve().parents[3] / "benchmarks/deepseek_v41/recover_engram_shards.py"
    spec = importlib.util.spec_from_file_location("engram_shard_recovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def case(tmp_path, recovery_module):
    source = tmp_path / "model"
    prefix_dir = source / "._____temp"
    prefix_dir.mkdir(parents=True)
    recovery_dir = source / ".recovery"
    recovery_dir.mkdir()
    filename = "model-00047-of-00048.safetensors"
    prefix = prefix_dir / filename
    prefix.write_bytes(b"prefix")
    content = b"prefix-and-suffix"
    info = {
        "number": 47,
        "filename": filename,
        "prefix_path": str(prefix),
        "prefix_identity": recovery_module.identity(prefix),
        "prefix_bytes": 6,
        "total_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    recovery = recovery_module.Recovery(source, recovery_dir, "http://127.0.0.1:7897")
    return recovery, info, content


class Response:
    def __init__(self, content, status=206, content_range=None):
        self.status_code = status
        self.headers = {"Content-Range": content_range, "Content-Length": str(len(content))}
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def iter_content(self, _):
        yield self.content


def fake_session(monkeypatch, module, response, calls):
    class Session:
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return response

    monkeypatch.setattr(module.requests, "Session", Session)


def test_resume_appends_only_missing_suffix(case, recovery_module, monkeypatch):
    recovery, info, content = case
    suffix = recovery.directory / (info["filename"] + ".suffix")
    suffix.write_bytes(content[6:9])
    calls = []
    fake_session(
        monkeypatch,
        recovery_module,
        Response(content[9:], content_range=f"bytes 9-{len(content) - 1}/{len(content)}"),
        calls,
    )
    recovery.download(info)
    assert suffix.read_bytes() == content[6:]
    assert len(calls) == 1
    assert calls[0][1]["headers"]["Range"] == f"bytes=9-{len(content) - 1}"
    assert recovery_module.REVISION in calls[0][0]
    recovery.check_source(info)


def test_complete_suffix_is_never_downloaded_again(case, recovery_module, monkeypatch):
    recovery, info, content = case
    suffix = recovery.directory / (info["filename"] + ".suffix")
    suffix.write_bytes(content[6:])
    calls = []
    fake_session(monkeypatch, recovery_module, Response(b"bad"), calls)
    recovery.download(info)
    assert calls == []
    assert suffix.read_bytes() == content[6:]


def test_bounded_ranges_continue_at_exact_byte_offset(case, recovery_module, monkeypatch):
    recovery, info, content = case
    monkeypatch.setattr(recovery_module, "RANGE_BYTES", 3)
    calls = []

    class Session:
        def get(self, url, **kwargs):
            start, end = map(int, kwargs["headers"]["Range"].removeprefix("bytes=").split("-"))
            calls.append((start, end))
            return Response(content[start : end + 1], content_range=f"bytes {start}-{end}/{len(content)}")

    monkeypatch.setattr(recovery_module.requests, "Session", Session)
    recovery.download(info)
    assert calls == [(6, 8), (9, 11), (12, 14), (15, 16)]
    assert (recovery.directory / (info["filename"] + ".suffix")).read_bytes() == content[6:]


@pytest.mark.parametrize("status,range_value", [(200, None), (206, "bytes 0-9/10"), (404, None)])
def test_wrong_response_never_appends(case, recovery_module, monkeypatch, status, range_value):
    recovery, info, content = case
    suffix = recovery.directory / (info["filename"] + ".suffix")
    suffix.write_bytes(content[6:9])
    fake_session(monkeypatch, recovery_module, Response(b"evil", status, range_value), [])
    with pytest.raises(RuntimeError, match="Range"):
        recovery.download(info)
    assert suffix.read_bytes() == content[6:9]


def test_assembly_verifies_and_publishes_without_changing_prefix(case, recovery_module):
    recovery, info, content = case
    (recovery.directory / (info["filename"] + ".suffix")).write_bytes(content[6:])
    recovery.assemble(info)
    complete = recovery.directory / info["filename"]
    target = recovery.root / info["filename"]
    assert complete.read_bytes() == target.read_bytes() == content
    assert complete.stat().st_ino == target.stat().st_ino
    assert recovery_module.identity(Path(info["prefix_path"])) == info["prefix_identity"]
    assert not complete.with_suffix(complete.suffix + ".assembling").exists()
    recovery.assemble(info)
    assert target.read_bytes() == content


def test_existing_user_target_is_preserved(case):
    recovery, info, content = case
    (recovery.directory / (info["filename"] + ".suffix")).write_bytes(content[6:])
    target = recovery.root / info["filename"]
    target.write_bytes(b"user-owned-file")
    recovery.assemble(info)
    assert target.read_bytes() == b"user-owned-file"
    assert (recovery.directory / info["filename"]).read_bytes() == content


def test_corrupt_suffix_never_publishes(case):
    recovery, info, content = case
    suffix = recovery.directory / (info["filename"] + ".suffix")
    suffix.write_bytes(b"x" * (len(content) - 6))
    with pytest.raises(RuntimeError, match="SHA256"):
        recovery.assemble(info)
    assert not (recovery.root / info["filename"]).exists()
    assert not (recovery.directory / info["filename"]).exists()
    assert suffix.read_bytes() == b"x" * (len(content) - 6)


def test_changed_original_prefix_stops_before_network(case, recovery_module, monkeypatch):
    recovery, info, _ = case
    Path(info["prefix_path"]).write_bytes(b"prefix-has-grown")
    calls = []
    fake_session(monkeypatch, recovery_module, Response(b""), calls)
    with pytest.raises(RuntimeError, match="original prefix changed"):
        recovery.download(info)
    assert calls == []


def test_disk_space_failure_precedes_assembly(case, recovery_module, monkeypatch):
    recovery, info, content = case
    (recovery.directory / (info["filename"] + ".suffix")).write_bytes(content[6:])
    monkeypatch.setattr(recovery_module, "RESERVE_BYTES", 2**100)
    with pytest.raises(RuntimeError, match="insufficient space"):
        recovery.assemble(info)
    assert not (recovery.root / info["filename"]).exists()
    assert not (recovery.directory / (info["filename"] + ".assembling")).exists()
