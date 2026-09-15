# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks; fake published files never invoke the real converter."""

import json
import runpy
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def setup_watcher(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[3] / "benchmarks/deepseek_v41/watch_conversion_tail.py"))
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    names = [f"model-{number:05d}-of-00048.safetensors" for number in range(1, 49)]
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict(enumerate(names))}))
    manifest = {"shards": dict.fromkeys(names[:46], {}), "complete": False}

    def save():
        (output / "conversion_manifest.json").write_text(json.dumps(manifest))

    save()
    calls = []

    def converter(command, *, check):
        assert check is True
        assert command[-3:] == ["--allow-incomplete", "--threads", "8"]
        calls.append(command)
        for name in names[-2:]:
            if (source / name).is_file():
                manifest["shards"][name] = {}
        manifest["complete"] = len(manifest["shards"]) == 48
        if manifest["complete"]:
            (output / "config.json").write_text("{}")
            (output / "model.safetensors.index.json").write_text("{}")
        save()

    watcher = module["ConversionTailWatcher"](
        source,
        output,
        "python",
        "converter.py",
        expected_files=[(name, 4) for name in names[-2:]],
        run=converter,
    )
    return watcher, source, output, names, manifest, calls, save


def test_no_final_file_never_calls_converter(setup_watcher):
    watcher, source, _, names, _, calls, _ = setup_watcher
    (source / (names[-2] + ".partial")).write_bytes(b"abcd")
    for _ in range(3):
        assert watcher.step() is False
    assert not calls


def test_each_published_shard_converts_once_then_completes(setup_watcher):
    watcher, source, _, names, _, calls, _ = setup_watcher
    (source / names[-2]).write_bytes(b"abcd")
    assert watcher.step() is False
    assert watcher.step() is False
    assert len(calls) == 1
    (source / names[-1]).write_bytes(b"abcd")
    assert watcher.step() is True
    assert watcher.step() is True
    assert len(calls) == 2


def test_both_published_together_require_one_serial_call(setup_watcher):
    watcher, source, _, names, _, calls, _ = setup_watcher
    for name in names[-2:]:
        (source / name).write_bytes(b"abcd")
    assert watcher.step() is True
    assert len(calls) == 1


def test_existing_complete_conversion_exits_without_call(setup_watcher):
    watcher, source, output, names, manifest, calls, save = setup_watcher
    for name in names[-2:]:
        (source / name).write_bytes(b"abcd")
        manifest["shards"][name] = {}
    manifest["complete"] = True
    save()
    for name in ("config.json", "model.safetensors.index.json"):
        (output / name).write_text("{}")
    assert watcher.step() is True
    assert not calls


@pytest.mark.parametrize("failure", ["converter", "no_progress", "wrong_size", "symlink"])
def test_failure_stops_without_retry(setup_watcher, failure):
    watcher, source, _, names, _, calls, _ = setup_watcher
    final = source / names[-2]
    if failure == "symlink":
        target = source / "download.partial"
        target.write_bytes(b"abcd")
        final.symlink_to(target)
    else:
        final.write_bytes(b"abc" if failure == "wrong_size" else b"abcd")

    def broken_converter(command, *, check):
        calls.append(command)
        if failure == "converter":
            raise subprocess.CalledProcessError(1, command)

    watcher.run = broken_converter
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        watcher.step()
    count = len(calls)
    with pytest.raises(ValueError):
        watcher.step()
    assert len(calls) == count


def test_incomplete_output_publication_is_not_success(setup_watcher):
    watcher, source, _, names, manifest, calls, save = setup_watcher
    for name in names[-2:]:
        (source / name).write_bytes(b"abcd")
        manifest["shards"][name] = {}
    manifest["complete"] = True
    save()
    with pytest.raises(ValueError, match="config/index"):
        watcher.step()
    assert not calls
