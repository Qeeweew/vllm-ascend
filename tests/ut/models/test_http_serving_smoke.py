# SPDX-License-Identifier: Apache-2.0
"""CPU protocol and owned-process cleanup checks; no listening socket or NPU."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "benchmarks/deepseek_v41/smoke_http_serving.py"
SPEC = importlib.util.spec_from_file_location("smoke_http_serving", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_sse_comments_multiline_frames_and_done():
    lines = [b": keep-alive\n", b"\n", b'data: {"choices":\n', b"data: []}\n", b"\n", b"data: [DONE]\n", b"\n"]
    assert list(smoke.sse_events(lines)) == ['{"choices":\n[]}', "[DONE]"]


def test_sse_truncation_and_bound_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="incomplete"):
        list(smoke.sse_events([b"data: truncated\n"]))
    monkeypatch.setattr(smoke, "MAX_RESPONSE_BYTES", 4)
    with pytest.raises(ValueError, match="bounded"):
        list(smoke.sse_events([b"data: too large\n"]))


def test_completion_requires_real_token_and_finite_probability_evidence():
    result = {
        "status": 200,
        "body": {
            "usage": {"completion_tokens": 4},
            "choices": [
                {"token_ids": [1, 2, 3, 4], "finish_reason": "length", "logprobs": {"token_logprobs": [-1.0] * 4}}
            ],
        },
    }
    assert smoke.validate_completion(result)["token_ids"] == [1, 2, 3, 4]
    result["body"]["choices"][0]["logprobs"]["token_logprobs"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        smoke.validate_completion(result)


def test_zero_parent_exit_does_not_hide_internal_force_kill_or_resource_leaks(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "Process manager: force killing remaining process EngineCore pid 123\n"
        "resource_tracker: There appear to be 8 leaked semaphore objects\n"
    )
    result = smoke.audit_cleanup_log(log)
    assert not result["clean"] and len(result["internal_forced_lines"]) == 1
    assert len(result["resource_leak_warnings"]) == 1
    log.write_text("[shutdown] Executor: all workers exited gracefully\n")
    assert smoke.audit_cleanup_log(log)["clean"]


def test_cleanup_releases_its_child_without_signaling_an_unrelated_session():
    code = (
        "import signal,time; signal.signal(signal.SIGINT, lambda *_: exit(0)); "
        "print('ready',flush=True); time.sleep(60)"
    )
    owned = subprocess.Popen([sys.executable, "-c", code], start_new_session=True, stdout=subprocess.PIPE, text=True)
    unrelated = subprocess.Popen(
        [sys.executable, "-c", code], start_new_session=True, stdout=subprocess.PIPE, text=True
    )
    try:
        assert owned.stdout.readline().strip() == unrelated.stdout.readline().strip() == "ready"
        result = smoke.cleanup_server(owned, 3)
        assert result["clean"] and result["signals"] == ["parent_SIGINT"]
        assert unrelated.poll() is None
    finally:
        for process in (owned, unrelated):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
            process.stdout.close()
