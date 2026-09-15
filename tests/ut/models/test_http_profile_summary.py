# SPDX-License-Identifier: Apache-2.0
"""CPU checks for interpreting overlapping exported profiler records."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "benchmarks/deepseek_v41/summarize_http_profile.py"
SPEC = importlib.util.spec_from_file_location("v41_profile_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_duration_sums_preserve_overlapping_records_and_ignore_flow_events(tmp_path):
    output = tmp_path / "ASCEND_PROFILER_OUTPUT"
    output.mkdir()
    events = [
        {"ph": "X", "name": "MODEL_EXECUTE", "dur": 100},
        {"ph": "s", "name": "MODEL_EXECUTE"},
        {"ph": "X", "name": "EngramGate", "dur": 5},
    ]
    (output / "trace_view.json").write_text(json.dumps({"traceEvents": events}))
    (output / "kernel_details.csv").write_text("Type,Duration(us)\nA,80\nA,70\nB,5\n")
    summary = MODULE.summarize_rank(tmp_path)
    assert summary["duration_sum_us"] == 155
    assert summary["kernel_types"]["A"] == {"count": 2, "duration_sum_us": 150}
    assert summary["coverage"]["graph_model_execute"] == 1
    assert summary["coverage"]["engram_gate"] == 1
    assert not summary["coverage"]["vision_fused_infer_types"]


def test_missing_rank_cannot_pass_as_tp8_profile(tmp_path):
    with pytest.raises(ValueError, match="rank 0"):
        MODULE.summarize(tmp_path)
