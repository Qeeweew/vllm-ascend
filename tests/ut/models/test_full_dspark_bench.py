# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the real DSpark serving benchmark controller."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks/deepseek_v41"


def load_script():
    spec = importlib.util.spec_from_file_location("bench_full_dspark", BENCHMARKS / "bench_full_dspark.py")
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(BENCHMARKS), *sys.path]):
        spec.loader.exec_module(module)
    return module


bench = load_script()


def arguments(tmp_path):
    return SimpleNamespace(
        checkpoint=tmp_path / "full",
        output=tmp_path,
        port=18141,
        dspark_tokens=5,
        disable_dspark=False,
        profile_after_bench=False,
        profile_warmup=False,
        native_decode=False,
        fused_rope=False,
        fused_cache_store=False,
        fused_router=False,
        chunk_size=128,
        kv_gib=2,
    )


def test_real_cli_accepts_dspark_graph_and_benchmark_commands(tmp_path):
    import torch
    from vllm.benchmarks.serve import add_cli_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    args = arguments(tmp_path)
    server = bench.serve_values(args, "owned-model")
    assert bench.validate_server_args(server)["npu_initialized"] is False
    spec = json.loads(server[server.index("--speculative-config") + 1])
    assert spec == {"method": "dspark", "num_speculative_tokens": 5}
    compilation = json.loads(server[server.index("--compilation-config") + 1])
    assert compilation["cudagraph_capture_sizes"] == [6, 12, 18, 24, 30, 36, 42, 48]
    assert "--worker-cls" not in server and "--enforce-eager" not in server
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    command = bench.bench_command(args, "owned-model", 1024, 8, 16, "case.json")
    parsed = parser.parse_args(command[5:])
    assert parsed.num_prompts == 16 and parsed.max_concurrency == 8
    assert parsed.random_input_len == 1024 and parsed.random_output_len == 128
    assert parsed.ready_check_timeout_sec == 0 and parsed.num_warmups == 0
    assert parsed.ignore_eos and parsed.extra_body == {"temperature": 0}
    assert not torch.npu.is_initialized()


def test_prometheus_filter_and_per_position_aggregation():
    raw = """
# TYPE vllm:spec_decode_num_drafts counter
vllm:spec_decode_num_drafts_total{model_name="owned",engine="0"} 10
vllm:spec_decode_num_drafts_total{model_name="foreign",engine="0"} 999
# TYPE vllm:spec_decode_num_accepted_tokens_per_pos counter
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="owned",engine="0",position="0"} 8
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="owned",engine="1",position="0"} 2
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="owned",engine="0",position="1"} 3
"""
    assert bench.parse_metrics(raw, "owned") == {
        bench.DRAFTS: 10,
        bench.PER_POSITION + ":0": 10,
        bench.PER_POSITION + ":1": 3,
    }


def test_ar_comparison_has_no_speculation_and_separate_profiler(tmp_path):
    args = arguments(tmp_path)
    args.disable_dspark = args.profile_after_bench = True
    server = bench.serve_values(args, "owned-ar")
    assert "--speculative-config" not in server
    assert bench.validate_server_args(server)["npu_initialized"] is False
    compilation = json.loads(server[server.index("--compilation-config") + 1])
    assert compilation["cudagraph_capture_sizes"] == list(range(1, 9))
    profiler = json.loads(server[server.index("--profiler-config") + 1])
    assert profiler["profiler"] == "torch" and profiler["max_iterations"] == 24
    command = bench.bench_command(args, "owned-ar", 128, 1, 16, "case.json")
    assert "--profile" not in command
    assert bench.metric_delta({}, {bench.SUCCESS: 16}, 16, dspark_enabled=False)["speculative_decoding"] is False
    with pytest.raises(ValueError, match="Speculative decoding ran"):
        bench.metric_delta({}, {bench.SUCCESS: 16, bench.DRAFTS: 1}, 16, dspark_enabled=False)


def test_metric_delta_excludes_warmup_and_uses_draft_token_denominator():
    before = {bench.SUCCESS: 8, bench.DRAFTS: 100, bench.DRAFT_TOKENS: 500, bench.ACCEPTED: 250}
    after = {bench.SUCCESS: 24, bench.DRAFTS: 110, bench.DRAFT_TOKENS: 550, bench.ACCEPTED: 270}
    result = bench.metric_delta(before, after, 16)
    assert result["draft_token_acceptance_rate"] == 0.4
    assert result["accepted_tokens_per_draft"] == 2
    assert result["mean_acceptance_length_including_bonus"] == 3


@pytest.mark.parametrize(
    "after",
    [
        {bench.SUCCESS: 16},  # An ordinary autoregressive server must fail.
        {bench.SUCCESS: 17, bench.DRAFTS: 1, bench.DRAFT_TOKENS: 5, bench.ACCEPTED: 1},
        {bench.SUCCESS: 16, bench.DRAFTS: 1, bench.DRAFT_TOKENS: 5, bench.ACCEPTED: 6},
        {bench.SUCCESS: 16, bench.DRAFTS: -1, bench.DRAFT_TOKENS: 5, bench.ACCEPTED: 1},
    ],
)
def test_missing_reset_or_contaminated_metrics_fail(after):
    with pytest.raises(ValueError):
        bench.metric_delta({}, after, 16)
