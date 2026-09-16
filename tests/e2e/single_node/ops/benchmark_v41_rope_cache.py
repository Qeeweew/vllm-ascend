# SPDX-License-Identifier: Apache-2.0
"""Whole-chain rotary/cache benchmark with fixed numerical and latency gates."""

import argparse
import hashlib
import importlib
import json
import math
import statistics
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.ops.cache_v41 import write_index_cache_v41, write_main_cache_v41
from vllm_ascend.ops.v41_rope_cache import v41_index_cache_store, v41_main_cache_store, v41_rope
from vllm_ascend.utils import bootstrap_custom_op_env


def baseline_rope(x, positions, cos, sin, inverse=False):
    c, s = cos[positions], sin[positions]
    if x.ndim == 3:
        c, s = c[:, None], s[:, None]
    if inverse:
        s = -s
    pairs = x[..., -64:].float().unflatten(-1, (-1, 2))
    a, b = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2).bfloat16()
    return torch.cat((x[..., :-64], rotated), dim=-1)


def summarize(rounds):
    samples = [value for group in rounds for value in group]
    medians = [statistics.median(group) for group in rounds]
    return {
        "median_us": statistics.median(samples),
        "p95_us": sorted(samples)[math.ceil(len(samples) * 0.95) - 1],
        "round_medians_us": medians,
        "round_median_spread": (max(medians) - min(medians)) / statistics.median(medians),
        "samples_us": rounds,
    }


def measure(call, samples, unroll):
    events = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(samples)]
    for start, end in events:
        start.record()
        call()
        end.record()
    torch.npu.synchronize()
    return [start.elapsed_time(end) * 1000 / unroll for start, end in events]


def make_call(kind, native, x, positions, slots, cos, sin, ratio):
    if kind == "rope":
        output = torch.empty_like(x)
        if native:
            return lambda: v41_rope(x, positions, cos, sin, output), (output,)
        return lambda: baseline_rope(x, positions, cos, sin), ()
    blocks = max(1, (x.shape[0] + 31) // 32)
    width = x.shape[-1]
    if kind == "index":
        # Match the runner's shared raw pages, including the FP16 scale suffix.
        stride = 32 * (width + 2)
        raw = torch.zeros(blocks * stride, device=x.device, dtype=torch.uint8)
        cache = raw.view(torch.int8).as_strided((blocks, 32, 1, width), (stride, width, width, 1))
        scales = raw.view(torch.float16).as_strided((blocks, 32, 1), (stride // 2, 1, 1), 32 * width // 2)
    else:
        cache = torch.zeros((blocks, 32, 1, width), device=x.device, dtype=torch.bfloat16)
        scales = None
    if native:
        if kind == "index":
            return lambda: v41_index_cache_store(x, positions, slots, cos, sin, cache, scales, compress_ratio=ratio), (
                cache,
                scales,
            )
        return lambda: v41_main_cache_store(x, positions, slots, cos, sin, cache, compress_ratio=ratio), (cache,)

    def baseline():
        rotated = baseline_rope(x, positions // ratio * ratio, cos, sin)
        if kind == "index":
            quantized, scale = torch_npu.npu_dynamic_quant(rotated, dst_type=torch.int8)
            write_index_cache_v41(
                cache, scales, quantized, scale.half(), slots, positions=positions, compress_ratio=ratio
            )
            return cache, scales
        return write_main_cache_v41(cache, rotated, slots, positions=positions, compress_ratio=ratio)

    return baseline, (cache, scales) if kind == "index" else (cache,)


def run_case(args, kind, tokens, heads, width, ratio):
    torch.npu.reset_peak_memory_stats()
    generator = torch.Generator().manual_seed(4141)
    shape = (tokens, heads, width) if kind == "rope" and heads != 1 else (tokens, width)
    x = torch.randn(shape, generator=generator).bfloat16().npu()
    positions = (torch.arange(tokens, dtype=torch.int64) * ratio + ratio - 1).npu()
    slots = torch.arange(tokens, dtype=torch.int64).npu()
    angles = torch.randn((max(2048, tokens * ratio), 32), generator=generator)
    cos, sin = angles.cos().npu(), angles.sin().npu()
    calls, outputs, graphs, raw_calls = {}, {}, [], []
    for name, native in (("baseline", False), ("candidate", True)):
        call, destinations = make_call(kind, native, x, positions, slots, cos, sin, ratio)
        result = call()
        outputs[name] = destinations if kind != "rope" else (result,)
        raw_calls.append(call)
        for _ in range(10):
            call()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(args.unroll):
                call()
        graphs.append(graph)
        calls[name] = graph.replay
    for actual, expected in zip(outputs["candidate"], outputs["baseline"]):
        assert torch.equal(actual.cpu(), expected.cpu()), "Bitwise baseline gate failed before timing"
    rounds = {"baseline": [], "candidate": []}
    for round_index in range(5):
        for name in ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline"):
            for _ in range(3):
                calls[name]()
            torch.npu.synchronize()
            rounds[name].append(measure(calls[name], args.samples, args.unroll))
    baseline, candidate = (summarize(rounds[name]) for name in ("baseline", "candidate"))
    limit = 0.90 if tokens in (1, 4, 128) else 1.03
    passed = all(candidate[key] <= baseline[key] * limit for key in ("median_us", "p95_us"))
    passed &= baseline["round_median_spread"] <= 0.03 and candidate["round_median_spread"] <= 0.03
    return {
        "kind": kind,
        "tokens": tokens,
        "heads": heads,
        "width": width,
        "compress_ratio": ratio,
        "baseline": baseline,
        "candidate": candidate,
        "latency_ratio_limit": limit,
        "passed": passed,
        "peak_allocated_bytes": torch.npu.max_memory_allocated(),
        "peak_reserved_bytes": torch.npu.max_memory_reserved(),
        "user_workspace_bytes": 0,
        "cann_workspace_bytes": args.cann_workspace_bytes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", nargs="+", choices=("rope", "main", "index"), default=["rope", "main", "index"])
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 4, 5, 16, 64, 128, 1024])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--unroll", type=int, default=32)
    parser.add_argument("--cann-workspace-bytes", type=int, required=True)
    parser.add_argument("--artifact-run", type=Path, required=True)
    parser.add_argument("--artifact-sha256", required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.samples < 20
        or args.unroll < 1
        or min(args.tokens) < 1
        or args.cann_workspace_bytes < 0
    ):
        parser.error("new output, >=20 samples, positive tokens and unroll required")
    bootstrap_custom_op_env(include_vendor_lib=True)
    if hashlib.sha256(args.artifact_run.read_bytes()).hexdigest() != args.artifact_sha256:
        raise ValueError("Artifact SHA256 does not match the supplied full-build package")
    extension = importlib.import_module("vllm_ascend.vllm_ascend_C")
    torch.npu.set_device(0)
    report = {
        "status": "running",
        "scope": "complete replaced RoPE or RoPE/quantization/cache-store chain; GEMM/RMSNorm excluded",
        "graph_unroll": args.unroll,
        "rounds": 5,
        "cases": [],
        "extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
        "artifact_run": str(args.artifact_run.resolve()),
        "artifact_sha256": args.artifact_sha256,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for kind in args.kind:
        shapes = [(1, 128), (1, 512), (8, 512), (32, 128)] if kind == "rope" else [(1, 128 if kind == "index" else 512)]
        for heads, width in shapes:
            for ratio in (1,) if kind == "rope" else (1, 2):
                for tokens in args.tokens:
                    report["active_case"] = [kind, tokens, heads, width, ratio]
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    report["cases"].append(run_case(args, kind, tokens, heads, width, ratio))
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "passed" if all(case["passed"] for case in report["cases"]) else "failed_gate"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
