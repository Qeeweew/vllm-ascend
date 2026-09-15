# SPDX-License-Identifier: Apache-2.0
"""Reproducible Engram post-GEMM benchmark with graph and eager baselines."""

import argparse
import hashlib
import json
import platform
import statistics
from functools import partial
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.engram_gate import engram_gate, engram_gate_reference
from vllm_ascend.utils import bootstrap_custom_op_env


def composed_baseline(hidden, kv, q_weight, k_weight, token_mask):
    """Device-only equivalent for finite benchmark inputs.

    torch.copysign currently falls back to CPU on NPU, so use the equivalent
    comparison/select for the nonzero dot products in these random workloads.
    The independent correctness oracle retains the exact model copysign.
    """
    h = hidden.float()
    key = kv[:, :20480].float().reshape(-1, 4, 5120)
    value = kv[:, 20480:].float().unsqueeze(1)
    weight = q_weight.float() * k_weight.float()
    rstd = torch.rsqrt(h.square().mean(-1) + 1e-20) * torch.rsqrt(key.square().mean(-1) + 1e-20)
    dot = ((h * weight) * key).sum(-1) * rstd * 5120**-0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.sigmoid(torch.where(dot < 0, -magnitude, magnitude))
    result = (h + gate.unsqueeze(-1) * value).bfloat16()
    return torch.where(token_mask[:, None, None], result, hidden)


def latency(fn, graph_mode, repeats):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    calls_per_replay = 32 if graph_mode else 1
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(calls_per_replay):
                fn()
        fn = graph.replay
    # Ten invocations per event pair amortize timestamp precision. Keep the
    # sample distribution, rather than reporting a single optimistic minimum.
    samples = []
    for _ in range(repeats):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(10):
            fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / (10 * calls_per_replay))
    return {"median_us": statistics.median(samples), "p95_us": sorted(samples)[int(0.95 * (len(samples) - 1))]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    device = torch.device(f"npu:{args.device}")
    kernel_dir = (
        Path(vllm_ascend.__file__).parent
        / "_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/engram_gate"
    )
    kernel_fingerprints = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in kernel_dir.glob("*.o")}
    if not kernel_fingerprints:
        raise RuntimeError(f"cannot fingerprint installed EngramGate kernel in {kernel_dir}")
    torch.manual_seed(41)
    rows = []
    for tokens in [1, 2, 8, 16, 32, 64, 256, 1024]:
        inputs = (
            torch.randn(tokens, 4, 5120, dtype=torch.bfloat16, device=device),
            torch.randn(tokens, 25600, dtype=torch.bfloat16, device=device),
            torch.randn(4, 5120, dtype=torch.bfloat16, device=device),
            torch.randn(4, 5120, dtype=torch.bfloat16, device=device),
            torch.ones(tokens, dtype=torch.bool, device=device),
        )
        output = torch.empty_like(inputs[0])
        expected = engram_gate_reference(*(tensor.cpu() for tensor in inputs)).float()
        actual = engram_gate(*inputs, output=output).cpu().float()
        nrmse = float((actual - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-20))
        accuracy_passed = nrmse < 2e-4 and bool(torch.isclose(actual, expected, rtol=0.008, atol=0.002).all())
        for graph_mode in [False, True]:
            custom = latency(partial(engram_gate, *inputs, output=output), graph_mode, args.repeats)
            baseline = latency(partial(composed_baseline, *inputs), graph_mode, args.repeats)
            speedup = baseline["median_us"] / custom["median_us"]
            # Launch-sensitive decode acceptance. Prefill has a separate
            # throughput gate; neither allows a regression vs the reference.
            budget_us = 25 if tokens <= 16 else 40 if tokens <= 64 else 250
            passed = (accuracy_passed and speedup >= 2 and custom["median_us"] <= budget_us) if graph_mode else None
            row = {
                "tokens": tokens,
                "graph": graph_mode,
                "custom": custom,
                "reference": baseline,
                "speedup": speedup,
                "nrmse": nrmse,
                "accuracy_passed": accuracy_passed,
                "budget_us": budget_us,
                "passed": passed,
            }
            print(json.dumps(row), flush=True)
            rows.append(row)
    report = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "machine": platform.machine(),
        "device": str(device),
        "kernel_sha256": kernel_fingerprints,
        "samples": args.repeats,
        "calls_per_graph": 32,
        "timed_replays_per_sample": 10,
        "warmup_calls": 5,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.enforce and not all(row["passed"] for row in rows if row["graph"]):
        raise SystemExit("Engram gate performance acceptance failed; see raw report")


if __name__ == "__main__":
    main()
