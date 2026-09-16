# SPDX-License-Identifier: Apache-2.0
"""Whole-router alternating graph timing and single-call msprof op workload."""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.fused_moe.router.fused_topk_router import select_deepseek_v4_vision_experts
from vllm_ascend.ops.v41_moe_router import v41_moe_router

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("router_cases", ROOT / "tests/e2e/single_node/ops/test_v41_moe_router.py")
CASES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CASES)


def percentile95(values):
    return sorted(values)[max(0, (95 * len(values) + 99) // 100 - 1)]


def summarize(rounds):
    medians = [statistics.median(values) for values in rounds]
    all_values = [value for values in rounds for value in values]
    return {
        "rounds_us": rounds,
        "median_us": statistics.median(all_values),
        "p95_us": percentile95(all_values),
        "round_median_spread": (max(medians) - min(medians)) / statistics.median(medians),
    }


def measure(graph, samples, unroll):
    values = []
    for _ in range(samples):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / unroll)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 5, 16, 64, 128, 1024])
    parser.add_argument("--experts", type=int, nargs="+", default=[384, 128])
    parser.add_argument("--modes", nargs="+", default=["hash", "mixed", "dynamic"])
    parser.add_argument("--profile", choices=["fused", "baseline"])
    parser.add_argument("--unroll", type=int, default=32)
    parser.add_argument("--scaling", type=float, default=1.5)
    args = parser.parse_args()
    if args.unroll < 1:
        parser.error("--unroll must be positive")
    torch.npu.set_device(args.device)
    results = {
        "accepted": False,
        "cases": [],
        "gate": {
            "rounds": 5,
            "samples": 20,
            "unroll": args.unroll,
            "routed_scaling_factor": args.scaling,
            "priority_median_and_p95_speedup": 1 / 0.9,
            "other_max_latency_ratio": 1.03,
            "max_round_spread": 0.03,
            "weight_rtol": 2e-6,
            "weight_atol": 2e-7,
            "ids": "exact",
        },
    }
    for experts in args.experts:
        k = 6 if experts == 384 else 3
        for rows in args.rows:
            for mode in args.modes:
                inputs = CASES.device_args(CASES.case(rows, experts, k, mode))
                weights = torch.empty((rows, k), device="npu")
                ids = torch.empty((rows, k), dtype=torch.int32, device="npu")

                def fused(inputs=inputs, weights=weights, ids=ids, k=k, scaling=args.scaling):
                    v41_moe_router(*inputs, weights, ids, k, True, scaling)

                def baseline(inputs=inputs, k=k, scaling=args.scaling):
                    x, token, mask, table, text_bias, image_bias = inputs
                    return select_deepseek_v4_vision_experts(
                        x, token, table, image_bias, text_bias, k, True, scaling, image_token_mask=mask
                    )

                fused()
                expected_weights, expected_ids = baseline()
                torch.testing.assert_close(ids.cpu().long(), expected_ids.cpu(), rtol=0, atol=0)
                torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=2e-6, atol=2e-7)
                if args.profile:
                    operation = fused if args.profile == "fused" else baseline
                    for _ in range(5):
                        operation()
                    torch.npu.synchronize()
                    operation()
                    torch.npu.synchronize()
                    results["cases"].append({"experts": experts, "rows": rows, "mode": mode, "profile": args.profile})
                    continue
                callables = {"fused": fused, "baseline": baseline}
                graphs, outputs, peaks = {}, {}, {}
                for name, operation in callables.items():
                    for _ in range(5):
                        operation()
                    torch.npu.synchronize()
                    torch.npu.reset_peak_memory_stats()
                    before = torch.npu.memory_allocated()
                    outputs[name] = operation()
                    torch.npu.synchronize()
                    peaks[name] = torch.npu.max_memory_allocated() - before
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        for _ in range(args.unroll):
                            outputs[name] = operation()
                    graphs[name] = graph
                measurements = {"fused": [], "baseline": []}
                for round_id in range(5):
                    order = ("baseline", "fused") if round_id % 2 == 0 else ("fused", "baseline")
                    for name in order:
                        measurements[name].append(measure(graphs[name], 20, args.unroll))
                summary = {name: summarize(values) for name, values in measurements.items()}
                required_ratio = 0.9 if rows in (1, 4, 128) else 1.03
                ratios = {
                    metric: summary["fused"][metric] / summary["baseline"][metric] for metric in ("median_us", "p95_us")
                }
                passed = all(ratio <= required_ratio for ratio in ratios.values()) and all(
                    value["round_median_spread"] <= 0.03 for value in summary.values()
                )
                results["cases"].append(
                    {
                        "experts": experts,
                        "rows": rows,
                        "mode": mode,
                        "timing": summary,
                        "latency_ratios": ratios,
                        "single_call_peak_extra_bytes": peaks,
                        "passed": passed,
                    }
                )
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                del graphs, outputs, inputs, weights, ids
    results["accepted"] = not args.profile and all(case["passed"] for case in results["cases"])
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    if not args.profile and not results["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
