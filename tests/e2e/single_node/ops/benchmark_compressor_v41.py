# SPDX-License-Identifier: Apache-2.0
"""Correct PyTorch baseline versus CompressorV41; GEMM is excluded explicitly."""

import argparse
import hashlib
import json
import math
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from test_compressor_v41 import make_case, run_reference

from vllm_ascend.ops.compressor_v41 import compressor_v41
from vllm_ascend.utils import bootstrap_custom_op_env


def torch_baseline(cpu_case, tensors, output, ratio):
    """Capture-safe baseline with known host segment metadata, no device reads."""
    raw, _, slots, _, _, weight, state = tensors
    boundaries = cpu_case[3].tolist()
    closed, boundary_rows, boundary_slots, tail_tokens, tail_slots = [], [], [], [], []
    for req, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start == end:
            continue
        first = int(cpu_case[1][start])
        if first % 2:
            boundary_rows.append(len(closed))
            boundary_slots.append(req * state.shape[1] + (first - 1) % state.shape[1])
        closed.extend(t for t in range(start, end) if int(cpu_case[1][t]) % 2)
        for token in range(max(start, end - state.shape[1]), end):
            tail_tokens.append(token)
            tail_slots.append(req * state.shape[1] + (first + token - start) % state.shape[1])

    # Specialize known host metadata once, outside timing/capture, and process
    # all requests together. This is a stronger baseline than a Python loop
    # dispatching separate softmax/norm/scatter kernels for each request.
    def indices(values):
        return torch.tensor(values, dtype=torch.int64, device=raw.device)

    closing = indices(closed)
    previous_rows = indices([max(0, token - 1) for token in closed])
    boundary_idx, boundary_prev = indices(boundary_rows), indices(boundary_slots)
    tail_idx, tail_dst = indices(tail_tokens), indices(tail_slots)

    def norm(x):
        rounded = x.bfloat16().float()
        return (rounded * (rounded.square().mean(-1, keepdim=True) + 1e-20).rsqrt() * weight.float()).bfloat16()

    def run():
        output.zero_()
        if ratio == 1:
            output.copy_(torch.where((slots >= 0)[:, None], norm(raw), 0))
            return
        if closed:
            current = raw.index_select(0, closing)
            previous = raw.index_select(0, previous_rows)
            if boundary_rows:
                previous.index_copy_(0, boundary_idx, state.view(-1, 1024).index_select(0, boundary_prev))
            pair = torch.stack((previous, current), dim=1)
            pooled = (pair[:, :, :512] * pair[:, :, 512:].softmax(dim=1)).sum(dim=1)
            output.index_copy_(0, closing, norm(pooled))
        if tail_tokens:
            state.view(-1, 1024).index_copy_(0, tail_dst, raw.index_select(0, tail_idx))

    return run


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def summarize(rounds):
    flat = [sample for group in rounds for sample in group]
    medians = [statistics.median(group) for group in rounds]
    return {
        "median_us": statistics.median(flat),
        "p95_us": percentile(flat, 0.95),
        "round_medians_us": medians,
        "round_median_spread_fraction": (max(medians) - min(medians)) / statistics.median(medians),
        "samples_us": rounds,
    }


def measure(fn, iterations, operations_per_call=1):
    pairs = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(iterations)]
    for start, end in pairs:
        start.record()
        fn()
        end.record()
    torch.npu.synchronize()
    return [start.elapsed_time(end) * 1000 / operations_per_call for start, end in pairs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--graph-unroll", type=int, default=32)
    parser.add_argument("--decode-parity", choices=["mixed", "even", "odd"], default="mixed")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 4096])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--source-sha", default="unrecorded")
    parser.add_argument(
        "--hardware-notes", default="unrecorded; record power/frequency/device isolation before acceptance"
    )
    args = parser.parse_args()
    if args.rounds < 5 or args.iterations < 20 or args.graph_unroll < 1:
        parser.error("at least 5 rounds and 20 iterations are required")
    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.npu.set_device(0)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "host": platform.node(),
        "source_sha": args.source_sha,
        "hardware_notes": args.hardware_notes,
        "device": torch.npu.get_device_name(0),
        "graph": args.graph,
        "operations_per_graph": args.graph_unroll if args.graph else 1,
        "decode_parity": args.decode_parity,
        "scope": "vector compression + state + RMSNorm; excludes GEMM and cache insertion",
        "baseline": "batched torch gather/softmax/RMSNorm/scatter with metadata precomputed outside timing",
        "gate": "median <= 1.03x baseline, p95 <= 1.05x baseline, equal-weight geometric speedup >= 1.10x",
        "cases": [],
    }
    repo = Path(__file__).resolve().parents[4]
    files = [Path(__file__).resolve(), repo / "vllm_ascend/ops/compressor_v41.py"]
    files.extend(sorted((repo / "csrc/attention/compressor_v41").glob("op_*/*.*")))
    report["source_sha256"] = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    for ratio in (1, 2):
        for tokens in args.tokens:
            modes = ["prefill", "decode"] if tokens <= 128 else ["prefill"]
            for mode in modes:
                lengths = [tokens] if mode == "prefill" else [1] * tokens
                starts = [0] if mode == "prefill" else [i % 2 for i in range(tokens)]
                if mode == "decode" and args.decode_parity != "mixed":
                    starts = [int(args.decode_parity == "odd")] * tokens
                cpu_case = make_case(ratio, lengths, starts, padding=3)
                expected, expected_state = run_reference(cpu_case, ratio)
                runs, outputs, buffers, graphs, captured_callables = {}, {}, {}, [], {}
                for name in ("torch_reference", "ascendc"):
                    tensors = [t.npu() for t in cpu_case]
                    output = torch.empty(expected.shape, dtype=torch.bfloat16, device="npu")
                    if name == "ascendc":

                        def run(tensors=tensors, output=output, ratio=ratio):
                            compressor_v41(*tensors, output, ratio)
                    else:
                        run = torch_baseline(cpu_case, tensors, output, ratio)
                    run()
                    torch.testing.assert_close(output.cpu(), expected, rtol=8e-3, atol=2e-3)
                    torch.testing.assert_close(tensors[-1].cpu(), expected_state, rtol=0, atol=0)
                    for _ in range(20):
                        run()
                    if args.graph:
                        # Closure-owned index tensors are graph inputs allocated
                        # before capture; retaining only graph/state/output does
                        # not keep those external allocations alive.
                        captured_callables[name] = run
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph):
                            for _ in range(args.graph_unroll):
                                run()
                        graphs.append(graph)
                        run = graph.replay
                        run()
                        torch.testing.assert_close(output.cpu(), expected, rtol=8e-3, atol=2e-3)
                        torch.testing.assert_close(tensors[-1].cpu(), expected_state, rtol=0, atol=0)
                    runs[name], outputs[name], buffers[name] = run, output, tensors
                samples = {name: [] for name in runs}
                for round_id in range(args.rounds):
                    order = list(runs) if round_id % 2 == 0 else list(reversed(runs))
                    for name in order:
                        samples[name].append(
                            measure(runs[name], args.iterations, args.graph_unroll if args.graph else 1)
                        )
                result = {"ratio": ratio, "tokens": tokens, "bucket": tokens + 3, "mode": mode}
                result.update({name: summarize(value) for name, value in samples.items()})
                baseline, candidate = result["torch_reference"], result["ascendc"]
                result["speedup"] = baseline["median_us"] / candidate["median_us"]
                result["per_case_gate_passed"] = (
                    candidate["median_us"] <= baseline["median_us"] * 1.03
                    and candidate["p95_us"] <= baseline["p95_us"] * 1.05
                )
                result["noise_gate_passed"] = (
                    max(baseline["round_median_spread_fraction"], candidate["round_median_spread_fraction"]) < 0.03
                )
                report["cases"].append(result)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"CR{ratio} {mode} T={tokens}: {candidate['median_us']:.3f} us, {result['speedup']:.2f}x",
                    flush=True,
                )
    report["geometric_speedup"] = math.exp(statistics.mean(math.log(case["speedup"]) for case in report["cases"]))
    report["performance_gate_passed"] = report["geometric_speedup"] >= 1.10 and all(
        case["per_case_gate_passed"] and case["noise_gate_passed"] for case in report["cases"]
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
