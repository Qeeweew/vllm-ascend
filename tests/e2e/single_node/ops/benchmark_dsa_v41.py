# SPDX-License-Identifier: Apache-2.0
"""Native V4.1 attention versus batched gather + CANN BMM/softmax.

Device timings exclude scheduler construction and cache writes for BOTH
paths. The baseline includes sparse gather and shares a single softmax.
No acceptance speedup is assumed: raw rounds and ratios are saved.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from test_dsa_v41 import check, device_case, forward, make_case, metadata, reference

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401

from vllm_ascend.ops.dsa_v41 import AscendDSAV41Ops  # noqa: E402


def baseline(case, ratio):
    offsets = torch.arange(128, device="npu", dtype=torch.int32)

    def gather(cache, table, positions):
        safe = positions.clamp_min(0)
        physical = table.gather(1, (safe // cache.shape[1]).long()) * cache.shape[1] + safe % cache.shape[1]
        return cache.reshape(-1, 512).index_select(0, physical.flatten().long()).reshape(*positions.shape, 512)

    def run():
        positions = case["lengths"][:, None] - 128 + offsets
        keys = gather(case["swa"], case["swa_bt"], positions)
        valid = positions >= 0
        if ratio:
            ids = case["indices"][:, 0]
            keys = torch.cat((keys, gather(case["cmp"], case["cmp_bt"], ids)), dim=1)
            valid = torch.cat((valid, ids >= 0), dim=1)
        score = torch.bmm(case["q"], keys.transpose(1, 2), out_dtype=torch.float32) * 512**-0.5
        score = score.masked_fill(~valid[:, None], -torch.inf)
        sink = case["sinks"][None, :, None].expand(score.shape[0], -1, -1)
        probabilities = torch.cat((score, sink), dim=-1).softmax(-1)[:, :, :-1].contiguous().bfloat16()
        return torch.bmm(probabilities, keys)

    return run


def measure(run, count, flush=None, unroll=1):
    times = []
    for _ in range(count):
        if flush is not None:
            flush.bitwise_xor_(1)
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / unroll)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--graph-unroll", type=int, default=32)
    parser.add_argument("--cold", action="store_true", help="Read/write 512 MiB outside each timed interval")
    args = parser.parse_args()
    # Flush before every measured attention, not only the first of a graph.
    if args.cold:
        args.graph_unroll = 1
    torch.npu.set_device(2)
    torch.set_num_threads(8)
    flush = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device="npu") if args.cold else None
    results = []
    for ratio in (0, 1, 2):
        for batch in (1, 4, 8):
            case = make_case(ratio, [1] * batch, [2049 + i for i in range(batch)], sparse=True)
            expected, _ = reference(case, ratio)
            device, ops = device_case(case), AscendDSAV41Ops(ratio)
            meta = metadata(ops, device)

            def native(ops=ops, device=device, meta=meta):
                return forward(ops, device, meta, lse=False)[0]

            composed = baseline(device, ratio)
            check(native(), expected)
            check(composed(), expected)
            for mode in ("eager", "graph"):
                runs = [native, composed]
                graphs = []
                if mode == "graph":
                    for run in runs:
                        for _ in range(3):
                            run()
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph):
                            for _ in range(args.graph_unroll):
                                run()
                        graphs.append(graph)
                    runs = [graph.replay for graph in graphs]
                for run in runs:
                    for _ in range(args.warmup):
                        run()
                torch.npu.synchronize()
                rounds = [[], []]
                for repetition in range(5):
                    for index in (repetition % 2, 1 - repetition % 2):
                        rounds[index].append(
                            measure(runs[index], args.samples, flush, args.graph_unroll if mode == "graph" else 1)
                        )
                medians = [statistics.median(sum(times, [])) for times in rounds]
                p95 = [sorted(sum(times, []))[int(0.95 * len(sum(times, [])))] for times in rounds]
                conservative_speedup = min(map(statistics.median, rounds[1])) / max(map(statistics.median, rounds[0]))
                row = dict(
                    ratio=ratio,
                    batch=batch,
                    mode=mode,
                    graph_unroll=args.graph_unroll if mode == "graph" else 1,
                    native_us=medians[0],
                    composed_us=medians[1],
                    native_p95_us=p95[0],
                    composed_p95_us=p95[1],
                    speedup=medians[1] / medians[0],
                    conservative_round_speedup=conservative_speedup,
                    rounds_us=rounds,
                )
                results.append(row)
                print({k: v for k, v in row.items() if k != "rounds_us"}, flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(dict(device="910B3 device 2", cold=args.cold, results=results), indent=2) + "\n"
                )


if __name__ == "__main__":
    main()
