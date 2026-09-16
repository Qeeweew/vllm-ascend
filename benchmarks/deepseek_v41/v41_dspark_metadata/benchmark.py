# SPDX-License-Identifier: Apache-2.0
"""Measure caller-owned AscendC scheduling against native AICPU metadata."""

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--unroll", type=int, default=64)
    args = parser.parse_args()
    torch.npu.set_device(0)
    rows = []
    for batch in (1, 2, 4, 8, 16, 32, 64, 128):
        tokens = 5 * batch
        cu = torch.arange(0, tokens + 1, 5, dtype=torch.int32, device="npu")
        lengths = torch.full((batch,), 4096, dtype=torch.int32, device="npu")
        spans = torch.full((tokens, 1), 133, dtype=torch.int32, device="npu")
        schedule = torch.empty(1024, dtype=torch.int32, device="npu")

        def candidate(cu=cu, lengths=lengths, spans=spans, schedule=schedule):
            torch.ops._C_ascend.v41_dspark_metadata(cu, lengths, spans, schedule)

        def baseline(cu=cu, lengths=lengths, spans=spans, batch=batch, tokens=tokens):
            return torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
                num_heads_q=8,
                num_heads_kv=1,
                head_dim=512,
                cu_seqlens_q=cu,
                seqused_ori_kv=lengths,
                ori_topk_length=spans,
                batch_size=batch,
                max_seqlen_q=tokens,
                max_seqlen_ori_kv=4096,
                max_seqlen_cmp_kv=0,
                ori_topk=256,
                cmp_topk=0,
                cmp_ratio=0,
                ori_mask_mode=0,
                cmp_mask_mode=3,
                ori_win_left=132,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_BBND",
                has_ori_kv=True,
                has_cmp_kv=False,
            )

        for _ in range(5):
            candidate()
            baseline()
        torch.npu.synchronize()
        elapsed = {"candidate_host_us": [], "aicpu_host_us": [], "candidate_graph_us": []}
        for _ in range(args.repeats):
            for name, function in (("candidate_host_us", candidate), ("aicpu_host_us", baseline)):
                started = perf_counter()
                for _ in range(args.unroll):
                    function()
                torch.npu.synchronize()
                elapsed[name].append((perf_counter() - started) * 1e6 / args.unroll)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(args.unroll):
                candidate()
        for _ in range(3):
            graph.replay()
        for _ in range(args.repeats):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            elapsed["candidate_graph_us"].append(start.elapsed_time(end) * 1000 / args.unroll)
        rows.append(
            {
                "batch": batch,
                "tokens": tokens,
                "span": 133,
                "samples": elapsed,
                "median_us": {k: statistics.median(v) for k, v in elapsed.items()},
            }
        )
        args.output.write_text(
            json.dumps(
                {
                    "rows": rows,
                    "unroll": args.unroll,
                    "scope": "metadata only; host latency includes enqueue and final sync",
                },
                indent=2,
            )
        )
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
