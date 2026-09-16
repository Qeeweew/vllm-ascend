# SPDX-License-Identifier: Apache-2.0
"""One fused CR1 consumer invocation for msprof op, followed by CPU oracle.

Run through an isolated OPP runner. No source-selector invocation is included,
so the QLI kernel filter uniquely identifies this consumer. Synthetic candidate
blocks diagnose the pipeline; source/CR2 are separate pending workloads.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from benchmark_indexer_v41 import AscendIndexerV41Ops
from test_indexer_v41 import build_metadata, device_case, make_case
from test_indexer_v41_candidate import candidate_device_case
from test_indexer_v41_fused import assert_query_rows, row_candidates, selector_call
from test_indexer_v41_paged_unique import unique_candidates


def useful_qk_work(case, candidates):
    positions = 0
    per_query = []
    page_size = case["k"].shape[1]
    for request, length in enumerate(case["sk"].tolist()):
        begin, end = case["cu"][request : request + 2].tolist()
        for row in range(begin, end):
            visible = max(0, min(case["bt"].shape[1] * page_size, length - (end - begin) + row - begin + 1))
            blocks = candidates[row].unique().long()
            blocks = blocks[(blocks >= 0) & (blocks * 8 < visible)]
            pages = case["bt"][request, blocks * 8 // page_size]
            blocks = blocks[(pages >= 0) & (pages < case["k"].shape[0])]
            count = int((visible - blocks * 8).clamp(0, 8).sum())
            positions += count
            per_query.append(count)
    return dict(
        valid_positions=positions, valid_positions_per_query=per_query, useful_qk_flops=2 * 32 * 128 * positions
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=32771)
    parser.add_argument("--queries", nargs="+", type=int, default=[1])
    parser.add_argument("--contexts", nargs="+", type=int)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--candidate-mode", type=int, choices=(2, 4), default=2)
    parser.add_argument("--candidate-fixture", type=Path)
    parser.add_argument("--legacy", action="store_true", help="Force generic v3 with zero row offsets")
    args = parser.parse_args()
    contexts = args.contexts or [args.length] * len(args.queries)
    if len(contexts) != len(args.queries):
        parser.error("Provide one context length per request")
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    case = make_case(1, args.queries, contexts, padding=args.padding)
    device = candidate_device_case(case)
    candidates = (
        unique_candidates(case, holes=False) if args.candidate_mode == 4 else row_candidates(case, adversarial=False)
    )
    fixture_sha = None
    if args.candidate_fixture is not None:
        fixture = torch.load(args.candidate_fixture, map_location="cpu", weights_only=True)
        if (fixture["queries"], fixture["contexts"], fixture["padding"]) != (args.queries, contexts, args.padding):
            raise ValueError("Candidate fixture shape/context mismatch")
        candidates = fixture["candidates"]
        device = device_case(case)  # Exactly the benchmark cache layout.
        fixture_sha = hashlib.sha256(args.candidate_fixture.read_bytes()).hexdigest()
    info = build_metadata(AscendIndexerV41Ops(1, "consumer"), device, max_q=max(args.queries), max_k=max(contexts))
    offset = torch.zeros((case["q"].shape[0], 1), dtype=torch.int32, device="npu") if args.legacy else None
    actual, _ = selector_call(
        device,
        info,
        candidates.to(device["q"].device),
        legacy_offset=offset,
        candidate_mode=2 if args.legacy else args.candidate_mode,
    )
    assert_query_rows(case, candidates, actual.cpu())
    print(
        json.dumps(
            dict(
                status="correctness_passed",
                implementation="legacy_native" if args.legacy else "fused",
                candidate_mode=args.candidate_mode,
                candidate_fixture_sha256=fixture_sha,
                cache_layout=dict(key_stride=list(device["k"].stride()), scale_stride=list(device["ks"].stride())),
                queries=args.queries,
                contexts=contexts,
                padding=args.padding,
                **useful_qk_work(case, candidates),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
