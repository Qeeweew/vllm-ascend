# SPDX-License-Identifier: Apache-2.0
"""Run under each OPP package to audit unaffected CR2/B8/B32 selectors."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from benchmark_indexer_v41 import measure, summarize
from benchmark_indexer_v41_fused import loaded_candidate_libraries
from test_indexer_v41 import build_metadata, check_outputs, device_case, make_case, select

from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerV41Ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--opp-root", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4097, 32771, 131075])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous results; choose a new output")
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    report = dict(status="incomplete", opp_root=str(args.opp_root.resolve()), cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    for ratio, batch, mode in ((2, 1, "off"), (1, 8, "consumer"), (1, 32, "consumer")):
        for length in args.lengths:
            report["active_case"] = dict(ratio=ratio, batch=batch, length=length, mode=mode)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            case = make_case(ratio, [1] * batch, [length] * batch)
            device = device_case(case)
            ops = AscendIndexerV41Ops(ratio, mode)
            info = build_metadata(ops, device, max_q=1, max_k=length)
            candidates = None
            if mode == "consumer":
                source = AscendIndexerV41Ops(1, "source")
                _, candidates = select(source, device, info)

            run = partial(select, ops, device, info, candidates)

            idx, blocks = run()
            check_outputs(
                case, ratio, mode, idx.cpu(), blocks.cpu(), candidates.cpu() if candidates is not None else None
            )
            report["loaded_libraries"] = loaded_candidate_libraries(args.opp_root.resolve())
            for _ in range(3):
                run()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(64):
                    output = run()
            stats = summarize([measure(graph.replay, 64) for _ in range(3)])
            record = dict(**report["active_case"], statistics=stats, noise_gate=stats["spread"] <= 0.03)
            if baseline:
                original = next(
                    row["statistics"]
                    for row in baseline["cases"]
                    if all(row[key] == record[key] for key in ("ratio", "batch", "length", "mode"))
                )
                record["latency_gate"] = all(stats[key] <= 1.03 * original[key] for key in ("median_us", "p95_us"))
            report["cases"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            del graph, output, run, idx, blocks, device, ops, info, candidates
            torch.npu.synchronize()
            torch.npu.empty_cache()
    report.pop("active_case", None)
    report["status"] = (
        "passed" if all(row["noise_gate"] and row.get("latency_gate", True) for row in report["cases"]) else "failed"
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
