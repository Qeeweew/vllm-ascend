# SPDX-License-Identifier: Apache-2.0
"""T-driven fused CR1 consumer benchmark, including whole-selector postprocess.

Run through run_indexer_v41_fused.py with an isolated OPP. This produces
measurements, never automatic acceptance. Run uncontented for performance
claims; --correctness-only is suitable during a shared-device diagnostic.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from benchmark_indexer_v41 import measure, summarize
from benchmark_indexer_v41_fused import loaded_candidate_libraries, memory_snapshot, sha256
from test_indexer_v41 import build_metadata, device_case, make_case
from test_indexer_v41_fused import assert_query_rows, row_candidates, selector_call
from test_indexer_v41_paged_unique import unique_candidates

from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerV41Ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--package-manifest", required=True, type=Path)
    parser.add_argument("--opp-root", required=True, type=Path)
    parser.add_argument("--queries", nargs="+", type=int, required=True)
    parser.add_argument("--contexts", nargs="+", type=int, required=True)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--candidate-mode", type=int, choices=(2, 4), default=2)
    parser.add_argument("--candidate-input", choices=("synthetic", "source"), default="synthetic")
    parser.add_argument("--save-candidates", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--unroll", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--shared-device", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Keep previous evidence; choose a fresh output path")
    if len(args.queries) != len(args.contexts) or any(value < 0 for value in args.queries + args.contexts):
        raise ValueError("One nonnegative context length per nonnegative query count is required")
    manifest = json.loads(args.package_manifest.read_text())
    if sha256(Path(manifest["package"])) != manifest["sha256"]:
        raise ValueError("Package fingerprint mismatch")
    torch.set_num_threads(8)
    torch.npu.set_device(args.device)
    report = dict(
        status="incomplete",
        acceptance="pending",
        package=manifest,
        queries=args.queries,
        contexts=args.contexts,
        padding=args.padding,
        shared_device=args.shared_device,
        candidate_mode=args.candidate_mode,
        candidate_input=args.candidate_input,
        scope="CR1 consumer only; CR1 source / CR2 prefill pending",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    case = make_case(1, args.queries, args.contexts, padding=args.padding)
    device = device_case(case)
    candidates_cpu = (
        unique_candidates(case, holes=False) if args.candidate_mode == 4 else row_candidates(case, adversarial=False)
    )
    candidates = candidates_cpu.npu()
    info = build_metadata(AscendIndexerV41Ops(1, "consumer"), device, max_q=max(args.queries), max_k=max(args.contexts))
    if args.candidate_input == "source":
        source = AscendIndexerV41Ops(1, "source")
        _, candidates = source.select_topk(
            device["q"],
            device["w"],
            device["qs"],
            device["k"],
            device["ks"],
            info,
        )
        candidates_cpu = candidates.cpu()
        for row in candidates_cpu[:, 0]:
            valid = row[row >= 0]
            if valid.unique().numel() != valid.numel():
                raise AssertionError("Actual source produced duplicate valid candidate IDs")
    if args.save_candidates is not None:
        if args.save_candidates.exists():
            raise ValueError("Preserve previous candidate fixture; choose a fresh path")
        torch.save(
            dict(queries=args.queries, contexts=args.contexts, padding=args.padding, candidates=candidates_cpu),
            args.save_candidates,
        )
        report["candidate_fixture"] = dict(
            path=str(args.save_candidates.resolve()), sha256=sha256(args.save_candidates)
        )
    report["cache_layout"] = dict(key_stride=list(device["k"].stride()), scale_stride=list(device["ks"].stride()))
    offsets = torch.zeros((case["q"].shape[0], 1), dtype=torch.int32, device="npu")
    runs = dict(
        fused=partial(selector_call, device, info, candidates, candidate_mode=args.candidate_mode),
        native=partial(selector_call, device, info, candidates, legacy_offset=offsets),
    )
    if args.candidate_mode == 4:
        runs["paged_candidate"] = partial(selector_call, device, info, candidates)
    report["correctness"] = {}
    for name, run in runs.items():
        assert_query_rows(case, candidates_cpu, run()[0].cpu())
        report["correctness"][name] = "passed"
        save()
    report["loaded_libraries"] = loaded_candidate_libraries(args.opp_root.resolve())
    row_workspace_bytes = manifest.get("row_workspace_bytes", 156704)
    report["user_workspace_formula"] = (
        manifest.get("workspace_formula", f"{row_workspace_bytes}*T + 64*producer_cores")
        + "; CANN fixed workspace additional"
    )
    if not args.correctness_only:
        report["memory"] = {name: memory_snapshot(run) for name, run in runs.items()}
        graphs, owners, samples = {}, {}, {name: [] for name in runs}
        for name, run in runs.items():
            for _ in range(3):
                run()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(args.unroll):
                    output = run()
            graphs[name], owners[name] = graph, output
        for iteration in range(args.rounds):
            order = list(runs) if iteration % 2 == 0 else list(reversed(runs))
            for name in order:
                samples[name].append(measure(graphs[name].replay, args.unroll))
        report["statistics"] = {name: summarize(values) for name, values in samples.items()}
        for values in report["statistics"].values():
            values["tokens_per_second"] = sum(args.queries) * 1e6 / values["median_us"]
        report["raw_samples"] = samples
    report["status"] = "correctness_passed" if args.correctness_only else "measurements_complete"
    save()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
