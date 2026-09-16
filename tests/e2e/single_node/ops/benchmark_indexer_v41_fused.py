# SPDX-License-Identifier: Apache-2.0
"""Whole-selector fused QLI gates, live controls, and incremental HBM.

The installed original package must remain preserved. Run with the separately
built candidate OPP selected for this process only, after correctness tests.
No command in this script installs a package or changes a device reservation.
"""

import argparse
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch_npu
from benchmark_indexer_v41 import dense_baseline, measure, summarize
from indexer_v41_candidate_reference import assert_candidate_selection, candidate_reference
from test_indexer_v41 import build_metadata, device_case, make_case, select
from test_indexer_v41_fused import selector_call

from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerV41Ops
from vllm_ascend.ops.indexer_v41_candidate import CandidateIndexerB1

USER_WORKSPACE_BYTES = 163840


def gates(statistics, frozen):
    fused, split, dense = (statistics[name] for name in ("fused", "split", "dense"))
    return dict(
        split_latency_gate=(
            fused["median_us"] <= 0.9 * split["median_us"] and fused["p95_us"] <= 0.9 * split["p95_us"]
        ),
        live_dense_gate=fused["median_us"] <= dense["median_us"] and fused["p95_us"] <= 1.05 * dense["p95_us"],
        frozen_dense_gate=(fused["median_us"] <= frozen["median_us"] and fused["p95_us"] <= 1.05 * frozen["p95_us"]),
        live_native_gate=(
            fused["median_us"] <= statistics["original_native"]["median_us"]
            and fused["p95_us"] <= statistics["original_native"]["p95_us"]
        ),
        noise_gate=max(s["spread"] for s in statistics.values()) <= 0.03,
    )


def memory_snapshot(run, persistent_bytes=0):
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    reserved_before = torch.npu.memory_reserved()
    torch.npu.reset_peak_memory_stats()
    output = run()
    torch.npu.synchronize()
    result = dict(
        persistent_bytes=persistent_bytes,
        incremental_peak_allocated_bytes=torch.npu.max_memory_allocated() - before + persistent_bytes,
        total_peak_allocated_bytes=torch.npu.max_memory_allocated(),
        total_peak_reserved_bytes=torch.npu.max_memory_reserved(),
        allocated_before_bytes=before,
        reserved_before_bytes=reserved_before,
    )
    del output
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loaded_candidate_libraries(opp_root):
    paths = {
        Path(line.split()[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/" in line and "libcust_op" in line
    }
    selected = [path for path in paths if path.is_relative_to(opp_root)]
    if not any(path.name == "libcust_opapi.so" for path in selected):
        raise AssertionError(f"Candidate ACLNN library was not mapped from {opp_root}: {paths}")
    return {str(path): sha256(path) for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--cann-api-workspace-bytes",
        type=int,
        required=True,
        help="GetLibApiWorkSpaceSize verified for this CANN/architecture; 16777216 on current 9.1/2201",
    )
    parser.add_argument("--lengths", nargs="+", type=int, default=[4097, 32771, 131075])
    parser.add_argument("--package", required=True, type=Path, help="Separately built .run package")
    parser.add_argument("--package-sha256", required=True, help="Expected fingerprint of the tested .run package")
    parser.add_argument("--opp-root", required=True, type=Path, help="Isolated installed candidate vendor directory")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output path; retain previous failures")
    args.opp_root = args.opp_root.resolve(strict=True)
    actual_sha = sha256(args.package)
    if actual_sha != args.package_sha256:
        raise ValueError("Package SHA256 mismatch")
    first_opp = Path(os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")[0]).resolve()
    if first_opp != args.opp_root:
        raise ValueError(f"Candidate OPP must be selected before bootstrap: {first_opp}")
    root = Path(__file__).parents[4]
    frozen_path = root / "benchmarks/deepseek_v41/indexer_v41/candidate_frozen_baseline.json"
    frozen = json.loads(frozen_path.read_text())
    source_dir = root / "csrc/attention/quant_lightning_indexer_v2"
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.suffix in {".cpp", ".h"}
    }
    report = dict(
        status="incomplete",
        package_sha256=actual_sha,
        package_path=str(args.package.resolve()),
        candidate_opp=str(args.opp_root),
        opp_sha256={
            str(path.relative_to(args.opp_root)): sha256(path)
            for path in sorted(args.opp_root.rglob("*"))
            if path.is_file()
        },
        source_sha256=hashes,
        cases=[],
        user_workspace_bytes=USER_WORKSPACE_BYTES,
        cann_api_workspace_bytes=args.cann_api_workspace_bytes,
        total_native_workspace_bytes=USER_WORKSPACE_BYTES + args.cann_api_workspace_bytes,
        live_original_control="Same v3 implementation with explicit zero output offset; generic dispatch",
        environment=dict(
            device_index=args.device,
            torch=str(torch.__version__),
            torch_npu=str(torch_npu.__version__),
            rounds=3,
            samples=12,
            candidate_unroll=64,
            dense_unroll=4,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.set_num_threads(8)
    torch.npu.set_device(args.device)
    torch.npu.matmul.allow_hf32 = False
    report["environment"]["device"] = torch.npu.get_device_name(args.device)
    for length in args.lengths:
        report["active_case"] = dict(length=length, stage="correctness")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        case = make_case(1, [1], [length])
        device = device_case(case)
        source = AscendIndexerV41Ops(1, "source")
        info = build_metadata(source, device, max_q=1, max_k=length)
        _, candidates = select(source, device, info)
        split = CandidateIndexerB1(length, device["q"].device)
        zero_offset = torch.zeros((1, 1), dtype=torch.int32, device=device["q"].device)
        runs = dict(
            fused=partial(selector_call, device, info, candidates),
            original_native=partial(selector_call, device, info, candidates, legacy_offset=zero_offset),
            split=partial(split, device["q"], device["k"], device["w"], device["qs"], device["ks"], info, candidates),
            dense=dense_baseline(device, "consumer", candidates),
        )
        reference = candidate_reference(case, candidates.cpu())
        for run in runs.values():
            assert_candidate_selection(run()[0].cpu(), reference)
        report["loaded_libraries"] = loaded_candidate_libraries(args.opp_root)
        report["active_case"]["stage"] = "memory_and_latency"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        persistent = sum(t.numel() * t.element_size() for t in vars(split).values() if isinstance(t, torch.Tensor))
        memory = {name: memory_snapshot(run, persistent if name == "split" else 0) for name, run in runs.items()}
        graphs, owners, samples = {}, {}, {}
        for name, run in runs.items():
            for _ in range(3):
                run()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(4 if name == "dense" else 64):
                    output = run()
            graphs[name], owners[name], samples[name] = graph, output, []
        for round_id in range(3):
            order = list(runs) if round_id % 2 == 0 else list(reversed(runs))
            for name in order:
                samples[name].append(measure(graphs[name].replay, 4 if name == "dense" else 64))
        statistics = {name: summarize(values) for name, values in samples.items()}
        baseline = next(row["dense"] for row in frozen["cases"] if row["batch"] == 1 and row["length"] == length)
        acceptance = gates(statistics, baseline)
        peak = memory["fused"]["incremental_peak_allocated_bytes"]
        acceptance["memory_gate"] = USER_WORKSPACE_BYTES <= 256 * 1024 and all(
            peak < memory[name]["incremental_peak_allocated_bytes"] for name in ("split", "original_native")
        )
        record = dict(length=length, statistics=statistics, memory=memory, gates=acceptance)
        report["cases"].append(record)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(dict(length=length, gates=acceptance, fused=statistics["fused"])), flush=True)
        del graphs, owners, runs, graph, output, run, split, source
        torch.npu.synchronize()
        torch.npu.empty_cache()
    report.pop("active_case", None)
    report["status"] = "passed" if all(all(r["gates"].values()) for r in report["cases"]) else "failed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
