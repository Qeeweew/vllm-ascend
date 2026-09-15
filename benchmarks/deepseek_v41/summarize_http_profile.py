# SPDX-License-Identifier: Apache-2.0
"""Summarize exported TP8 timelines; kernel sums are not wall-clock latency."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize_rank(directory):
    trace = directory / "ASCEND_PROFILER_OUTPUT/trace_view.json"
    kernels = directory / "ASCEND_PROFILER_OUTPUT/kernel_details.csv"
    events = json.loads(trace.read_text())
    if isinstance(events, dict):
        events = events["traceEvents"]
    counts = Counter(event.get("name") for event in events if event.get("ph") == "X")
    with kernels.open() as stream:
        rows = list(csv.DictReader(stream))
    if not events or not rows:
        raise ValueError(f"Empty exported trace or kernels: {directory}")
    by_type = defaultdict(lambda: {"count": 0, "duration_sum_us": 0.0})
    for row in rows:
        group = by_type[row["Type"]]
        group["count"] += 1
        group["duration_sum_us"] += float(row["Duration(us)"])
    return {
        "directory": str(directory.resolve()),
        "trace_events": len(events),
        "kernel_rows": len(rows),
        "duration_sum_us": sum(value["duration_sum_us"] for value in by_type.values()),
        "kernel_types": dict(sorted(by_type.items(), key=lambda item: -item[1]["duration_sum_us"])),
        "coverage": {
            "graph_model_execute": counts["MODEL_EXECUTE"],
            "graph_execute_api": counts["AscendCL@aclmdlRIExecuteAsync"],
            "native_w4a16": counts["fused_moe_small_bs_w4a16_bf16_8"],
            "engram_gate": counts["EngramGate"],
            "compressor": counts["CompressorV41"],
            "cpu_h2d_copy": counts["acl_memcpy_host_to_device"],
            "cpu_d2h_copy": counts["acl_memcpy_device_to_host"],
            "hccl_allreduce_api": counts["HcclAllreduce"],
            "vision_fused_infer_types": sorted(name for name in by_type if "InferAttention" in name),
        },
    }


def summarize(root):
    ranks = {}
    for rank in range(8):
        directories = list(root.glob(f"*tp{rank}_*rank{rank}_*_ascend_pt"))
        if len(directories) != 1:
            raise ValueError(f"Expected exactly one trace directory for rank {rank}")
        ranks[str(rank)] = summarize_rank(directories[0])
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            artifacts.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest})
    return {
        "scope": "three_real_language_layers_small_synthetic_engram_warm_http_requests",
        "artifact_root": str(root.resolve()),
        "interpretation": (
            "Durations are summed profiler observations and may overlap across streams. "
            "They are not request latency, throughput, or full-model performance gates. "
            "Copy counts include all model metadata and cannot isolate Engram transfers."
        ),
        "ranks": ranks,
        "artifacts": artifacts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.profile_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"ranks": len(result["ranks"]), "artifacts": len(result["artifacts"])}))


if __name__ == "__main__":
    main()
