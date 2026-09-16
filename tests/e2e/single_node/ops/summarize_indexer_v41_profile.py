# SPDX-License-Identifier: Apache-2.0
"""Summarize every core in one msprof op PipeUtilization capture.

Counters overlap and cannot be summed as wall time. This tool preserves raw
rows and fingerprints; it does not classify a performance run as accepted.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def summarize_profile(directory):
    pipe = directory / "PipeUtilization.csv"
    basic = directory / "OpBasicInfo.csv"
    with pipe.open() as stream:
        rows = list(csv.DictReader(stream))
    with basic.open() as stream:
        operations = list(csv.DictReader(stream))
    if not rows or len(operations) != 1:
        raise ValueError("Expected one profiled operation and nonempty per-core counters")
    identities = [(row["block_id"], row["sub_block_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate core rows; split multiple captures before summarizing")
    groups = {}
    counter_warnings = []
    wall_us = float(operations[0]["Task Duration(us)"])
    for engine in ("cube0", "vector0", "vector1"):
        selected = [row for row in rows if row["sub_block_id"] == engine]
        if not selected:
            raise ValueError(f"Missing {engine} counters")
        metrics = {}
        for name in rows[0]:
            if not name or name in ("block_id", "sub_block_id"):
                continue
            values = [float(row[name]) for row in selected if row.get(name) not in (None, "", "NA")]
            if values:
                metrics[name] = dict(min=min(values), mean=sum(values) / len(values), max=max(values))
        groups[engine] = dict(cores=len(selected), metrics=metrics)
        time_metric = "aic_time(us)" if engine == "cube0" else "aiv_time(us)"
        # Counter time and task time use different measurement boundaries. A
        # generous 20% margin flags gross anomalies without interpreting small
        # boundary differences as broken counters. Preserve every raw row.
        if time_metric in metrics and metrics[time_metric]["max"] > wall_us * 1.2:
            counter_warnings.append(
                f"{engine} {time_metric} max={metrics[time_metric]['max']:.3f} exceeds "
                f"task duration {wall_us:.3f} us by more than 20%; "
                "do not interpret utilization ratios until a fresh capture confirms the counters"
            )
    return dict(
        scope="All captured cores; PipeUtilization counters overlap",
        directory=str(directory.resolve()),
        sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (pipe, basic)},
        operation=operations[0],
        groups=groups,
        counter_warnings=counter_warnings,
        per_core=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous evidence; choose a fresh output path")
    report = summarize_profile(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"operation": report["operation"], "groups": report["groups"]}), flush=True)


if __name__ == "__main__":
    main()
