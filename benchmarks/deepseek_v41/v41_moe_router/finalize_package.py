# SPDX-License-Identifier: Apache-2.0
"""Install a completed package in an isolated artifact and audit its provenance."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def only(paths):
    paths = list(paths)
    if len(paths) != 1:
        raise RuntimeError(f"Expected one artifact, found {paths}")
    return paths[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--build-log", type=Path, required=True)
    args = parser.parse_args()
    source_file = args.source / "moe/v41_moe_router/op_kernel/v41_moe_router.cpp"
    snapshot_file = args.snapshot / "moe/v41_moe_router/op_kernel/v41_moe_router.cpp"
    copied_file = args.snapshot / "build/binary/ascend910b/src/v41_moe_router/op_kernel/v41_moe_router.cpp"
    hashes = [digest(path) for path in (source_file, snapshot_file, copied_file)]
    if len(set(hashes)) != 1:
        raise RuntimeError("Raw, snapshot and copied sources differ; refuse stale binary installation")
    kernel = only((args.snapshot / "build/binary/ascend910b/bin/v41_moe_router").glob("*.o"))
    package = only((args.snapshot / "build").glob("cann-ops-transformer-v41_router_candidate_*.run"))
    args.artifact.mkdir(parents=True, exist_ok=True)
    copied_package = args.artifact / package.name
    shutil.copy2(package, copied_package)
    with (args.artifact / "install.log").open("w") as log:
        subprocess.run(
            ["bash", str(copied_package), "--quiet", f"--install-path={args.artifact / 'opp'}"],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    installed_source = only((args.artifact / "opp").rglob("v41_moe_router.cpp"))
    installed_kernel = only((args.artifact / "opp").rglob(kernel.name))
    if digest(installed_source) != hashes[0] or digest(installed_kernel) != digest(kernel):
        raise RuntimeError("Installed source or kernel differs from build output")
    provenance = {
        "package": str(copied_package),
        "package_sha256": digest(copied_package),
        "kernel_sha256": digest(kernel),
        "source_sha256": hashes[0],
        "raw_source": str(source_file),
        "snapshot_source": str(snapshot_file),
        "copied_source": str(copied_file),
        "installed_source": str(installed_source),
        "built_kernel": str(kernel),
        "installed_kernel": str(installed_kernel),
        "build_log": str(args.build_log),
        "build_log_sha256": digest(args.build_log),
        "native_accuracy_accepted": False,
        "native_performance_accepted": False,
    }
    (args.artifact / "package.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
