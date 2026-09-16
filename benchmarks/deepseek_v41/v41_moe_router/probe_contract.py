# SPDX-License-Identifier: Apache-2.0
"""Small, baseline-only NPU probe; run only in a root-approved NPU window."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    values = torch.tensor(
        [
            -104.0,
            -100.0,
            -90.0,
            -80.0,
            -40.0,
            -20.0,
            -17.0,
            -16.0,
            -15.0,
            -8.0,
            -1.0,
            0.0,
            1.0,
            19.999998,
            20.0,
            20.000002,
            40.0,
            90.0,
            1.0e30,
        ],
        dtype=torch.float32,
    )
    device_values = values.npu()
    baseline = torch.nn.functional.softplus(device_values)
    naive = torch.where(device_values > 20.0, device_values, device_values.clamp_max(20.0).exp().add(1.0).log())
    results = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "softplus": {
            "inputs": values.tolist(),
            "baseline": baseline.cpu().tolist(),
            "exp_add_log": naive.cpu().tolist(),
            "baseline_bits": baseline.view(torch.int32).cpu().tolist(),
            "sqrt_baseline": baseline.sqrt().cpu().tolist(),
        },
        "topk": [],
    }
    for experts, top_k in ((128, 3), (384, 6)):
        for rows in (1, 4, 128):
            for kind in ("all_equal", "cutoff_tie", "block_tie", "near_tie"):
                scores = torch.zeros((rows, experts), dtype=torch.float32)
                if kind == "cutoff_tie":
                    scores[:, : top_k - 1] = torch.arange(top_k, 1, -1).float()
                elif kind == "block_tie":
                    scores[:, 31:65] = 1.0
                    scores[:, -4:] = 1.0
                elif kind == "near_tie":
                    scores[:] = 1.0
                    scores[:, ::3] = torch.nextafter(torch.tensor(1.0), torch.tensor(float("inf")))
                device_scores = scores.npu()
                repeats = [torch.topk(device_scores, top_k, sorted=True).indices.cpu().tolist() for _ in range(3)]
                expected = torch.argsort(scores, descending=True, stable=True)[:, :top_k].tolist()
                results["topk"].append(
                    {
                        "experts": experts,
                        "k": top_k,
                        "rows": rows,
                        "kind": kind,
                        "ids": repeats[0],
                        "repeated_identical": repeats.count(repeats[0]) == len(repeats),
                        "ties_index_ascending": repeats[0] == expected,
                    }
                )
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
