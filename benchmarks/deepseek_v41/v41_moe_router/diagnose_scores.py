# SPDX-License-Identifier: Apache-2.0
"""Numerical diagnosis through hash rows, bypassing dynamic selection/normalization."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.v41_moe_router import v41_moe_router


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for experts, k in ((128, 3), (384, 6)):
        for pattern in ("random", "negative_tail", "threshold"):
            generator = torch.Generator().manual_seed(71)
            if pattern == "random":
                row = torch.randn(experts, generator=generator) * 4
            elif pattern == "negative_tail":
                row = torch.linspace(-104, -8, experts)
            else:
                row = torch.linspace(19.99, 20.01, experts)
            rows = (experts + k - 1) // k
            logits = row.repeat(rows, 1).npu()
            tokens = torch.arange(rows, dtype=torch.int64, device="npu")
            mask = torch.zeros(rows, dtype=torch.bool, device="npu")
            table = torch.arange(rows * k, dtype=torch.int32, device="npu").remainder(experts).reshape(rows, k)
            bias = torch.zeros(experts, device="npu")
            weights = torch.empty((rows, k), device="npu")
            ids = torch.empty((rows, k), dtype=torch.int32, device="npu")
            v41_moe_router(logits, tokens, mask, table, None, bias, weights, ids, k, False)
            expected = torch.nn.functional.softplus(logits[0]).sqrt().cpu()
            actual = weights.flatten()[:experts].cpu()
            ulps = (actual.view(torch.int32).long() - expected.view(torch.int32).long()).abs()
            largest = torch.topk(ulps, min(16, experts)).indices
            results.append(
                {
                    "experts": experts,
                    "pattern": pattern,
                    "exact": int(ulps.eq(0).sum()),
                    "max_ulp": int(ulps.max()),
                    "max_abs": float((actual - expected).abs().max()),
                    "worst": [
                        {
                            "logit": float(row[i]),
                            "fused": float(actual[i]),
                            "baseline": float(expected[i]),
                            "ulp": int(ulps[i]),
                        }
                        for i in largest
                    ],
                }
            )
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
