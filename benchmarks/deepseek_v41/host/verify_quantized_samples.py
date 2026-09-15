# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

torch.set_num_threads(8)
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
parser.add_argument("--output", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32"))
parser.add_argument("--max-shards", type=int, default=3)
args = parser.parse_args()
src = args.source
dst = args.output
index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
manifest = json.loads((dst / "conversion_manifest.json").read_text())
results = []
for filename in list(manifest["shards"])[: args.max_shards]:
    with safe_open(dst / filename, framework="pt") as out:
        names = out.keys()
        experts = [k for k in names if k.endswith(".weight_packed")]
        dense = [k for k in names if k.endswith(".weight") and k.removesuffix(".weight") + ".scale" in index]
        selected = experts[:3] + experts[-3:] + dense[:3]
        for key in selected:
            stem = key.removesuffix(".weight_packed") if key.endswith(".weight_packed") else key.removesuffix(".weight")
            name = stem + ".weight"
            with (
                safe_open(src / index[name], framework="pt") as inp,
                safe_open(src / index[stem + ".scale"], framework="pt") as scales,
            ):
                w = inp.get_slice(name)[:32]
                s = scales.get_slice(stem + ".scale")[
                    : 32 if key.endswith(".weight_packed") or ".engram.embed" in key else 1
                ].float()
            if key.endswith(".weight_packed"):
                bits = w.view(torch.uint8).long()
                codes = torch.stack((bits & 15, bits >> 4), -1).flatten(-2)
                lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
                original = (lut[codes] * s.repeat_interleave(32, -1)).bfloat16().float()
                groups = original.unflatten(-1, (-1, 32))
                expected_scale = torch.empty(groups.shape[:-1], dtype=torch.bfloat16)
                for row in range(groups.shape[0]):
                    for group in range(groups.shape[1]):
                        vals = groups[row, group]
                        a, b = abs(float(vals.min())), abs(float(vals.max()))
                        value = a / 8 if a > b else (-b / 8 if b > a else -b / 7)
                        expected_scale[row, group] = value if value else torch.finfo(torch.float32).eps
                actual_scale = out.get_slice(stem + ".weight_scale")[:32]
                assert torch.equal(expected_scale, actual_scale), key
                packed = out.get_slice(key)[:32].long()
                q = torch.stack([((packed >> (4 * i)) & 15) - 8 for i in range(8)], -1).flatten(-2)
                expected_q = torch.round(groups / expected_scale.float().unsqueeze(-1)).clamp(-8, 7).flatten(-2)
                assert torch.equal(q, expected_q), key
                approx = q * actual_scale.float().repeat_interleave(32, -1)
                nrmse = float((approx - original).square().mean().sqrt() / original.square().mean().sqrt())
                results.append({"tensor": key, "rows": 32, "exact_rtn": True, "nrmse": nrmse})
            else:
                rowblock = 1 if ".engram.embed" in key else 32
                expected = (
                    w.float() * s.repeat_interleave(rowblock, 0).repeat_interleave(32, 1)[: w.shape[0], : w.shape[1]]
                ).bfloat16()
                assert torch.equal(out.get_slice(key)[:32], expected), key
                results.append({"tensor": key, "rows": 32, "exact_bf16": True})
print(json.dumps({"validated_shards": min(len(manifest["shards"]), args.max_shards), "samples": results}, indent=2))
