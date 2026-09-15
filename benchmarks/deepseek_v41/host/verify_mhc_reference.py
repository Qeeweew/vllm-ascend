# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()

torch.set_num_threads(8)
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
args = parser.parse_args()
root = args.source
index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
results = []
for layer in (0, 1, 2, 20, 39):
    params = []
    for field in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base"):
        name = f"layers.{layer}.{field}"
        with safe_open(root / index[name], framework="pt") as reader:
            params.append(reader.get_tensor(name))
    fn, scale, base = params
    for batch in (1, 16):
        gen = torch.Generator().manual_seed(414 + layer)
        x = torch.randn((batch, 4, 5120), generator=gen).bfloat16()
        incoming = torch.rand((batch, 4), generator=gen)
        flat = x.float().flatten(1)
        mix = (flat @ fn.t()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-20)
        pre = (mix[:, :4] * scale[0] + base[:4]).sigmoid() + 1e-6
        post = 2 * (mix[:, 4:8] * scale[1] + base[4:8]).sigmoid()
        comb = (mix[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4).softmax(-1) + 1e-6
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
        for _ in range(19):
            comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
            comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
        y = (incoming[:, :, None] * x.float()).sum(1).bfloat16()
        actual = torch.ops._C_ascend.npu_hc_pre_v3(
            x.npu(),
            fn.npu(),
            scale.npu(),
            base.npu(),
            incoming.npu(),
            hc_mult=4,
            hc_sinkhorn_iters=20,
            norm_eps=1e-20,
            hc_eps=1e-6,
        )
        entry = {"layer": layer, "batch": batch}
        for name, got, want in zip(("y", "post", "comb", "pre"), actual, (y, post, comb, pre)):
            delta = got.cpu().float() - want.float()
            entry[name] = {
                "nrmse": float(delta.square().mean().sqrt() / want.float().square().mean().sqrt()),
                "maxabs": float(delta.abs().max()),
            }
        results.append(entry)
print(json.dumps(results, indent=2))
