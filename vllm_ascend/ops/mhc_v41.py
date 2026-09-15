# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 delayed pre-mix using the existing AscendC HcPre v3 interface.

The incoming pre-mix collapses this sublayer's input. The pre-mix produced by
its projection belongs to the next sublayer, not this one. FP32 control weights
are retained; HcPre's Cube projection uses HF32, which must be included in model
quality validation (the CPU reference below intentionally does not emulate it).
"""

import torch


def mhc_pre_delayed(hidden, hc_fn, hc_scale, hc_base, pre_mix, norm_eps=1e-20, hc_eps=1e-6, iterations=20):
    if hidden.ndim != 3 or hidden.shape[1:] != (4, 5120) or hidden.dtype != torch.bfloat16:
        raise ValueError("V4.1 mHC input must be BF16 [tokens,4,5120]")
    for tensor, shape in ((hc_fn, (24, 20480)), (hc_scale, (3,)), (hc_base, (24,)), (pre_mix, (hidden.shape[0], 4))):
        if tensor.shape != shape or tensor.dtype != torch.float32 or tensor.device != hidden.device:
            raise ValueError("V4.1 mHC controls must be FP32 with matching shapes and device")
    if any(not t.is_contiguous() for t in (hidden, hc_fn, hc_scale, hc_base, pre_mix)):
        raise ValueError("V4.1 mHC inputs must be contiguous")
    return torch.ops._C_ascend.npu_hc_pre_v3(
        hidden,
        hc_fn,
        hc_scale,
        hc_base,
        pre_mix,
        hc_mult=4,
        hc_sinkhorn_iters=iterations,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
    )


def mhc_post(hidden, residual, post, comb):
    return torch.ops._C_ascend.npu_hc_post(
        hidden.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)
    ).squeeze(0)


def mhc_collapse(hidden, pre_mix):
    """Terminal collapse with the last FFN's pre-mix; there is no learned head."""
    return (hidden.float() * pre_mix.unsqueeze(-1)).sum(1).to(hidden.dtype)


def mhc_pre_delayed_reference(hidden, hc_fn, hc_scale, hc_base, pre_mix, norm_eps=1e-20, hc_eps=1e-6, iterations=20):
    """Strict FP32 model arithmetic, without HF32 truncation of control weights."""
    flat = hidden.float().flatten(1)
    projected = (flat @ hc_fn.t()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    pre = (projected[:, :4] * hc_scale[0] + hc_base[:4]).sigmoid() + hc_eps
    post = 2 * (projected[:, 4:8] * hc_scale[1] + hc_base[4:8]).sigmoid()
    comb = (projected[:, 8:] * hc_scale[2] + hc_base[8:]).reshape(-1, 4, 4).softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return mhc_collapse(hidden, pre_mix), post, comb, pre
