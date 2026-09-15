# SPDX-License-Identifier: Apache-2.0
"""Independent post-projection Engram gate for V4.1 (no GEMM or collectives)."""

import math

import torch


def _validate(hidden, kv, q_weight, k_weight, token_mask, output, eps):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if hidden.ndim != 3 or hidden.shape[1:] != (4, 5120):
        raise ValueError("hidden must have shape [T, 4, 5120]")
    tokens = hidden.shape[0]
    expected = (
        (hidden, (tokens, 4, 5120), torch.bfloat16, "hidden"),
        (kv, (tokens, 25600), torch.bfloat16, "kv"),
        (q_weight, (4, 5120), torch.bfloat16, "q_weight"),
        (k_weight, (4, 5120), torch.bfloat16, "k_weight"),
        (token_mask, (tokens,), torch.bool, "token_mask"),
    )
    if output is not None:
        expected += ((output, hidden.shape, torch.bfloat16, "output"),)
    for tensor, shape, dtype, name in expected:
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
        if tensor.device != hidden.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {hidden.device}")


def engram_gate_reference(hidden, kv, q_weight, k_weight, token_mask, eps=1e-20):
    """Model reference with the original FP32 association and BF16 boundary.

    False mask rows pass through exactly. Inputs must be finite on active rows;
    masked padding may contain arbitrary kv values because it is not consumed.
    This function is for correctness and a performance baseline, not serving.
    """
    _validate(hidden, kv, q_weight, k_weight, token_mask, None, eps)
    h = hidden.float()
    key = kv[:, : 4 * 5120].float().reshape(-1, 4, 5120)
    value = kv[:, 4 * 5120 :].float().unsqueeze(1)
    weight = q_weight.float() * k_weight.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = ((h * weight) * key).sum(-1) * rstd * 5120**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    result = (h + gate.unsqueeze(-1) * value).bfloat16()
    return torch.where(token_mask[:, None, None], result, hidden)


def engram_gate(hidden, kv, q_weight, k_weight, token_mask, eps=1e-20, *, output=None):
    """Run the AIV gate after external wkv GEMM and TP reduction.

    `token_mask` is mandatory and must occupy a stable device buffer for graph
    replay. An optional preallocated `output` supports caller-owned storage.
    Hidden/output may alias, but output must not alias kv or weight buffers.
    """
    _validate(hidden, kv, q_weight, k_weight, token_mask, output, eps)
    if hidden.device.type != "npu":
        raise ValueError("engram_gate requires NPU tensors")
    if output is None:
        output = torch.empty_like(hidden)
    if hidden.shape[0]:
        torch.ops._C_ascend.engram_gate(hidden, kv, q_weight, k_weight, token_mask, output, eps)
    return output
