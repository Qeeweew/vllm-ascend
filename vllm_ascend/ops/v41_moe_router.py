# SPDX-License-Identifier: Apache-2.0
"""Experimental V4.1 router entry point; production dispatch remains gated."""

import math

import torch


def validate_v41_moe_router(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    image_mask: torch.Tensor,
    tid2eid: torch.Tensor | None,
    text_bias: torch.Tensor | None,
    image_bias: torch.Tensor,
    weights: torch.Tensor,
    expert_ids: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
) -> None:
    """Metadata validation only: safe to execute while capturing an NPU graph."""
    if logits.ndim != 2 or (logits.shape[1], top_k) not in ((384, 6), (128, 3)):
        raise ValueError("V4.1 router supports [T,384]/K6 and [T,128]/K3")
    rows, experts = logits.shape
    specs = [
        ("logits", logits, (rows, experts), torch.float32),
        ("token_ids", token_ids, (rows,), torch.int64),
        ("image_mask", image_mask, (rows,), torch.bool),
        ("image_bias", image_bias, (experts,), torch.float32),
        ("weights", weights, (rows, top_k), torch.float32),
        ("expert_ids", expert_ids, (rows, top_k), torch.int32),
    ]
    if tid2eid is not None:
        if tid2eid.ndim != 2 or tid2eid.shape[0] < 1:
            raise ValueError("tid2eid must be a nonempty [vocabulary,K] table")
        specs.append(("tid2eid", tid2eid, (tid2eid.shape[0], top_k), torch.int32))
    if text_bias is not None:
        specs.append(("text_bias", text_bias, (experts,), torch.float32))
    for name, tensor, shape, dtype in specs:
        if tensor.shape != shape or tensor.dtype != dtype or tensor.device != logits.device:
            raise ValueError(f"{name} must have shape {shape}, dtype {dtype}, and the logits device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not math.isfinite(routed_scaling_factor):
        raise ValueError("routed_scaling_factor must be finite")
    # C++ binding must repeat overlap checks for direct torch.ops callers.
    for output in (weights, expert_ids):
        for _, tensor, _, _ in specs:
            if tensor is not output and torch._C._overlaps(output, tensor):
                raise ValueError("router outputs must not alias another input or output")


def v41_moe_router(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    image_mask: torch.Tensor,
    tid2eid: torch.Tensor | None,
    text_bias: torch.Tensor | None,
    image_bias: torch.Tensor,
    weights: torch.Tensor,
    expert_ids: torch.Tensor,
    top_k: int = 6,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
) -> None:
    """Fill caller-owned outputs. Finite values and valid text lookup IDs required.

    Image rows never read token IDs or the lookup table. For defensive native
    memory safety only, invalid text table indices write an entire zero/-1 row;
    callers must never pass such sentinel rows to expert execution.
    """
    validate_v41_moe_router(
        logits,
        token_ids,
        image_mask,
        tid2eid,
        text_bias,
        image_bias,
        weights,
        expert_ids,
        top_k,
        routed_scaling_factor,
    )
    if logits.device.type != "npu":
        raise ValueError("v41_moe_router requires NPU tensors")
    if logits.shape[0] == 0:
        return
    torch.ops._C_ascend.v41_moe_router(
        logits,
        token_ids,
        image_mask,
        tid2eid,
        text_bias,
        image_bias,
        weights,
        expert_ids,
        top_k,
        renormalize,
        routed_scaling_factor,
    )
