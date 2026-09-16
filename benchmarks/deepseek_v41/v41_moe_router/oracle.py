# SPDX-License-Identifier: Apache-2.0
"""Independent scalar numerical oracle; CPU topk is deliberately not used."""

import math
import struct

import torch


def fp32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def sqrt_softplus(value: float) -> float:
    softplus = value if value > 20.0 else fp32(math.log1p(math.exp(value)))
    return fp32(math.sqrt(softplus))


def scalar_router(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    image_mask: torch.Tensor,
    table: torch.Tensor | None,
    text_bias: torch.Tensor | None,
    image_bias: torch.Tensor,
    top_k: int,
    renormalize: bool = True,
    scaling: float = 1.0,
    *,
    tie_order: str = "reject",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reject dynamic ties until an NPU probe establishes the tie contract.

    The explicit ascending-index option supports testing the proposed candidate,
    but is not evidence of the installed NPU backend's tie ordering.
    """
    if any(t.device.type != "cpu" for t in (logits, token_ids, image_mask, image_bias)):
        raise ValueError("scalar oracle requires CPU inputs")
    weights = torch.zeros((len(logits), top_k), dtype=torch.float32)
    ids = torch.full((len(logits), top_k), -1, dtype=torch.int32)
    for row, values in enumerate(logits.tolist()):
        image = bool(image_mask[row])
        scores = [sqrt_softplus(x) for x in values]
        if table is not None and not image:
            token = int(token_ids[row])
            if not 0 <= token < len(table):
                continue
            selected = table[token].tolist()
            if any(index < 0 or index >= len(values) for index in selected):
                continue
        else:
            bias = (
                image_bias.tolist() if image else (text_bias.tolist() if text_bias is not None else [0.0] * len(values))
            )
            biased = [fp32(score + offset) for score, offset in zip(scores, bias)]
            selected = sorted(range(len(values)), key=lambda index: (-biased[index], index))[:top_k]
            if tie_order == "reject":
                cutoff = biased[selected[-1]]
                if len(set(biased[index] for index in selected)) != top_k or biased.count(cutoff) > 1:
                    raise ValueError("dynamic tie requires an explicitly frozen NPU tie contract")
            elif tie_order != "index_ascending":
                raise ValueError("unsupported tie contract")
        chosen = [scores[index] for index in selected]
        denominator = 0.0
        for score in chosen:
            denominator = fp32(denominator + score)
        denominator = max(denominator, torch.finfo(torch.float32).tiny)
        for column, (index, score) in enumerate(zip(selected, chosen)):
            value = fp32(score / denominator) if renormalize else score
            weights[row, column] = fp32(value * fp32(scaling))
            ids[row, column] = index
    return weights, ids
