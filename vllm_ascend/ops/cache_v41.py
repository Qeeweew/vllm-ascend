# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-shape V4.1 paged cache stores using the existing AscendC scatter.

GEMM, normalization, RoPE and quantization are independent operations. Main
compressed values must already use group-FIRST RoPE; a CR2 group is published
only at its LAST token. Valid active slots must be unique within an invocation.
"""

import torch


def _validate_cache(cache: torch.Tensor, dtype: torch.dtype, width: int) -> None:
    if cache.dtype != dtype or cache.ndim not in (3, 4):
        raise ValueError(f"V4.1 cache must be {dtype} [blocks,block_size,{width}] or [blocks,block_size,1,{width}]")
    if cache.shape[-1] != width or (cache.ndim == 4 and cache.shape[2] != 1):
        raise ValueError("V4.1 cache has an invalid singleton head or feature width")
    if cache.shape[0] < 1 or cache.shape[1] < 1:
        raise ValueError("V4.1 cache requires a nonempty page allocation")
    # Native scatter accepts gapped page strides, not inner-axis permutations.
    expected = 1
    for axis in range(cache.ndim - 1, 0, -1):
        if cache.shape[axis] > 1 and cache.stride(axis) != expected:
            raise ValueError("V4.1 cache inner axes must be contiguous; only page axis 0 may be strided")
        expected *= cache.shape[axis]
    if cache.stride(0) < expected:
        raise ValueError("V4.1 cache pages must not overlap")


def _validate_values(cache: torch.Tensor, values: torch.Tensor, tokens: int) -> torch.Tensor:
    width = cache.shape[-1]
    if values.dtype != cache.dtype or values.device != cache.device:
        raise ValueError("V4.1 cache values must match cache dtype and device")
    if values.shape not in ((tokens, width), (tokens, 1, width)) or not values.is_contiguous():
        raise ValueError("V4.1 cache values must be contiguous [T,D] or [T,1,D]")
    return values.view(tokens, *cache.shape[2:])


def _indices(
    cache: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor | None, compress_ratio: int
) -> torch.Tensor:
    if compress_ratio not in (1, 2):
        raise ValueError("V4.1 cache compress_ratio must be 1 or 2")
    if slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64) or slots.device != cache.device:
        raise ValueError("V4.1 slots must be device integer [T] physical compressed cache slots")
    if compress_ratio == 2 and positions is None:
        raise ValueError("CR2 stores require original token positions to select group ends")
    valid = (slots >= 0) & (slots < cache.shape[0] * cache.shape[1])
    if positions is not None:
        if positions.shape != slots.shape or positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("V4.1 positions must be integer [T]")
        if positions.device != slots.device:
            raise ValueError("V4.1 positions and slots must share a device")
        valid = valid & (positions >= 0) & ((positions + 1) % compress_ratio == 0)
    # (-1,0) flattens to a negative row and is skipped by AscendC. Never map
    # padding to a real cache row, including the final row (-1 in Python).
    page = torch.where(valid, slots // cache.shape[1], -1)
    offset = torch.where(valid, slots % cache.shape[1], 0)
    return torch.stack((page, offset), dim=-1)


def write_main_cache_v41(
    cache: torch.Tensor,
    values: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
    compress_ratio: int = 1,
) -> torch.Tensor:
    """Store BF16 SWA or CR1/2 main latent; leave invalid slots untouched.

    Slots already address the physical cache in compressed-position units;
    this helper does not divide a slot by the compression ratio. Negative or
    out-of-capacity slots are ignored. Returned tensor aliases ``cache``.
    """
    _validate_cache(cache, torch.bfloat16, 512)
    values = _validate_values(cache, values, slot_mapping.numel())
    indices = _indices(cache, slot_mapping, positions, compress_ratio)
    if slot_mapping.numel():
        torch.ops._C_ascend.npu_scatter_nd_update_sk(cache, indices, values)
    return cache


def write_index_cache_v41(
    key_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    keys: torch.Tensor,
    scales: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
    compress_ratio: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Store prequantized INT8 index K and FP16 scales at identical slots.

    Cache scale layout is [blocks,block_size,1] or [blocks,block_size,1,1].
    Quantization and key group-first RoPE must complete before this call.
    """
    _validate_cache(key_cache, torch.int8, 128)
    _validate_cache(scale_cache, torch.float16, 1)
    if key_cache.shape[:2] != scale_cache.shape[:2] or key_cache.device != scale_cache.device:
        raise ValueError("V4.1 index keys and scales must share page dimensions and device")
    tokens = slot_mapping.numel()
    keys = _validate_values(key_cache, keys, tokens)
    if scales.shape == (tokens,):
        scales = scales.view(tokens, 1)
    scales = _validate_values(scale_cache, scales, tokens)
    indices = _indices(key_cache, slot_mapping, positions, compress_ratio)
    if tokens:
        torch.ops._C_ascend.npu_scatter_nd_update_sk(key_cache, indices, keys)
        torch.ops._C_ascend.npu_scatter_nd_update_sk(scale_cache, indices, scales)
    return key_cache, scale_cache
