# SPDX-License-Identifier: Apache-2.0
"""V4.1 rotary primitives with caller-owned destinations and frozen rounding.

Projection GEMMs and RMSNorm are independent. CPU references are explicit test
oracles, never implicit runtime fallbacks. Only host tensor metadata is checked
by production wrappers; positions and slots are consumed on device.
"""

import torch

ROTARY_WIDTH = 64
TABLE_WIDTH = ROTARY_WIDTH // 2


def _reject_aliases(inputs, outputs):
    for index, output in enumerate(outputs):
        for other in (*inputs, *outputs[:index]):
            if output.numel() and other.numel() and torch._C._is_alias_of(output, other):
                raise ValueError("V4.1 rotary destinations must not share storage with another argument")


def _validate_rope(x, positions, cos, sin):
    if x.ndim not in (2, 3) or x.shape[-1] not in (128, 512):
        raise ValueError("RoPE input must be BF16 [T,D] or [T,H,D], D=128/512")
    if x.ndim == 3 and x.shape[1] not in (1, 8, 32):
        raise ValueError("RoPE supports H=1/8/32")
    if positions.shape != (x.shape[0],):
        raise ValueError("positions must be INT64 [T]")
    if cos.ndim != 2 or cos.shape[0] < 1 or cos.shape[1] != TABLE_WIDTH or sin.shape != cos.shape:
        raise ValueError("cos/sin must be FP32 [max_positions,32]")
    for tensor, dtype in ((x, torch.bfloat16), (positions, torch.int64), (cos, torch.float32), (sin, torch.float32)):
        if tensor.dtype != dtype or tensor.device != x.device or not tensor.is_contiguous():
            raise ValueError("RoPE inputs require declared dtypes, contiguous layouts and a common device")


def v41_rope(x, positions, cos, sin, output, *, inverse=False):
    """Rotate valid positions; copy the full input row for out-of-table positions.

    Invalid positions never index either table. Input/output sharing any storage
    is rejected even when their particular views do not overlap. Every output
    row is initialized, including padding. No destination allocation occurs.
    """
    _validate_rope(x, positions, cos, sin)
    if output.shape != x.shape or output.dtype != x.dtype or output.device != x.device or not output.is_contiguous():
        raise ValueError("output must have the contiguous input shape, dtype and device")
    _reject_aliases((x, positions, cos, sin), (output,))
    if x.shape[0]:
        torch.ops._C_ascend.v41_rope(x, positions, cos, sin, output, inverse)
    return output


def v41_rope_reference(x, positions, cos, sin, *, inverse=False):
    """CPU oracle matching separate FP32 products and the final BF16 cast."""
    _validate_rope(x, positions, cos, sin)
    if x.device.type != "cpu":
        raise ValueError("The rotary reference is CPU-only")
    output = x.clone()
    valid = (positions >= 0) & (positions < cos.shape[0])
    chosen = positions[valid]
    c, s = cos[chosen], sin[chosen]
    if x.ndim == 3:
        c, s = c[:, None], s[:, None]
    if inverse:
        s = -s
    pairs = x[valid, ..., -ROTARY_WIDTH:].float().unflatten(-1, (-1, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)
    output[valid, ..., -ROTARY_WIDTH:] = rotated.bfloat16()
    return output


def _validate_cache(cache, dtype, width, device):
    if cache.ndim not in (3, 4) or cache.dtype != dtype or cache.device != device:
        raise ValueError("Cache must have the declared dtype/device and shape [B,P,D] or [B,P,1,D]")
    if cache.shape[-1] != width or (cache.ndim == 4 and cache.shape[2] != 1) or min(cache.shape[:2]) < 1:
        raise ValueError("Cache requires nonempty pages, singleton head and the declared feature width")
    expected = 1
    for axis in range(cache.ndim - 1, 0, -1):
        if cache.shape[axis] > 1 and cache.stride(axis) != expected:
            raise ValueError("Cache inner axes must be contiguous")
        expected *= cache.shape[axis]
    if cache.stride(0) < expected:
        raise ValueError("Cache pages must not overlap")


def _validate_index_destinations(inputs, key_cache, scale_cache):
    if key_cache.shape[:2] != scale_cache.shape[:2]:
        raise ValueError("Key and scale cache page dimensions must match")
    # The runner packs keys and scales into disjoint regions of every raw page.
    # Reject all input aliases, but permit this specific shared-output layout.
    _reject_aliases(inputs, (key_cache,))
    _reject_aliases(inputs, (scale_cache,))
    if not torch._C._is_alias_of(key_cache, scale_cache):
        return
    stride = key_cache.stride(0) * key_cache.element_size()
    if stride != scale_cache.stride(0) * scale_cache.element_size():
        raise ValueError("Shared key/scale storage requires equal byte page strides")
    delta = (scale_cache.storage_offset() * scale_cache.element_size() - key_cache.storage_offset()) % stride
    key_bytes = key_cache.shape[1] * 128
    scale_bytes = scale_cache.shape[1] * 2
    if delta < key_bytes or delta + scale_bytes > stride:
        raise ValueError("Shared key/scale storage requires disjoint regions within each page")


def _validate_store(x, positions, slots, cos, sin, width, compress_ratio):
    _validate_rope(x, positions, cos, sin)
    if x.ndim != 2 or x.shape[1] != width:
        raise ValueError(f"Store input must be BF16 [T,{width}]")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")
    if (
        slots.shape != positions.shape
        or slots.dtype != torch.int64
        or slots.device != x.device
        or not slots.is_contiguous()
    ):
        raise ValueError("slots must be contiguous INT64 [T] on the input device")


def v41_main_cache_store(x, positions, slots, cos, sin, cache, *, compress_ratio=1):
    """Rotate pre-RoPE BF16 main values and publish valid physical cache slots.

    CR2 publishes odd original positions using their preceding even table row.
    Slots are already compressed; they are never divided by compress_ratio.
    Invalid table positions/slots and incomplete groups leave cache untouched.
    Caller guarantees that active destinations within a launch are unique.
    """
    _validate_store(x, positions, slots, cos, sin, 512, compress_ratio)
    _validate_cache(cache, torch.bfloat16, 512, x.device)
    _reject_aliases((x, positions, slots, cos, sin), (cache,))
    if x.shape[0]:
        torch.ops._C_ascend.v41_main_cache_store(x, positions, slots, cos, sin, cache, compress_ratio)
    return cache


def v41_index_cache_store(key, positions, slots, cos, sin, key_cache, scale_cache, *, compress_ratio=1):
    """Rotate then round BF16 before dynamic INT8 quantization and cache stores.

    Scale is computed in FP32 and stored as FP16 after quantized values have
    been computed. This wrapper does not perform projection or RMSNorm.
    """
    _validate_store(key, positions, slots, cos, sin, 128, compress_ratio)
    _validate_cache(key_cache, torch.int8, 128, key.device)
    _validate_cache(scale_cache, torch.float16, 1, key.device)
    _validate_index_destinations((key, positions, slots, cos, sin), key_cache, scale_cache)
    if key.shape[0]:
        torch.ops._C_ascend.v41_index_cache_store(
            key, positions, slots, cos, sin, key_cache, scale_cache, compress_ratio
        )
    return key_cache, scale_cache


def v41_main_cache_store_reference(x, positions, slots, cos, sin, cache, *, compress_ratio=1):
    """CPU mutation oracle with explicit logical page addressing."""
    _validate_store(x, positions, slots, cos, sin, 512, compress_ratio)
    _validate_cache(cache, torch.bfloat16, 512, x.device)
    if x.device.type != "cpu":
        raise ValueError("The cache reference is CPU-only")
    group_positions = torch.div(positions, compress_ratio, rounding_mode="floor") * compress_ratio
    rotated = v41_rope_reference(x, group_positions, cos, sin)
    destinations = set()
    for token, (position, group_position, slot) in enumerate(
        zip(positions.tolist(), group_positions.tolist(), slots.tolist())
    ):
        if (
            position < 0
            or group_position >= cos.shape[0]
            or slot < 0
            or slot >= cache.shape[0] * cache.shape[1]
            or (position + 1) % compress_ratio
        ):
            continue
        if slot in destinations:
            raise ValueError("Active cache destinations must be unique")
        destinations.add(slot)
        cache[slot // cache.shape[1], slot % cache.shape[1]].copy_(rotated[token].view_as(cache[0, 0]))
    return cache


def v41_index_cache_store_reference(key, positions, slots, cos, sin, key_cache, scale_cache, *, compress_ratio=1):
    """CPU mathematical reference, including zero rows and physical addressing.

    Ascend Vector division differs at some FP32 half boundaries; native bitwise
    acceptance must use the original CANN dynamic quantizer for that stage.
    """
    _validate_store(key, positions, slots, cos, sin, 128, compress_ratio)
    _validate_cache(key_cache, torch.int8, 128, key.device)
    _validate_cache(scale_cache, torch.float16, 1, key.device)
    _validate_index_destinations((key, positions, slots, cos, sin), key_cache, scale_cache)
    if key.device.type != "cpu":
        raise ValueError("The index cache reference is CPU-only")
    group_positions = torch.div(positions, compress_ratio, rounding_mode="floor") * compress_ratio
    rounded = v41_rope_reference(key, group_positions, cos, sin).float()
    maximum = rounded.abs().amax(-1, keepdim=True)
    multiplier = torch.where(maximum == 0, 0, 127.0 / maximum)
    quantized = (rounded * multiplier).round().clamp(-128, 127).to(torch.int8)
    scales = (maximum[:, 0] * torch.tensor(1.0 / 127.0, dtype=torch.float32)).half()
    destinations = set()
    for token, (position, group_position, slot) in enumerate(
        zip(positions.tolist(), group_positions.tolist(), slots.tolist())
    ):
        if (
            position < 0
            or group_position >= cos.shape[0]
            or slot < 0
            or slot >= key_cache.shape[0] * key_cache.shape[1]
            or (position + 1) % compress_ratio
        ):
            continue
        if slot in destinations:
            raise ValueError("Active cache destinations must be unique")
        destinations.add(slot)
        page, within = divmod(slot, key_cache.shape[1])
        key_cache[page, within].copy_(quantized[token].view_as(key_cache[0, 0]))
        scale_cache[page, within].fill_(scales[token])
    return key_cache, scale_cache
