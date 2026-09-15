# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 compression after the independent projection GEMM.

The production call is an out operator: it allocates nothing, reads no device
scalar on the CPU, mutates the FP32 ring, and writes every latent row. Invalid
and non-boundary rows are zero. The reference is deliberately CPU-only and is
never used as an implicit production fallback.
"""

import math

import torch

HEAD_DIM = 512
STATE_DIM = 2 * HEAD_DIM


def _validate(
    kv_score: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    norm_weight: torch.Tensor,
    state_cache: torch.Tensor,
    latent_out: torch.Tensor,
    compress_ratio: int,
    eps: float,
) -> None:
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    expected_dtype = torch.bfloat16 if compress_ratio == 1 else torch.float32
    if kv_score.ndim != 2 or kv_score.shape[1] != HEAD_DIM * compress_ratio:
        raise ValueError("kv_score must have shape [T, 512 * compress_ratio]")
    tokens = kv_score.shape[0]
    specs = (
        (kv_score, expected_dtype, "kv_score"),
        (positions, torch.int64, "positions"),
        (slot_mapping, torch.int64, "slot_mapping"),
        (query_start_loc, torch.int32, "query_start_loc"),
        (token_to_req_indices, torch.int32, "token_to_req_indices"),
        (norm_weight, torch.bfloat16, "norm_weight"),
        (state_cache, torch.float32, "state_cache"),
        (latent_out, torch.bfloat16, "latent_out"),
    )
    for tensor, dtype, name in specs:
        if tensor.dtype != dtype or tensor.device != kv_score.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous {dtype} on {kv_score.device}")
    if positions.shape != (tokens,) or slot_mapping.shape != (tokens,):
        raise ValueError("positions and slot_mapping must have shape [T]")
    if norm_weight.shape != (HEAD_DIM,) or latent_out.shape != (tokens, HEAD_DIM):
        raise ValueError("norm_weight / latent_out must have shape [512] / [T, 512]")
    if query_start_loc.ndim != 1 or token_to_req_indices.ndim != 1:
        raise ValueError("request metadata must be one-dimensional")
    if compress_ratio == 2:
        if token_to_req_indices.shape != (tokens,) or query_start_loc.numel() < 1:
            raise ValueError("CR2 requires token request IDs and request boundaries")
        if state_cache.ndim != 3 or state_cache.shape[2] != STATE_DIM:
            raise ValueError("CR2 state_cache must have shape [blocks, capacity, 1024]")
        capacity = state_cache.shape[1]
        if capacity < 8 or capacity & (capacity - 1):
            raise ValueError("CR2 ring capacity must be a power of two >= 8")


def compressor_v41(
    kv_score: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    norm_weight: torch.Tensor,
    state_cache: torch.Tensor,
    latent_out: torch.Tensor,
    compress_ratio: int,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Compress projected tokens into a static [T,512] pre-RoPE buffer.

    CR1 takes BF16 [T,512], and ignores state / request metadata (empty tensors
    of their declared dtype are sufficient). CR2 takes FP32 [T,1024], laid out
    [kv512, score512]. Its state is FP32 [blocks,capacity,1024]. Slots encode
    block*capacity + position%capacity; -1 skips a token. Each request occupies
    a contiguous segment and its own ring block, with consecutive positions.
    The scheduler must supply initialized history before a chunk starts odd.
    Capacity must exceed the number of speculative tokens plus one.

    Inputs and outputs must not overlap, except that state is updated in place.
    Valid rows and metadata are refreshed outside each graph replay; no Python
    control flow depends on their contents. GEMM, RoPE, and cache insertion are
    intentionally separate operators.
    """
    _validate(
        kv_score,
        positions,
        slot_mapping,
        query_start_loc,
        token_to_req_indices,
        norm_weight,
        state_cache,
        latent_out,
        compress_ratio,
        eps,
    )
    if kv_score.shape[0] == 0:
        return latent_out
    torch.ops._C_ascend.compressor_v41(
        kv_score,
        positions,
        slot_mapping,
        query_start_loc,
        token_to_req_indices,
        norm_weight,
        state_cache,
        latent_out,
        compress_ratio,
        eps,
    )
    return latent_out


def compressor_v41_reference(
    kv_score: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    norm_weight: torch.Tensor,
    state_cache: torch.Tensor,
    compress_ratio: int,
    eps: float = 1e-20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU correctness oracle; return latent and a new ring without mutation.

    Normalization follows inference/model.py: pool in FP32, round to BF16,
    promote to FP32 for RMSNorm, and round the weighted result to BF16. The
    ring is modeled as a snapshot plus final tail updates rather than copying
    the kernel's task scheduling. Metadata values are checked here only.
    """
    if kv_score.device.type != "cpu":
        raise ValueError("the correctness reference accepts CPU tensors only")
    latent = torch.zeros((kv_score.shape[0], HEAD_DIM), dtype=torch.bfloat16)
    _validate(
        kv_score,
        positions,
        slot_mapping,
        query_start_loc,
        token_to_req_indices,
        norm_weight,
        state_cache,
        latent,
        compress_ratio,
        eps,
    )
    state = state_cache.clone()

    def normalize(value: torch.Tensor) -> torch.Tensor:
        value = value.to(torch.bfloat16).float()
        return (value * torch.rsqrt(value.square().mean() + eps) * norm_weight.float()).to(torch.bfloat16)

    if compress_ratio == 1:
        for token in range(kv_score.shape[0]):
            if slot_mapping[token] >= 0:
                latent[token] = normalize(kv_score[token])
        return latent, state

    capacity = state.shape[1]
    starts = query_start_loc.tolist()
    if not starts or starts[0] != 0 or any(a > b for a, b in zip(starts, starts[1:])):
        raise ValueError("request boundaries must start at zero and be monotone")
    if starts[-1] > len(kv_score):
        raise ValueError("request boundary exceeds token bucket")
    owned_blocks: set[int] = set()
    for req, (start, end) in enumerate(zip(starts, starts[1:])):
        valid = [t for t in range(start, end) if slot_mapping[t] >= 0]
        if not valid:
            continue
        if valid != list(range(start, end)):
            raise ValueError("padding must be outside actual request segments")
        block = int(slot_mapping[start]) // capacity
        if block >= state.shape[0] or block in owned_blocks:
            raise ValueError("requests need distinct valid state blocks")
        owned_blocks.add(block)
        first_pos = int(positions[start])
        for token in range(start, end):
            pos = int(positions[token])
            slot = int(slot_mapping[token])
            if int(token_to_req_indices[token]) != req or pos != first_pos + token - start:
                raise ValueError("request IDs / consecutive positions do not match boundaries")
            if slot != block * capacity + pos % capacity or pos < 0:
                raise ValueError("invalid ring slot mapping")
            if pos % 2 == 1:
                previous = kv_score[token - 1] if token > start else state_cache[block, (pos - 1) % capacity]
                pair = torch.stack((previous, kv_score[token]))
                pooled = (pair[:, :HEAD_DIM] * pair[:, HEAD_DIM:].softmax(dim=0)).sum(dim=0)
                latent[token] = normalize(pooled)
        for token in range(max(start, end - capacity), end):
            state[block, int(positions[token]) % capacity] = kv_score[token]
    return latent, state
