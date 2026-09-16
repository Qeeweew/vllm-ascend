# SPDX-License-Identifier: Apache-2.0
"""Exercise compiled schemas and Meta kernels through run_v41_small_ops.py."""

from contextlib import nullcontext

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.fixture(autouse=True)
def require_candidate_extension():
    if not hasattr(torch.ops._C_ascend, "v41_rope"):
        pytest.skip("Load the isolated small-op extension with run_v41_small_ops.py")


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("tokens", [0, 1, 17])
def test_compiled_rope_and_paged_stores(fake, tokens):
    with FakeTensorMode() if fake else nullcontext():
        device = "cpu" if fake else "meta"
        positions = torch.empty(tokens, dtype=torch.int64, device=device)
        slots = torch.empty_like(positions)
        cos = torch.empty((32, 32), dtype=torch.float32, device=device)
        sin = torch.empty_like(cos)
        q = torch.empty((tokens, 32, 128), dtype=torch.bfloat16, device=device)
        out = torch.empty_like(q)
        assert torch.ops._C_ascend.v41_rope(q, positions, cos, sin, out, True) is None
        assert out.shape == q.shape and out.dtype == q.dtype
        main = torch.empty((tokens, 512), dtype=torch.bfloat16, device=device)
        cache = torch.empty((8, 32, 1, 512), dtype=torch.bfloat16, device=device)[1::2]
        assert torch.ops._C_ascend.v41_main_cache_store(main, positions, slots, cos, sin, cache, 2) is None
        assert cache.storage_offset() == 32 * 512 and cache.stride(0) == 2 * 32 * 512
        key = torch.empty((tokens, 128), dtype=torch.bfloat16, device=device)
        keys = torch.empty((8, 32, 1, 128), dtype=torch.int8, device=device)[1::2]
        scales = torch.empty((8, 32, 1, 1), dtype=torch.float16, device=device)[1::2]
        assert torch.ops._C_ascend.v41_index_cache_store(key, positions, slots, cos, sin, keys, scales, 1) is None
        assert keys.dtype == torch.int8 and scales.dtype == torch.float16


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("tokens,experts,topk,hash_route", [(0, 384, 6, False), (1, 384, 6, True), (17, 128, 3, False)])
def test_compiled_router_meta(fake, tokens, experts, topk, hash_route):
    with FakeTensorMode() if fake else nullcontext():
        device = "cpu" if fake else "meta"
        logits = torch.empty((tokens, experts), dtype=torch.float32, device=device)
        ids = torch.empty(tokens, dtype=torch.int64, device=device)
        mask = torch.empty(tokens, dtype=torch.bool, device=device)
        bias = torch.empty(experts, dtype=torch.float32, device=device)
        table = torch.empty((32, topk), dtype=torch.int32, device=device) if hash_route else None
        weights = torch.empty((tokens, topk), dtype=torch.float32, device=device)
        selected = torch.empty((tokens, topk), dtype=torch.int32, device=device)
        assert (
            torch.ops._C_ascend.v41_moe_router(
                logits, ids, mask, table, None if hash_route else bias, bias, weights, selected, topk, True, 2.0
            )
            is None
        )
        assert weights.shape == selected.shape == (tokens, topk)


def test_compiled_schemas_preserve_output_mutations():
    for name, outputs in {
        "v41_rope": ("output",),
        "v41_main_cache_store": ("cache",),
        "v41_index_cache_store": ("key_cache", "scale_cache"),
        "v41_moe_router": ("weights", "expert_ids"),
        "v41_dspark_metadata": ("schedule",),
    }.items():
        schema = getattr(torch.ops._C_ascend, name).default._schema
        assert not schema.returns
        for argument in schema.arguments:
            assert bool(argument.alias_info and argument.alias_info.is_write) == (argument.name in outputs)
        assert torch._C._dispatch_has_kernel_for_dispatch_key(f"_C_ascend::{name}", "Meta")
        assert torch._C._dispatch_has_kernel_for_dispatch_key(f"_C_ascend::{name}", "PrivateUse1")


@pytest.mark.parametrize("fake", [False, True])
def test_compiled_index_store_accepts_runner_packed_pages(fake):
    with FakeTensorMode() if fake else nullcontext():
        device = "cpu" if fake else "meta"
        rows, pitch, offset = 32, 8192, 64
        raw = torch.empty(offset + 4 * pitch, dtype=torch.uint8, device=device)
        keys = raw.view(torch.int8).as_strided((4, rows, 1, 128), (pitch, 128, 128, 1), offset)
        scales = raw.view(torch.float16).as_strided((4, rows, 1), (pitch // 2, 1, 1), (offset + rows * 128) // 2)
        x = torch.empty((5, 128), dtype=torch.bfloat16, device=device)
        positions = torch.empty(5, dtype=torch.int64, device=device)
        cos = torch.empty((256, 32), dtype=torch.float32, device=device)
        assert torch.ops._C_ascend.v41_index_cache_store(x, positions, positions, cos, cos, keys, scales, 1) is None
        overlap = raw.view(torch.float16).as_strided((4, rows, 1), (pitch // 2, 1, 1), offset // 2)
        with pytest.raises(RuntimeError, match="must not overlap"):
            torch.ops._C_ascend.v41_index_cache_store(x, positions, positions, cos, cos, keys, overlap, 1)


def test_compiled_meta_rejects_aliasing_and_wrong_dtype():
    x = torch.empty((1, 128), dtype=torch.bfloat16, device="meta")
    pos = torch.empty(1, dtype=torch.int64, device="meta")
    table = torch.empty((8, 32), dtype=torch.float32, device="meta")
    with pytest.raises(RuntimeError, match="share input storage"):
        torch.ops._C_ascend.v41_rope(x, pos, table, table, x)
    with pytest.raises(RuntimeError, match="dtype/device mismatch"):
        torch.ops._C_ascend.v41_rope(x, pos.int(), table, table, torch.empty_like(x))


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("batch,tokens", [(0, 0), (1, 5), (32, 160), (4096, 32768)])
def test_compiled_dspark_metadata_contract(fake, batch, tokens):
    with FakeTensorMode() if fake else nullcontext():
        device = "cpu" if fake else "meta"
        cu_q = torch.empty(batch + 1, dtype=torch.int32, device=device)
        lengths = torch.empty(batch, dtype=torch.int32, device=device)
        topk = torch.empty((tokens, 1), dtype=torch.int32, device=device)
        schedule = torch.empty(1024, dtype=torch.int32, device=device)
        assert torch.ops._C_ascend.v41_dspark_metadata(cu_q, lengths, topk, schedule) is None


@pytest.mark.parametrize("fake", [False, True])
def test_compiled_dspark_metadata_rejects_invalid_buffers(fake):
    with FakeTensorMode() if fake else nullcontext():
        device = "cpu" if fake else "meta"
        cu_q = torch.empty(3, dtype=torch.int32, device=device)
        lengths = torch.empty(2, dtype=torch.int32, device=device)
        topk = torch.empty((10, 1), dtype=torch.int32, device=device)
        schedule = torch.empty(1024, dtype=torch.int32, device=device)
        op = torch.ops._C_ascend.v41_dspark_metadata
        with pytest.raises(RuntimeError, match="boundaries"):
            op(cu_q[:-1], lengths, topk, schedule)
        with pytest.raises(RuntimeError, match="1024 INT32"):
            op(cu_q, lengths, topk, schedule[:-1])
        with pytest.raises(RuntimeError, match="share input storage"):
            op(schedule[:3], lengths, topk, schedule)
        with pytest.raises(RuntimeError, match="contiguous"):
            op(cu_q, lengths, topk, torch.empty(2048, dtype=torch.int32, device=device)[::2])
        with pytest.raises(RuntimeError, match="dtype/device"):
            op(cu_q.long(), lengths, topk, schedule)
        with pytest.raises(RuntimeError, match="32768 query"):
            op(cu_q, lengths, torch.empty((32769, 1), dtype=torch.int32, device=device), schedule)
