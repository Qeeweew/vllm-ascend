# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_ascend.ops.v41_rope_cache import (
    v41_index_cache_store,
    v41_index_cache_store_reference,
    v41_main_cache_store,
    v41_main_cache_store_reference,
    v41_rope,
    v41_rope_reference,
)


@pytest.mark.parametrize("shape", [(0, 128), (1, 512), (4, 8, 512), (16, 32, 128)])
@pytest.mark.parametrize("inverse", [False, True])
def test_rope_cpu_contract(shape, inverse):
    generator = torch.Generator().manual_seed(410)
    x = torch.randn(shape, generator=generator).bfloat16()
    positions = torch.arange(shape[0], dtype=torch.int64)
    angles = torch.randn((32, 32), generator=generator)
    cos, sin = angles.cos(), angles.sin()
    actual = v41_rope_reference(x, positions, cos, sin, inverse=inverse)
    assert torch.equal(actual[..., :-64], x[..., :-64])
    # A scalar pair oracle does not share the kernel's gather/layout algorithm.
    for token in range(shape[0]):
        for head in range(shape[1] if len(shape) == 3 else 1):
            row = x[token, head] if len(shape) == 3 else x[token]
            out = actual[token, head] if len(shape) == 3 else actual[token]
            for pair in range(32):
                even, odd = row[-64 + 2 * pair].float(), row[-63 + 2 * pair].float()
                c, s = cos[token, pair], sin[token, pair] * (-1 if inverse else 1)
                assert out[-64 + 2 * pair] == (even * c - odd * s).bfloat16()
                assert out[-63 + 2 * pair] == (even * s + odd * c).bfloat16()


def test_rope_invalid_positions_copy_input():
    x = torch.randn((4, 8, 128)).bfloat16()
    positions = torch.tensor([-1, 0, 3, 4], dtype=torch.int64)
    cos, sin = torch.ones(4, 32), torch.zeros(4, 32)
    assert torch.equal(v41_rope_reference(x, positions, cos, sin), x)


def test_rope_rejects_alias_and_layout():
    x = torch.zeros((2, 128), dtype=torch.bfloat16)
    positions = torch.arange(2, dtype=torch.int64)
    cos, sin = torch.ones(2, 32), torch.zeros(2, 32)
    with pytest.raises(ValueError, match="share storage"):
        v41_rope(x, positions, cos, sin, x)
    with pytest.raises(ValueError, match="share storage"):
        v41_rope(x, positions, cos, sin, x.view_as(x))
    with pytest.raises(ValueError, match="contiguous"):
        v41_rope(torch.zeros((2, 256), dtype=torch.bfloat16)[:, ::2], positions, cos, sin, torch.empty_like(x))


def test_empty_rope_skips_native_launch():
    x = torch.empty((0, 128), dtype=torch.bfloat16)
    output = torch.empty_like(x)
    with patch("torch.ops._C_ascend.v41_rope", create=True) as native:
        assert v41_rope(x, torch.empty(0, dtype=torch.int64), torch.ones(2, 32), torch.zeros(2, 32), output) is output
        native.assert_not_called()


@pytest.mark.parametrize("ratio", [1, 2])
def test_main_store_compressed_slots_and_guards(ratio):
    base = torch.full((2 * 4 * 512 + 31,), 7, dtype=torch.bfloat16)
    cache = base.as_strided((2, 4, 1, 512), (4 * 512 + 7, 512, 512, 1), storage_offset=3)
    original = base.clone()
    values = torch.arange(6).float()[:, None].expand(6, 512).contiguous().bfloat16()
    positions = torch.tensor([1, 2, 3, -1, 4, 9], dtype=torch.int64)
    slots = torch.tensor([7, 6, 5, 4, -1, 3], dtype=torch.int64)
    v41_main_cache_store_reference(
        values, positions, slots, torch.ones(8, 32), torch.zeros(8, 32), cache, compress_ratio=ratio
    )
    active = [0, 1, 2] if ratio == 1 else [0, 2]
    expected = original.clone()
    for token in active:
        slot = slots[token].item()
        begin = 3 + (slot // 4) * (4 * 512 + 7) + (slot % 4) * 512
        expected[begin : begin + 512] = values[token]
    assert torch.equal(base, expected)


def test_index_store_ties_to_even_and_zero():
    values = torch.zeros((2, 128), dtype=torch.bfloat16)
    values[0, :5] = torch.tensor([127, 0.5, 1.5, -0.5, -1.5])
    keys = torch.full((1, 4, 128), -7, dtype=torch.int8)
    scales = torch.full((1, 4, 1), -7, dtype=torch.float16)
    v41_index_cache_store_reference(
        values,
        torch.tensor([1, 3]),
        torch.tensor([3, 1]),
        torch.ones(4, 32),
        torch.zeros(4, 32),
        keys,
        scales,
        compress_ratio=2,
    )
    assert keys[0, 3, :5].tolist() == [127, 0, 2, 0, -2]
    assert keys[0, 1].count_nonzero() == 0
    assert scales[0, 3, 0] == 1 and scales[0, 1, 0] == 0
    assert (keys[0, [0, 2]] == -7).all() and (scales[0, [0, 2]] == -7).all()


@pytest.mark.parametrize("scale_first", [False, True])
@pytest.mark.parametrize("prefix,gap", [(0, 0), (6, 18)])
@pytest.mark.parametrize("scale_page_shift", [0, 1])
def test_index_packed_page_storage(scale_first, prefix, gap, scale_page_shift):
    page, blocks = 4, 2
    stride = page * 130 + gap
    raw = torch.full((prefix + (blocks + scale_page_shift) * stride + 16,), 0xA5, dtype=torch.uint8)
    key_offset = prefix + (page * 2 if scale_first else 0)
    scale_offset = prefix + (0 if scale_first else page * 128) + scale_page_shift * stride
    keys = raw.view(torch.int8).as_strided((blocks, page, 1, 128), (stride, 128, 128, 1), key_offset)
    scales = raw.view(torch.float16).as_strided((blocks, page, 1), (stride // 2, 1, 1), scale_offset // 2)
    before = raw.clone()
    key = torch.ones((2, 128), dtype=torch.bfloat16)
    args = key, torch.tensor([1, 3]), torch.tensor([0, 7]), torch.ones(4, 32), torch.zeros(4, 32)
    with patch("torch.ops._C_ascend.v41_index_cache_store", create=True) as native:
        result = v41_index_cache_store(*args, keys, scales)
        assert result[0] is keys and result[1] is scales
        native.assert_called_once()
    v41_index_cache_store_reference(*args, keys, scales)
    expected = before.clone()
    for slot in (0, 7):
        block, row = divmod(slot, page)
        start = key_offset + block * stride + row * 128
        expected[start : start + 128] = 127
        start = scale_offset + block * stride + row * 2
        expected[start : start + 2] = torch.tensor([1 / 127], dtype=torch.float16).view(torch.uint8)
    assert torch.equal(raw, expected)


@pytest.mark.parametrize("scale_offset,scale_stride", [(510, 528), (526, 528), (512, 530)])
def test_index_packed_page_rejects_overlap_or_stride(scale_offset, scale_stride):
    raw = torch.zeros(2048, dtype=torch.uint8)
    keys = raw.view(torch.int8).as_strided((2, 4, 128), (528, 128, 1))
    scales = raw.view(torch.float16).as_strided((2, 4, 1), (scale_stride // 2, 1, 1), scale_offset // 2)
    with pytest.raises(ValueError, match="Shared key/scale storage"):
        v41_index_cache_store(
            torch.ones((1, 128), dtype=torch.bfloat16),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.ones(2, 32),
            torch.zeros(2, 32),
            keys,
            scales,
        )


@pytest.mark.parametrize("kind", ["rope", "main", "index"])
@pytest.mark.parametrize("aliased", [False, True])
def test_fake_tensor_wrapper_storage_identity(kind, aliased):
    # FakeTensor storage pointers are all zero. Storage identity must still
    # distinguish unrelated arguments and views that share an allocation.
    with FakeTensorMode():
        width = 512 if kind == "main" else 128
        x = torch.empty((2, width), dtype=torch.bfloat16)
        positions = torch.empty(2, dtype=torch.int64)
        cos, sin = torch.empty(4, 32), torch.empty(4, 32)
        if kind == "rope":
            output = x.view_as(x) if aliased else torch.empty_like(x)
            call = lambda: v41_rope(x, positions, cos, sin, output)
            op_name = "v41_rope"
        elif kind == "main":
            cache = x.view(1, 2, width) if aliased else torch.empty((1, 2, width), dtype=x.dtype)
            slots = torch.empty_like(positions)
            call = lambda: v41_main_cache_store(x, positions, slots, cos, sin, cache)
            op_name = "v41_main_cache_store"
        else:
            cache = x.view(torch.int8).view(1, 4, 128) if aliased else torch.empty((1, 4, 128), dtype=torch.int8)
            scales = torch.empty((1, 4, 1), dtype=torch.float16)
            slots = torch.empty_like(positions)
            call = lambda: v41_index_cache_store(x, positions, slots, cos, sin, cache, scales)
            op_name = "v41_index_cache_store"
        with patch(f"torch.ops._C_ascend.{op_name}", create=True) as native:
            if aliased:
                with pytest.raises(ValueError, match="share storage"):
                    call()
                native.assert_not_called()
            else:
                call()
                native.assert_called_once()


@pytest.mark.parametrize("scale_offset", [512, 510])
def test_fake_tensor_wrapper_packed_cache_regions(scale_offset):
    with FakeTensorMode():
        raw = torch.empty(1056, dtype=torch.uint8)
        keys = raw.view(torch.int8).as_strided((2, 4, 1, 128), (528, 128, 128, 1))
        scales = raw.view(torch.float16).as_strided((2, 4, 1), (264, 1, 1), scale_offset // 2)
        args = (
            torch.empty((2, 128), dtype=torch.bfloat16),
            torch.empty(2, dtype=torch.int64),
            torch.empty(2, dtype=torch.int64),
            torch.empty(4, 32),
            torch.empty(4, 32),
            keys,
            scales,
        )
        with patch("torch.ops._C_ascend.v41_index_cache_store", create=True) as native:
            if scale_offset == 512:
                result = v41_index_cache_store(*args)
                assert result[0] is keys and result[1] is scales
                native.assert_called_once()
            else:
                with pytest.raises(ValueError, match="Shared key/scale storage"):
                    v41_index_cache_store(*args)
                native.assert_not_called()
