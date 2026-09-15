# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.cache_v41 import write_index_cache_v41, write_main_cache_v41


def test_fixed_shape_indices_mask_padding_and_group_incomplete_rows(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_scatter_nd_update_sk", lambda *args: calls.append(args), raising=False
    )
    cache = torch.empty((4, 32, 1, 512), dtype=torch.bfloat16)
    slots = torch.tensor([-1, 0, 31, 32, 127, 128, 2, 3], dtype=torch.int64)
    positions = torch.tensor([1, 0, 1, 3, 7, 9, -1, 10], dtype=torch.int64)
    values = torch.empty((8, 512), dtype=torch.bfloat16)
    result = write_main_cache_v41(cache, values, slots, positions=positions, compress_ratio=2)
    assert result is cache
    assert len(calls) == 1
    _, indices, updates = calls[0]
    expected = torch.tensor([[-1, 0], [-1, 0], [0, 31], [1, 0], [3, 31], [-1, 0], [-1, 0], [-1, 0]])
    torch.testing.assert_close(indices, expected, atol=0, rtol=0)
    assert indices.shape == (8, 2) and updates.shape == (8, 1, 512)


def test_index_keys_and_scales_use_same_indices_and_validate_before_writes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_scatter_nd_update_sk", lambda *args: calls.append(args), raising=False
    )
    key_cache = torch.empty((4, 32, 1, 128), dtype=torch.int8)
    scale_cache = torch.empty((4, 32, 1), dtype=torch.float16)
    keys, scales = torch.empty((2, 128), dtype=torch.int8), torch.empty(2, dtype=torch.float16)
    slots = torch.tensor([0, -1], dtype=torch.int32)
    with pytest.raises(ValueError, match="dtype"):
        write_index_cache_v41(key_cache, scale_cache, keys, scales.float(), slots)
    assert not calls
    write_index_cache_v41(key_cache, scale_cache, keys, scales, slots)
    assert len(calls) == 2 and calls[0][1] is calls[1][1]
    assert calls[0][2].shape == (2, 1, 128) and calls[1][2].shape == (2, 1)


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_legacy_ratios_are_rejected(ratio):
    with pytest.raises(ValueError, match="compress_ratio"):
        write_main_cache_v41(
            torch.empty((1, 32, 512), dtype=torch.bfloat16),
            torch.empty((0, 512), dtype=torch.bfloat16),
            torch.empty(0, dtype=torch.int64),
            compress_ratio=ratio,
        )


def test_empty_and_invalid_tensor_contracts(monkeypatch):
    def no_native(*args):
        pytest.fail("Empty/invalid calls must not reach native scatter")

    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", no_native, raising=False)
    cache = torch.empty((1, 32, 1, 512), dtype=torch.bfloat16)
    empty = torch.empty((0, 512), dtype=torch.bfloat16)
    slots = torch.empty(0, dtype=torch.int64)
    assert write_main_cache_v41(cache, empty, slots) is cache
    with pytest.raises(ValueError, match="group ends"):
        write_main_cache_v41(cache, empty, slots, compress_ratio=2)
    with pytest.raises(ValueError, match="slots"):
        write_main_cache_v41(cache, empty, slots.float())
    with pytest.raises(ValueError, match="positions"):
        write_main_cache_v41(cache, empty, slots, positions=torch.empty(0))
    with pytest.raises(ValueError, match="inner axes"):
        write_main_cache_v41(torch.empty((1, 64, 1, 512), dtype=torch.bfloat16)[:, ::2], empty, slots)
    with pytest.raises(ValueError, match="overlap"):
        write_main_cache_v41(cache.expand(2, -1, -1, -1), empty, slots)
    with pytest.raises(ValueError):
        write_main_cache_v41(cache.float(), empty, slots)
    with pytest.raises(ValueError, match="dtype and device"):
        write_main_cache_v41(cache, empty.float(), slots)
    with pytest.raises(ValueError, match="contiguous"):
        write_main_cache_v41(cache, torch.empty((512, 2), dtype=torch.bfloat16).T, torch.zeros(2, dtype=torch.int64))
