# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec

from vllm_ascend.models.deepseek_v4.compressor import CompressorV41MetadataBuilder, CompressorV41StateCache


def test_metadata_device_boundaries_padding_and_reused_storage():
    spec = CircularBufferSpec(block_size=8, num_kv_heads=1, head_size=1024, head_size_v=0, dtype=torch.float32)
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=8))
    builder = CompressorV41MetadataBuilder(spec, ["test"], config, torch.device("cpu"))
    # Includes a zero-length request. CPU query boundaries are deliberately
    # stale to simulate adaptive verification updating them only on device.
    common = SimpleNamespace(
        slot_mapping=torch.zeros(8, dtype=torch.int64),
        positions=torch.tensor([5, 6, 7, 10, 11, 0, 0, 0]),
        query_start_loc=torch.tensor([0, 3, 3, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2, 5], dtype=torch.int32),
        block_table_tensor=torch.tensor([[2], [9], [4]], dtype=torch.int32),
        num_actual_tokens=5,
    )
    result = builder.build(0, common)
    assert result.slot_mapping.tolist() == [21, 22, 23, 34, 35, -1, -1, -1]
    assert result.token_to_req_indices[:5].tolist() == [0, 0, 0, 2, 2]
    ptr = result.slot_mapping.data_ptr()
    common.query_start_loc = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    common.block_table_tensor = torch.tensor([[7], [2], [1]], dtype=torch.int32)
    result = builder.build(0, common)
    assert result.slot_mapping.data_ptr() == ptr
    assert result.slot_mapping.tolist() == [61, 62, 15, 10, 11, -1, -1, -1]


@pytest.mark.parametrize("drafts,capacity", [(0, 8), (5, 8), (7, 16), (15, 32)])
def test_state_capacity_and_duplicate_registration(drafts, capacity):
    context = {}
    config = SimpleNamespace(
        num_speculative_tokens=drafts, compilation_config=SimpleNamespace(static_forward_context=context)
    )
    with patch("vllm_ascend.models.deepseek_v4.compressor.get_current_vllm_config", return_value=config):
        cache = CompressorV41StateCache("layer2.compressor.state")
        assert cache.block_size == capacity
        assert context[cache.prefix] is cache
        with pytest.raises(ValueError, match="Duplicate"):
            CompressorV41StateCache(cache.prefix)
    storage = torch.zeros((3, 1, capacity, 1024), dtype=torch.float32)
    cache.bind_kv_cache(storage)
    assert cache.kv_cache.data_ptr() == storage.data_ptr()
    assert cache.kv_cache.shape == (3, capacity, 1024)
    spec = cache.get_kv_cache_spec(config)
    assert not spec.prefix_cacheable and not spec.uses_slot_mapping
    with pytest.raises(ValueError, match="FP32"):
        cache.bind_kv_cache(storage.bfloat16())


def test_full_graph_backend_update_hook_keeps_tensor_addresses_and_contents():
    from vllm_ascend.models.deepseek_v4.compressor import CompressorV41Backend

    # The runner traverses every backend during full graph replay, including
    # this storage-only backend. The inherited get_impl_cls raises otherwise.
    impl = CompressorV41Backend.get_impl_cls()
    slots = torch.tensor([8, 9, -1])
    before = slots.clone()
    address = slots.data_ptr()
    metadata = SimpleNamespace(slot_mapping=slots)
    assert impl.update_graph_params(None, 3, {"ring": metadata}, None) is None
    assert slots.data_ptr() == address
    torch.testing.assert_close(slots, before, rtol=0, atol=0)
