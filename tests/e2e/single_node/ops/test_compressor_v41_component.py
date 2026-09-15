# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent GEMM -> ring metadata -> vector compressor integration."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import vllm_ascend.vllm_ascend_C  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.v1.kv_cache_interface import CircularBufferSpec

from vllm_ascend.models.deepseek_v4.compressor import (
    CompressorV41,
    CompressorV41Metadata,
    CompressorV41MetadataBuilder,
)
from vllm_ascend.ops.compressor_v41 import compressor_v41_reference


@pytest.mark.parametrize("ratio", [1, 2])
def test_real_component_projection_and_compression(ratio):
    config = VllmConfig()
    device = torch.device("npu")
    with (
        set_current_vllm_config(config),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1),
    ):
        module = CompressorV41(5120, ratio, 1e-20, f"test{ratio}").to(device)
    generator = torch.Generator().manual_seed(534)
    weight = torch.randn((512 * ratio, 5120), generator=generator).mul_(0.01).bfloat16()
    hidden = torch.randn((5, 5120), generator=generator).bfloat16()
    with torch.no_grad():
        module.fused_wkv_wgate.weight.copy_(weight)
        module.norm.weight.fill_(1)
    projected = module.project(hidden.to(device))
    reference_projection = hidden.float() @ weight.float().t()
    expected_dtype = torch.float32 if ratio == 2 else torch.bfloat16
    assert projected.dtype == expected_dtype
    torch.testing.assert_close(
        projected.cpu(),
        reference_projection.to(expected_dtype),
        rtol=2e-4 if ratio == 2 else 0.01,
        atol=2e-5 if ratio == 2 else 0.01,
    )
    positions = torch.arange(5, dtype=torch.int64)
    slots = positions.clone()
    starts = torch.tensor([0, 5], dtype=torch.int32)
    requests = torch.zeros(5, dtype=torch.int32)
    state = torch.zeros((1, 8, 1024), dtype=torch.float32) if ratio == 2 else torch.empty(0, dtype=torch.float32)
    if ratio == 2:
        module.state_cache.bind_kv_cache(state[:, None].to(device))
    metadata = CompressorV41Metadata(slots.to(device), starts.to(device), requests.to(device))
    out = torch.empty((5, 512), dtype=torch.bfloat16, device=device)
    reference, expected_state = compressor_v41_reference(
        projected.cpu(), positions, slots, starts, requests, torch.ones(512, dtype=torch.bfloat16), state, ratio
    )
    module(projected, positions.to(device), metadata, out)
    torch.testing.assert_close(out.cpu(), reference, rtol=0.01, atol=0.01)
    if ratio == 2:
        torch.testing.assert_close(module.state_cache.kv_cache.cpu(), expected_state, rtol=0, atol=0)


def test_device_metadata_search_tracks_changed_boundaries():
    device = torch.device("npu")
    spec = CircularBufferSpec(block_size=8, num_kv_heads=1, head_size=1024, head_size_v=0, dtype=torch.float32)
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=8))
    builder = CompressorV41MetadataBuilder(spec, ["test"], config, device)
    common = SimpleNamespace(
        slot_mapping=torch.zeros(8, dtype=torch.int64, device=device),
        positions=torch.tensor([5, 6, 7, 10, 11, 0, 0, 0], device=device),
        query_start_loc=torch.tensor([0, 3, 3, 5], dtype=torch.int32, device=device),
        block_table_tensor=torch.tensor([[2], [9], [4]], dtype=torch.int32, device=device),
        num_actual_tokens=5,
    )
    for boundaries, want in [
        ([0, 3, 3, 5], [21, 22, 23, 34, 35, -1, -1, -1]),
        ([0, 2, 2, 5], [21, 22, 39, 34, 35, -1, -1, -1]),
        ([0, 2, 2, 4], [21, 22, 39, 34, -1, -1, -1, -1]),
    ]:
        common.query_start_loc.copy_(torch.tensor(boundaries, dtype=torch.int32))
        result = builder.build(0, common)
        assert result.slot_mapping.cpu().tolist() == want
