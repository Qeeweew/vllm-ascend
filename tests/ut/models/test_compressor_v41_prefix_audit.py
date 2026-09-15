# SPDX-License-Identifier: Apache-2.0
"""CPU evidence for prefix-hit boundaries and private CR2 ring lifetime."""

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_interface import (
    AscendV41IndexerCacheSpec,
    AscendV41MainCacheSpec,
    AscendV41SWACacheSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.ops.compressor_v41 import compressor_v41_reference
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator


def _coordinator(ascend):
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    init_none_hash(sha256)
    specs = [
        AscendV41MainCacheSpec(
            block_size=32, tokens_per_state=ratio, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
        )
        for ratio in (1, 2)
    ] + [
        AscendV41IndexerCacheSpec(
            block_size=32, tokens_per_state=ratio, num_kv_heads=1, head_size=128, dtype=torch.int8
        )
        for ratio in (1, 2)
    ]
    specs.extend(
        [
            AscendV41SWACacheSpec(
                block_size=32, sliding_window=128, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
            ),
            CircularBufferSpec(block_size=8, num_kv_heads=1, head_size=1024, head_size_v=0, dtype=torch.float32),
        ]
    )
    cls = AscendHybridKVCacheCoordinator if ascend else HybridKVCacheCoordinator
    return cls(
        kv_cache_config=KVCacheConfig(
            num_blocks=128,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec([f"group{i}"], spec) for i, spec in enumerate(specs)],
        ),
        max_model_len=256,
        max_in_flight_tokens=256,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=32,
        hash_block_size=32,
    )


def _request(name, length):
    return Request(
        request_id=name,
        prompt_token_ids=list(range(length)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(32, sha256),
    )


def _compress(raw, start, end, state):
    positions = torch.arange(start, end, dtype=torch.int64)
    return compressor_v41_reference(
        raw[start:end].contiguous(),
        positions,
        positions % 8,
        torch.tensor([0, end - start], dtype=torch.int32),
        torch.zeros(end - start, dtype=torch.int32),
        torch.ones(512, dtype=torch.bfloat16),
        state,
        2,
    )


@pytest.mark.parametrize("ascend", [False, True], ids=["upstream", "ascend"])
@pytest.mark.parametrize("length", [32, 33, 41, 65, 97])
def test_real_coordinator_prefix_hit_is_even_with_new_private_ring(ascend, length):
    coordinator = _coordinator(ascend)
    writer = _request("writer", length)
    for manager in coordinator.single_type_managers:
        manager.allocate_new_blocks(writer.request_id, length, length)
        manager.cache_blocks(writer, length, replay_boundaries=(length - 1,))
    blocks, hit, _ = coordinator.find_longest_cache_hit(writer.block_hashes, length - 1)
    assert hit == (length - 1) // 32 * 32
    assert hit % 2 == 0
    assert blocks[-1] == []
    assert not coordinator.enable_partial_hash_hits
    assert all(5 not in group.group_ids for group in coordinator.attention_groups)
    ring = coordinator.single_type_managers[-1]
    ring.add_local_computed_blocks("reader", blocks[-1], hit, 0)
    ring.allocate_new_blocks("reader", length, length)
    assert len(ring.req_to_blocks["reader"]) == 1
    assert ring.req_to_blocks["reader"][0] is not ring.req_to_blocks["writer"][0]

    raw = torch.randn(length + 4, 1024, generator=torch.Generator().manual_seed(7))
    expected, _ = _compress(raw, 0, length + 4, torch.zeros(1, 8, 1024))
    # NaN poison proves no stale row contributes, including one-token tail
    # replay and the following decode chunk that starts at an odd position.
    tail, state = _compress(raw, hit, length, torch.full((1, 8, 1024), float("nan")))
    decode, _ = _compress(raw, length, length + 4, state)
    actual = torch.cat((tail, decode))
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected[hit:], rtol=0, atol=0)


@pytest.mark.parametrize("start", [1, 31, 33, 65])
def test_odd_resume_without_history_is_a_real_counterexample(start):
    raw = torch.randn(start + 3, 1024, generator=torch.Generator().manual_seed(11))
    expected, _ = _compress(raw, 0, start + 3, torch.zeros(1, 8, 1024))
    poisoned, _ = _compress(raw, start, start + 3, torch.full((1, 8, 1024), float("nan")))
    assert torch.isnan(poisoned[0]).all()
    assert torch.isfinite(expected[start]).all()
    # Zero initialization conceals the invalid read but does not restore
    # the predecessor projection; recomputing its even position does.
    zeroed, _ = _compress(raw, start, start + 3, torch.zeros(1, 8, 1024))
    assert not torch.equal(zeroed[0], expected[start])
    replayed, _ = _compress(raw, start - 1, start + 3, torch.full((1, 8, 1024), float("nan")))
    torch.testing.assert_close(replayed[1:], expected[start:], rtol=0, atol=0)


def test_smoke_chunk_pattern_preserves_odd_start_history():
    chunks = [32, 9, 2, 2, 1]
    raw = torch.randn(sum(chunks), 1024, generator=torch.Generator().manual_seed(23))
    state = torch.full((1, 8, 1024), float("nan"))
    expected, _ = _compress(raw, 0, len(raw), state)
    start = 0
    outputs = []
    for chunk in chunks:
        output, state = _compress(raw, start, start + chunk, state)
        outputs.append(output)
        start += chunk
    torch.testing.assert_close(torch.cat(outputs), expected, rtol=0, atol=0)
