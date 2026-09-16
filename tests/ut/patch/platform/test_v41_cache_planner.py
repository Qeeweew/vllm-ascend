# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import vllm.v1.core.kv_cache_utils as upstream
from vllm.config import CacheConfig
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec

import vllm_ascend.patch.platform.patch_kv_cache_utils as planner
from vllm_ascend.core.kv_cache_interface import (
    AscendV41IndexerCacheSpec,
    AscendV41MainCacheSpec,
    AscendV41SWACacheSpec,
    register_ascend_kv_cache_specs,
)

PAGE_BYTES = 32768


def make_config():
    cache_config = CacheConfig(block_size=32)
    cache_config.kv_cache_layout = "LBHNC"
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=cache_config,
        attention_config=SimpleNamespace(hisparse_config=None),
        model_config=SimpleNamespace(max_model_len=131072),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        max_in_flight_tokens=4096,
    )


def make_specs(count=1):
    roles = {
        "main1": AscendV41MainCacheSpec(block_size=32, num_kv_heads=1, head_size=512, dtype=torch.bfloat16),
        "main2": AscendV41MainCacheSpec(
            block_size=32, tokens_per_state=2, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
        ),
        "index1": AscendV41IndexerCacheSpec(block_size=32, num_kv_heads=1, head_size=128, dtype=torch.int8),
        "index2": AscendV41IndexerCacheSpec(
            block_size=32, tokens_per_state=2, num_kv_heads=1, head_size=128, dtype=torch.int8
        ),
        "swa": AscendV41SWACacheSpec(
            block_size=32, sliding_window=128, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
        ),
        "ring": CircularBufferSpec(block_size=8, num_kv_heads=1, head_size=1024, head_size_v=0, dtype=torch.float32),
    }
    return {f"{role}.{i}": spec for role, spec in roles.items() for i in range(count)}


@pytest.fixture(autouse=True)
def register_specs():
    register_ascend_kv_cache_specs()


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("ring_first", [False, True])
def test_grouping_preserves_all_cache_roles_and_scheduler_units(count, ring_first):
    config, specs = make_config(), make_specs(count)
    if ring_first:
        specs = dict(reversed(list(specs.items())))
    original_pages = {name: spec.page_size_bytes for name, spec in specs.items()}
    groups = upstream.get_kv_cache_groups(config, specs)
    grouped = {name: group.kv_cache_spec for group in groups for name in group.layer_names}
    assert sorted(grouped) == sorted(specs)
    assert sum(len(group.layer_names) for group in groups) == len(specs)
    assert len(groups) == 6
    for name, original in specs.items():
        cache = grouped[name]
        assert type(cache) is type(original)
        assert cache.page_size_bytes == PAGE_BYTES
        assert cache.block_size == original.block_size
        assert specs[name].page_size_bytes == original_pages[name]
        if isinstance(cache, CircularBufferSpec):
            assert cache == original
            assert cache.prefix_cacheable is False
            assert cache.uses_slot_mapping is False
            assert cache.max_num_blocks_per_req(config, 131072) == 1
        else:
            assert cache.tokens_per_state == original.tokens_per_state
            assert cache.physical_block_size == 32 // original.tokens_per_state
            assert cache.real_page_size_bytes == original.real_page_size_bytes
    assert grouped["main2.0"].real_page_size_bytes == 16384
    assert grouped["index1.0"].real_page_size_bytes == 4160
    assert grouped["index2.0"].real_page_size_bytes == 2080
    assert grouped["swa.0"].max_admission_blocks_per_request(4096, 131072) == 133


def test_unequal_role_counts_preserve_layers():
    specs = make_specs(3)
    del specs["index1.2"]
    groups = upstream.get_kv_cache_groups(make_config(), specs)
    assert sorted(name for group in groups for name in group.layer_names) == sorted(specs)
    assert all(len(group.layer_names) <= 3 for group in groups)


@pytest.mark.parametrize("hook", ["_get_glm5_next_kv_cache_groups", "_ascend_get_packed_kv_cache_groups"])
def test_v41_dispatch_bypasses_legacy_grouping(monkeypatch, hook):
    fail = Mock(side_effect=AssertionError("legacy grouping was called"))
    for name in ("_orig_get_kv_cache_groups", "_orig_get_packed_kv_cache_groups", "group_and_unify_kv_cache_specs"):
        monkeypatch.setattr(planner, name, fail)
    assert len(getattr(planner, hook)(make_config(), make_specs())) == 6
    fail.assert_not_called()


def test_legacy_unifier_declines_v41():
    assert planner.group_and_unify_kv_cache_specs(make_specs()) is None


def test_non_v41_public_dispatch_unchanged(monkeypatch):
    expected = object()
    original = Mock(return_value=expected)
    monkeypatch.setattr(planner, "_orig_get_kv_cache_groups", original)
    config = make_config()
    specs = {"old": FullAttentionSpec(block_size=32, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)}
    assert upstream.get_kv_cache_groups(config, specs) is expected
    original.assert_called_once_with(config, specs)


@pytest.mark.parametrize(
    "name,changes",
    [
        ("main1.0", {"block_size": 64}),
        ("swa.0", {"block_size": 64}),
        ("index1.0", {"page_size_padded": 65536}),
        ("ring.0", {"block_size": 16}),
        ("ring.0", {"page_size_padded": 65536}),
        ("ring.0", {"head_size": 512}),
        ("ring.0", {"dtype": torch.bfloat16}),
        ("ring.0", {"head_size_v": 1024}),
    ],
)
def test_incompatible_pages_and_ring_geometry_fail_closed(name, changes):
    specs = make_specs()
    specs[name] = replace(specs[name], **changes)
    with pytest.raises(ValueError, match="32 KiB|capacity-8"):
        upstream.get_kv_cache_groups(make_config(), specs)


def test_unsupported_mixed_cache_fails_closed():
    specs = make_specs()
    specs["legacy"] = FullAttentionSpec(block_size=32, num_kv_heads=1, head_size=512, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Unsupported cache legacy"):
        upstream.get_kv_cache_groups(make_config(), specs)


def test_disabled_hybrid_manager_fails_closed():
    config = make_config()
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    with pytest.raises(ValueError, match="hybrid cache manager"):
        upstream.get_kv_cache_groups(config, make_specs())


def test_historical_vllm_api_fails_closed(monkeypatch):
    monkeypatch.setattr(planner, "vllm_version_is", lambda _: True)
    with pytest.raises(ValueError, match="main descriptor API"):
        upstream.get_kv_cache_groups(make_config(), make_specs())


@pytest.mark.parametrize("layer_count", [1, 3])
def test_real_standard_planner_shared_backing_descriptor_geometry(monkeypatch, layer_count):
    fail = Mock(side_effect=AssertionError("legacy V4 allocation was called"))
    monkeypatch.setattr(planner, "_get_kv_cache_config_deepseek_v4_main", fail)
    config, specs = make_config(), make_specs(layer_count)
    groups = upstream.get_kv_cache_groups(config, specs)
    pool_bytes = layer_count * PAGE_BYTES
    budget = pool_bytes * 17 + 3
    planned = upstream.get_kv_cache_config_from_groups(config, groups, budget)
    assert planned.num_blocks == 17
    assert upstream._pool_bytes_per_block(groups) == pool_bytes
    assert {tensor.size for tensor in planned.kv_cache_tensors} == {pool_bytes * 17}
    assert sorted(name for tensor in planned.kv_cache_tensors for name in tensor.layers) == sorted(specs)
    for tensor in planned.kv_cache_tensors:
        assert tensor.block_stride == PAGE_BYTES
        assert tensor.layer_stride == PAGE_BYTES * 17
        assert tensor.offset == 0
        end = tensor.offset + (len(tensor.layers) - 1) * tensor.layer_stride + 16 * tensor.block_stride + PAGE_BYTES
        assert end <= tensor.size <= budget
    fail.assert_not_called()


@pytest.mark.parametrize("layout", ["BLNHC", "BLHNC", "BHLNC", "LHBNC"])
def test_noncompact_layout_fails_before_standard_allocation(layout):
    config = make_config()
    config.cache_config.kv_cache_layout = layout
    with pytest.raises(ValueError, match="layer-compact and block-compact"):
        upstream.get_kv_cache_groups(config, make_specs(3))


def test_upstream_default_layout_is_compatible():
    from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts

    config = make_config()
    default_layout = get_supported_kv_cache_layouts([])[0]
    assert default_layout.name == "LBNHC"
    config.cache_config.kv_cache_layout = default_layout.name
    groups = upstream.get_kv_cache_groups(config, make_specs(3))
    planned = upstream.get_kv_cache_config_from_groups(config, groups, 17 * 3 * PAGE_BYTES)
    assert planned.num_blocks == 17
    assert all(tensor.block_stride == PAGE_BYTES for tensor in planned.kv_cache_tensors)


@pytest.mark.parametrize("tokens", range(1, 9))
def test_dspark_query_length_sets_real_ring_pages_and_runner_views(tokens):
    # Use the actual planner and runner allocation/reshape functions; a larger
    # ring is genuine contiguous state, never a padded capacity-eight view.
    from tests.ut.worker.test_model_runner_v1 import TestNPUModelRunnerKVCache
    from vllm_ascend.models.deepseek_v4.compressor import CompressorV41StateCache

    config, specs = make_config(), make_specs()
    config.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=tokens)
    capacity = 8 if tokens <= 6 else 16
    page_bytes = capacity * 1024 * 4
    specs["ring.0"] = replace(specs["ring.0"], block_size=capacity)
    groups = upstream.get_kv_cache_groups(config, specs)
    grouped = {name: group.kv_cache_spec for group in groups for name in group.layer_names}
    assert all(spec.page_size_bytes == page_bytes for spec in grouped.values())
    assert grouped["ring.0"].real_page_size_bytes == page_bytes
    assert grouped["ring.0"].page_size_padded is None
    assert grouped["swa.0"].extra_retained_tokens == tokens
    assert specs["swa.0"].extra_retained_tokens == 0
    plan = upstream.get_kv_cache_config_from_groups(config, groups, page_bytes * 17)
    assert plan.num_blocks == 17
    assert all(tensor.block_stride == page_bytes for tensor in plan.kv_cache_tensors)

    runner = TestNPUModelRunnerKVCache()._build_runner()
    runner._get_layer_kv_cache_specs = lambda _: grouped
    runner._kv_cache_spec_attn_group_iterator = lambda: [
        SimpleNamespace(backend=None, kv_cache_spec=spec, layer_names=[name]) for name, spec in grouped.items()
    ]
    raw = runner._allocate_kv_cache_tensors(plan)
    for value in raw.values():
        value.zero_()
    caches = runner._reshape_kv_cache_tensors(plan, raw)
    ring = caches["ring.0"]
    assert ring.shape == (17, 1, capacity, 1024)
    assert ring.is_contiguous() and ring.stride(0) == capacity * 1024
    assert ring.data_ptr() == raw["ring.0"].data_ptr()
    ring[2, 0, -1].fill_(3.25)
    actual = raw["ring.0"][3 * page_bytes - 4096 : 3 * page_bytes].view(torch.float32)
    assert torch.all(actual == 3.25)
    assert torch.all(ring[1] == 0) and torch.all(ring[3] == 0)
    state = CompressorV41StateCache.__new__(CompressorV41StateCache)
    torch.nn.Module.__init__(state)
    state.block_size = capacity
    state.bind_kv_cache(ring)
    assert state.kv_cache.shape == (17, capacity, 1024)
    assert state.kv_cache.is_contiguous()

    for name in ("main1.0", "main2.0", "swa.0"):
        key = caches[name]
        assert key.stride(0) * key.element_size() == page_bytes
        assert key.shape[1] == grouped[name].physical_block_size
    keys, scales = caches["index1.0"]
    assert keys.stride(0) == scales.stride(0) * 2 == page_bytes
    assert scales.data_ptr() - keys.data_ptr() == grouped["index1.0"].scale_offset_bytes
    keys[4].fill_(7)
    scales[4].fill_(0.25)
    index_page = raw["index1.0"][4 * page_bytes : 5 * page_bytes]
    assert torch.all(index_page[grouped["index1.0"].real_page_size_bytes :] == 0)


@pytest.mark.parametrize("tokens", [7, 8])
def test_large_dspark_rejects_old_or_fake_padded_ring(tokens):
    config = make_config()
    config.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=tokens)
    for padding in (None, 65536):
        specs = make_specs()
        specs["ring.0"] = replace(specs["ring.0"], page_size_padded=padding)
        with pytest.raises(ValueError, match="capacity-16"):
            upstream.get_kv_cache_groups(config, specs)


@pytest.mark.parametrize("tokens", range(1, 9))
def test_planned_compressor_ring_retains_pair_after_every_rejection(tokens):
    from vllm_ascend.ops.compressor_v41 import compressor_v41_reference

    config, specs = make_config(), make_specs()
    config.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=tokens)
    specs["ring.0"] = replace(specs["ring.0"], block_size=8 if tokens <= 6 else 16)
    groups = upstream.get_kv_cache_groups(config, specs)
    capacity = next(group.kv_cache_spec.block_size for group in groups if "ring.0" in group.layer_names)
    rng = torch.Generator().manual_seed(4141)
    initial, proposed, corrected = (torch.randn((n, 1024), generator=rng) for n in (31, tokens + 1, 3))
    weight = torch.ones(512, dtype=torch.bfloat16)
    empty = torch.zeros((1, capacity, 1024))

    def run(raw, start, state):
        positions = torch.arange(start, start + raw.shape[0], dtype=torch.int64)
        return compressor_v41_reference(
            raw,
            positions,
            positions % capacity,
            torch.tensor([0, raw.shape[0]], dtype=torch.int32),
            torch.zeros(raw.shape[0], dtype=torch.int32),
            weight,
            state,
            2,
        )

    _, prefix_state = run(initial, 0, empty)
    _, speculative_state = run(proposed, 31, prefix_state)
    for accepted in range(tokens + 1):
        resume = 32 + accepted
        actual, _ = run(corrected, resume, speculative_state)
        committed = torch.cat((initial, proposed[: accepted + 1], corrected))
        expected, _ = run(committed, 0, empty)
        torch.testing.assert_close(actual, expected[-3:], rtol=0, atol=0)
