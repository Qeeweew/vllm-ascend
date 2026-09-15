# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendV41IndexerCacheSpec,
    AscendV41MainCacheSpec,
    AscendV41SWACacheSpec,
    get_storage_block_size,
    register_ascend_kv_cache_specs,
)


def spec(kind="main", ratio=1, **kwargs):
    cls = {"main": AscendV41MainCacheSpec, "index": AscendV41IndexerCacheSpec, "swa": AscendV41SWACacheSpec}[kind]
    arguments = dict(
        block_size=32 * ratio,
        tokens_per_state=ratio,
        num_kv_heads=1,
        head_size=128 if kind == "index" else 512,
        dtype=torch.int8 if kind == "index" else torch.bfloat16,
    )
    if kind == "swa":
        arguments["sliding_window"] = 128
    arguments.update(kwargs)
    return cls(**arguments)


@pytest.mark.parametrize("kind,content_bytes", [("main", 1024), ("index", 130), ("swa", 1024)])
@pytest.mark.parametrize("ratio", [1, 2])
def test_physical_geometry_state_and_page_sizes(kind, content_bytes, ratio):
    if kind == "swa" and ratio == 2:
        with pytest.raises(ValueError, match="uncompressed"):
            spec(kind, ratio)
        return
    cache = spec(kind, ratio)
    assert cache.physical_block_size == 32
    assert get_storage_block_size(cache) == 32
    assert cache.num_states == 32
    assert cache.state_content_size_bytes == content_bytes
    assert cache.unpadded_page_size_bytes == cache.real_page_size_bytes == cache.page_size_bytes == 32 * content_bytes
    assert cache.indexes_kv_by_block_stride
    assert cache.model_version == "deepseek_v41"
    assert cache.cache_layout.startswith("v41_")
    group = UniformTypeKVCacheSpecs(block_size=cache.block_size, kv_cache_specs={"layer": cache})
    assert get_storage_block_size(group) == 32
    if kind == "index":
        assert cache.scale_offset_bytes == 32 * 128
        assert cache.scale_offset_bytes + 32 * 2 == cache.page_size_bytes


@pytest.mark.parametrize("kind", ["main", "index", "swa"])
def test_page_padding_alignment_and_block_resize_survive_dataclass_replace(kind):
    cache = spec(kind, page_size_padded=131072, alignment=512)
    # Upstream MLA post-init would shrink a larger common page when real size
    # is not already aligned. Index 32*130=4160 makes that regression visible.
    assert cache.page_size_bytes == 131072
    resized = replace(cache, block_size=cache.block_size * 2)
    assert resized.physical_block_size == 64 and get_storage_block_size(resized) == 64
    assert resized.page_size_bytes == 131072
    merged = type(cache).merge([cache, replace(cache)])
    assert merged == cache and type(merged) is type(cache)
    with pytest.raises(ValueError, match="padded pages"):
        replace(cache, page_size_padded=2)
    with pytest.raises(ValueError, match="padded pages"):
        replace(cache, page_size_padded=131073)
    if kind == "index":
        aligned = spec(kind, alignment=512)
        assert aligned.page_size_bytes == 4608
        assert aligned.real_page_size_bytes == 4160


def test_swa_keeps_current_chunk_plus_127_and_extra_speculative_rows():
    cache = spec("swa")
    config = SimpleNamespace(
        max_in_flight_tokens=4096,
        model_config=SimpleNamespace(max_model_len=131072),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    expected_blocks = (4096 + 127 + 31) // 32 + 1
    assert cache.max_admission_blocks_per_request(4096, 131072) == expected_blocks
    assert cache.max_memory_usage_bytes(config) == expected_blocks * cache.page_size_bytes
    assert expected_blocks > 128 // 32
    retained = replace(cache, extra_retained_tokens=65)
    assert retained.max_admission_blocks_per_request(4096, 131072) == (4096 + 127 + 65 + 31) // 32 + 1
    assert retained.max_admission_blocks_per_request(4096, 100) == (100 + 31) // 32 + 1
    assert cache.max_num_blocks_per_req(config, 131072) == 131072 // 32


def test_full_cache_budget_uses_raw_token_capacity_with_cr2():
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=131073),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    for kind in ("main", "index"):
        cache = spec(kind, 2)
        assert cache.max_memory_usage_bytes(config) == ((131073 + 63) // 64) * cache.page_size_bytes
        assert cache.max_num_blocks_per_req(config, 131073) == (131073 + 63) // 64


def test_spec_registry_and_uniform_groups_do_not_erase_layout_or_ratio():
    register_ascend_kv_cache_specs()
    main, index, swa = spec(), spec("index"), spec("swa")
    for cache, manager in ((main, FullAttentionManager), (index, FullAttentionManager), (swa, SlidingWindowManager)):
        assert KVCacheSpecRegistry.get_manager_class(cache) is manager
        assert KVCacheSpecRegistry.get_uniform_type_base_spec(cache) is type(cache)
        assert cache.is_uniform_with_collection({"a": cache, "b": replace(cache)})
        assert not cache.is_uniform_with_collection(
            {
                "legacy": AscendMLAAttentionSpec(
                    block_size=32,
                    num_kv_heads=1,
                    head_size=512,
                    dtype=torch.bfloat16,
                )
            }
        )
    assert not main.is_uniform_with_collection({"index": index})
    assert not main.is_uniform_with_collection({"compressed": spec("main", 2)})
    assert not swa.is_uniform_with_collection({"retained": replace(swa, extra_retained_tokens=1)})
    with pytest.raises(ValueError, match="identical"):
        AscendV41MainCacheSpec.merge([main, spec("main", 2)])
    with pytest.raises(ValueError, match="identical"):
        AscendV41MainCacheSpec.merge([main, index])
    with pytest.raises(ValueError, match="identical"):
        AscendV41SWACacheSpec.merge([swa, replace(swa, extra_retained_tokens=1)])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tokens_per_state": 4},
        {"tokens_per_state": 2, "block_size": 33},
        {"block_size": 0},
        {"dtype": torch.float16},
        {"head_size": 448},
        {"num_kv_heads": 2},
        {"head_size_v": 64},
        {"state_content_bytes": 1025},
        {"num_head_slots": 2},
        {"scale_dim": 1},
        {"cache_sparse_sfa_c8": True},
        {"alignment": 3},
        {"model_version": "deepseek_v4"},
        {"indexes_kv_by_block_stride": False},
        {"cache_dtype_str": "fp8"},
    ],
)
def test_main_rejects_incompatible_geometry(kwargs):
    with pytest.raises(ValueError):
        spec(**kwargs)


def test_index_and_swa_reject_wrong_scale_or_retention_contracts():
    for kwargs in ({"scale_dim": 2}, {"scale_dtype": torch.int8}, {"dtype": torch.bfloat16}, {"store_on_host": True}):
        with pytest.raises(ValueError):
            spec("index", **kwargs)
    for kwargs in ({"sliding_window": 256}, {"extra_retained_tokens": -1}, {"compress_ratio": 2}):
        with pytest.raises(ValueError):
            spec("swa", **kwargs)
