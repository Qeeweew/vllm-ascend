# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.config import CacheConfig, EngramConfig, ParallelConfig

import vllm_ascend.patch.platform.patch_engram_config as patch
from vllm_ascend.utils import refresh_block_size


def make_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="DeepseekV41ForCausalLM",
            hf_text_config=SimpleNamespace(engram_layer_ids=[1, 14]),
        ),
        parallel_config=ParallelConfig(tensor_parallel_size=8),
        engram_config=None,
        load_config=SimpleNamespace(load_format="safetensors"),
        speculative_config=None,
    )


@pytest.fixture(autouse=True)
def npu_platform(monkeypatch):
    monkeypatch.setattr(patch, "current_platform", SimpleNamespace(device_type="npu"))


def test_default_pinned_tp_offload_and_explicit_config_identity():
    config = make_config()
    patch._resolve_and_verify_engram_config(config)
    assert config.engram_config.cpu_offload
    assert config.load_config.safetensors_load_strategy == "lazy"
    original = config.engram_config
    patch._resolve_and_verify_engram_config(config)
    assert config.engram_config is original


@pytest.mark.parametrize(
    "kwargs",
    [{"cpu_offload": False}, {"embedding_across_dp": True}, {"dp_shared_memory": True}],
)
def test_unsupported_storage_modes_rejected(kwargs):
    config = make_config()
    config.engram_config = EngramConfig(**kwargs)
    with pytest.raises(ValueError, match="pinned CPU Engram"):
        patch._resolve_and_verify_engram_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tensor_parallel_size", 4, "TP8"),
        ("pipeline_parallel_size", 2, "PP1"),
        ("data_parallel_size", 2, "DP1"),
        ("prefill_context_parallel_size", 2, "context parallelism"),
        ("decode_context_parallel_size", 2, "context parallelism"),
        ("enable_expert_parallel", True, "replicated token"),
        ("enable_elastic_ep", True, "replicated token"),
        ("enable_dbo", True, "DBO"),
    ],
)
def test_unsupported_parallel_geometry(field, value, message):
    config = make_config()
    setattr(config.parallel_config, field, value)
    with pytest.raises(ValueError, match=message):
        patch._resolve_and_verify_engram_config(config)


def test_speculation_requires_separate_model_integration():
    config = make_config()
    config.speculative_config = object()
    with pytest.raises(ValueError, match="speculative model integration"):
        patch._resolve_and_verify_engram_config(config)


@pytest.mark.parametrize("strategy", ["eager", "torchao"])
def test_loader_cannot_materialize_full_host_table_in_each_worker(strategy):
    config = make_config()
    config.load_config.safetensors_load_strategy = strategy
    with pytest.raises(ValueError, match="lazy or prefetch"):
        patch._resolve_and_verify_engram_config(config)


@pytest.mark.parametrize("load_format", ["fastsafetensors", "instanttensor", "netloader"])
def test_device_direct_loaders_cannot_receive_host_tables(load_format):
    config = make_config()
    config.load_config.load_format = load_format
    with pytest.raises(ValueError, match="safetensors loader"):
        patch._resolve_and_verify_engram_config(config)


def test_explicit_shared_page_cache_prefetch_is_preserved():
    config = make_config()
    config.load_config.safetensors_load_strategy = "prefetch"
    patch._resolve_and_verify_engram_config(config)
    assert config.load_config.safetensors_load_strategy == "prefetch"


@pytest.mark.parametrize("case", ["cuda", "other_model", "no_model", "no_layers"])
def test_unrelated_paths_delegate_to_upstream(monkeypatch, case):
    config = make_config()
    if case == "cuda":
        monkeypatch.setattr(patch, "current_platform", SimpleNamespace(device_type="cuda"))
    elif case == "other_model":
        config.model_config.architecture = "Qwen4ExpForCausalLM"
    elif case == "no_model":
        config.model_config = None
    else:
        config.model_config.hf_text_config.engram_layer_ids = []
    original = Mock()
    monkeypatch.setattr(patch, "_ORIGINAL_RESOLVE", original)
    patch._resolve_and_verify_engram_config(config)
    original.assert_called_once_with(config)


@pytest.mark.parametrize("block_size", [None, 32])
def test_v41_chunked_prefill_preserves_32kib_state_page(block_size):
    config = make_config()
    config.model_config.hf_config = SimpleNamespace(model_type="deepseek_v41")
    config.cache_config = CacheConfig(block_size=block_size, enable_prefix_caching=True)
    config.scheduler_config = SimpleNamespace(enable_chunked_prefill=True)
    refresh_block_size(config)
    assert config.cache_config.block_size == 32
    assert config.cache_config.kv_cache_layout == "LBNHC"
    config.cache_config.kv_cache_layout = "LBHNC"
    refresh_block_size(config)
    assert config.cache_config.kv_cache_layout == "LBHNC"


@pytest.mark.parametrize("block_size", [16, 64, 128])
def test_v41_unsupported_state_page_fails_early(block_size):
    config = make_config()
    config.model_config.hf_config = SimpleNamespace(model_type="deepseek_v41")
    config.cache_config = CacheConfig(block_size=block_size)
    config.scheduler_config = SimpleNamespace(enable_chunked_prefill=True)
    with pytest.raises(ValueError, match="block_size=32"):
        refresh_block_size(config)
