# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.models.deepseek_v4.compressor import CompressorV41StateCache
from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache


def test_worker_binding_keeps_raw_connector_tensor_and_unpacks_compressor_ring():
    with set_current_vllm_config(VllmConfig()):
        cache = CompressorV41StateCache("model.layers.2.self_attn.compressor.state_cache")
    raw = torch.zeros((3, 1, 8, 1024), dtype=torch.float32)
    runner_caches = []
    bind_kv_cache({cache.prefix: raw}, {cache.prefix: cache}, runner_caches)
    assert runner_caches[0] is raw
    assert cache.kv_cache.shape == (3, 8, 1024)
    assert cache.kv_cache.data_ptr() == raw.data_ptr()
    cache.kv_cache[2, 7, 13] = 42
    assert raw[2, 0, 7, 13] == 42


def test_legacy_layers_without_binding_hook_still_receive_raw_tensor():
    layer = SimpleNamespace()
    raw = torch.zeros((2, 16, 1, 512), dtype=torch.bfloat16)
    runner_caches = []
    bind_kv_cache({"model.layers.1.attn": raw}, {"model.layers.1.attn": layer}, runner_caches)
    assert layer.kv_cache is raw
    assert runner_caches[0] is raw
