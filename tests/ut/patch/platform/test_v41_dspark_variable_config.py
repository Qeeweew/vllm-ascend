# SPDX-License-Identifier: Apache-2.0
"""Actual EngineArgs configuration of V4.1 parallel draft query lengths on CPU."""

import json
from types import SimpleNamespace

import pytest
import torch
from vllm.engine.arg_utils import EngineArgs

from vllm_ascend.patch.platform import patch_speculative_config as patch


@pytest.fixture(scope="module")
def config_directory(tmp_path_factory):
    path = tmp_path_factory.mktemp("v41_dspark_config")
    config = {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "dtype": "bfloat16",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "text_config": {
            "model_type": "deepseek_v41_text",
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1280,
            "o_lora_rank": 1024,
            "vocab_size": 129280,
            "n_routed_experts": 384,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 2304,
            "max_position_embeddings": 1048576,
            "sliding_window": 128,
            "num_nextn_predict_layers": 3,
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [37, 38, 39],
            "dspark_noise_token_id": 128799,
            "dspark_n_routed_experts": 128,
            "dspark_num_experts_per_tok": 3,
            "engram_layer_ids": [],
            "hc_mult": 4,
            "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        },
        "vision_config": {"num_hidden_layers": 0},
    }
    (path / "config.json").write_text(json.dumps(config) + "\n")
    return path


def engine_config(directory, tokens):
    return EngineArgs(
        model=str(directory),
        tokenizer=str(directory),
        tensor_parallel_size=8,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=1,
        dtype="bfloat16",
        enforce_eager=True,
        block_size=32,
        async_scheduling=False,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0},
        speculative_config={"method": "dspark", "num_speculative_tokens": tokens},
    ).create_engine_config()


@pytest.fixture(scope="module")
def original_target_config(config_directory):
    return engine_config(config_directory, 5).model_config.hf_config.to_dict()


@pytest.mark.parametrize("tokens", range(1, 9))
def test_real_engine_args_accepts_v41_parallel_query_lengths(config_directory, original_target_config, tokens):
    before = (config_directory / "config.json").read_bytes()
    config = engine_config(config_directory, tokens)
    draft = config.speculative_config.draft_model_config
    assert config.speculative_config.num_speculative_tokens == tokens
    assert config.speculative_config.parallel_drafting
    assert draft.architectures == ["DSparkV41DraftModel"]
    assert draft.hf_config.n_predict == tokens
    assert draft.hf_config.dspark_block_size == 5
    assert draft.hf_config.num_nextn_predict_layers == 3
    assert config.model_config.hf_config.to_dict() == original_target_config
    assert (config_directory / "config.json").read_bytes() == before
    assert torch.npu.is_initialized() is False


@pytest.mark.parametrize("tokens", [6, 7, 8])
def test_upstream_mtp_reuse_rule_rejects_same_real_dspark_configs(monkeypatch, config_directory, tokens):
    # Counterfactual uses the unchanged upstream method inside the same real
    # EngineArgs path, proving why the architecture-specific normalization exists.
    monkeypatch.setattr(patch.SpeculativeConfig, "update_arch_", patch._orig_update_arch)
    with pytest.raises(ValueError, match="must be divisible"):
        engine_config(config_directory, tokens)
    assert torch.npu.is_initialized() is False


@pytest.mark.parametrize(
    "method,model_type,architecture,tokens",
    [
        ("mtp", "deepseek_v41", "DSparkV41DraftModel", 7),
        ("dspark", "deepseek_v4", "DSparkDraftModel", 7),
        ("dspark", "qwen3", "Qwen3DSparkModel", 7),
        ("dspark", "deepseek_v41", "DeepseekV41ForCausalLM", 7),
        ("dspark", "deepseek_v41", "DSparkV41DraftModel", None),
    ],
)
def test_other_architectures_methods_and_default_are_unchanged(monkeypatch, method, model_type, architecture, tokens):
    calls = []
    monkeypatch.setattr(patch, "_orig_update_arch", calls.append)
    draft = SimpleNamespace(model_type=model_type, architectures=[architecture], n_predict=5, dspark_block_size=5)
    config = SimpleNamespace(
        method=method, num_speculative_tokens=tokens, draft_model_config=SimpleNamespace(hf_config=draft)
    )
    patch._dspark_update_arch(config)
    assert calls == [config]
    assert draft.n_predict == draft.dspark_block_size == 5
