# SPDX-License-Identifier: Apache-2.0
"""Checkpoint routing tests without constructing the 40-layer model.

Dense fusion and TP slicing use real vLLM linear parameter loaders. Expert
tests retain the real upstream expert-name mapping and record the callbacks
that own packed-weight, scale and shape sharding.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.model_executor import parameter as parameter_module
from vllm.model_executor.layers import linear as linear_module

from vllm_ascend.models.deepseek_v4 import model as model_module
from vllm_ascend.ops import linear as ascend_linear_module


def attach(root, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        if not hasattr(root, part):
            root.add_module(part, nn.Module())
        root = getattr(root, part)
    setattr(root, parts[-1], value)
    return value


def param(shape=(4,), dtype=torch.bfloat16):
    return nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)


@pytest.fixture
def model(monkeypatch):
    # Deliberately use nonzero TP rank: replicated fused projections must not
    # accidentally inherit rank 3 slicing from a global parameter default.
    for module in (model_module, linear_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 3)
    for module in (model_module, linear_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 8)
    # Full UT bootstrap installs Ascend linear subclasses. Keep their genuine
    # parameter loaders while supplying the absent distributed group choice.
    monkeypatch.setattr(
        ascend_linear_module,
        "get_parallel_op",
        lambda disable_tp, *args: (None, 0, 1) if disable_tp else (None, 3, 8),
    )
    monkeypatch.setattr(model_module, "get_ascend_config", lambda: SimpleNamespace(mix_placement=False))
    monkeypatch.setattr(model_module, "is_pp_missing_parameter", lambda *args: False)
    monkeypatch.setattr(model_module, "enable_dsa_cp", lambda: False)
    monkeypatch.setattr(model_module.rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: False)
    result = model_module.AscendDeepseekV41ForCausalLM.__new__(model_module.AscendDeepseekV41ForCausalLM)
    nn.Module.__init__(result)
    result.model = nn.Module()
    result.config = SimpleNamespace(
        n_routed_experts=2, n_shared_experts=1, num_attention_heads=64, num_hidden_layers=40
    )
    result.num_redundant_experts = 0
    return result


def merged(output_sizes, *, replicated=True):
    return linear_module.MergedColumnParallelLinear(
        8, output_sizes, bias=False, params_dtype=torch.bfloat16, quant_config=None, disable_tp=replicated
    )


@pytest.mark.parametrize("prefix", ["", "model."])
def test_replicated_query_kv_and_compressor_fusions(model, prefix):
    attention = attach(model, "model.layers.2.self_attn.fused_wqa_wkv", merged([5, 3]))
    compressor = attach(model, "model.layers.2.self_attn.compressor.fused_wkv_wgate", merged([3, 3]))
    values = [torch.full((rows, 8), float(i + 1), dtype=torch.bfloat16) for i, rows in enumerate((5, 3, 3, 3))]
    names = ["wq_a", "wkv", "compressor.wkv", "compressor.wgate"]
    loaded = model.load_weights((f"{prefix}layers.2.attn.{name}.weight", value) for name, value in zip(names, values))
    torch.testing.assert_close(attention.weight, torch.cat(values[:2]))
    torch.testing.assert_close(compressor.weight, torch.cat(values[2:]))
    assert loaded == {
        "model.layers.2.self_attn.fused_wqa_wkv.weight",
        "model.layers.2.self_attn.compressor.fused_wkv_wgate.weight",
    }
    assert attention.tp_rank == compressor.tp_rank == 0


def test_cr1_compressor_loads_only_the_latent_projection(model):
    compressor = attach(model, "model.layers.24.self_attn.compressor.fused_wkv_wgate", merged([3]))
    value = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
    assert model.load_weights([("layers.24.attn.compressor.wkv.weight", value)]) == {
        "model.layers.24.self_attn.compressor.fused_wkv_wgate.weight"
    }
    torch.testing.assert_close(compressor.weight, value)


def test_engram_host_tables_are_skipped_but_gate_and_projection_are_loaded(model):
    names = ("q_weight", "k_weight", "wkv.weight")
    values = [torch.full((4,), float(i + 1), dtype=torch.bfloat16) for i in range(3)]
    for name in names:
        attach(model, f"model.layers.1.engram.{name}", param())
    loaded = model.load_weights(
        [("layers.1.engram.embed.weight", torch.empty(0)), ("model.layers.1.engram.embed.scale", torch.empty(0))]
        + [(f"layers.1.engram.{name}", value) for name, value in zip(names, values)]
    )
    assert loaded == {f"model.layers.1.engram.{name}" for name in names}
    for name, value in zip(names, values):
        torch.testing.assert_close(dict(model.named_parameters())[f"model.layers.1.engram.{name}"], value)


@pytest.mark.parametrize("source,shard,destination", [("w1", "w1", "w13"), ("w3", "w3", "w13"), ("w2", "w2", "w2")])
@pytest.mark.parametrize(
    "suffix,dtype,shape",
    [
        ("weight_packed", torch.int32, (16, 8)),
        ("weight_scale", torch.bfloat16, (16, 2)),
        ("weight_shape", torch.int32, (2,)),
    ],
)
def test_routed_expert_suffixes_delegate_unchanged_to_tp_callback(
    model, source, shard, destination, suffix, dtype, shape
):
    name = f"model.layers.0.mlp.experts.routed_experts.{destination}_{suffix}"
    target = attach(model, name, param((1,), dtype))
    loader = Mock(return_value=True)
    target.weight_loader = loader
    weight = torch.ones(shape, dtype=dtype)
    loaded = model.load_weights([(f"layers.0.ffn.experts.1.{source}.{suffix}", weight)])
    assert loaded == {name}
    loader.assert_called_once()
    args, kwargs = loader.call_args
    assert args[0] is target and args[1] is weight and args[2] == name
    assert kwargs == {"shard_id": shard, "expert_id": 1, "return_success": True}


def test_shared_expert_gate_up_uses_real_tp_slicing(model):
    fused = attach(model, "model.layers.0.mlp.shared_experts.gate_up_proj", merged([16, 16], replicated=False))
    gate = torch.arange(128, dtype=torch.bfloat16).reshape(16, 8)
    up = -gate
    loaded = model.load_weights(
        [("layers.0.ffn.shared_experts.w1.weight", gate), ("layers.0.ffn.shared_experts.w3.weight", up)]
    )
    torch.testing.assert_close(fused.weight, torch.cat((gate[6:8], up[6:8])))
    assert loaded == {"model.layers.0.mlp.shared_experts.gate_up_proj.weight"}


def test_dense_head_norm_router_and_hc_mapping(model):
    mapping = {
        "embed.weight": "model.embed_tokens.weight",
        "head.weight": "lm_head.weight",
        "norm.weight": "model.norm.weight",
        "layers.0.attn_norm.weight": "model.layers.0.input_layernorm.weight",
        "layers.0.ffn_norm.weight": "model.layers.0.post_attention_layernorm.weight",
        "layers.0.attn.q_norm.weight": "model.layers.0.self_attn.q_norm.weight",
        "layers.0.attn.kv_norm.weight": "model.layers.0.self_attn.kv_norm.weight",
        "layers.0.attn.indexer.weights_proj.weight": "model.layers.0.self_attn.indexer.weights_proj.weight",
        "layers.0.ffn.gate.bias": "model.layers.0.mlp.gate.e_score_correction_bias",
        "layers.0.ffn.gate.bias_vl": "model.layers.0.mlp.gate.bias_vl",
        "layers.0.ffn.gate.weight": "model.layers.0.mlp.gate.weight",
        "layers.0.hc_attn_fn": "model.layers.0.hc_attn_fn",
        "layers.0.hc_ffn_scale": "model.layers.0.hc_ffn_scale",
    }
    values = {}
    for i, (source, target) in enumerate(mapping.items()):
        attach(model, target, param())
        values[source] = torch.full((4,), i + 1, dtype=torch.bfloat16)
    assert model.load_weights(values.items()) == set(mapping.values())
    for source, target in mapping.items():
        torch.testing.assert_close(dict(model.named_parameters())[target], values[source])


def test_attention_sinks_are_sliced_by_rank(model):
    sink = attach(model, "model.layers.0.self_attn.attn_sink", param((8,), torch.float32))
    model.load_weights([("layers.0.attn.attn_sink", torch.arange(64, dtype=torch.float32))])
    torch.testing.assert_close(sink, torch.arange(24, 32, dtype=torch.float32))


@pytest.mark.parametrize("name", ["wq_b", "wo_a"])
def test_dense_column_projections_keep_tp_loader(model, name):
    projection = attach(
        model,
        f"model.layers.0.self_attn.{name}",
        linear_module.ColumnParallelLinear(8, 16, bias=False, params_dtype=torch.bfloat16),
    )
    weight = torch.arange(128, dtype=torch.bfloat16).reshape(16, 8)
    model.load_weights([(f"layers.0.attn.{name}.weight", weight)])
    torch.testing.assert_close(projection.weight, weight[6:8])


def test_output_projection_keeps_tp_row_loader(model):
    projection = attach(
        model,
        "model.layers.0.self_attn.wo_b",
        linear_module.RowParallelLinear(16, 8, bias=False, params_dtype=torch.bfloat16),
    )
    weight = torch.arange(128, dtype=torch.bfloat16).reshape(8, 16)
    model.load_weights([("layers.0.attn.wo_b.weight", weight)])
    torch.testing.assert_close(projection.weight, weight[:, 6:8])


@pytest.mark.parametrize(
    "name",
    [
        "vision.blocks.0.weight",
        "mtp.layers.0.weight",
        "aligner.w1.weight",
        "aligner.w2.bias",
        "image_start",
        "image_end",
        "image_newline",
    ],
)
def test_text_loader_skips_separate_vision_and_draft_towers(model, name):
    assert model.load_weights([(name, torch.empty(0))]) == set()
