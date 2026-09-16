# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU draft contract: no distributed startup, NPU initialization or CUDA model."""

import json
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.model_executor import parameter as parameter_module
from vllm.model_executor.layers import linear as linear_module

from vllm_ascend.models.deepseek_v4 import dspark as draft_module
from vllm_ascend.models.deepseek_v4 import model as model_module
from vllm_ascend.ops import linear as ascend_linear_module
from vllm_ascend.ops.mhc_v41 import mhc_pre_delayed_reference


def attach(root, name, value):
    parts = name.split(".")
    for part in parts[:-1]:
        if not hasattr(root, part):
            root.add_module(part, nn.Module())
        root = getattr(root, part)
    setattr(root, parts[-1], value)
    return value


def parameter(shape=(4,), dtype=torch.bfloat16):
    return nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)


def draft_shell(experts=128):
    draft = draft_module.DSparkDeepseekV41ForCausalLM.__new__(draft_module.DSparkDeepseekV41ForCausalLM)
    nn.Module.__init__(draft)
    draft.config = SimpleNamespace(n_routed_experts=experts, num_attention_heads=64)
    draft.model = nn.Module()
    draft.model.num_dspark_layers = 3
    draft.model.confidence_head = nn.Identity()
    return draft


@pytest.fixture
def draft(monkeypatch):
    for module in (draft_module, linear_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 3)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(
        ascend_linear_module,
        "get_parallel_op",
        lambda disable_tp, *args: (None, 0, 1) if disable_tp else (None, 3, 8),
    )
    return draft_shell()


def test_config_draft_expert_counts_are_isolated_from_target():
    original = SimpleNamespace(
        hidden_size=5120,
        hc_mult=4,
        dspark_block_size=5,
        vocab_size=129280,
        dspark_n_routed_experts=128,
        dspark_num_experts_per_tok=3,
        n_routed_experts=384,
        num_experts_per_tok=6,
        vision_n_layers=32,
        num_hash_layers=3,
        engram_layer_ids=[1, 14],
    )
    clone = draft_module._v41_dspark_config(original)
    assert (clone.n_routed_experts, clone.num_experts_per_tok) == (128, 3)
    assert clone.vision_n_layers == clone.num_hash_layers == 0
    assert clone.engram_layer_ids == []
    assert (original.n_routed_experts, original.num_experts_per_tok) == (384, 6)
    assert original.engram_layer_ids == [1, 14]
    original.dspark_n_routed_experts = 384
    with pytest.raises(ValueError, match="128-expert"):
        draft_module._v41_dspark_config(original)


@pytest.mark.parametrize(
    "source,destination",
    [
        ("mtp.0.main_proj.weight", "model.main_proj.weight"),
        ("mtp.0.main_norm.weight", "model.main_norm.weight"),
        ("mtp.2.norm.weight", "model.norm.weight"),
        ("mtp.2.markov_head.embed.weight", "model.markov_head.markov_w1.weight"),
        ("mtp.2.markov_head.head.weight", "model.markov_head.markov_w2.weight"),
        ("mtp.2.confidence_head.proj.weight", "model.confidence_head.proj.weight"),
        ("mtp.1.attn_norm.weight", "model.layers.1.input_layernorm.weight"),
        ("mtp.1.ffn_norm.weight", "model.layers.1.post_attention_layernorm.weight"),
        ("mtp.1.ffn.gate.bias", "model.layers.1.mlp.gate.e_score_correction_bias"),
        ("mtp.0.hc_attn_fn", "model.layers.0.hc_attn_fn"),
    ],
)
def test_released_names_and_model_prefix(draft, source, destination):
    assert draft._remap_dspark_name(source) == destination
    assert draft._remap_dspark_name("model." + source) == destination


@pytest.mark.parametrize("name", ["mtp.1.main_proj.weight", "mtp.0.norm.weight", "mtp.3.attn.wkv.weight"])
def test_invalid_stage_placement_rejected(draft, name):
    with pytest.raises(ValueError):
        draft._remap_dspark_name(name)


def test_only_target_weights_and_explicit_vision_bias_are_skipped(draft):
    assert (
        draft.load_weights(
            [
                ("embed.weight", torch.empty(0)),
                ("head.weight", torch.empty(0)),
                ("mtp.0.ffn.gate.bias_vl", torch.empty(0)),
            ]
        )
        == set()
    )
    for name in ("mtp.0.hc_head_fn", "mtp.0.engram.embed.weight", "mtp.0.embed.weight"):
        with pytest.raises(ValueError, match="Unexpected"):
            draft.load_weights([(name, torch.zeros(4))])
    assert draft.has_own_embed_tokens is False and draft.has_own_lm_head is False
    assert draft.get_draft_attn_causal() == [False, False, False]


def test_real_linear_loaders_replicate_qkv_and_tp_slice_shared_experts(draft):
    qkv = linear_module.MergedColumnParallelLinear(
        8,
        [5, 3],
        bias=False,
        params_dtype=torch.bfloat16,
        disable_tp=True,
    )
    shared = linear_module.MergedColumnParallelLinear(
        8,
        [16, 16],
        bias=False,
        params_dtype=torch.bfloat16,
    )
    attach(draft, "model.layers.0.self_attn.fused_wqa_wkv", qkv)
    attach(draft, "model.layers.0.mlp.shared_experts.gate_up_proj", shared)
    query = torch.arange(40, dtype=torch.bfloat16).reshape(5, 8)
    kv = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
    gate = torch.arange(128, dtype=torch.bfloat16).reshape(16, 8)
    weights = [
        ("mtp.0.attn.wq_a.weight", query),
        ("mtp.0.attn.wkv.weight", kv),
        ("mtp.0.ffn.shared_experts.w1.weight", gate),
        ("mtp.0.ffn.shared_experts.w3.weight", -gate),
    ]
    draft.load_weights(weights)
    torch.testing.assert_close(qkv.weight, torch.cat((query, kv)), rtol=0, atol=0)
    torch.testing.assert_close(shared.weight, torch.cat((gate[6:8], -gate[6:8])), rtol=0, atol=0)
    with pytest.raises(ValueError, match="Missing 1"):
        draft.load_weights(weights[:-1])


@pytest.mark.parametrize(
    "suffix,dtype",
    [
        ("weight_packed", torch.int32),
        ("weight_scale", torch.bfloat16),
        ("weight_shape", torch.int32),
    ],
)
def test_all_128_expert_slices_consumed_without_repacking(draft, suffix, dtype):
    callbacks = {}
    for bank in ("w13", "w2"):
        p = attach(draft, f"model.layers.2.mlp.experts.routed_experts.{bank}_{suffix}", parameter((1,), dtype))
        p.weight_loader = callbacks[bank] = Mock(return_value=True)
    inputs = [
        (f"mtp.2.ffn.experts.{expert}.{projection}.{suffix}", torch.tensor([-1, 1], dtype=dtype))
        for expert in range(128)
        for projection in ("w1", "w2", "w3")
    ]
    draft.load_weights(inputs)
    assert callbacks["w13"].call_count == 256 and callbacks["w2"].call_count == 128
    final = callbacks["w13"].call_args
    assert final.kwargs == {"shard_id": "w3", "expert_id": 127, "return_success": True}
    assert final.args[1] is inputs[-1][1]
    with pytest.raises(ValueError, match="Missing 1"):
        draft.load_weights(inputs[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        draft.load_weights(inputs + inputs[:1])
    with pytest.raises(ValueError, match="Unexpected"):
        draft.load_weights(inputs + [(f"mtp.2.ffn.experts.128.w1.{suffix}", inputs[0][1])])
    callbacks["w2"].return_value = False
    with pytest.raises(ValueError, match="did not consume"):
        draft.load_weights(inputs)


def test_loader_rejects_unconverted_dtype_and_missing_head(draft):
    attach(draft, "model.norm.weight", parameter())
    with pytest.raises(ValueError, match="Missing"):
        draft.load_weights([])
    with pytest.raises(ValueError, match="dtype"):
        draft.load_weights([("mtp.2.norm.weight", torch.zeros(4, dtype=torch.float16))])


def test_sink_tp_slice_and_markov_names_do_not_match_w1_fusion(draft):
    sink = attach(draft, "model.layers.1.self_attn.attn_sink", parameter((8,), torch.float32))
    markov = attach(draft, "model.markov_head.markov_w1.weight", parameter())
    draft.load_weights(
        [
            ("mtp.1.attn.attn_sink", torch.arange(64, dtype=torch.float32)),
            ("mtp.2.markov_head.embed.weight", torch.arange(4, dtype=torch.bfloat16)),
        ]
    )
    torch.testing.assert_close(sink, torch.arange(24, 32, dtype=torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(markov, torch.arange(4, dtype=torch.bfloat16), rtol=0, atol=0)


class ScaleOperation(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states=None, positions=None, **kwargs):
        if "image_token_mask" in kwargs:
            assert not kwargs["image_token_mask"].any()
        return (hidden_states.float() * self.scale).to(hidden_states.dtype)


def test_three_real_v41_blocks_use_delayed_mix_and_prenorm_terminal_hidden(monkeypatch):
    # Replace only device HC primitives with CPU arithmetic, keeping the real
    # V4.1 decoder and new draft loop. Zero projection weights make control
    # mixes analytic, permitting an independent recurrence without HC helpers.
    monkeypatch.setattr(model_module, "mhc_pre_delayed", mhc_pre_delayed_reference)
    monkeypatch.setattr(model_module, "RMSNorm", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(
        model_module,
        "mhc_post",
        lambda x, residual, post, comb: (
            torch.einsum("tsi,tsj->tij", comb, residual.float()) + post[..., None] * x[:, None].float()
        ).to(x.dtype),
    )
    config = SimpleNamespace(hidden_size=5120, rms_norm_eps=1e-20, hc_eps=1e-6, hc_sinkhorn_iters=4)
    blocks = []
    for index in range(3):
        layer = model_module.DeepseekV41DecoderLayer(config, ScaleOperation(0.125), ScaleOperation(-0.0625))
        layer.input_layernorm = nn.Identity()
        layer.post_attention_layernorm = nn.Identity()
        for name, value in layer.named_parameters():
            value.data.zero_()
            if name.endswith("_base"):
                value.data[:4] = torch.tensor([-2.0, -0.5, 0.75, 1.5]) + index * 0.25
        blocks.append(layer)
    model = draft_module.DeepseekV41DSparkModel.__new__(draft_module.DeepseekV41DSparkModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList(blocks)
    model.hidden_size, model.hc_mult = 5120, 4
    model.max_position = 256
    embedded = torch.linspace(-1, 1, 10240).reshape(2, 5120).to(torch.bfloat16)
    ids = torch.tensor([128799, 129264])  # Both remain text in the drafter.
    positions = torch.tensor([127, 128])
    actual = model(ids, positions, inputs_embeds=embedded)
    expected = embedded[:, None].expand(-1, 4, -1).contiguous()
    pre = torch.tensor([[1.0, 0, 0, 0]]).expand(2, -1)
    for index in range(3):
        for factor in (0.125, -0.0625):
            x = (expected.float() * pre[..., None]).sum(1).to(torch.bfloat16)
            x = (x.float() * factor).to(torch.bfloat16)
            mixing = torch.full((2, 4, 4), 0.25 + 1e-6)
            mixing /= mixing.sum(-2, keepdim=True) + 1e-6
            for _ in range(3):
                mixing /= mixing.sum(-1, keepdim=True) + 1e-6
                mixing /= mixing.sum(-2, keepdim=True) + 1e-6
            expected = (torch.einsum("tsi,tsj->tij", mixing, expected.float()) + x[:, None].float()).to(torch.bfloat16)
            pre = (torch.tensor([-2.0, -0.5, 0.75, 1.5]) + index * 0.25).sigmoid()[None] + 1e-6
    expected = (expected.float() * pre[..., None]).sum(1).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not any("hc_head" in name for name, _ in model.named_parameters())


def test_draft_rope_padding_preserves_virtual_positions():
    observed = []
    rope_cache = torch.arange(129)

    class Layer(nn.Module):
        def forward(self, positions, hidden, pre, **kwargs):
            observed.append(rope_cache[positions])
            return hidden, pre

    model = draft_module.DeepseekV41DSparkModel.__new__(draft_module.DeepseekV41DSparkModel)
    nn.Module.__init__(model)
    model.hidden_size, model.hc_mult, model.max_position = 8, 4, 129
    model.layers = nn.ModuleList([Layer(), Layer(), Layer()])
    positions = torch.tensor([126, 127, 128, 129, 130, -1])
    original = positions.clone()
    embedded = torch.ones((6, 8), dtype=torch.bfloat16)
    output = model(torch.zeros(6, dtype=torch.int64), positions, inputs_embeds=embedded)
    assert torch.equal(positions, original)
    assert len(observed) == 3
    assert all(torch.equal(value, torch.tensor([126, 127, 128, 0, 0, 0])) for value in observed)
    assert torch.equal(output, embedded)


def test_context_uses_same_projected_target_and_each_layer_slots(monkeypatch):
    model = draft_module.DeepseekV41DSparkModel.__new__(draft_module.DeepseekV41DSparkModel)
    nn.Module.__init__(model)
    model.hidden_size = 4
    layers, observed = [], []
    for stage in range(3):
        attn = nn.Module()
        attn.q_rank = 2
        attn.fused_wqa_wkv = lambda x, stage=stage: torch.cat((x[:, :2], x * (stage + 1)), dim=1)
        attn.kv_norm = nn.Identity()
        attn.rotate = lambda x, pos: x + pos[:, None]
        attn.swa_cache_layer = SimpleNamespace(prefix=f"layer{stage}", kv_cache=torch.zeros(1))
        layer = nn.Module()
        layer.self_attn = attn
        layers.append(layer)
    model.layers = nn.ModuleList(layers)
    monkeypatch.setattr(
        draft_module, "write_main_cache_v41", lambda cache, x, slots: observed.append((cache, x, slots))
    )
    context = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    positions = torch.tensor([31, 32])
    slots = [torch.tensor([5, 8]), None, torch.tensor([31, 1])]
    model.precompute_and_store_context_kv(context, positions, slots)
    assert len(observed) == 2
    for entry, stage in zip(observed, (0, 2)):
        assert entry[2] is slots[stage]
        torch.testing.assert_close(entry[1], context * (stage + 1) + positions[:, None], rtol=0, atol=0)
    observed.clear()
    model.precompute_and_store_context_kv(context, positions)
    assert not observed  # Profiling projection allocates no cache writes.
    with pytest.raises(ValueError, match="one context slot"):
        model.precompute_and_store_context_kv(context, positions, slots[:1])


def test_real_converted_draft_headers_match_released_contract(draft):
    root = Path("/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32")
    if not (root / "model-00044-of-00048.safetensors").exists():
        pytest.skip("Local converted V4.1 checkpoint is not available")
    counts = []
    for stage in range(3):
        with (root / f"model-{44 + stage:05d}-of-00048.safetensors").open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        names = {key for key in header if key.startswith(f"mtp.{stage}.")}
        counts.append(len(names))
        experts = set()
        for name in names:
            mapped = draft._remap_dspark_name(name)
            assert mapped is not None or name.endswith(".ffn.gate.bias_vl")
            if ".experts." in name:
                experts.add(int(name.split(".experts.")[1].split(".")[0]))
                assert header[name]["dtype"] == ("BF16" if name.endswith("weight_scale") else "I32")
        assert experts == set(range(128))
        assert header[f"mtp.{stage}.ffn.gate.weight"]["shape"] == [128, 5120]
    assert counts == [1176, 1174, 1178]


def test_cpu_suite_did_not_initialize_npu():
    assert not torch.npu.is_initialized()


def test_constructor_uses_only_v41_blocks_and_explicit_draft_attention(monkeypatch):
    target = SimpleNamespace(
        hidden_size=5120,
        hc_mult=4,
        dspark_block_size=5,
        vocab_size=129280,
        dspark_n_routed_experts=128,
        dspark_num_experts_per_tok=3,
        n_routed_experts=384,
        num_experts_per_tok=6,
        vision_n_layers=32,
        num_hash_layers=3,
        engram_layer_ids=[1, 14],
        rms_norm_eps=1e-20,
        hc_eps=1e-6,
        hc_sinkhorn_iters=4,
        dspark_target_layer_ids=[37, 38, 39],
        num_nextn_predict_layers=3,
        num_hidden_layers=40,
        compress_ratios=[0] * 43,
        dspark_markov_rank=256,
    )
    parallel = SimpleNamespace(
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        enable_expert_parallel=False,
        use_sequence_parallel_moe=False,
        enable_eplb=False,
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=target)),
        parallel_config=parallel,
        model_config=SimpleNamespace(max_model_len=512),
        quant_config=object(),
    )
    seen = []

    class DraftAttention(nn.Module):
        def __init__(self, actual_config, layer_id, max_position, prefix, **kwargs):
            super().__init__()
            assert kwargs["is_draft_layer"] is True
            assert actual_config.n_routed_experts == 128 and actual_config.num_experts_per_tok == 3
            assert max_position == 512
            seen.append((layer_id, prefix))
            self.register_buffer("rope_cos", torch.ones(1))
            self.register_buffer("rope_sin", torch.zeros(1))

    def draft_moe(actual_config, actual_parallel, quant_config, **kwargs):
        assert actual_config.n_routed_experts == 128 and actual_config.vision_n_layers == 0
        assert actual_parallel is parallel and kwargs["is_draft_layer"] is True
        return nn.Identity()

    monkeypatch.setattr(draft_module, "DeepseekV41Attention", DraftAttention)
    monkeypatch.setattr(draft_module, "DeepseekV4MoE", draft_moe)
    for name in ("ReplicatedLinear", "VocabParallelEmbedding", "RMSNorm", "DSparkMarkovHead", "DSparkConfidenceHead"):
        monkeypatch.setattr(draft_module, name, lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(model_module, "RMSNorm", lambda *args, **kwargs: nn.Identity())
    result = draft_module.DeepseekV41DSparkModel(vllm_config=config, prefix="draft.model")
    assert seen == [(i, f"draft.model.layers.{i}.self_attn") for i in (40, 41, 42)]
    assert all(type(layer) is model_module.DeepseekV41DecoderLayer and layer.engram is None for layer in result.layers)
    assert target.n_routed_experts == 384 and target.vision_n_layers == 32
    parallel.enable_expert_parallel = True
    with pytest.raises(ValueError, match="does not support EP"):
        draft_module.DeepseekV41DSparkModel(vllm_config=config)
