# SPDX-License-Identifier: Apache-2.0
"""Forty-layer publication/lifetime regression using CPU trace operations.

Real modules, cache registration, metadata, cache-write masks and forward
control flow are retained. Projection, compression, quantization and attention
math are replaced: this is not a checkpoint precision or performance test.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter as parameter_module
from vllm.model_executor.layers import linear as linear_module
from vllm.v1.kv_cache_interface import CircularBufferSpec

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import AscendV41IndexerCacheSpec, AscendV41SWACacheSpec
from vllm_ascend.models.deepseek_v4 import model as model_module
from vllm_ascend.ops import linear as ascend_linear_module
from vllm_ascend.patch.platform.patch_kv_cache_utils import _get_deepseek_v41_kv_cache_groups

# Snapshot of DeepSeek-V4.1-Flash text_config; the trailing three CR0 entries
# belong to draft layers and must not become backbone cache owners.
KV_SOURCES = (2, 8, 14, 20)
INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
KV_OWNER = (None,) * 2 + (2,) * 6 + (8,) * 6 + (14,) * 6 + (20,) * 20
TOPK_OWNER = (None,) * 2 + (2,) * 6 + (8,) * 6 + (14,) * 6 + (20,) * 4 + (24,) * 4 + (28,) * 4 + (32,) * 4 + (36,) * 4


@pytest.fixture
def chain(monkeypatch, request):
    candidate_enabled = getattr(request, "param", False)
    monkeypatch.setattr(
        model_module, "get_ascend_config", lambda: SimpleNamespace(enable_indexer_candidate_decode=candidate_enabled)
    )
    for module in (model_module, linear_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(
        ascend_linear_module, "get_parallel_op", lambda disable_tp, *args: (None, 0, 1 if disable_tp else 8)
    )
    monkeypatch.setattr(ascend_linear_module, "get_replicated_op", lambda *args: (None, 0, 1))
    hf = SimpleNamespace(
        hidden_size=32,
        num_attention_heads=64,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=8,
        rms_norm_eps=1e-6,
        index_n_heads=32,
        index_head_dim=128,
        kv_source_layer_ids=KV_SOURCES,
        index_source_layer_ids=INDEX_SOURCES,
        candidate_source_layer_id=20,
        compress_ratios=[0] * 2 + [2] * 18 + [1] * 20 + [0] * 3,
    )
    config = VllmConfig()
    config.cache_config.block_size = 32
    config.cache_config.kv_cache_layout = "LBNHC"
    config.scheduler_config.max_num_batched_tokens = 4
    config.scheduler_config.max_num_seqs = 1
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.model_config = SimpleNamespace(max_model_len=2048, hf_config=hf, hf_text_config=hf)
    config.parallel_config.tensor_parallel_size = 8
    topk = torch.full((4, 1, 512), -999, dtype=torch.int32)
    candidates = torch.full((4, 1, 2048), -999, dtype=torch.int32)
    rope = (torch.ones(2048, 32), torch.zeros(2048, 32))
    with set_current_vllm_config(config):
        modules = [
            model_module.DeepseekV41Attention(
                hf,
                layer,
                2048,
                f"model.layers.{layer}.self_attn",
                rope,
                vllm_config=config,
                topk_buffer=topk,
                candidate_buffer=candidates,
            )
            for layer in range(40)
        ]
    context = config.compilation_config.static_forward_context
    specs = {name: layer.get_kv_cache_spec(config) for name, layer in context.items()}
    groups = _get_deepseek_v41_kv_cache_groups(config, specs)
    builders = [
        context[group.layer_names[0]]
        .get_attn_backend()
        .get_builder_cls()(group.kv_cache_spec, group.layer_names, config, torch.device("cpu"))
        for group in groups
    ]
    # Separate CPU allocations test source identity, not scheduler-pool aliasing.
    for name, layer in context.items():
        spec = specs[name]
        if isinstance(spec, CircularBufferSpec):
            cache = torch.zeros(1, 1, 8, 1024)
        elif isinstance(spec, AscendV41IndexerCacheSpec):
            key = torch.zeros(64, spec.physical_block_size, 1, 128, dtype=torch.int8)
            cache = (key, torch.zeros(key.shape[:-1], dtype=torch.float16))
        else:
            pages = 8 if isinstance(spec, AscendV41SWACacheSpec) else 64
            cache = torch.full((pages, spec.physical_block_size, 1, 512), -9, dtype=torch.bfloat16)
        layer.bind_kv_cache(cache)
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_sparse_flash_mla_metadata",
        lambda **kwargs: torch.zeros(1024, dtype=torch.int32),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_quant_lightning_indexer_v2_metadata",
        lambda **kwargs: torch.zeros(1024, dtype=torch.int32),
        raising=False,
    )
    return SimpleNamespace(
        modules=modules,
        config=config,
        context=context,
        specs=specs,
        groups=groups,
        builders=builders,
        topk=topk,
        candidates=candidates,
    )


def test_all_backbone_cache_and_projection_owners(chain):
    assert len(chain.context) == 51
    assert sum(len(group.layer_names) for group in chain.groups) == 51
    assert {name for group in chain.groups for name in group.layer_names} == set(chain.context)
    assert all(group.kv_cache_spec.page_size_bytes == 32768 for group in chain.groups)
    for layer, module in enumerate(chain.modules):
        assert (module.compressor is not None) == (layer in KV_SOURCES)
        assert (module.indexer is not None) == (layer in INDEX_SOURCES)
        if module.indexer is not None:
            assert hasattr(module.indexer, "wk") == (layer in KV_SOURCES)
            assert hasattr(module.indexer, "k_norm") == (layer in KV_SOURCES)
        if module.compressor is not None:
            assert (module.compressor.state_cache is not None) == (layer in (2, 8, 14))
        if KV_OWNER[layer] is not None:
            assert module._kv_source_prefix == f"model.layers.{KV_OWNER[layer]}.self_attn"


@pytest.mark.parametrize("chain", [False, True], indirect=True)
def test_candidate_workspaces_only_on_enabled_index_consumers(chain):
    consumers = (24, 28, 32, 36)
    enabled = model_module.get_ascend_config().enable_indexer_candidate_decode
    for layer, module in enumerate(chain.modules):
        if module.selector is None:
            continue
        workspace = module.selector._candidate_selector
        if enabled and layer in consumers:
            assert module.selector.candidate_max_context == chain.config.model_config.max_model_len
            assert workspace.key.device == module.indexer.wq_b.weight.device
            ptr = workspace.key.data_ptr()
            module.selector.prepare_candidate_workspace(workspace.key.device)
            assert workspace.key.data_ptr() == ptr
        else:
            assert workspace is None


def metadata_for_step(chain, positions):
    tokens = positions.numel()
    metadata = {}
    for group, builder in zip(chain.groups, chain.builders):
        spec = group.kv_cache_spec
        table = torch.arange(64, dtype=torch.int32).unsqueeze(0)
        if isinstance(spec, CircularBufferSpec):
            table = torch.zeros(1, 1, dtype=torch.int32)
        elif isinstance(spec, AscendV41SWACacheSpec):
            # Preserve logical columns while retiring old SWA pages. The
            # current chunk plus 127 previous tokens straddles up to six pages.
            first_page = (int(positions[0]) - 127) // 32
            last_page = int(positions[-1]) // 32
            table[:] %= 8
            table[:, :first_page] = -1
            table[:, last_page + 1 :] = -1
        common = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, tokens], dtype=torch.int32),
            seq_lens=torch.tensor([int(positions[-1]) + 1], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=tokens,
            max_query_len=tokens,
            max_seq_len=int(positions[-1]) + 1,
            block_table_tensor=table,
            slot_mapping=torch.full((tokens,), -1, dtype=torch.int64),
            positions=positions,
            is_prefilling=torch.tensor([tokens > 1]),
        )
        value = builder.build(0, common)
        metadata.update({name: value for name in group.layer_names})
    return metadata


@torch.inference_mode()
@pytest.mark.parametrize("fused_stores", [False, True])
def test_full_chain_top512_candidates_and_swa_retirement_across_steps(chain, monkeypatch, fused_stores):
    trace = []
    state = SimpleNamespace(epoch=0, metadata=None)
    monkeypatch.setattr(model_module, "get_forward_context", lambda: SimpleNamespace(attn_metadata=state.metadata))

    def scatter(cache, indices, values):
        valid = indices[:, 0] >= 0
        cache[indices[valid, 0], indices[valid, 1]] = values[valid]
        return cache

    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", scatter, raising=False)

    def fused_main(x, positions, slots, cos, sin, cache, *, compress_ratio=1):
        # Synthetic rotate below adds one; the fused store must receive values
        # BEFORE rotation and ORIGINAL positions, including CR2 odd boundaries.
        assert torch.equal(positions, state.metadata[chain.modules[0].prefix + ".swa_cache"].positions)
        model_module.write_main_cache_v41(
            cache, (x + 1).contiguous(), slots, positions=positions, compress_ratio=compress_ratio
        )

    def fused_index(x, positions, slots, cos, sin, keys, scales, *, compress_ratio=1):
        assert torch.equal(positions, state.metadata[chain.modules[0].prefix + ".swa_cache"].positions)
        model_module.write_index_cache_v41(
            keys,
            scales,
            (x + 1).to(torch.int8),
            torch.ones(x.shape[0], dtype=torch.float16),
            slots,
            positions=positions,
            compress_ratio=compress_ratio,
        )

    monkeypatch.setattr(model_module, "v41_main_cache_store", fused_main)
    monkeypatch.setattr(model_module, "v41_index_cache_store", fused_index)

    def install(layer, module):
        module.enable_fused_cache_store = fused_stores

        def inputs(hidden, positions, *, rotate_kv=True):
            tokens = hidden.shape[0]
            return (
                hidden[:, :16],
                torch.zeros(tokens, 8, 512, dtype=torch.bfloat16),
                torch.full((tokens, 512), layer + (80 if rotate_kv else 79), dtype=torch.bfloat16),
            )

        monkeypatch.setattr(module, "project_inputs", inputs)
        monkeypatch.setattr(module, "rotate", lambda value, positions: (value + 1).contiguous())
        monkeypatch.setattr(module, "project_output", lambda value, positions: value[:, 0, :32])
        if module.compressor is not None:
            monkeypatch.setattr(module.compressor, "project", lambda hidden: hidden)

            def compress(projected, positions, metadata, latent):
                trace.append((state.epoch, "compress", layer))
                latent.fill_(layer)
                return latent

            def key(latent):
                # The index projection must see the pre-RoPE compressor latent.
                assert torch.all(latent == layer)
                return latent[:, :128].contiguous()

            monkeypatch.setattr(module.compressor, "forward", compress)
            monkeypatch.setattr(module.indexer, "project_key", key)
        if module.indexer is not None:
            monkeypatch.setattr(
                module.indexer,
                "project_query",
                lambda hidden, qr: (
                    torch.zeros(hidden.shape[0], 32, 128, dtype=torch.bfloat16),
                    torch.zeros(hidden.shape[0], 32, dtype=torch.float16),
                ),
            )
            monkeypatch.setattr(
                module.selector,
                "quantize",
                lambda value: (
                    value.to(torch.int8),
                    torch.ones(value.shape[:-1], dtype=torch.float16),
                ),
            )

            def select(query, weights, scale, key_cache, scale_cache, metadata, candidates):
                trace.append((state.epoch, "index", layer))
                source = chain.modules[KV_OWNER[layer]]
                assert key_cache is source.index_cache_layer.kv_cache[0]
                assert scale_cache is source.index_cache_layer.kv_cache[1]
                if layer > 20:
                    assert candidates.data_ptr() == chain.candidates.data_ptr()
                    assert torch.all(candidates == state.epoch + 1)
                else:
                    assert candidates is None
                start = layer + 16 * state.epoch
                indices = torch.arange(start, start + 512, dtype=torch.int32)[None, None]
                result = indices.expand(query.shape[0], 1, 512).contiguous()
                # A non-source result must never overwrite candidate layer 20.
                candidate = torch.full(
                    (query.shape[0], 1, 2048), state.epoch + 1 if layer == 20 else -777, dtype=torch.int32
                )
                return result, candidate

            monkeypatch.setattr(module.selector, "select_topk", select)

        def attention(query, swa_cache, sink, metadata, *, cmp_cache, cmp_indices):
            trace.append((state.epoch, "attention", layer))
            swa_metadata = state.metadata[f"{module.prefix}.swa_cache"]
            assert torch.all(swa_cache.view(-1, 512)[swa_metadata.slot_mapping] == layer + 80)
            assert metadata.swa_block_table[0, 0] == -1
            first_live = (int(swa_metadata.positions[0]) - 127) // 32
            assert torch.all(metadata.swa_block_table[0, :first_live] == -1)
            if KV_OWNER[layer] is None:
                assert cmp_cache is cmp_indices is None
            else:
                source = chain.modules[KV_OWNER[layer]]
                assert cmp_cache is source.main_cache_layer.kv_cache
                start = TOPK_OWNER[layer] + 16 * state.epoch
                expected = torch.arange(start, start + 512, dtype=torch.int32)[None, None].expand(
                    query.shape[0], 1, 512
                )
                torch.testing.assert_close(cmp_indices, expected)
                assert torch.all(cmp_indices[..., -1] < (swa_metadata.positions + 1)[:, None] // module.compress_ratio)
                main = state.metadata[f"{source.prefix}.main_cache"]
                valid = main.slot_mapping >= 0
                assert torch.all(cmp_cache.view(-1, 512)[main.slot_mapping[valid]] == KV_OWNER[layer] + 1)
            if layer >= 20:
                assert torch.all(chain.candidates[: query.shape[0]] == state.epoch + 1)
            return query, None

        monkeypatch.setattr(module.sparse, "forward", attention)

    for layer, module in enumerate(chain.modules):
        install(layer, module)
    for epoch, positions in enumerate((torch.arange(1150, 1154), torch.tensor([1154]), torch.arange(1155, 1158))):
        state.epoch = epoch
        state.metadata = metadata_for_step(chain, positions)
        prior_topk, prior_candidates = chain.topk.clone(), chain.candidates.clone()
        hidden = torch.zeros(positions.numel(), 32, dtype=torch.bfloat16)
        for module in chain.modules:
            module(positions, hidden)
        torch.testing.assert_close(chain.topk[positions.numel() :], prior_topk[positions.numel() :])
        torch.testing.assert_close(chain.candidates[positions.numel() :], prior_candidates[positions.numel() :])
        events = [(kind, layer) for generation, kind, layer in trace if generation == epoch]
        assert [layer for kind, layer in events if kind == "compress"] == list(KV_SOURCES)
        assert [layer for kind, layer in events if kind == "index"] == list(INDEX_SOURCES)
        assert [layer for kind, layer in events if kind == "attention"] == list(range(40))
        for source in KV_SOURCES:
            assert (
                events.index(("compress", source))
                < events.index(("index", source))
                < events.index(("attention", source))
            )
