# SPDX-License-Identifier: Apache-2.0
"""Planner -> runner views -> registered caches -> metadata -> NPU graph.

Projection ranks/hidden width are reduced to keep the wiring test lightweight;
native attention/cache/indexer dimensions remain H8/D512/D128. Production
projection dimensions and performance have separate component coverage.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import test_dsa_v41 as attention_test
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import CircularBufferSpec
from vllm.v1.worker.utils import extract_layer_index

from vllm_ascend.models.deepseek_v4.model import DeepseekV41Attention
from vllm_ascend.ops.compressor_v41 import compressor_v41_reference
from vllm_ascend.patch.platform.patch_kv_cache_utils import _get_deepseek_v41_kv_cache_groups
from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.fixture(scope="module")
def runtime():
    attention_test.runtime.__wrapped__()


def small_config():
    return SimpleNamespace(
        hidden_size=32,
        num_attention_heads=64,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=8,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        compress_rope_theta=160000,
        rope_scaling=dict(factor=16, original_max_position_embeddings=65536),
        index_n_heads=32,
        index_head_dim=128,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        compress_ratios=[0] * 2 + [2] * 18 + [1] * 20,
    )


def allocate_and_bind(config, device):
    context = config.compilation_config.static_forward_context
    specs = {name: layer.get_kv_cache_spec(config) for name, layer in context.items()}
    groups = _get_deepseek_v41_kv_cache_groups(config, specs)
    plan = get_kv_cache_config_from_groups(config, groups, 16 * 1024 * 1024)
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = device
    runner.vllm_config = config
    runner.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    runner.runner_only_attn_layers = set()
    attn_groups = [
        SimpleNamespace(
            layer_names=group.layer_names,
            kv_cache_spec=group.kv_cache_spec,
            backend=context[group.layer_names[0]].get_attn_backend(),
        )
        for group in groups
    ]
    runner._kv_cache_spec_attn_group_iterator = lambda: attn_groups
    raw = runner._allocate_kv_cache_tensors(plan)
    views = runner._reshape_kv_cache_tensors(plan, raw)
    runner_caches = []
    bind_kv_cache(views, context, runner_caches, kv_cache_groups=groups)
    ordered_names = sorted(views, key=extract_layer_index)
    assert len(runner_caches) == len(views)
    assert all(cache is views[name] for cache, name in zip(runner_caches, ordered_names))
    builders = [
        group.backend.get_builder_cls()(group.kv_cache_spec, group.layer_names, config, device) for group in attn_groups
    ]
    assert len({tensor.untyped_storage().data_ptr() for tensor in raw.values()}) == 1
    return plan, raw, views, builders


def build_metadata(plan, builders, positions, cu, lengths, *, changed=False):
    metadata = {}
    for group_id, (group, builder) in enumerate(zip(plan.kv_cache_groups, builders)):
        # Cache groups overlay the same backing: page IDs must differ between
        # groups, as in the scheduler's shared block pool. Zero is reserved.
        page = group_id + 1 + (len(builders) if changed else 0)
        assert page < plan.num_blocks
        columns = 1 if isinstance(group.kv_cache_spec, CircularBufferSpec) else 4
        table = torch.full((1, columns), -1, dtype=torch.int32, device=positions.device)
        table[:, 0] = page
        common = SimpleNamespace(
            positions=positions,
            query_start_loc=cu,
            seq_lens=lengths,
            num_reqs=1,
            num_actual_tokens=positions.numel(),
            max_seq_len=positions.numel(),
            slot_mapping=torch.full_like(positions, 999999),
            block_table_tensor=table,
        )
        built = builder.build(0, common)
        for name in group.layer_names:
            metadata[name] = built
    return metadata


@pytest.mark.parametrize("layers", [(0, 1), (2, 3), (2, 8, 9), (20, 24, 25)])
@torch.inference_mode()
def test_registered_source_consumer_planner_views_and_graph(runtime, layers):
    device = torch.device("npu:2")
    hf = small_config()
    config = VllmConfig()
    config.cache_config.block_size = 32
    config.cache_config.kv_cache_layout = "LBNHC"
    config.cache_config.num_gpu_blocks_override = 64
    config.scheduler_config.max_num_batched_tokens = 8
    config.scheduler_config.max_num_seqs = 2
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.model_config = SimpleNamespace(max_model_len=128, hf_config=hf)
    config.parallel_config.tensor_parallel_size = 8
    tokens = 6
    topk = torch.full((8, 1, 512), -1, dtype=torch.int32, device=device)
    candidates = torch.full((8, 1, 2048), -1, dtype=torch.int32, device=device)
    generator = torch.Generator().manual_seed(719)
    with ExitStack() as stack:
        stack.enter_context(set_current_vllm_config(config))
        stack.enter_context(
            patch(
                "vllm_ascend.models.deepseek_v4.model.get_ascend_config",
                return_value=SimpleNamespace(enable_indexer_candidate_decode=False),
            )
        )
        for namespace in (
            "vllm.model_executor.layers.linear",
            "vllm.model_executor.parameter",
            "vllm_ascend.models.deepseek_v4.model",
        ):
            stack.enter_context(patch(f"{namespace}.get_tensor_model_parallel_world_size", return_value=8))
            stack.enter_context(patch(f"{namespace}.get_tensor_model_parallel_rank", return_value=0))
        stack.enter_context(
            patch("vllm.model_executor.layers.linear.tensor_model_parallel_all_reduce", side_effect=lambda value: value)
        )
        modules = {}
        for layer_id in layers:
            module = DeepseekV41Attention(
                hf,
                layer_id,
                128,
                f"model.layers.{layer_id}.self_attn",
                vllm_config=config,
                topk_buffer=topk,
                candidate_buffer=candidates,
            ).to(device)
            for name, parameter in module.named_parameters():
                value = torch.randn(parameter.shape, generator=generator).mul_(0.02).to(parameter.dtype)
                if "norm.weight" in name:
                    value.add_(1)
                parameter.copy_(value)
            modules[layer_id] = module
        plan, raw, views, builders = allocate_and_bind(config, device)
        positions = torch.arange(tokens, dtype=torch.int64, device=device)
        cu = torch.tensor([0, tokens], dtype=torch.int32, device=device)
        lengths = torch.tensor([tokens], dtype=torch.int32, device=device)
        metadata = build_metadata(plan, builders, positions, cu, lengths)
        context = SimpleNamespace(attn_metadata=metadata)
        stack.enter_context(patch("vllm_ascend.models.deepseek_v4.model.get_forward_context", return_value=context))
        resolved = {layer: module._resolve_batch(metadata) for layer, module in modules.items()}
        for layer, module in modules.items():
            batch = resolved[layer]
            assert batch.swa_cache is views[f"{module.prefix}.swa_cache"]
            if not module.compress_ratio:
                assert batch.main_cache is batch.indexer is batch.compressor is None
                continue
            expected_source = max(source for source in hf.kv_source_layer_ids if source <= layer)
            source = modules[expected_source]
            assert batch.main_cache is views[f"{source.prefix}.main_cache"]
            assert batch.index_cache is views[f"{source.prefix}.indexer.k_cache"][0]
            assert batch.index_scale_cache is views[f"{source.prefix}.indexer.k_cache"][1]
            assert batch.topk is topk and batch.candidates is candidates
            assert (batch.indexer is not None) == module.is_index_source
            assert (batch.compressor is not None) == module.is_kv_source
            if module.compressor is not None and module.compressor.state_cache is not None:
                state = module.compressor.state_cache
                assert batch.compressor is metadata[state.prefix]
                assert state.kv_cache.is_contiguous()
                assert state.kv_cache.data_ptr() == views[state.prefix].data_ptr()
        hidden = torch.randn((tokens, hf.hidden_size), generator=generator).bfloat16().to(device)

        def run():
            # Exercise the registered dictionary branch, never explicit batch.
            return {layer: module(positions, hidden) for layer, module in modules.items()}

        for _ in range(3):
            run()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = run()
        for changed in (False, True):
            hidden.copy_(torch.randn(hidden.shape, generator=generator).bfloat16())
            for tensor in raw.values():
                tensor.zero_()
            context.attn_metadata = build_metadata(plan, builders, positions, cu, lengths, changed=changed)
            graph.replay()
            for layer, module in modules.items():
                batch = module._resolve_batch(context.attn_metadata)
                swa_meta = context.attn_metadata[f"{module.prefix}.swa_cache"]
                swa_page = int(swa_meta.block_table[0, 0].cpu())
                _, query, swa = module.project_inputs(hidden, positions)
                torch.testing.assert_close(batch.swa_cache[swa_page, :tokens, 0].cpu(), swa.cpu(), rtol=0, atol=0)
                main = None
                if module.compress_ratio:
                    source = modules[max(source for source in hf.kv_source_layer_ids if source <= layer)]
                    source_batch = source._resolve_batch(context.attn_metadata)
                    projected = source.compressor.project(hidden).cpu()
                    latent, _ = compressor_v41_reference(
                        projected,
                        positions.cpu(),
                        positions.cpu(),
                        cu.cpu(),
                        torch.zeros(tokens, dtype=torch.int32),
                        source.compressor.norm.weight.cpu(),
                        torch.zeros((1, 8, 1024)) if module.compress_ratio == 2 else torch.empty(0),
                        module.compress_ratio,
                        hf.rms_norm_eps,
                    )
                    main = source.rotate(latent.to(device), positions // module.compress_ratio * module.compress_ratio)
                    main = main.cpu()[module.compress_ratio - 1 :: module.compress_ratio]
                    main_page = int(context.attn_metadata[f"{source.prefix}.main_cache"].block_table[0, 0].cpu())
                    torch.testing.assert_close(
                        source_batch.main_cache[main_page, : tokens // module.compress_ratio, 0].cpu(),
                        main,
                        rtol=0.01,
                        atol=0.01,
                    )
                    ids = batch.topk[:tokens].cpu()
                    for row in range(tokens):
                        visible = (row + 1) // module.compress_ratio
                        assert ids[row, 0, :visible].tolist() == list(range(visible))
                        assert torch.all(ids[row, 0, visible:] == -1)
                output = []
                for row in range(tokens):
                    kv = swa.cpu()[: row + 1].float()
                    if main is not None:
                        kv = torch.cat((kv, main[: (row + 1) // module.compress_ratio].float()))
                    logits = query.cpu()[row].float() @ kv.T / 512**0.5
                    logits = torch.cat((logits, module.attn_sink.cpu()[:, None]), dim=-1)
                    output.append(logits.softmax(-1)[:, :-1] @ kv)
                expected = module.project_output(torch.stack(output).bfloat16().to(device), positions)
                torch.testing.assert_close(actual[layer].cpu(), expected.cpu(), rtol=0.025, atol=0.035)
                error = (actual[layer].cpu().float() - expected.cpu().float()).square().mean().sqrt()
                norm = expected.cpu().float().square().mean().sqrt().clamp_min(1e-8)
                assert float(error / norm) < 0.012
