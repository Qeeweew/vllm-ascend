# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real projection/compressor/cache/indexer/attention chain at TP8 local shape."""

from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.models.deepseek_v4.compressor import CompressorV41Metadata
from vllm_ascend.models.deepseek_v4.model import DeepseekV41Attention, DeepseekV41AttentionBatch
from vllm_ascend.ops.compressor_v41 import compressor_v41_reference
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("layer", [0, 2, 20])
@torch.inference_mode()
def test_source_and_consumer_attention_reference_and_graph(layer):
    assert enable_custom_op()
    config = SimpleNamespace(
        hidden_size=5120,
        num_attention_heads=64,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=1280,
        o_lora_rank=1024,
        o_groups=8,
        rms_norm_eps=1e-20,
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
    device = torch.device("npu")
    ratio, tokens, page = config.compress_ratios[layer], 6, 32
    with ExitStack() as stack:
        stack.enter_context(set_current_vllm_config(VllmConfig()))
        for namespace in (
            "vllm.model_executor.layers.linear",
            "vllm.model_executor.parameter",
            "vllm_ascend.models.deepseek_v4.model",
        ):
            stack.enter_context(patch(f"{namespace}.get_tensor_model_parallel_world_size", return_value=8))
            stack.enter_context(patch(f"{namespace}.get_tensor_model_parallel_rank", return_value=0))
        stack.enter_context(
            patch("vllm.model_executor.layers.linear.tensor_model_parallel_all_reduce", side_effect=lambda x: x)
        )
        module = DeepseekV41Attention(config, layer, 128, f"model.layers.{layer}.attn").to(device)
        consumer = DeepseekV41Attention(
            config,
            layer + 1,
            128,
            f"model.layers.{layer + 1}.attn",
            rope_cache=(module.rope_cos, module.rope_sin),
        ).to(device)
        assert consumer.rope_cos.data_ptr() == module.rope_cos.data_ptr()
        assert consumer.rope_sin.data_ptr() == module.rope_sin.data_ptr()
        gen = torch.Generator().manual_seed(415)
        for model in (module, consumer):
            for name, parameter in model.named_parameters():
                value = torch.randn(parameter.shape, generator=gen).mul_(0.02).to(parameter.dtype)
                if "norm.weight" in name:
                    value.add_(1)
                parameter.copy_(value)
        positions = torch.arange(tokens, dtype=torch.int64, device=device)
        cu = torch.tensor([0, tokens], dtype=torch.int32, device=device)
        lengths = torch.tensor([tokens], dtype=torch.int32, device=device)
        table = torch.zeros((1, 1), dtype=torch.int32, device=device)
        metadata = module.sparse.build_metadata(
            cu,
            lengths,
            table,
            max_seqlen_q=tokens,
            max_seqlen_kv=tokens,
            cmp_block_table=table if ratio else None,
        )
        batch = DeepseekV41AttentionBatch(
            torch.zeros((1, page, 1, 512), dtype=torch.bfloat16, device=device),
            positions.clone(),
            metadata,
        )
        if ratio:
            batch.main_cache = torch.zeros_like(batch.swa_cache)
            batch.main_slots = positions // ratio
            batch.index_slots = positions // ratio
            batch.index_cache = torch.zeros((1, page, 1, 128), dtype=torch.int8, device=device)
            batch.index_scale_cache = torch.ones((1, page, 1), dtype=torch.float16, device=device)
            batch.topk = torch.full((tokens, 1, 512), -1, dtype=torch.int32, device=device)
            batch.candidates = torch.full((tokens, 1, 2048), -1, dtype=torch.int32, device=device)
            batch.compressor = CompressorV41Metadata(
                positions.clone(), cu, torch.zeros_like(positions, dtype=torch.int32)
            )
            batch.indexer = module.selector.build_metadata(
                cu,
                lengths // ratio,
                table,
                max_seqlen_q=tokens,
                max_seqlen_k=tokens,
                cmp_residual_k=lengths % ratio if ratio == 2 else None,
            )
            if ratio == 2:
                module.compressor.state_cache.bind_kv_cache(torch.zeros((1, 1, 8, 1024), device=device))
        second_batch = replace(batch, swa_cache=torch.zeros_like(batch.swa_cache))
        hidden = torch.randn((tokens, 5120), generator=gen).bfloat16().to(device)

        def run():
            first = module(positions, hidden, batch)
            second = consumer(positions, hidden, second_batch)
            return first, second

        for _ in range(3):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = run()
        for _ in range(2):
            hidden.copy_(torch.randn(hidden.shape, generator=gen).bfloat16())
            graph.replay()
            # Cache reference uses independent vector compression math. Main
            # and index consumers must both see the same pre-RoPE latent.
            if ratio:
                projected = module.compressor.project(hidden).cpu()
                latent, _ = compressor_v41_reference(
                    projected,
                    positions.cpu(),
                    positions.cpu(),
                    cu.cpu(),
                    torch.zeros(tokens, dtype=torch.int32),
                    module.compressor.norm.weight.cpu(),
                    torch.zeros((1, 8, 1024)) if ratio == 2 else torch.empty(0),
                    ratio,
                )
                rotated = module.rotate(latent.to(device), positions // ratio * ratio).cpu()
                expected_main = rotated[ratio - 1 :: ratio]
                torch.testing.assert_close(
                    batch.main_cache.cpu()[0, : tokens // ratio, 0], expected_main, rtol=0.01, atol=0.01
                )
                for row in range(tokens):
                    visible = (row + 1) // ratio
                    assert batch.topk.cpu()[row, 0, :visible].tolist() == list(range(visible))
                    assert (batch.topk.cpu()[row, 0, visible:] == -1).all()
            for model, buffers, got in zip((module, consumer), (batch, second_batch), actual):
                _, q, swa = model.project_inputs(hidden, positions)
                torch.testing.assert_close(buffers.swa_cache.cpu()[0, :tokens, 0], swa.cpu(), rtol=0, atol=0)
                out = []
                for row in range(tokens):
                    kv = swa.cpu()[: row + 1].float()
                    if ratio:
                        kv = torch.cat((kv, batch.main_cache.cpu()[0, : (row + 1) // ratio, 0].float()))
                    logits = q.cpu()[row].float() @ kv.T / 512**0.5
                    logits = torch.cat((logits, model.attn_sink.cpu()[:, None]), dim=-1)
                    out.append(logits.softmax(-1)[:, :-1] @ kv)
                expected = model.project_output(torch.stack(out).bfloat16().to(device), positions)
                torch.testing.assert_close(got.cpu(), expected.cpu(), rtol=0.025, atol=0.035)
