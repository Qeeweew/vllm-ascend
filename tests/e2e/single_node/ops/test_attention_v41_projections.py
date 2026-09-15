# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP8 local projection math and changing-position graph replay.

The output collective is replaced by identity to compare this rank's partial
output. This does not validate HCCL or the sparse-attention kernel itself.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.models.deepseek_v4.model import DeepseekV41AttentionProjections


def rotate_reference(x, cos, sin, inverse=False):
    pairs = torch.view_as_complex(x[..., -64:].float().reshape(*x.shape[:-1], 32, 2))
    frequencies = torch.complex(cos, -sin if inverse else sin)
    if x.ndim == 3:
        frequencies = frequencies[:, None]
    rotated = torch.view_as_real(pairs * frequencies).flatten(-2).bfloat16()
    return torch.cat((x[..., :-64], rotated), dim=-1)


@pytest.mark.parametrize("ratio", [0, 1, 2])
@torch.inference_mode()
def test_tp8_projection_reference_and_graph(ratio):
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
        rope_scaling=dict(factor=16, original_max_position_embeddings=65536, beta_fast=32, beta_slow=1),
    )
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
        module = DeepseekV41AttentionProjections(config, ratio, 1024, f"projection{ratio}").npu()
        generator = torch.Generator().manual_seed(1841)
        weights = {}
        for name, parameter in module.named_parameters():
            value = torch.randn(parameter.shape, generator=generator).mul_(0.02).to(parameter.dtype)
            if "norm.weight" in name:
                value.add_(1)
            parameter.copy_(value)
            weights[name] = value
        host = torch.randn((4, 5120), generator=generator).bfloat16()
        hidden = host.npu()
        positions = torch.tensor([0, 1, 126, 127], device="npu", dtype=torch.int64)
        cos, sin = module.rope_cos.cpu(), module.rope_sin.cpu()

        def reference(position):
            projected = (host.float() @ weights["fused_wqa_wkv.weight"].float().T).bfloat16()
            qr, kv = projected.split((1280, 512), dim=-1)

            def norm(x, name):
                x = x.float()
                return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20) * weights[name].float()).bfloat16()

            qr, kv = norm(qr, "q_norm.weight"), norm(kv, "kv_norm.weight")
            q = (qr.float() @ weights["wq_b.weight"].float().T).bfloat16().reshape(4, 8, 512)
            q = rotate_reference(q, cos[position], sin[position])
            kv = rotate_reference(kv, cos[position], sin[position])
            # Deterministic stand-in for attention output, to test inverse RoPE
            # and both output GEMMs separately from the attention kernel.
            output = rotate_reference(q, cos[position], sin[position], inverse=True).flatten(1)
            output = (output.float() @ weights["wo_a.weight"].float().T).bfloat16()
            output = (output.float() @ weights["wo_b.weight"].float().T).bfloat16()
            return qr, q, kv, output

        def run():
            qr, q, kv = module.project_inputs(hidden, positions)
            return qr, q, kv, module.project_output(q, positions)

        for _ in range(3):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = run()
        for ids in ([0, 1, 126, 127], [2, 129, 510, 1023]):
            positions.copy_(torch.tensor(ids))
            graph.replay()
            for actual, expected in zip(result, reference(ids)):
                error = (actual.cpu().float() - expected.float()).square().mean().sqrt()
                scale = expected.float().square().mean().sqrt().clamp_min(1e-10)
                assert error / scale < 0.008
                torch.testing.assert_close(actual.cpu(), expected, rtol=0.025, atol=0.035)
