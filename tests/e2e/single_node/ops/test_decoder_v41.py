# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A whole delayed-mHC block with deterministic attention and MoE sublayers."""

from types import SimpleNamespace

import torch
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.models.deepseek_v4.model import DeepseekV41DecoderLayer
from vllm_ascend.ops.mhc_v41 import mhc_pre_delayed_reference
from vllm_ascend.utils import enable_custom_op


class Attention(nn.Module):
    def forward(self, positions, hidden_states):
        return hidden_states * 0.25


class MoE(nn.Module):
    def forward(self, hidden_states, input_ids, hidden_states_fp32, image_token_mask=None):
        assert image_token_mask is self.expected_image_mask
        return hidden_states * 0.5


def test_decoder_carries_attention_mix_into_ffn_and_ffn_mix_to_next_layer():
    assert enable_custom_op()
    config = SimpleNamespace(hidden_size=5120, rms_norm_eps=1e-20, hc_eps=1e-6, hc_sinkhorn_iters=20)
    with set_current_vllm_config(VllmConfig()):
        module = DeepseekV41DecoderLayer(config, Attention(), MoE()).npu()
    image_mask = torch.tensor([False, True, False], device="npu")
    module.mlp.expected_image_mask = image_mask
    gen = torch.Generator().manual_seed(812)
    controls = {}
    with torch.no_grad():
        for key, param in module.named_parameters():
            if key.endswith("_fn"):
                value = torch.randn(param.shape, generator=gen) * 0.001
            elif key.endswith("_scale"):
                value = torch.tensor([0.2, 0.3, 0.1])
            elif key.endswith("_base"):
                value = torch.randn(param.shape, generator=gen) * 0.3
            else:
                value = torch.ones(param.shape)
            param.copy_(value)
            controls[key] = value
    hidden = torch.randn((3, 4, 5120), generator=gen).bfloat16()
    incoming = torch.tensor([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0.1, 0.2, 0.3, 0.4]])
    actual, next_pre = module(torch.arange(3).npu(), hidden.npu(), incoming.npu(), image_token_mask=image_mask)
    expected = hidden
    current_pre = incoming
    for part, mult in [("attn", 0.25), ("ffn", 0.5)]:
        collapsed, post, comb, new_pre = mhc_pre_delayed_reference(
            expected, controls[f"hc_{part}_fn"], controls[f"hc_{part}_scale"], controls[f"hc_{part}_base"], current_pre
        )
        normalized = (
            collapsed.float() * torch.rsqrt(collapsed.float().square().mean(-1, keepdim=True) + 1e-20)
        ).bfloat16()
        output = normalized * mult
        expected = (
            post[:, :, None] * output.float()[:, None, :]
            + (comb[:, :, :, None] * expected.float()[:, :, None, :]).sum(1)
        ).bfloat16()
        current_pre = new_pre
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.015, atol=0.015)
    torch.testing.assert_close(next_pre.cpu(), current_pre, rtol=2e-4, atol=2e-4)
