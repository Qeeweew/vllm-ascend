# SPDX-License-Identifier: Apache-2.0
"""Exercise the real Ascend wo_a loader before V4.1 grouped projection."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import ColumnParallelLinear

from vllm_ascend.models.deepseek_v4.model import DeepseekV41AttentionProjections
from vllm_ascend.ops import linear as ascend_linear


@pytest.mark.parametrize("tp_rank", [0, 3, 7])
@pytest.mark.parametrize("grouped_layout", [False, True])
def test_real_weight_loader_then_grouped_output(monkeypatch, tp_rank, grouped_layout):
    monkeypatch.setattr(
        ascend_linear, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda capability: False)
    )
    groups, rank, width, tp_size = 2, 3, 6, 8
    # Skip construction/distributed setup, retaining both real loader methods.
    linear = ascend_linear.AscendColumnParallelLinear.__new__(ascend_linear.AscendColumnParallelLinear)
    nn.Module.__init__(linear)
    linear.prefix = "model.layers.0.self_attn.wo_a"
    linear.tp_rank = tp_rank
    linear.tp_size = tp_size
    linear.n_local_groups = groups
    linear.o_lora_rank = rank
    linear.quant_config = None
    linear.weight = nn.Parameter(torch.empty(groups * rank, width, dtype=torch.bfloat16), requires_grad=False)
    linear.weight.output_dim = 0

    projection = DeepseekV41AttentionProjections.__new__(DeepseekV41AttentionProjections)
    nn.Module.__init__(projection)
    projection.local_groups, projection.o_rank, projection.group_width = groups, rank, width
    projection.rope_dim = 2
    projection.register_buffer("rope_cos", torch.ones(4, 1))
    projection.register_buffer("rope_sin", torch.zeros(4, 1))
    projection.wo_a = linear
    projection.wo_b = nn.Identity()
    generator = torch.Generator().manual_seed(41072)
    attention = torch.randn(4, groups, width, generator=generator).bfloat16()
    positions = torch.arange(4)
    for _ in range(2):
        checkpoint = torch.randn(tp_size * groups * rank, width, generator=generator).bfloat16()
        if grouped_layout:
            linear.weight_loader(linear.weight, checkpoint)
            assert linear.weight.shape == (groups, width, rank)
        else:
            ColumnParallelLinear.weight_loader(linear, linear.weight, checkpoint)
        local = checkpoint[tp_rank * groups * rank : (tp_rank + 1) * groups * rank]
        expected = torch.cat(
            [
                torch.nn.functional.linear(
                    attention[:, group].float(), local[group * rank : (group + 1) * rank].float()
                ).bfloat16()
                for group in range(groups)
            ],
            dim=1,
        )
        actual = projection.project_output(attention, positions)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
