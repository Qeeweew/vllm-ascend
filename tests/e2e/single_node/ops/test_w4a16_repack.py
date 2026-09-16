# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_npu

from vllm_ascend.ops.triton.int4_repack import repack_int4_moe
from vllm_ascend.quantization.methods.wna16.w4a16 import repack_experts_bounded


@pytest.mark.parametrize("outputs, inputs", [(576, 5120), (5120, 288)])
@pytest.mark.parametrize("repack", [repack_experts_bounded, repack_int4_moe])
def test_repack_exact_signed_scale_layout(outputs, inputs, repack):
    torch.manual_seed(41)
    q = torch.randint(-8, 8, (3, outputs, inputs), dtype=torch.int32)
    checkpoint = torch.zeros((3, outputs, inputs // 8), dtype=torch.int32)
    expected = torch.zeros((3, inputs, outputs // 8), dtype=torch.int32)
    q_kn = q.transpose(1, 2)
    for nibble in range(8):
        checkpoint |= (q[..., nibble::8] + 8) << (nibble * 4)
        expected |= (q_kn[..., nibble::8] & 15) << (nibble * 4)
    actual = repack(checkpoint.npu())
    torch_npu.npu.synchronize()
    assert torch.equal(actual.cpu(), expected)


def test_repack_rejects_non_int4():
    with pytest.raises(ValueError, match="4-bit"):
        repack_experts_bounded(torch.zeros(1, 64, 16, dtype=torch.int32), num_bits=8)


@pytest.mark.parametrize("inputs, outputs", [(5120, 576), (288, 5120)])
@pytest.mark.parametrize("rows_per_expert", [1, 16, 128])
def test_cann_omitted_offset_matches_explicit_zero(inputs, outputs, rows_per_expert):
    torch.manual_seed(4104)
    experts = 6
    q = torch.randint(-8, 8, (experts, inputs, outputs), dtype=torch.int32)
    scales = (torch.rand(experts, inputs // 32, outputs) * 0.015 + 0.001).bfloat16()
    scales[..., ::2].neg_()
    q[:, 0, 0] = -8
    packed = torch_npu.npu_convert_weight_to_int4pack(q.npu().flatten(0, 1)).view(experts, inputs, outputs // 8)
    scales = scales.npu()
    params = dict(
        x=[torch.randn(experts * rows_per_expert, inputs).bfloat16().npu()],
        weight=[packed],
        antiquant_scale=[scales],
        group_list=torch.full((experts,), rows_per_expert, dtype=torch.int64, device="npu"),
        group_list_type=1,
        group_type=0,
        split_item=2,
        output_dtype=torch.bfloat16,
    )
    explicit = torch_npu.npu_grouped_matmul(**params, antiquant_offset=[torch.zeros_like(scales)])[0]
    omitted = torch_npu.npu_grouped_matmul(**params)[0]
    torch_npu.npu.synchronize()
    assert torch.equal(explicit.cpu(), omitted.cpu())
