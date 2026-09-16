# SPDX-License-Identifier: Apache-2.0
"""Real V4.1 TP8 shapes; independent FP32 arithmetic, never FP16 partial sums."""

import pytest
import torch
import torch.nn.functional as F
import torch_npu

import vllm_ascend.vllm_ascend_C  # noqa: F401

GROUP_SIZE = 32
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 288
TOP_K = 6


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    q_kn = q.transpose(1, 2).contiguous()
    result = torch.zeros((*q_kn.shape[:-1], q_kn.shape[-1] // 8), dtype=torch.int32)
    for nibble in range(8):
        result |= (q_kn[..., nibble::8] & 15) << (4 * nibble)
    return result


def linear_reference(x, q, scales):
    # Form effective weights in FP32 to avoid importing the kernel's group
    # summation order or any accidental FP16/BF16 dequantization rounding.
    weights = q.float() * scales.t().float().repeat_interleave(GROUP_SIZE, dim=1)
    return F.linear(x.float(), weights)


def moe_reference(x, q13, s13, q2, s2, ids, routing, limit):
    output = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(q13.shape[0]):
        tokens, routes = (ids == expert).nonzero(as_tuple=True)
        if not tokens.numel():
            continue
        gate, up = linear_reference(x[tokens], q13[expert], s13[expert]).chunk(2, dim=-1)
        if limit:
            gate = gate.clamp(max=limit)
            up = up.clamp(-limit, limit)
        act = (F.silu(gate) * up).to(x.dtype)
        projected = linear_reference(act, q2[expert], s2[expert])
        output.index_add_(0, tokens, projected * routing[tokens, routes, None])
    return output.to(x.dtype)


def make_case(
    batch=1,
    dtype=torch.bfloat16,
    overflow=False,
    experts=TOP_K,
    top_k=TOP_K,
    hidden=HIDDEN_SIZE,
    inter=INTERMEDIATE_SIZE,
):
    torch.manual_seed(4100 + batch)
    x = (torch.randn(batch, hidden) * (100000 if overflow else 0.2)).to(dtype)
    q13 = torch.randint(-8, 8, (experts, 2 * inter, hidden), dtype=torch.int32)
    q2 = torch.randint(-8, 8, (experts, hidden, inter), dtype=torch.int32)
    s13 = (torch.rand(experts, hidden // GROUP_SIZE, 2 * inter) * 0.015 + 0.001).to(dtype)
    s2 = (torch.rand(experts, inter // GROUP_SIZE, hidden) * 0.015 + 0.001).to(dtype)
    s13[..., ::2].neg_()
    s2[..., ::2].neg_()
    q13[:, 0, 0] = -8
    q2[:, 0, -1] = -8  # Regression: last element of the ninth K group.
    ids = torch.arange(batch * top_k, dtype=torch.int32).reshape(batch, top_k) % experts
    routing = torch.softmax(torch.randn(batch, top_k), dim=-1).float() * 1.5
    args = (x, pack_int4(q13), s13, pack_int4(q2), s2, ids, routing)
    return args, (q13, q2)


def assert_accurate(actual, expected):
    actual = actual.cpu().float()
    expected = expected.float()
    assert torch.isfinite(actual).all()
    error = actual - expected
    nrmse = (error.norm() / expected.norm().clamp_min(1e-12)).item()
    max_scaled = (error.abs().max() / expected.abs().max().clamp_min(1e-12)).item()
    assert nrmse < 0.006, f"NRMSE={nrmse}"
    assert max_scaled < 0.015, f"max_scaled_error={max_scaled}"


@pytest.mark.parametrize("batch", [1, 2, 8, 32, 64])
@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_real_tp8_shape(batch, limit):
    args, (q13, q2) = make_case(batch)
    x, _, s13, _, s2, ids, routing = args
    expected = moe_reference(x, q13, s13, q2, s2, ids, routing, limit)
    actual = torch.ops._C_ascend.npu_w4a16_moe(*(t.npu() for t in args), limit)
    assert_accurate(actual, expected)


def test_bf16_range_and_clamp():
    args, (q13, q2) = make_case(overflow=True)
    x, _, s13, _, s2, ids, routing = args
    expected = moe_reference(x, q13, s13, q2, s2, ids, routing, 10.0)
    actual = torch.ops._C_ascend.npu_w4a16_moe(*(t.npu() for t in args), 10.0)
    assert_accurate(actual, expected)


def test_graph_replay_changed_routes():
    args, (q13, q2) = make_case(2)
    device_args = tuple(t.npu() for t in args)
    for _ in range(3):
        torch.ops._C_ascend.npu_w4a16_moe(*device_args, 10.0)
    torch_npu.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = torch.ops._C_ascend.npu_w4a16_moe(*device_args, 10.0)
    x, _, s13, _, s2, ids, routing = args
    for iteration in range(5):
        x.mul_(-0.9)
        ids.fill_(iteration % TOP_K)  # All routes target the same expert.
        ids[0, 0] = -1  # Invalid/padding route contributes exactly zero.
        routing.mul_(0.95)
        device_args[0].copy_(x)
        device_args[5].copy_(ids)
        device_args[6].copy_(routing)
        graph.replay()
        expected = moe_reference(x, q13, s13, q2, s2, ids, routing, 10.0)
        assert_accurate(output, expected)


def test_reject_non_group_aligned_input():
    args, _ = make_case()
    device_args = [t.npu() for t in args]
    device_args[0] = device_args[0][:, :-1].contiguous()
    with pytest.raises(RuntimeError, match="hidden"):
        torch.ops._C_ascend.npu_w4a16_moe(*device_args, 10.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,hidden,inter", [(2, 128, 32), (2, 128, 256), (2, 128, 1056), (128, 5120, 288)])
def test_contiguous_and_tiled_weight_reads(batch, hidden, inter, dtype):
    # Full-N DMA, split N=2112, full/partial K blocks, and the decode limit.
    args, (q13, q2) = make_case(batch, dtype=dtype, hidden=hidden, inter=inter)
    x, _, s13, _, s2, ids, routing = args
    device_args = tuple(t.npu() for t in args)
    expected = moe_reference(x, q13, s13, q2, s2, ids, routing, 10.0)
    output = torch.ops._C_ascend.npu_w4a16_moe(*device_args, 10.0)
    assert_accurate(output, expected)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = torch.ops._C_ascend.npu_w4a16_moe(*device_args, 10.0)
    ids.add_(1).remainder_(TOP_K)
    ids[0, 0] = -1
    device_args[5].copy_(ids)
    graph.replay()
    assert_accurate(output, moe_reference(x, q13, s13, q2, s2, ids, routing, 10.0))


@pytest.mark.parametrize("batch", [8, 24, 48, 72])
def test_dspark_draft_top3_graph(batch):
    args, (q13, q2) = make_case(batch, experts=128, top_k=3)
    device_args = tuple(t.npu() for t in args)
    for _ in range(3):
        torch.ops._C_ascend.npu_w4a16_moe(*device_args, 10.0)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = torch.ops._C_ascend.npu_w4a16_moe(*device_args, 10.0)
    x, _, s13, _, s2, ids, routing = args
    for concentrated in (False, True):
        if concentrated:
            ids.copy_(torch.arange(3, dtype=torch.int32).expand(batch, -1))
        else:
            ids.add_(37).remainder_(128)
        ids[0, 0] = -1
        x.mul_(-0.9)
        device_args[0].copy_(x)
        device_args[5].copy_(ids)
        graph.replay()
        expected = moe_reference(x, q13, s13, q2, s2, ids, routing, 10.0)
        assert_accurate(output, expected)
