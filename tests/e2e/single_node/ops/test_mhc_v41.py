# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm_ascend.ops.mhc_v41 import mhc_collapse, mhc_post, mhc_pre_delayed, mhc_pre_delayed_reference
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def npu_ops():
    assert enable_custom_op()


def inputs(batch):
    gen = torch.Generator().manual_seed(423)
    hidden = torch.randn((batch, 4, 5120), generator=gen).bfloat16()
    fn = torch.randn((24, 20480), generator=gen) * 20480**-0.5
    scale = torch.tensor([0.1, 0.1, 0.1])
    base = torch.randn(24, generator=gen) * 0.1
    pre = torch.rand((batch, 4), generator=gen)
    return hidden, fn, scale, base, pre


@pytest.mark.parametrize("batch", [1, 16, 128])
def test_delayed_collapse_and_fp32_control_reference(batch):
    cpu = inputs(batch)
    dev = tuple(t.npu() for t in cpu)
    y, post, comb, pre = mhc_pre_delayed(*dev)
    want = mhc_pre_delayed_reference(*cpu)
    torch.testing.assert_close(y.cpu(), want[0], rtol=0, atol=0)
    # Native HcPre uses HF32 for the projection. Compare with strict FP32
    # reference, not a reference that truncates weights to match the kernel.
    for actual, expected in zip((post, comb, pre), want[1:]):
        relative = (actual.cpu() - expected).square().mean().sqrt() / expected.square().mean().sqrt()
        assert relative < 2e-4
    other = dev[-1] * 0
    other[:, 2] = 1
    new_y, new_post, new_comb, new_pre = mhc_pre_delayed(*dev[:-1], other)
    assert torch.equal(new_y.cpu(), cpu[0][:, 2])
    for a, b in zip((new_post, new_comb, new_pre), (post, comb, pre)):
        assert torch.equal(a.cpu(), b.cpu())


def test_post_orientation_and_terminal_collapse():
    cpu = inputs(3)
    hidden, fn, scale, base, incoming = tuple(t.npu() for t in cpu)
    y, post, comb, next_pre = mhc_pre_delayed(hidden, fn, scale, base, incoming)
    output = mhc_post(y, hidden, post, comb)
    residual = hidden.cpu().float()
    expected = (
        post.cpu()[:, :, None] * y.cpu().float()[:, None, :]
        + (comb.cpu()[:, :, :, None] * residual[:, :, None, :]).sum(1)
    ).bfloat16()
    torch.testing.assert_close(output.cpu(), expected, rtol=0.01, atol=0.01)
    result = mhc_collapse(output, next_pre)
    expected_final = (output.cpu().float() * next_pre.cpu()[:, :, None]).sum(1).bfloat16()
    torch.testing.assert_close(result.cpu(), expected_final, rtol=0, atol=0)


def test_graph_replay_reads_new_incoming_mix():
    hidden, fn, scale, base, incoming = tuple(t.npu() for t in inputs(4))
    for _ in range(3):
        mhc_pre_delayed(hidden, fn, scale, base, incoming)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        y, post, comb, next_pre = mhc_pre_delayed(hidden, fn, scale, base, incoming)
    snapshots = []
    for head in (0, 3, 1, 2):
        incoming.zero_()
        incoming[:, head].fill_(1)
        graph.replay()
        snapshots.append((y.clone(), head))
    for actual, head in snapshots:
        assert torch.equal(actual.cpu(), hidden[:, head].cpu())
