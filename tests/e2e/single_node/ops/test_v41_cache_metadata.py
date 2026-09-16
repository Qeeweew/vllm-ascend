# SPDX-License-Identifier: Apache-2.0
"""Exact device preparation semantics, including changing graph replay inputs."""

import pytest
import torch
import torch_npu  # noqa: F401


def make_case(batch, tokens, ratio, input_columns=7, output_columns=11):
    positions = torch.arange(tokens, dtype=torch.int64) * 3 + 30
    if tokens:
        positions[-1] = -1
    boundaries = torch.linspace(0, max(tokens - 2, 0), batch + 1).to(torch.int32)
    if batch > 2:
        boundaries[-2] = boundaries[-1]  # trailing empty request
        boundaries[1] = 0  # leading empty request
    lengths = torch.arange(batch, dtype=torch.int32) * 127 + 33
    table = torch.arange(batch * input_columns, dtype=torch.int32).view(batch, input_columns) * 3 + 1
    if table.numel():
        table.flatten()[::3] = -1
    inputs = [positions, boundaries, lengths, table]
    outputs = [
        torch.empty_like(positions),
        torch.empty_like(boundaries),
        torch.empty_like(lengths),
        torch.empty((batch, output_columns), dtype=torch.int32),
        torch.empty(tokens, dtype=torch.int32),
        torch.empty_like(positions),
        torch.empty_like(lengths),
        torch.empty_like(lengths),
    ]
    return inputs, outputs, [16 * ratio, 16, ratio, ratio != 1]


def reference(inputs, outputs, attrs):
    pi, ci, li, ti = inputs
    po, co, lo, to, ro, so, cm, re = outputs
    logical, physical, ratio, compressed = attrs
    po.copy_(pi[: po.numel()])
    co.copy_(ci[: co.numel()])
    lo.copy_(li[: lo.numel()])
    to.fill_(-1)
    to[:, : ti.shape[1]].copy_(ti[: to.shape[0]])
    cm.copy_(torch.div(lo, ratio, rounding_mode="floor"))
    re.copy_(lo % ratio)
    if not lo.numel():
        ro.zero_()
        so.fill_(-1)
        return
    indices = torch.arange(po.numel(), dtype=torch.int32)
    ro.copy_(torch.searchsorted(co[1:], indices, right=True, out_int32=True))
    valid_token = (indices < co[-1]) & (po >= 0)
    valid = valid_token & (po // logical < to.shape[1])
    if compressed:
        valid &= (po + 1) % ratio == 0
    page = (po // logical).clamp(0, to.shape[1] - 1)
    block = to[ro.clamp(0, to.shape[0] - 1).long(), page]
    so.copy_(block.long() * physical + (po % logical) // ratio)
    so.masked_fill_(~valid | (block < 0), -1)
    ro.masked_fill_(~valid_token, -1)


@pytest.mark.parametrize(
    "batch,tokens",
    [(0, 0), (0, 8), (1, 1), (1, 128), (8, 8), (8, 24), (8, 48), (8, 72), (8, 128), (8, 1024), (17, 129)],
)
@pytest.mark.parametrize("ratio", [1, 2])
def test_exact_and_graph_replay(batch, tokens, ratio):
    inputs, outputs, attrs = make_case(batch, tokens, ratio)
    device_inputs = [x.npu() for x in inputs]
    device_outputs = [x.npu() for x in outputs]
    op = torch.ops._C_ascend.v41_cache_metadata
    op(*device_inputs, *device_outputs, *attrs)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        op(*device_inputs, *device_outputs, *attrs)
    for generation in range(3):
        inputs[0].add_(31)
        if tokens:
            inputs[0][-1] = -generation - 1
        inputs[2].add_(1)
        inputs[3].add_(2)
        if batch and generation == 2:
            inputs[1].zero_()
        for cpu, npu in zip(inputs, device_inputs):
            npu.copy_(cpu)
        graph.replay()
        reference(inputs, outputs, attrs)
        for expected, actual in zip(outputs, device_outputs):
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("columns", [0, 1, 3, 2049, 8193])
def test_long_and_partial_tables(columns):
    inputs, outputs, attrs = make_case(8, 72, 2, columns, columns + 17)
    inputs[0].copy_(torch.arange(72) * max(columns // 3, 1))
    inputs[2][0] = -1  # floor/remainder contract
    device_outputs = [x.npu() for x in outputs]
    torch.ops._C_ascend.v41_cache_metadata(*[x.npu() for x in inputs], *device_outputs, *attrs)
    reference(inputs, outputs, attrs)
    for expected, actual in zip(outputs, device_outputs):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_reject_strided_input_and_alias():
    inputs, outputs, attrs = make_case(8, 8, 2)
    inputs = [x.npu() for x in inputs]
    outputs = [x.npu() for x in outputs]
    bad_table = torch.empty((8, 14), dtype=torch.int32, device="npu")[:, ::2]
    with pytest.raises(RuntimeError, match="contiguous"):
        torch.ops._C_ascend.v41_cache_metadata(*inputs[:3], bad_table, *outputs, *attrs)
    outputs[0] = inputs[0]
    with pytest.raises(RuntimeError, match="share input storage"):
        torch.ops._C_ascend.v41_cache_metadata(*inputs, *outputs, *attrs)


@pytest.mark.parametrize("physical", [32, 256, 1024])
@pytest.mark.parametrize("ratio", [1, 2])
def test_block_boundaries(physical, ratio):
    inputs, outputs, attrs = make_case(8, 24, ratio)
    logical = physical * ratio
    attrs[:2] = [logical, physical]
    boundary_positions = torch.tensor(
        [
            -1,
            0,
            1,
            logical - 2,
            logical - 1,
            logical,
            logical + 1,
            7 * logical - 1,
            7 * logical,
            11 * logical - 1,
            11 * logical,
            11 * logical + 1,
        ],
        dtype=torch.int64,
    )
    inputs[0].copy_(boundary_positions.repeat(2))
    device_outputs = [x.npu() for x in outputs]
    torch.ops._C_ascend.v41_cache_metadata(*[x.npu() for x in inputs], *device_outputs, *attrs)
    reference(inputs, outputs, attrs)
    for expected, actual in zip(outputs, device_outputs):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
