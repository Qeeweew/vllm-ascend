# SPDX-License-Identifier: Apache-2.0
"""Native accuracy/graph tests; require root-installed isolated candidate."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.fused_moe.router.fused_topk_router import select_deepseek_v4_vision_experts
from vllm_ascend.ops.v41_moe_router import v41_moe_router

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "router_scalar_oracle", ROOT / "benchmarks/deepseek_v41/v41_moe_router/oracle.py"
)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def case(rows, experts, k, mode, seed=11):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn((rows, experts), generator=generator)
    token = torch.arange(rows, dtype=torch.int64).remainder(64)
    mask = torch.zeros(rows, dtype=torch.bool)
    if mode == "mixed":
        mask[1::2] = True
        token[1::2] = -12345
    table = torch.stack([torch.randperm(experts, generator=generator)[:k] for _ in range(64)]).int()
    text_bias = torch.randn(experts, generator=generator) * 0.1
    image_bias = torch.randn(experts, generator=generator) * 0.1
    if mode == "dynamic":
        table = None
    return logits, token, mask, table, text_bias, image_bias


def device_args(inputs):
    return tuple(value.npu() if value is not None else None for value in inputs)


def baseline(inputs, k, renormalize=True, scaling=1.0):
    x, token, mask, table, text_bias, image_bias = inputs
    return select_deepseek_v4_vision_experts(
        x, token, table, image_bias, text_bias, k, renormalize, scaling, image_token_mask=mask
    )


@pytest.mark.parametrize("experts,k", [(128, 3), (384, 6)])
@pytest.mark.parametrize("rows", [0, 1, 2, 4, 5, 16, 64, 128, 1024])
@pytest.mark.parametrize("mode", ["hash", "mixed", "dynamic"])
def test_native_router(rows, experts, k, mode):
    cpu = case(rows, experts, k, mode)
    args = device_args(cpu)
    weights = torch.empty((rows, k), device="npu")
    ids = torch.empty((rows, k), dtype=torch.int32, device="npu")
    v41_moe_router(*args, weights, ids, k, True, 2.5)
    expected_weights, expected_ids = baseline(args, k, True, 2.5)
    torch.testing.assert_close(ids.cpu().long(), expected_ids.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=2e-6, atol=2e-7)
    oracle_weights, oracle_ids = ORACLE.scalar_router(*cpu, k, True, 2.5)
    torch.testing.assert_close(ids.cpu(), oracle_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), oracle_weights, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("experts,k", [(128, 3), (384, 6)])
def test_changed_input_graph(experts, k):
    args = device_args(case(4, experts, k, "mixed"))
    weights = torch.empty((4, k), device="npu")
    ids = torch.empty((4, k), dtype=torch.int32, device="npu")
    for _ in range(3):
        v41_moe_router(*args, weights, ids, k)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        v41_moe_router(*args, weights, ids, k)
    pointers = [value.data_ptr() for value in (*args, weights, ids)]
    for seed in range(21, 29):
        changed = case(4, experts, k, "mixed", seed=seed)
        changed[2].logical_not_()
        changed[1].fill_(seed % 64)
        for destination, value in zip(args, changed):
            destination.copy_(value)
        graph.replay()
        expected_weights, expected_ids = baseline(args, k)
        torch.testing.assert_close(ids.cpu().long(), expected_ids.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=2e-6, atol=2e-7)
        assert pointers == [value.data_ptr() for value in (*args, weights, ids)]


@pytest.mark.parametrize("bad_table", [False, True])
def test_invalid_text_lookup_native(bad_table):
    cpu = list(case(2, 384, 6, "hash"))
    if bad_table:
        cpu[3][0, 3] = 384
    else:
        cpu[1][0] = -1
    cpu[2][1] = True
    cpu[1][1] = -999999
    args = device_args(cpu)
    weights = torch.empty((2, 6), device="npu")
    ids = torch.empty((2, 6), dtype=torch.int32, device="npu")
    v41_moe_router(*args, weights, ids)
    assert weights[0].cpu().eq(0).all()
    assert ids[0].cpu().eq(-1).all()
    assert ids[1].cpu().ge(0).all()


@pytest.mark.parametrize("experts,k", [(128, 3), (384, 6)])
@pytest.mark.parametrize("kind", ["all_equal", "cutoff_tie", "block_tie", "near_tie"])
def test_dynamic_ties_match_installed_topk(experts, k, kind):
    cpu = list(case(4, experts, k, "dynamic"))
    cpu[0].zero_()
    cpu[4].zero_()
    cpu[5].zero_()
    if kind == "cutoff_tie":
        cpu[4][: k - 1] = torch.arange(k, 1, -1).float()
    elif kind == "block_tie":
        cpu[4][31:65] = 1.0
        cpu[4][-4:] = 1.0
    elif kind == "near_tie":
        cpu[0][:, ::3] = torch.nextafter(torch.tensor(1.0), torch.tensor(float("inf")))
        cpu[0][:, 1::3] = 1.0
        cpu[0][:, 2::3] = torch.nextafter(torch.tensor(1.0), torch.tensor(float("-inf")))
    args = device_args(cpu)
    weights = torch.empty((4, k), device="npu")
    ids = torch.empty((4, k), dtype=torch.int32, device="npu")
    v41_moe_router(*args, weights, ids, k)
    expected_weights, expected_ids = baseline(args, k)
    torch.testing.assert_close(ids.cpu().long(), expected_ids.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("value", [-1000.0, -100.0, -40.0, -20.0, -16.0, 20.0, 100.0, 1e30])
@pytest.mark.parametrize("renormalize", [True, False])
def test_extreme_hash_scores_native(value, renormalize):
    cpu = list(case(1, 384, 6, "hash"))
    cpu[0].fill_(value)
    args = device_args(cpu)
    weights = torch.empty((1, 6), device="npu")
    ids = torch.empty((1, 6), dtype=torch.int32, device="npu")
    v41_moe_router(*args, weights, ids, 6, renormalize)
    expected_weights, expected_ids = baseline(args, 6, renormalize)
    torch.testing.assert_close(ids.cpu().long(), expected_ids.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("experts,k", [(128, 3), (384, 6)])
def test_cutoff_bias_cancellation_native(experts, k):
    cpu = list(case(4, experts, k, "dynamic", seed=71))
    args = list(device_args(cpu))
    # Construct cancellation using the real baseline's FP32 scores, without
    # assuming CPU transcendental kernels round exactly like the NPU backend.
    args[0].copy_(args[0][0:1].expand_as(args[0]).clone())
    score = torch.nn.functional.softplus(args[0][0]).sqrt()
    desired = torch.ones(experts, dtype=torch.float32, device="npu")
    desired[: k - 1] = 2.0
    desired[k - 1 : 2 * k] = 1.0 + torch.finfo(torch.float32).eps
    args[4].copy_(desired - score)
    weights = torch.empty((4, k), device="npu")
    ids = torch.empty((4, k), dtype=torch.int32, device="npu")
    v41_moe_router(*args, weights, ids, k)
    expected_weights, expected_ids = baseline(args, k)
    torch.testing.assert_close(ids.cpu().long(), expected_ids.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=2e-6, atol=2e-7)
