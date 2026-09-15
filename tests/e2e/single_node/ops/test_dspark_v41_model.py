# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the bounded real-draft diagnostic; these never select an NPU.

Actual TP8 execution is an explicitly scheduled benchmark command, not an
implicit pytest fixture. Reference primitives do not import production math.
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "dspark_v41_reference", ROOT / "benchmarks/deepseek_v41/dspark_v41_reference.py"
)
REFERENCE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REFERENCE
SPEC.loader.exec_module(REFERENCE)


@pytest.mark.parametrize("heads", [None, 2])
def test_adjacent_rope_scalar_complex_reference(heads):
    shape = (3, 8) if heads is None else (3, heads, 8)
    x = torch.arange(math.prod(shape)).reshape(shape).float().mul_(0.0625).bfloat16()
    positions = torch.tensor([0, 31, 128])
    actual = REFERENCE.rotate(x, positions, 4, 10000.0)
    expected = x.clone()
    for token, position in enumerate(positions.tolist()):
        for head in range(heads or 1):
            source = x[token] if heads is None else x[token, head]
            target = expected[token] if heads is None else expected[token, head]
            for pair in range(2):
                angle = position / (10000.0 ** (2 * pair / 4))
                value = complex(float(source[4 + 2 * pair]), float(source[5 + 2 * pair]))
                value *= complex(math.cos(angle), math.sin(angle))
                target[4 + 2 * pair], target[5 + 2 * pair] = value.real, value.imag
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[..., :4], x[..., :4], rtol=0, atol=0)


def test_noncausal_attention_includes_future_query_and_zero_value_sink():
    q = torch.zeros((5, 2, 8))
    kv = torch.arange(9 * 8).reshape(9, 8).float()
    sink = torch.tensor([0.0, math.log(2)])
    output = REFERENCE.dense_noncausal_attention(q, kv, sink)
    expected = torch.stack((kv.sum(0) / 10, kv.sum(0) / 11))[None].expand(5, -1, -1)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    # Changing the final future query key changes every query's result.
    changed = kv.clone()
    changed[-1] += 10
    assert (REFERENCE.dense_noncausal_attention(q, changed, sink) != output).all()


def test_hc_incoming_only_changes_current_collapse_and_post_orientation():
    hidden = torch.arange(32).reshape(1, 4, 8).bfloat16()
    fn = torch.zeros(24, 32)
    scale = torch.tensor([0.3, 0.4, 0.5])
    base = torch.linspace(-1, 1, 24)
    first = REFERENCE.hc_pre(hidden, torch.tensor([[1.0, 0, 0, 0]]), fn, scale, base, 1e-20, 1e-6, 20)
    second = REFERENCE.hc_pre(hidden, torch.tensor([[0.0, 0, 0, 1]]), fn, scale, base, 1e-20, 1e-6, 20)
    torch.testing.assert_close(first[0], hidden[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(second[0], hidden[:, 3], rtol=0, atol=0)
    for a, b in zip(first[1:], second[1:]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    comb = torch.zeros(1, 4, 4)
    for old in range(4):
        comb[0, old, (old + 1) % 4] = 1
    actual = REFERENCE.hc_post(torch.zeros(1, 8), hidden, torch.zeros(1, 4), comb)
    torch.testing.assert_close(actual, hidden[:, [3, 0, 1, 2]].float(), rtol=0, atol=0)


def pack(q):
    q = q.int() + 8
    output = torch.zeros((*q.shape[:-1], q.shape[-1] // 8), dtype=torch.int32)
    for digit in range(8):
        output |= q[..., digit::8] << (4 * digit)
    return output


def test_offset_binary_unpack_preserves_all_signed_codes():
    q = torch.arange(-8, 8).repeat(3, 2).reshape(3, 32)
    torch.testing.assert_close(REFERENCE.unpack(pack(q)), q.float(), rtol=0, atol=0)


def test_streamed_expert_reference_negative_scale_and_router_once():
    class Weights:
        def read(self, name, rows=None, columns=None):
            is_down = ".w2." in name
            shape = (64, 2304) if is_down else (2304, 64)
            if name.endswith("weight_packed"):
                value = pack(torch.ones(shape, dtype=torch.int32))
            else:
                value = torch.full((shape[0], shape[1] // 32), -0.125 if is_down else 0.125).bfloat16()
            if rows is not None:
                value = value[rows]
            if columns is not None:
                value = value[:, columns]
            return value

    x = torch.full((2, 64), 0.125).bfloat16()
    ids = torch.tensor([[0, 1, 2], [2, 3, 4]])
    routing = torch.tensor([[0.25, 0.5, 0.25], [0.5, 0.25, 0.25]])
    actual = REFERENCE.routed_moe(Weights(), 0, 3, x, ids, routing)
    # Gate/up dot = 1 exactly. Every local expert output is identical.
    active = torch.tensor(1.0).sigmoid().bfloat16().float()
    expected = torch.full((2, 64), float(active * -0.125 * 288)).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (actual < 0).all()


def test_no_npu_initialization():
    assert not (hasattr(torch, "npu") and torch.npu.is_initialized())
