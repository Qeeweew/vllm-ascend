# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORACLE = load_file("v41_router_oracle", ROOT / "benchmarks/deepseek_v41/v41_moe_router/oracle.py")
WRAPPER = load_file("v41_router_wrapper", ROOT / "vllm_ascend/ops/v41_moe_router.py")


def fixture(experts=384, k=6, rows=4):
    logits = torch.linspace(-8, 8, experts).repeat(rows, 1)
    tokens = torch.zeros(rows, dtype=torch.int64)
    mask = torch.zeros(rows, dtype=torch.bool)
    table = torch.arange(k - 1, -1, -1, dtype=torch.int32).repeat(2, 1)
    bias = torch.zeros(experts)
    weights = torch.empty((rows, k))
    ids = torch.empty((rows, k), dtype=torch.int32)
    return logits, tokens, mask, table, None, bias, weights, ids


@pytest.mark.parametrize("experts,k", [(384, 6), (128, 3)])
@pytest.mark.parametrize("renormalize", [False, True])
def test_hash_order_mask_and_unbiased_weights(experts, k, renormalize):
    x, token, mask, table, _, bias, _, _ = fixture(experts, k)
    token[1] = -999  # image rows never index the table
    mask[1] = True
    bias[experts - 2] = 100
    actual, ids = ORACLE.scalar_router(x, token, mask, table, None, bias, k, renormalize, 2.5)
    assert ids[0].tolist() == table[0].tolist()
    assert ids[1, 0] == experts - 2
    scores = torch.nn.functional.softplus(x).sqrt()
    expected = scores.gather(1, ids.long())
    if renormalize:
        expected = expected / expected.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    torch.testing.assert_close(actual, expected * 2.5, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("bad_token,bad_expert", [(-1, None), (2, None), (0, -1), (0, 384)])
def test_invalid_lookup_is_whole_row_sentinel(bad_token, bad_expert):
    x, token, mask, table, _, bias, _, _ = fixture(rows=1)
    token[0] = bad_token
    if bad_expert is not None:
        table[0, 2] = bad_expert
    weights, ids = ORACLE.scalar_router(x, token, mask, table, None, bias, 6)
    assert weights.eq(0).all()
    assert ids.eq(-1).all()


def test_extreme_scores_and_zero_weights():
    x, token, mask, table, _, bias, _, _ = fixture(rows=1)
    x.fill_(-1000)
    weights, ids = ORACLE.scalar_router(x, token, mask, table, None, bias, 6)
    assert weights.eq(0).all()
    assert ids.tolist() == table[:1].tolist()
    x[0, :6] = torch.tensor([-100, -40, 0, 20, 100, 1e30])
    weights, _ = ORACLE.scalar_router(x, token, mask, table, None, bias, 6, False)
    assert torch.isfinite(weights).all()
    assert weights[0, -1] > 0  # log1p must not discard the negative tail


def test_tie_must_be_explicit():
    x, token, mask, _, _, bias, _, _ = fixture(rows=1)
    x.zero_()
    with pytest.raises(ValueError, match="tie"):
        ORACLE.scalar_router(x, token, mask, None, None, bias, 6)
    _, ids = ORACLE.scalar_router(x, token, mask, None, None, bias, 6, tie_order="index_ascending")
    assert ids.tolist() == [list(range(6))]


@pytest.mark.parametrize("rows", [0, 1, 2, 4, 16, 64, 128, 1024])
def test_metadata_contract(rows):
    args = fixture(rows=rows)
    WRAPPER.validate_v41_moe_router(*args, 6, 1.0)


def test_output_alias_rejected():
    args = list(fixture(rows=1))
    args[6] = args[0][:, :6]
    with pytest.raises(ValueError, match="alias"):
        WRAPPER.validate_v41_moe_router(*args, 6, 1.0)


def test_wrong_mask_rejected():
    args = list(fixture())
    args[2] = args[2].int()
    with pytest.raises(ValueError, match="image_mask"):
        WRAPPER.validate_v41_moe_router(*args, 6, 1.0)


def test_literal_image_id_remains_text_under_explicit_mask():
    x, token, mask, table, _, bias, _, _ = fixture(rows=2)
    literal_image_id = 128815
    table = table[:1].repeat(literal_image_id + 1, 1)
    table[literal_image_id] = torch.tensor([20, 18, 16, 14, 12, 10], dtype=torch.int32)
    token[0] = literal_image_id
    token[1] = 0
    _, ids = ORACLE.scalar_router(x, token, mask, table, None, bias, 6)
    assert ids[0].tolist() == table[literal_image_id].tolist()
    assert ids[1].tolist() == table[0].tolist()


def test_bias_only_changes_selection_not_weight():
    x, token, mask, _, _, bias, _, _ = fixture(rows=1)
    mask[0] = True
    bias[:6] = torch.arange(200, 194, -1).float()
    weights, ids = ORACLE.scalar_router(x, token, mask, None, None, bias, 6, False)
    assert ids.tolist() == [list(range(6))]
    expected = torch.nn.functional.softplus(x[:, :6]).sqrt()
    torch.testing.assert_close(weights, expected, rtol=2e-6, atol=2e-7)
