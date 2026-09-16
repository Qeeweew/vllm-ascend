# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU fused hash/gather versus independent signed-int64 scalar arithmetic."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.engram_hash import HostEngramHasher, HostEngramLayout


@pytest.fixture
def op():
    if not hasattr(torch.ops._C_ascend, "engram_hash_gather_cpu"):
        try:
            importlib.import_module("vllm_ascend.vllm_ascend_C")
        except ImportError:
            pytest.skip("Native extension is not installed")
    if not hasattr(torch.ops._C_ascend, "engram_hash_gather_cpu"):
        pytest.skip("Rebuild the native CPU Engram extension")
    return torch.ops._C_ascend.engram_hash_gather_cpu


def fixture(rank=0, world=3, reverse=False):
    cfg = SimpleNamespace(
        engram_layer_ids=[1, 14],
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_vocab_size=11,
        engram_num_embeddings=[1000, 1000],
        engram_head_dim=8,
    )
    h = HostEngramHasher(HostEngramLayout.from_config(cfg), torch.tensor([0, 1, 2, 3, 1, 2]), 4, 0)
    heads, offsets, tables = [], [], []
    generator = torch.Generator().manual_seed(41)
    for layer in range(2):
        selected, _ = h.layout.head_shard(layer, rank, world)
        selected = tuple(reversed(selected)) if reverse else selected
        cursor, starts = 0, []
        for index in selected:
            starts.append(cursor)
            cursor += int(h.primes[layer].flatten()[index])
        heads.append(selected)
        offsets.append(starts)
        tables.append(torch.randn(cursor, 8, generator=generator).bfloat16())
    return h, torch.tensor(heads), torch.tensor(offsets), tables


def oracle(h, heads, offsets, tables, ids, cu, positions, prior, mask=None, prior_mask=None):
    result = [torch.empty(len(ids), heads.shape[1], t.shape[1], dtype=torch.bfloat16) for t in tables]
    for request, (first, last) in enumerate(zip(cu[:-1], cu[1:])):
        for row in range(first, last):
            window, blocked = [], False
            for shift in range(h.max_ngram):
                index = row - shift
                mapped = h.pad_id
                if index >= first:
                    blocked |= mask is not None and not bool(mask[index])
                    if not blocked:
                        mapped = int(h.token_map[ids[index]])
                else:
                    back = first - index - 1
                    if back < positions[request]:
                        blocked |= prior_mask is not None and not bool(prior_mask[request, back])
                        if not blocked:
                            mapped = int(h.token_map[prior[request, back]])
                window.append(mapped)
            for layer in range(len(tables)):
                rolling, values = 0, []
                for shift, mapped in enumerate(window):
                    rolling ^= (mapped * int(h.multipliers[layer, shift])) & ((1 << 64) - 1)
                    values.append(rolling - (1 << 64) if rolling >= 1 << 63 else rolling)
                for local, head in enumerate(heads[layer].tolist()):
                    order = head // h.primes.shape[2] + 1
                    index = int(offsets[layer, local]) + values[order] % int(h.primes[layer].flatten()[head])
                    result[layer][row, local] = tables[layer][index]
    return result


def run_case(op, h, heads, offsets, tables, ids, cu, positions, prior, mask=None, prior_mask=None):
    bucket = len(ids) + 3
    outputs = [torch.full((bucket + 2, heads.shape[1], t.shape[1]), 7, dtype=torch.bfloat16) for t in tables]
    pointers = [out.data_ptr() for out in outputs]
    op(
        ids,
        torch.tensor(cu, dtype=torch.int64),
        torch.tensor(positions, dtype=torch.int64),
        prior,
        h.token_map,
        h.multipliers,
        h.primes,
        heads,
        offsets,
        tables,
        outputs,
        h.pad_id,
        bucket,
        mask,
        prior_mask,
    )
    expected = oracle(h, heads, offsets, tables, ids, cu, positions, prior, mask, prior_mask)
    assert pointers == [out.data_ptr() for out in outputs]
    for out, want in zip(outputs, expected):
        torch.testing.assert_close(out[: len(ids)], want, rtol=0, atol=0)
        assert not out[len(ids) : bucket].count_nonzero()
        assert (out[bucket:] == 7).all()
    return outputs


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_ragged_tp_heads_masks_and_rollback(op, rank, reverse):
    h, heads, offsets, tables = fixture(rank, reverse=reverse)
    ids = torch.tensor([1, 2, 3, 4, 5, 1])
    cu, positions = [0, 2, 2, 6], [0, 77, 3]
    prior = torch.tensor([[-1, -1, -1], [-1, -1, -1], [2, 1, 3]])
    mask = torch.tensor([True, False, True, True, False, True])
    pmask = torch.tensor([[False, False, False], [False, False, False], [True, False, True]])
    for corrected in [False, True]:
        if corrected:
            ids[3] = 1
            prior[2, 0] = 4
        run_case(op, h, heads, offsets, tables, ids, cu, positions, prior, mask, pmask)


def test_empty_input_and_prefill(op):
    h, heads, offsets, tables = fixture()
    run_case(
        op,
        h,
        heads,
        offsets,
        tables,
        torch.empty(0, dtype=torch.int64),
        [0],
        [],
        torch.empty((0, 3), dtype=torch.int64),
    )
    ids = torch.arange(256) % 6
    run_case(op, h, heads, offsets, tables, ids, [0, 128, 256], [0, 3], torch.tensor([[-1, -1, -1], [1, 2, 3]]))


@pytest.mark.parametrize("invalid", ["token", "history", "head", "bucket", "boundaries"])
def test_reject_invalid_inputs(op, invalid):
    h, heads, offsets, tables = fixture()
    ids, cu, pos, prior = torch.tensor([1]), torch.tensor([0, 1]), torch.tensor([1]), torch.tensor([[2, -1, -1]])
    out = [torch.empty((2, 2, 8), dtype=torch.bfloat16) for _ in tables]
    if invalid == "token":
        ids[0] = -1
    elif invalid == "history":
        prior[0, 0] = -1
    elif invalid == "head":
        heads[0, 0] = 6
    elif invalid == "bucket":
        offsets[0, 0] = 1
    else:
        cu[0] = 1
    with pytest.raises(RuntimeError):
        op(ids, cu, pos, prior, h.token_map, h.multipliers, h.primes, heads, offsets, tables, out, h.pad_id, 2)
