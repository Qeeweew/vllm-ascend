# SPDX-License-Identifier: Apache-2.0
"""Fixed-buffer and independent-bucket contracts; NPU replay has a TP8 gate."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.spec_decode.dspark_v41_graph import DSparkV41GraphRunner, graph_buckets, select_bucket


def test_context_and_query_buckets_are_independent_and_bounded():
    context, query = graph_buckets(511), graph_buckets(7)
    assert context == (1, 2, 4, 8, 16, 32, 64, 128, 256, 511)
    assert query == (1, 2, 4, 7)
    assert [
        (select_bucket(rows, context), select_bucket(batch, query)) for rows, batch in ((33, 3), (129, 3), (129, 1))
    ] == [(64, 4), (256, 4), (256, 1)]
    for size in (0, -1, 512):
        with pytest.raises(ValueError):
            select_bucket(size, context)


def test_raw_auxiliary_rows_preserve_address_and_clear_padding():
    graph = DSparkV41GraphRunner.__new__(DSparkV41GraphRunner)
    graph.context_buckets = graph_buckets(16)
    graph.aux = torch.full((16, 12), 99.0)
    pointer = graph.aux.data_ptr()
    graph.stage_aux(torch.ones(9, 12), 9)
    assert graph.aux[:9].eq(1).all() and graph.aux[9:].eq(0).all()
    graph.stage_aux(torch.full((5, 12), 2.0), 5)
    assert graph.aux.data_ptr() == pointer
    assert graph.aux[:5].eq(2).all() and graph.aux[5:8].eq(0).all()
    with pytest.raises(ValueError, match="three target"):
        graph.stage_aux(torch.ones(5, 4), 5)


def test_graph_replay_stages_current_boundaries_pages_and_padding():
    graph = DSparkV41GraphRunner.__new__(DSparkV41GraphRunner)
    graph.ready = True
    graph.context_buckets, graph.query_buckets = graph_buckets(16), graph_buckets(4)
    graph.cu_q = torch.full((5,), 99, dtype=torch.int32)
    graph.lengths = torch.full((4,), 99, dtype=torch.int32)
    graph.tables = {2: torch.full((4, 8), 99, dtype=torch.int32)}
    graph.context, graph.query = object(), object()
    p = SimpleNamespace(
        num_query_per_req=5,
        _dflash_num_context=9,
        _context_positions_buffer=torch.full((16,), 3),
        _per_group_context_slot_mapping_buffers={2: torch.arange(16)},
        _per_group_query_slot_mapping_buffers={2: torch.arange(20)},
        input_ids=torch.full((20,), 99),
        positions=torch.full((20,), 99),
        parallel_drafting_token_id=7,
        draft_attn_groups=[SimpleNamespace(kv_cache_group_id=2)],
        _per_group_block_table_buffers={2: torch.arange(24, dtype=torch.int32).reshape(3, 8)},
    )
    common = SimpleNamespace(
        num_reqs=3, query_start_loc=torch.tensor([0, 5, 10, 15]), seq_lens=torch.tensor([14, 38, 134])
    )
    p.set_inputs_first_pass = MagicMock(return_value=(15, None, common, None))
    graph.proposer = p
    calls = []

    def call(wrapper, size, mode):
        calls.append((wrapper, size))
        return torch.arange(20).reshape(4, 5) if wrapper is graph.query else None

    graph._call = call
    result = graph.propose(5, None, None, None, None, None, common, None, None)
    assert result.shape == (3, 5)
    assert calls == [(graph.context, 16), (graph.query, 4)]
    assert graph.cu_q.tolist() == [0, 5, 10, 15, 15]
    assert graph.lengths.tolist() == [14, 38, 134, 0]
    assert graph.tables[2][3].eq(-1).all()
    assert p._context_positions_buffer[9:16].eq(0).all()
    assert p._per_group_context_slot_mapping_buffers[2][9:16].eq(-1).all()
    assert p.positions[15:20].eq(-1).all()
    assert p.input_ids[15:20].eq(7).all()
    assert p._per_group_query_slot_mapping_buffers[2][15:20].eq(-1).all()
    graph.ready = False
    with pytest.raises(RuntimeError, match="captured before"):
        graph.propose(5, None, None, None, None, None, common, None, None)


def test_context_callable_combines_aux_and_stores_each_layer_inside_capture():
    graph = DSparkV41GraphRunner.__new__(DSparkV41GraphRunner)
    graph.aux = torch.arange(16 * 12).reshape(16, 12).float()
    calls = []

    def combine(aux):
        calls.append("combine")
        return aux[:, :4] + 1

    def store(hidden, positions, slots):
        calls.append("store")
        assert torch.equal(hidden, graph.aux[:8, :4] + 1)
        assert positions.shape == (8,)
        assert len(slots) == 3
        assert slots[0].data_ptr() == slots[2].data_ptr()
        assert all(value[5:].eq(-1).all() for value in slots)

    graph.proposer = SimpleNamespace(
        model=SimpleNamespace(combine_hidden_states=combine, precompute_and_store_context_kv=store),
        _context_positions_buffer=torch.arange(16),
        _layer_group_idx=[2, 7, 2],
        _per_group_context_slot_mapping_buffers={gid: torch.tensor([0, 1, 2, 3, 4] + [-1] * 11) for gid in (2, 7)},
    )
    assert graph._run_context(8) == ()
    assert calls == ["combine", "store"]
