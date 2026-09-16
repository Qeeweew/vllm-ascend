# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.dsa_v41 import AscendDSAV41Ops, build_dspark_v41_swa_indices


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_metadata_and_native_contract(ratio):
    ops = AscendDSAV41Ops(ratio)
    lengths = torch.tensor([129, 130], dtype=torch.int32)
    offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
    table = torch.zeros((2, 5), dtype=torch.int32)
    schedule = torch.zeros(1024, dtype=torch.int32)
    with patch("torch.ops._C_ascend.npu_sparse_flash_mla_metadata", return_value=schedule, create=True) as build:
        metadata = ops.build_metadata(
            offsets,
            lengths,
            table,
            max_seqlen_q=1,
            max_seqlen_kv=160,
            cmp_block_table=table if ratio else None,
        )
    args = build.call_args.kwargs
    assert args["cmp_ratio"] == ratio
    assert args["ori_win_left"] == 127
    assert args["ori_win_right"] == 0
    assert args["layout_kv"] == "PA_BBND"
    if ratio:
        torch.testing.assert_close(metadata.seqused_cmp_kv, lengths // ratio)
    if ratio == 2:
        torch.testing.assert_close(metadata.cmp_residual_kv, torch.tensor([1, 0], dtype=torch.int32))
    else:
        assert metadata.cmp_residual_kv is None
    query = torch.empty((2, 8, 512), dtype=torch.bfloat16)
    cache = torch.empty((1, 32, 1, 512), dtype=torch.bfloat16)
    sinks = torch.empty(8)
    indices = torch.full((2, 1, 512), -1, dtype=torch.int32)
    with patch("torch.ops._C_ascend.npu_sparse_flash_mla", return_value=(query, torch.empty(0)), create=True) as native:
        actual, _ = ops.forward(
            query,
            cache,
            sinks,
            metadata,
            cmp_cache=cache if ratio else None,
            cmp_indices=indices if ratio else None,
        )
    assert actual is query
    assert native.call_args.kwargs["cmp_sparse_indices"] is (indices if ratio else None)
    assert "ori_sparse_indices" not in native.call_args.kwargs
    assert native.call_args.kwargs["cmp_mask_mode"] == 3
    assert native.call_args.kwargs["softmax_scale"] == 512**-0.5


@pytest.mark.parametrize("ratio", [-1, 3, 4, 128])
def test_reject_old_v4_ratios(ratio):
    with pytest.raises(ValueError, match="compress_ratio"):
        AscendDSAV41Ops(ratio)


def test_reject_missing_compressed_table():
    with pytest.raises(ValueError, match="compressed block table"):
        AscendDSAV41Ops(2).build_metadata(
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
            torch.zeros((1, 1), dtype=torch.int32),
            max_seqlen_q=1,
            max_seqlen_kv=32,
        )


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("prefix", [0, 1, 127, 128, 129, 1024])
def test_dspark_visibility_matches_official_window(prefix, tokens):
    length = prefix + tokens
    pages = (length + 31) // 32
    table = torch.arange(pages, dtype=torch.int32).flip(0)[None]
    offsets = torch.tensor([0, tokens], dtype=torch.int32)
    lengths = torch.tensor([length], dtype=torch.int32)
    indices = torch.empty((tokens + 2, 1, 256), dtype=torch.int32)
    spans = torch.empty((tokens + 2, 1), dtype=torch.int32)
    # No value-dependent host reads are permitted in this helper.
    with (
        patch.object(torch.Tensor, "item", side_effect=AssertionError("host read")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host read")),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host read")),
    ):
        actual, actual_spans = build_dspark_v41_swa_indices(
            table,
            offsets,
            lengths,
            page_size=32,
            num_cache_blocks=pages,
            indices_output=indices,
            lengths_output=spans,
        )
    assert actual is indices and actual_spans is spans
    # CPU oracle enumerates the official logical visibility predicate. Page
    # permutation must NOT affect these IDs: native arch22 applies pagination.
    visible = [position for position in range(length) if position >= prefix - 128]
    expected = torch.full_like(indices, -1)
    expected[:tokens, 0, : len(visible)] = torch.tensor(visible, dtype=torch.int32)
    assert torch.equal(actual, expected)
    assert spans[:, 0].tolist() == [len(visible)] * tokens + [0, 0]


def test_dspark_empty_requests_invalid_pages_and_stable_outputs():
    table = torch.tensor([[1, -1, 3, 0, 2], [0, 0, 0, 0, 0], [99, 0, 1, 2, 3]], dtype=torch.int32)
    offsets = torch.tensor([0, 5, 5, 10], dtype=torch.int32)
    lengths = torch.tensor([133, 0, 9], dtype=torch.int32)
    indices, spans = torch.empty((12, 1, 256), dtype=torch.int32), torch.empty((12, 1), dtype=torch.int32)
    addresses = (indices.data_ptr(), spans.data_ptr())
    for _ in range(2):
        build_dspark_v41_swa_indices(
            table, offsets, lengths, page_size=32, num_cache_blocks=4, indices_output=indices, lengths_output=spans
        )
        assert (indices.data_ptr(), spans.data_ptr()) == addresses
        assert torch.all(indices[:5, 0, 32:64] == -1)
        assert torch.equal(indices[0, 0, 64:133], torch.arange(64, 133, dtype=torch.int32))
        assert torch.all(indices[5:] == -1)
        # Invalid pages make holes; span must not truncate later valid columns.
        assert spans[:, 0].tolist() == [133] * 5 + [9] * 5 + [0, 0]


def test_dspark_metadata_and_forward_explicit_contract():
    ops = AscendDSAV41Ops(0)
    offsets, lengths = torch.tensor([0, 5], dtype=torch.int32), torch.tensor([133], dtype=torch.int32)
    table = torch.zeros((1, 5), dtype=torch.int32)
    indices, spans = torch.empty((5, 1, 256), dtype=torch.int32), torch.empty((5, 1), dtype=torch.int32)
    with patch(
        "torch.ops._C_ascend.npu_sparse_flash_mla_metadata", return_value=torch.zeros(1024), create=True
    ) as build:
        meta = ops.build_metadata(
            offsets,
            lengths,
            table,
            max_seqlen_q=5,
            max_seqlen_kv=160,
            draft_swa_indices=indices,
            draft_swa_lengths=spans,
        )
    assert build.call_args.kwargs["ori_mask_mode"] == 0
    assert build.call_args.kwargs["ori_topk"] == 256
    assert build.call_args.kwargs["ori_topk_length"] is spans
    assert meta.seqused_kv is lengths  # Keep caller-owned length updates visible during graph replay.
    query = torch.empty((5, 8, 512), dtype=torch.bfloat16)
    cache = torch.empty((1, 32, 1, 512), dtype=torch.bfloat16)
    with patch("torch.ops._C_ascend.npu_sparse_flash_mla", return_value=(query, torch.empty(0)), create=True) as native:
        ops.forward(query, cache, torch.zeros(8), meta)
    assert native.call_args.kwargs["ori_sparse_indices"] is indices
    assert native.call_args.kwargs["ori_topk_length"] is spans
    assert native.call_args.kwargs["ori_mask_mode"] == 0
    assert native.call_args.kwargs["ori_win_left"] == 255
    with pytest.raises(ValueError, match="CR0"):
        AscendDSAV41Ops(1).build_metadata(
            offsets,
            lengths,
            table,
            max_seqlen_q=5,
            max_seqlen_kv=160,
            cmp_block_table=table,
            draft_swa_indices=indices,
            draft_swa_lengths=spans,
        )
    with pytest.raises(ValueError, match="together"):
        ops.build_metadata(offsets, lengths, table, max_seqlen_q=5, max_seqlen_kv=160, draft_swa_indices=indices)


@pytest.mark.parametrize("maximum", [128, 129])
@pytest.mark.parametrize("rejected", range(6))
def test_dspark_end_queries_do_not_shift_virtual_prefix(maximum, rejected):
    pages = (maximum + 31) // 32
    prefix = maximum - rejected
    indices = torch.empty((7, 1, 256), dtype=torch.int32)
    lengths = torch.empty((7, 1), dtype=torch.int32)
    with patch.object(torch.Tensor, "item", side_effect=AssertionError("host read")):
        build_dspark_v41_swa_indices(
            torch.arange(pages, dtype=torch.int32).flip(0)[None],
            torch.tensor([0, 5], dtype=torch.int32),
            torch.tensor([prefix + 5], dtype=torch.int32),
            page_size=32,
            num_cache_blocks=pages,
            indices_output=indices,
            lengths_output=lengths,
            max_model_len=maximum,
        )
    visible = list(range(max(0, prefix - 128), min(prefix + 5, maximum)))
    for row in range(7):
        if row < min(rejected, 5):
            assert indices[row, 0, : len(visible)].tolist() == visible
            assert lengths[row] == len(visible)
        else:
            assert torch.all(indices[row] == -1)
            assert lengths[row] == 0


def test_dspark_zero_span_masks_uninitialized_native_output_and_lse():
    ops = AscendDSAV41Ops(0)
    indices = torch.full((5, 1, 256), -1, dtype=torch.int32)
    spans = torch.tensor([[129], [129], [0], [0], [0]], dtype=torch.int32)
    with patch("torch.ops._C_ascend.npu_sparse_flash_mla_metadata", return_value=torch.zeros(1024), create=True):
        metadata = ops.build_metadata(
            torch.tensor([0, 5], dtype=torch.int32),
            torch.tensor([129], dtype=torch.int32),
            torch.zeros((1, 5), dtype=torch.int32),
            max_seqlen_q=5,
            max_seqlen_kv=129,
            draft_swa_indices=indices,
            draft_swa_lengths=spans,
        )
    assert metadata.seqused_kv.tolist() == [129]
    native_output = torch.full((5, 8, 512), float("nan"), dtype=torch.bfloat16)
    native_output[:2].fill_(3)
    native_lse = torch.full((1, 5, 8), float("nan"))
    native_lse[:, :2].fill_(4)
    with patch("torch.ops._C_ascend.npu_sparse_flash_mla", return_value=(native_output, native_lse), create=True):
        output, lse = ops.forward(
            torch.empty_like(native_output),
            torch.empty((5, 32, 1, 512), dtype=torch.bfloat16),
            torch.zeros(8),
            metadata,
            return_softmax_lse=True,
        )
    assert torch.all(output[:2] == 3) and torch.all(lse[:, :2] == 4)
    assert torch.all(output[2:] == 0) and torch.all(lse[:, 2:] == 0)
