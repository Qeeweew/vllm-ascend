# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.dsa_v41 import (
    AscendV41CacheBackend,
    AscendV41CacheMetadataBuilder,
    make_v41_attention_metadata,
    make_v41_indexer_metadata,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import AscendV41IndexerCacheSpec, AscendV41MainCacheSpec, AscendV41SWACacheSpec


def make_builder(role, ratio=1, device="cpu", max_model_len=256):
    spec_type = {"main": AscendV41MainCacheSpec, "index": AscendV41IndexerCacheSpec, "swa": AscendV41SWACacheSpec}[role]
    args = dict(
        block_size=32 * ratio,
        num_kv_heads=1,
        head_size=128 if role == "index" else 512,
        dtype=torch.int8 if role == "index" else torch.bfloat16,
        head_size_v=0,
    )
    spec_fields = {field.name for field in fields(spec_type)}
    if "tokens_per_state" in spec_fields:
        args["tokens_per_state"] = ratio
    if "compress_ratio" in spec_fields:
        args["compress_ratio"] = ratio
    if role == "swa":
        args["sliding_window"] = 128
    spec = spec_type(**args)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=4),
        model_config=SimpleNamespace(max_model_len=max_model_len, hf_config=SimpleNamespace(num_attention_heads=64)),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
    )
    return AscendV41CacheMetadataBuilder(spec, ["layer.cache"], config, torch.device(device))


def make_common(device="cpu", changed=False):
    # Deliberately wrong CPU boundaries catch accidental dependence on host
    # request lengths. The device second request starts after token 2 or 3.
    return SimpleNamespace(
        positions=torch.tensor(
            [62, 63, 64, 3, 4, 5, 100] if not changed else [63, 64, 3, 4, 5, 6, 100], dtype=torch.int64, device=device
        ),
        query_start_loc=torch.tensor([0, 3, 6] if not changed else [0, 2, 6], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, 1, 6], dtype=torch.int32),
        seq_lens=torch.tensor([65, 6] if not changed else [65, 7], dtype=torch.int32, device=device),
        block_table_tensor=torch.tensor([[4, 7, 9, 10], [2, 5, 8, 11]], dtype=torch.int32, device=device),
        # Garbage generic slot IDs must not be used to address compressed KV.
        slot_mapping=torch.full((7,), 99999, dtype=torch.int64, device=device),
        num_reqs=2,
        num_actual_tokens=7,
    )


def make_execution_common(query_lengths, prefilling):
    starts = torch.tensor([0, *torch.tensor(query_lengths).cumsum(0).tolist()], dtype=torch.int32)
    tokens = sum(query_lengths)
    return AscendCommonAttentionMetadata(
        query_start_loc=starts.clone(),
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor([32 + length for length in query_lengths], dtype=torch.int32),
        num_reqs=len(query_lengths),
        num_actual_tokens=tokens,
        max_query_len=max(query_lengths, default=0),
        max_seq_len=36,
        block_table_tensor=torch.zeros((len(query_lengths), 2), dtype=torch.int32),
        slot_mapping=torch.full((4,), -1, dtype=torch.int64),
        positions=torch.tensor([32] * tokens + [-1] * (4 - tokens), dtype=torch.int64),
        is_prefilling=torch.tensor(prefilling, dtype=torch.bool) if prefilling is not None else None,
    )


@pytest.fixture
def native_metadata():
    with (
        patch(
            "torch.ops._C_ascend.npu_sparse_flash_mla_metadata",
            create=True,
            side_effect=lambda **kw: torch.full((1024,), kw["cmp_ratio"], dtype=torch.int32),
        ) as attention,
        patch(
            "torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata",
            create=True,
            return_value=torch.full((1024,), 42, dtype=torch.int32),
        ) as index,
        patch(
            "torch.ops._C_ascend.v41_dspark_metadata",
            create=True,
            side_effect=lambda cu, lengths, topk, output: output.fill_(99),
        ) as draft,
    ):
        yield attention, index, draft


@pytest.mark.parametrize("role,ratio", [("swa", 1), ("main", 1), ("main", 2), ("index", 1), ("index", 2)])
def test_physical_slots_and_device_boundaries(native_metadata, role, ratio):
    builder = make_builder(role, ratio)
    meta = builder.build(0, make_common())
    assert meta.slot_mapping.tolist() == (
        [254, 255, 288, 67, 68, 69, -1] if ratio == 1 else [-1, 159, -1, 65, -1, 66, -1]
    )
    assert meta.token_to_req_indices.tolist() == [0, 0, 0, 1, 1, 1, -1]
    assert meta.block_table.is_contiguous()
    if role != "swa":
        assert meta.seqused_cmp_kv.tolist() == [65 // ratio, 6 // ratio]
    if ratio == 2:
        assert meta.cmp_residual_kv.tolist() == [1, 0]
    attention, index, _ = native_metadata
    assert (attention.call_count, index.call_count) == ((0, 1) if role == "index" else (1, 0))


def test_rebuild_preserves_graph_addresses_and_refreshes_every_derived_buffer(native_metadata):
    builder = make_builder("main", 2)
    first = builder.build(0, make_common())
    addresses = {
        field.name: getattr(first, field.name).data_ptr()
        for field in fields(first)
        if isinstance(getattr(first, field.name), torch.Tensor)
    }
    second = builder.build(0, make_common(changed=True))
    for name, address in addresses.items():
        assert getattr(second, name).data_ptr() == address
    assert first.slot_mapping.tolist() == [159, -1, 65, -1, 66, -1, -1]
    assert first.cmp_residual_kv.tolist() == [1, 1]
    assert first.token_to_req_indices.tolist() == [0, 0, 1, 1, 1, 1, -1]


def test_assemble_source_and_consumer_without_rescheduling(native_metadata):
    common = make_common()
    swa = make_builder("swa").build(0, common)
    main = make_builder("main", 2).build(0, common)
    index = make_builder("index", 2).build(0, common)
    consumer = make_v41_attention_metadata(swa, main)
    selector = make_v41_indexer_metadata(index)
    assert consumer.schedule is main.schedule
    assert consumer.swa_block_table is swa.block_table
    assert consumer.cmp_block_table is main.block_table
    assert selector.qli_metadata is index.schedule
    assert selector.block_table is index.block_table
    assert native_metadata[0].call_count == 2
    assert native_metadata[1].call_count == 1


def test_missing_pages_and_padding_are_never_written(native_metadata):
    common = make_common()
    common.block_table_tensor[0, 0] = -1
    common.positions[5] = 9999
    meta = make_builder("main", 2).build(0, common)
    assert meta.slot_mapping.tolist() == [-1, -1, -1, 65, -1, -1, -1]


def test_empty_batch_does_not_launch_aicpu(native_metadata):
    common = make_common()
    common.num_reqs = 0
    common.query_start_loc = common.query_start_loc[:1]
    common.seq_lens = common.seq_lens[:0]
    common.block_table_tensor = common.block_table_tensor[:0]
    meta = make_builder("main", 2).build(0, common)
    assert torch.all(meta.slot_mapping == -1)
    assert not native_metadata[0].called


def test_rebased_window_table_is_rejected_before_native_schedule(native_metadata):
    common = make_common()
    common.max_seq_len = 129
    # Four physical SWA pages cover 128 logical tokens. Rebasing these to
    # the most recent window would make absolute-position lookup incorrect.
    with pytest.raises(ValueError, match="full logical block-table"):
        make_builder("swa").build(0, common)
    assert not native_metadata[0].called


def test_oversized_bucket_is_rejected_before_buffer_copy(native_metadata):
    common = make_common()
    common.slot_mapping = torch.empty(17, dtype=torch.int64)
    with pytest.raises(ValueError, match="preallocated metadata capacity"):
        make_builder("swa").build(0, common)
    assert not native_metadata[0].called


def test_backend_layouts_preserve_contiguous_state_per_layer():
    layouts = AscendV41CacheBackend.supported_kv_cache_layouts()
    assert tuple(layout.name for layout in layouts) == ("LBNHC", "LBHNC")
    assert all(layout.is_layer_compact and layout.is_block_compact for layout in layouts)


@pytest.mark.parametrize("role,ratio", [("swa", 1), ("main", 2), ("index", 1)])
@pytest.mark.parametrize(
    "query_lengths,prefilling,expected",
    [
        ([1], [False], (0, 1)),
        ([1], [True], (1, 0)),
        ([1, 1], [False, True], (1, 1)),
        ([1, 3], [False, True], (1, 1)),
        ([2, 2], [False, False], (0, 4)),
        ([3, 1], [False, True], (1, 3)),
        ([1, 1, 0], [False, False, False], (0, 2)),
        ([1], None, (1, 0)),
        ([], [], (0, 0)),
    ],
)
def test_host_execution_classification(native_metadata, role, ratio, query_lengths, prefilling, expected):
    common = make_execution_common(query_lengths, prefilling)
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("no dispatch D2H")):
        meta = make_builder(role, ratio).build(0, common)
    assert (meta.num_prefills, meta.num_decode_tokens) == expected


def test_capture_overrides_stale_prefill_flags_without_mutating_common(native_metadata):
    builder = make_builder("swa")
    common = make_execution_common([1, 1], [True, True])
    meta = builder.build_for_cudagraph_capture(common)
    assert (meta.num_prefills, meta.num_decode_tokens) == (0, 2)
    assert common.is_prefilling.tolist() == [True, True]
    # A subsequent real one-token prefill must restore the fallback branch.
    real = builder.build(0, common)
    assert (real.num_prefills, real.num_decode_tokens) == (2, 0)
    torch.testing.assert_close(meta.slot_mapping, real.slot_mapping)
    assert meta.slot_mapping.data_ptr() == real.slot_mapping.data_ptr()


def test_capture_long_queries_do_not_select_decode(native_metadata):
    common = make_execution_common([4], [True])
    meta = make_builder("swa").build_for_cudagraph_capture(common)
    assert (meta.num_prefills, meta.num_decode_tokens) == (1, 0)


def make_draft_common(device="cpu"):
    return SimpleNamespace(
        positions=torch.tensor([128, 129, 130, 131, 132, -1, -1], device=device),
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([133], dtype=torch.int32, device=device),
        block_table_tensor=torch.tensor([[4, 3, 2, 1, 0]], dtype=torch.int32, device=device),
        slot_mapping=torch.full((7,), -1, dtype=torch.int64, device=device),
        num_reqs=1,
        num_actual_tokens=5,
        max_query_len=5,
        max_seq_len=133,
        causal=False,
    )


def bind_draft_cache(builder, cache):
    builder.vllm_config.compilation_config = SimpleNamespace(
        static_forward_context={"layer.cache": SimpleNamespace(kv_cache=cache)}
    )


def test_draft_builder_is_explicit_and_target_stays_causal(native_metadata):
    common = make_draft_common()
    target = make_builder("swa")
    target_meta = target.build(0, common)
    assert target_meta.draft_swa_indices is None
    assert native_metadata[0].call_args.kwargs["ori_mask_mode"] == 4
    draft = make_builder("swa")
    bind_draft_cache(draft, torch.empty((5, 32, 1, 512), dtype=torch.bfloat16))
    draft.enable_dspark_device_metadata(8)
    first = draft.build(0, common)
    converted = make_v41_attention_metadata(first)
    assert converted.draft_swa_indices is first.draft_swa_indices
    assert converted.draft_swa_lengths is first.draft_swa_lengths
    assert first.draft_swa_lengths[:, 0].tolist() == [133] * 5 + [0, 0]
    assert first.draft_swa_indices[0, 0, :133].tolist() == list(range(133))
    assert native_metadata[0].call_count == 1  # Only the target used AICPU.
    assert native_metadata[2].call_args.args[2] is first.draft_swa_lengths
    assert native_metadata[2].call_args.args[3] is first.schedule
    assert first.schedule.eq(99).all()
    addresses = (first.draft_swa_indices.data_ptr(), first.draft_swa_lengths.data_ptr(), first.schedule.data_ptr())
    common.block_table_tensor[0, 4] = 99
    again = draft.build(0, common)
    assert addresses == (
        again.draft_swa_indices.data_ptr(),
        again.draft_swa_lengths.data_ptr(),
        again.schedule.data_ptr(),
    )
    assert torch.all(again.draft_swa_indices[:, :, 128:] == -1)
    assert torch.all(again.slot_mapping == -1)
    main = make_builder("main").build(0, common)
    with pytest.raises(ValueError, match="compressed"):
        make_v41_attention_metadata(first, main)


def test_draft_builder_rejects_wrong_role_causal_and_capacity(native_metadata):
    with pytest.raises(ValueError, match="CR0"):
        make_builder("main").enable_dspark_device_metadata(8)
    draft = make_builder("swa")
    with pytest.raises(ValueError, match="capacity"):
        draft.enable_dspark_device_metadata(17)
    draft.enable_dspark_device_metadata(8)
    draft.enable_dspark_device_metadata(8)
    with pytest.raises(ValueError, match="cannot change"):
        draft.enable_dspark_device_metadata(7)
    common = make_draft_common()
    common.causal = True
    with pytest.raises(ValueError, match="causal=False"):
        draft.build(0, common)


@pytest.mark.parametrize("heads,requests,queries", [(4, 4, 8), (8, 4097, 8), (8, 4, 32769)])
def test_draft_schedule_rejects_unsupported_partition_before_allocation(heads, requests, queries):
    builder = make_builder("swa")
    builder.num_heads = heads
    builder.max_requests = requests
    with pytest.raises(ValueError, match="schedule requires H8"):
        builder.enable_dspark_device_metadata(queries)
    assert builder.draft_swa_indices is None
    assert builder.draft_swa_lengths is None


@pytest.mark.parametrize("maximum", [128, 129])
@pytest.mark.parametrize("rejected", range(6))
def test_draft_maximum_context_retains_valid_queries(native_metadata, maximum, rejected):
    builder = make_builder("swa", max_model_len=maximum)
    pages = (maximum + 31) // 32
    table = torch.arange(3 * pages, dtype=torch.int32).flip(0).reshape(3, pages)
    bind_draft_cache(builder, torch.empty((3 * pages, 32, 1, 512), dtype=torch.bfloat16))
    builder.enable_dspark_device_metadata(16)
    prefixes = [maximum - rejected, 17, maximum - 2]
    positions = [p + i for p in prefixes for i in range(5)] + [maximum + 9]
    common = SimpleNamespace(
        positions=torch.tensor(positions),
        query_start_loc=torch.tensor([0, 5, 10, 15], dtype=torch.int32),
        seq_lens=torch.tensor([p + 5 for p in prefixes], dtype=torch.int32),
        block_table_tensor=table,
        slot_mapping=torch.empty(16, dtype=torch.int64),
        num_reqs=3,
        num_actual_tokens=15,
        max_query_len=5,
        max_seq_len=max(prefixes) + 5,
        causal=False,
    )
    metadata = builder.build(0, common)
    assert metadata.seqused_kv.tolist() == [min(p + 5, maximum) for p in prefixes]
    assert common.seq_lens.tolist() == [p + 5 for p in prefixes]
    assert metadata.positions.tolist() == positions  # Logical positions are never clamped.
    assert native_metadata[2].call_args.args[1] is metadata.seqused_kv
    for row, position in enumerate(positions):
        valid = row < 15 and position < maximum
        if not valid:
            assert metadata.slot_mapping[row] == -1
            assert metadata.token_to_req_indices[row] == -1
            assert metadata.draft_swa_lengths[row] == 0
            assert torch.all(metadata.draft_swa_indices[row] == -1)
            continue
        request = row // 5
        expected_slot = int(table[request, position // 32]) * 32 + position % 32
        assert metadata.slot_mapping[row] == expected_slot
        visible = list(range(max(prefixes[request] - 128, 0), min(prefixes[request] + 5, maximum)))
        assert metadata.draft_swa_lengths[row] == len(visible)
        assert metadata.draft_swa_indices[row, 0, : len(visible)].tolist() == visible


def test_draft_empty_middle_request_has_zero_native_length(native_metadata):
    builder = make_builder("swa", max_model_len=129)
    bind_draft_cache(builder, torch.empty((5, 32, 1, 512), dtype=torch.bfloat16))
    builder.enable_dspark_device_metadata(16)
    common = make_draft_common()
    common.num_reqs, common.num_actual_tokens, common.max_seq_len = 3, 10, 132
    common.positions = torch.tensor([124, 125, 126, 127, 128, 127, 128, 129, 130, 131, -1])
    common.slot_mapping = torch.empty(11, dtype=torch.int64)
    common.query_start_loc = torch.tensor([0, 5, 5, 10], dtype=torch.int32)
    common.seq_lens = torch.tensor([129, 17, 132], dtype=torch.int32)
    common.block_table_tensor = common.block_table_tensor.repeat(3, 1)
    metadata = builder.build(0, common)
    assert metadata.seqused_kv.tolist() == [129, 0, 129]
    assert common.seq_lens.tolist() == [129, 17, 132]
    assert metadata.token_to_req_indices.tolist() == [0, 0, 0, 0, 0, 2, 2, -1, -1, -1, -1]
    assert metadata.draft_swa_lengths[:, 0].tolist() == [129] * 7 + [0] * 4


@pytest.mark.parametrize("draft_tokens", range(1, 9))
def test_explicit_uniform_speculative_capture_and_real_prefill(native_metadata, draft_tokens):
    tokens = draft_tokens + 1
    common = make_execution_common([tokens], [True])
    common.slot_mapping = torch.full((tokens,), -1, dtype=torch.int64)
    builder = make_builder("swa")
    captured = builder.build_for_cudagraph_capture(common, uniform_decode=True)
    assert (captured.num_prefills, captured.num_decode_tokens) == (0, tokens)
    assert common.is_prefilling.tolist() == [True]
    prompt = builder.build(0, common)
    assert (prompt.num_prefills, prompt.num_decode_tokens) == (1, 0)
    common.is_prefilling.fill_(False)
    verification = builder.build(0, common)
    assert (verification.num_prefills, verification.num_decode_tokens) == (0, tokens)
