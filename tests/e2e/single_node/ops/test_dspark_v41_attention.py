# SPDX-License-Identifier: Apache-2.0
"""K5 noncausal attention against the official logical visibility predicate."""

from types import SimpleNamespace

import pytest
import torch
from test_dsa_v41 import check, device_case, make_case


def reference(case, maximum=None):
    output = torch.zeros_like(case["q"], dtype=torch.float32)
    lse = torch.zeros((1, case["q"].shape[0], 8), dtype=torch.float32)
    for request, sequence_length in enumerate(case["lengths"].tolist()):
        begin, end = case["cu"][request : request + 2].tolist()
        prefix_length = sequence_length - (end - begin)
        stop = sequence_length if maximum is None else min(sequence_length, maximum)
        positions = [p for p in range(stop) if p >= prefix_length - 128]
        # Gather directly through the CPU page table, independent of builder
        # layout/searchsorted and native logical-index interpretation.
        keys = [case["swa"][case["swa_bt"][request, p // 32], p % 32, 0] for p in positions]
        if begin == end:
            continue
        kv = torch.stack(keys).float()
        for row in range(begin, end):
            if maximum is not None and prefix_length + row - begin >= maximum:
                continue
            score = case["q"][row].float() @ kv.T / 512**0.5
            score = torch.cat((score, case["sinks"][:, None]), -1)
            output[row] = score.softmax(-1)[:, :-1] @ kv
            lse[0, row] = score.logsumexp(-1)
    return output, lse


@pytest.fixture(scope="module")
def runtime():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    from vllm_ascend.ops.dsa_v41 import AscendDSAV41Ops, build_dspark_v41_swa_indices

    torch.set_num_threads(8)
    torch.npu.set_device(0)
    return AscendDSAV41Ops, build_dspark_v41_swa_indices


def prepare(runtime, case):
    ops_type, build = runtime
    ops = ops_type(0)
    indices = torch.empty((case["q"].shape[0], 1, 256), dtype=torch.int32, device=case["q"].device)
    spans = torch.empty(indices.shape[:2], dtype=torch.int32, device=indices.device)

    def refresh_indices():
        build(
            case["swa_bt"],
            case["cu"],
            case["lengths"],
            page_size=32,
            num_cache_blocks=case["swa"].shape[0],
            indices_output=indices,
            lengths_output=spans,
        )

    def refresh_schedule():
        return ops.build_metadata(
            case["cu"],
            case["lengths"],
            case["swa_bt"],
            max_seqlen_q=5,
            max_seqlen_kv=case["swa_bt"].shape[1] * 32,
            draft_swa_indices=indices,
            draft_swa_lengths=spans,
        )

    refresh_indices()
    meta = refresh_schedule()

    def run():
        refresh_indices()
        return ops.forward(case["q"], case["swa"], case["sinks"], meta, return_softmax_lse=True)

    return run, refresh_indices, refresh_schedule, meta


@pytest.mark.parametrize("lengths,sink", [([5], 0), ([132], 0), ([133], 0), ([134], 0), ([1029, 19], 0), ([133], 20)])
def test_dspark_noncausal_oracle(runtime, lengths, sink):
    case = make_case(0, [5] * len(lengths), lengths, sink=sink)
    device = device_case(case, gapped=True)
    run, _, _, _ = prepare(runtime, device)
    actual, lse = run()
    expected, expected_lse = reference(case)
    check(actual, expected)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.015, rtol=0.003)


def test_dspark_graph_refreshes_query_lengths_pages_and_keys(runtime):
    case = make_case(0, [5, 5], [134, 69])
    # Two additional graph bucket rows are padding, never valid queries.
    case["q"] = torch.cat((case["q"], torch.zeros_like(case["q"][:2])))
    device = device_case(case, gapped=True)
    run, refresh_indices, refresh_schedule, meta = prepare(runtime, device)
    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual, lse = run()
    pointers = (meta.draft_swa_indices.data_ptr(), meta.draft_swa_lengths.data_ptr(), meta.schedule.data_ptr())
    for lengths, offsets in [([134, 69], [0, 5, 10]), ([133, 19], [0, 5, 10]), ([0, 5], [0, 0, 5])]:
        case["lengths"].copy_(torch.tensor(lengths, dtype=torch.int32))
        case["cu"].copy_(torch.tensor(offsets, dtype=torch.int32))
        case["swa_bt"] = case["swa_bt"].flip(0).roll(1, 1)
        case["q"] = case["q"].roll(1, 1)
        case["swa"] = -case["swa"]
        for name in ("lengths", "cu", "swa_bt", "q", "swa"):
            device[name].copy_(case[name])
        refresh_indices()
        meta.schedule.copy_(refresh_schedule().schedule)
        graph.replay()
        expected, expected_lse = reference(case)
        check(actual[: offsets[-1]], expected[: offsets[-1]])
        torch.testing.assert_close(lse.cpu()[:, : offsets[-1]], expected_lse[:, : offsets[-1]], atol=0.015, rtol=0.003)
        assert torch.all(meta.draft_swa_indices.cpu()[offsets[-1] :] == -1)
        assert torch.all(meta.draft_swa_lengths.cpu()[offsets[-1] :] == 0)
        assert pointers == (
            meta.draft_swa_indices.data_ptr(),
            meta.draft_swa_lengths.data_ptr(),
            meta.schedule.data_ptr(),
        )


def test_draft_cache_builder_refreshes_captured_attention(runtime):
    from tests.ut.attention.test_dsa_v41_metadata import bind_draft_cache, make_builder
    from vllm_ascend.attention.dsa_v41 import make_v41_attention_metadata

    case = make_case(0, [5, 5], [134, 69])
    device = device_case(case, gapped=True)
    ops = runtime[0](0)
    builder = make_builder("swa", device="npu:0")
    bind_draft_cache(builder, device["swa"])
    builder.enable_dspark_device_metadata(16)

    def common():
        positions = torch.cat([torch.arange(n - 5, n) for n in case["lengths"].tolist()]).npu()
        return SimpleNamespace(
            positions=positions,
            query_start_loc=device["cu"],
            seq_lens=device["lengths"],
            block_table_tensor=device["swa_bt"],
            slot_mapping=torch.empty(10, dtype=torch.int64, device="npu"),
            num_reqs=2,
            num_actual_tokens=10,
            max_query_len=5,
            max_seq_len=134,
            causal=False,
        )

    metadata = make_v41_attention_metadata(builder.build(0, common()))
    for _ in range(3):
        ops.forward(device["q"], device["swa"], device["sinks"], metadata)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual, _ = ops.forward(device["q"], device["swa"], device["sinks"], metadata)
    pointers = (
        metadata.draft_swa_indices.data_ptr(),
        metadata.draft_swa_lengths.data_ptr(),
        metadata.schedule.data_ptr(),
    )
    for lengths in ([134, 69], [133, 5], [19, 129]):
        case["lengths"].copy_(torch.tensor(lengths, dtype=torch.int32))
        case["swa_bt"] = case["swa_bt"].roll(2, 1)
        case["q"] = -case["q"]
        case["swa"] = case["swa"].roll(1, 0)
        for name in ("lengths", "swa_bt", "q", "swa"):
            device[name].copy_(case[name])
        builder.build(0, common())
        graph.replay()
        check(actual, reference(case)[0])
        assert pointers == (
            metadata.draft_swa_indices.data_ptr(),
            metadata.draft_swa_lengths.data_ptr(),
            metadata.schedule.data_ptr(),
        )


@pytest.mark.parametrize("maximum", [128, 129])
def test_maximum_context_mixed_requests_cache_writes_and_graph(runtime, maximum):
    """Real builder/cache writer/native attention, with CPU enumeration oracle."""
    from tests.ut.attention.test_dsa_v41_metadata import bind_draft_cache, make_builder
    from vllm_ascend.attention.dsa_v41 import make_v41_attention_metadata
    from vllm_ascend.ops.cache_v41 import write_main_cache_v41

    case = make_case(0, [5, 5, 5], [maximum] * 3)
    case["q"] = torch.cat((case["q"], torch.ones_like(case["q"][:1])))
    device = device_case(case, gapped=True)
    builder = make_builder("swa", device="npu:0", max_model_len=maximum)
    bind_draft_cache(builder, device["swa"])
    builder.enable_dspark_device_metadata(16)
    positions = torch.zeros(16, dtype=torch.int64, device="npu:0")
    common = SimpleNamespace(
        positions=positions,
        query_start_loc=device["cu"],
        seq_lens=device["lengths"],
        block_table_tensor=device["swa_bt"],
        slot_mapping=torch.empty(16, dtype=torch.int64, device="npu:0"),
        num_reqs=3,
        num_actual_tokens=15,
        max_query_len=5,
        max_seq_len=maximum + 5,
        causal=False,
    )
    keys = torch.randn((16, 512), generator=torch.Generator().manual_seed(4129)).bfloat16()
    device_keys = keys.npu()

    def refresh(rejected, empty_middle=False):
        prefixes = [maximum - rejected, 17, maximum - 2]
        query_counts = [5, 0 if empty_middle else 5, 5]
        offsets = [0, query_counts[0], sum(query_counts[:2]), sum(query_counts)]
        logical_positions = [p + i for p, count in zip(prefixes, query_counts) for i in range(count)]
        logical_positions += [maximum + 5] * (16 - len(logical_positions))
        case["lengths"].copy_(torch.tensor([p + n for p, n in zip(prefixes, query_counts)], dtype=torch.int32))
        case["cu"].copy_(torch.tensor(offsets, dtype=torch.int32))
        case["swa_bt"].copy_(case["swa_bt"].roll(1, 1).flip(0))
        case["q"].neg_()
        case["swa"].neg_()
        for name in ("lengths", "cu", "swa_bt", "q", "swa"):
            device[name].copy_(case[name])
        positions.copy_(torch.tensor(logical_positions, dtype=torch.int64))
        common.num_actual_tokens = offsets[-1]
        metadata = builder.build(0, common)
        # CPU writes only valid logical queries, independently of device slots.
        valid_rows = []
        for request, count in enumerate(query_counts):
            for local in range(count):
                row, logical = offsets[request] + local, prefixes[request] + local
                if logical < maximum:
                    page = int(case["swa_bt"][request, logical // 32])
                    case["swa"][page, logical % 32, 0] = keys[row]
                    valid_rows.append(row)
        assert torch.equal(metadata.positions.cpu(), torch.tensor(logical_positions))
        assert metadata.seqused_kv.cpu().tolist() == [
            min(p + n, maximum) if n else 0 for p, n in zip(prefixes, query_counts)
        ]
        invalid = torch.ones(16, dtype=torch.bool)
        invalid[valid_rows] = False
        assert torch.all(metadata.slot_mapping.cpu()[invalid] == -1)
        assert torch.all(metadata.draft_swa_lengths.cpu()[invalid] == 0)
        assert torch.all(metadata.draft_swa_indices.cpu()[invalid] == -1)
        return metadata, valid_rows, invalid

    raw_meta, _, _ = refresh(0)
    metadata = make_v41_attention_metadata(raw_meta)
    ops = runtime[0](0)

    def run():
        write_main_cache_v41(device["swa"], device_keys, raw_meta.slot_mapping)
        return ops.forward(device["q"], device["swa"], device["sinks"], metadata, return_softmax_lse=True)

    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, lse = run()
    pointers = [t.data_ptr() for t in (raw_meta.slot_mapping, metadata.seqused_kv, metadata.draft_swa_indices)]
    for rejected, empty_middle in [(n, False) for n in range(6)] + [(2, True)]:
        _, valid_rows, invalid = refresh(rejected, empty_middle)
        graph.replay()
        expected, expected_lse = reference(case, maximum)
        check(output[valid_rows], expected[valid_rows])
        torch.testing.assert_close(lse.cpu()[:, valid_rows], expected_lse[:, valid_rows], atol=0.015, rtol=0.003)
        torch.testing.assert_close(device["swa"].cpu(), case["swa"], rtol=0, atol=0)
        # Both out-of-context queries and graph bucket padding are deterministic.
        assert torch.count_nonzero(output.cpu()[invalid]) == 0
        assert torch.count_nonzero(lse.cpu()[:, invalid]) == 0
        assert pointers == [
            t.data_ptr() for t in (raw_meta.slot_mapping, metadata.seqused_kv, metadata.draft_swa_indices)
        ]


@pytest.mark.parametrize("maximum", [128, 129])
def test_all_queries_beyond_maximum_are_safe_graph_padding(runtime, maximum):
    from tests.ut.attention.test_dsa_v41_metadata import bind_draft_cache, make_builder
    from vllm_ascend.attention.dsa_v41 import make_v41_attention_metadata
    from vllm_ascend.ops.cache_v41 import write_main_cache_v41

    case = make_case(0, [5], [maximum])
    device = device_case(case)
    builder = make_builder("swa", device="npu:0", max_model_len=maximum)
    bind_draft_cache(builder, device["swa"])
    builder.enable_dspark_device_metadata(5)
    common = SimpleNamespace(
        positions=torch.arange(maximum, maximum + 5, device="npu:0"),
        query_start_loc=device["cu"],
        seq_lens=torch.tensor([maximum + 5], dtype=torch.int32, device="npu:0"),
        block_table_tensor=device["swa_bt"],
        slot_mapping=torch.empty(5, dtype=torch.int64, device="npu:0"),
        num_reqs=1,
        num_actual_tokens=5,
        max_query_len=5,
        max_seq_len=maximum + 5,
        causal=False,
    )
    raw_meta = builder.build(0, common)
    metadata = make_v41_attention_metadata(raw_meta)
    ops = runtime[0](0)
    keys = torch.ones((5, 512), dtype=torch.bfloat16, device="npu:0")

    def run():
        write_main_cache_v41(device["swa"], keys, raw_meta.slot_mapping)
        return ops.forward(device["q"], device["swa"], device["sinks"], metadata, return_softmax_lse=True)

    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, lse = run()
    for _ in range(2):
        graph.replay()
        assert torch.count_nonzero(output.cpu()) == 0
        assert torch.count_nonzero(lse.cpu()) == 0
        assert torch.all(raw_meta.slot_mapping.cpu() == -1)
        assert torch.equal(device["swa"].cpu(), case["swa"])
