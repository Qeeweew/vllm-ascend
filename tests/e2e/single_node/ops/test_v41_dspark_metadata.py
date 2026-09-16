# SPDX-License-Identifier: Apache-2.0
"""AscendC schedules checked by SMLA execution, FP32 oracle and graph replay."""

from dataclasses import replace

import pytest
import torch
from test_dsa_v41 import check, device_case, make_case


@pytest.fixture(scope="module")
def runtime():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from vllm_ascend.ops.dsa_v41 import AscendDSAV41Ops, build_dspark_v41_swa_indices

    torch.set_num_threads(4)
    torch.npu.set_device(0)  # Select physical device with ASCEND_RT_VISIBLE_DEVICES.
    if not hasattr(torch.ops._C_ascend, "v41_dspark_metadata"):
        pytest.fail("Load the complete isolated extension with the metadata binding")
    return AscendDSAV41Ops(0), build_dspark_v41_swa_indices


def assert_schedule(schedule, offsets):
    """Interpret the consumer's half-open BN2/M bounds and prove exact coverage."""
    rows = schedule.cpu().tolist()
    assert len(rows) == 1024
    assert not any(rows[20 * 9 :]), "Unused AIC/FD and trailing storage must be cleared"
    covered, loads = [], []
    for core in range(20):
        enabled, bs, ms, ss, be, me, se, fd, max_s2 = rows[core * 9 : (core + 1) * 9]
        if not enabled:
            assert [enabled, bs, ms, ss, be, me, se, fd, max_s2] == [0] * 9
            continue
        assert enabled == 1 and ss == se == fd == max_s2 == 0
        if core == 0:
            assert bs == ms == 0
        stop_batch = be + int(me != 0)
        tasks = []
        for batch in range(bs, stop_batch):
            count = offsets[batch + 1] - offsets[batch]
            start = ms if batch == bs else 0
            stop = me if batch == stop_batch - 1 and me else count
            assert 0 <= start <= stop <= count
            tasks.extend(range(offsets[batch] + start, offsets[batch] + stop))
        assert tasks
        covered.extend(tasks)
        loads.append(len(tasks))
    assert covered == list(range(offsets[-1])), "Every query must execute once, in order"
    assert len(loads) == min(20, offsets[-1])
    if loads:
        assert max(loads) - min(loads) <= 1


def oracle(case, indices, spans):
    output = torch.zeros_like(case["q"], dtype=torch.float32)
    lse = torch.zeros((1, case["q"].shape[0], 8))
    for request in range(case["lengths"].numel()):
        start, stop = case["cu"][request : request + 2].tolist()
        for row in range(start, stop):
            span = int(spans[row, 0])
            if span == 0:
                continue
            ids = indices[row, 0, :span].long()
            ids = ids[(ids >= 0) & (ids < int(case["lengths"][request]))]
            pages = case["swa_bt"][request, ids // 32].long()
            keys = case["swa"][pages, ids % 32, 0].float()
            scores = case["q"][row].float() @ keys.T / 512**0.5
            scores = torch.cat((scores, case["sinks"][:, None]), dim=-1)
            output[row] = scores.softmax(-1)[:, :-1] @ keys
            lse[0, row] = scores.logsumexp(-1)
    return output, lse


def visibility_oracle(case, max_model_len):
    """Independently enumerate prefix128 + K noncausal logical candidates."""
    tokens = case["q"].shape[0]
    indices = torch.full((tokens, 1, 256), -1, dtype=torch.int32)
    spans = torch.zeros((tokens, 1), dtype=torch.int32)
    for request, length in enumerate(case["lengths"].tolist()):
        first, last = case["cu"][request : request + 2].tolist()
        count = last - first
        if count == 0:
            continue
        prefix = length - count
        start = max(0, prefix - 128)
        end = min(length, max_model_len)
        ids = torch.arange(start, end, dtype=torch.int32)
        for row in range(first, last):
            if prefix + row - first >= max_model_len:
                continue
            indices[row, 0, : ids.numel()] = ids
            spans[row] = ids.numel()
    return indices, spans


def make_inputs(batch, span, ragged=False, draft_tokens=5):
    query_counts = [0 if ragged and request % 3 == 0 else draft_tokens for request in range(batch)]
    if not any(query_counts):
        query_counts[-1] = draft_tokens
    case = make_case(0, query_counts, [256 if count else 0 for count in query_counts])
    case["q"] = torch.cat((case["q"], torch.zeros((3, 8, 512), dtype=torch.bfloat16)))
    tokens = case["q"].shape[0]
    indices = torch.full((tokens, 1, 256), -1, dtype=torch.int32)
    spans = torch.zeros((tokens, 1), dtype=torch.int32)
    total = sum(query_counts)
    indices[:total, 0, :span] = torch.arange(span)
    spans[:total] = span
    # In-range padded rows test that schedule coverage does not reinterpret
    # visibility. The wrapper must still initialize their outputs to zero.
    if ragged:
        spans[1::4] = 0
        indices[1::4] = -1
    return case, indices, spans


def make_metadata(ops, case, indices, spans):
    native = ops.build_metadata(
        case["cu"],
        case["lengths"],
        case["swa_bt"],
        max_seqlen_q=case["q"].shape[0],
        max_seqlen_kv=case["swa_bt"].shape[1] * 32,
        draft_swa_indices=indices,
        draft_swa_lengths=spans,
    )
    return native, replace(native, schedule=torch.full((1024,), -77, dtype=torch.int32, device="npu"))


def run_candidate(ops, case, metadata):
    torch.ops._C_ascend.v41_dspark_metadata(
        case["cu"],
        case["lengths"],
        metadata.draft_swa_lengths,
        metadata.schedule,
    )
    return ops.forward(case["q"], case["swa"], case["sinks"], metadata, return_softmax_lse=True)


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("span", [0, 1, 133, 256])
@pytest.mark.parametrize("ragged", [False, True])
def test_schedule_native_and_fp32_oracles(runtime, batch, span, ragged):
    ops, _ = runtime
    host, indices, spans = make_inputs(batch, span, ragged)
    case = device_case(host, gapped=True)
    native, candidate = make_metadata(ops, case, indices.npu(), spans.npu())
    actual, lse = run_candidate(ops, case, candidate)
    baseline, baseline_lse = ops.forward(case["q"], case["swa"], case["sinks"], native, return_softmax_lse=True)
    expected, expected_lse = oracle(host, indices, spans)
    assert_schedule(candidate.schedule, host["cu"].tolist())
    check(actual, expected)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.015, rtol=0.003)
    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
    torch.testing.assert_close(lse, baseline_lse, atol=0, rtol=0)


@pytest.mark.parametrize(
    "batch,draft_tokens",
    [(batch, k) for batch in (1, 4, 16) for k in (1, 2, 3, 4, 5, 6, 7, 8)] + [(batch, 5) for batch in (2, 8, 32)],
)
def test_changed_graph_captures_schedule_and_attention(runtime, batch, draft_tokens):
    ops, build_indices = runtime
    host = make_case(0, [draft_tokens] * batch, [256] * batch)
    host["q"] = torch.cat((host["q"], torch.zeros((3, 8, 512), dtype=torch.bfloat16)))
    case = device_case(host)
    tokens = host["q"].shape[0]
    indices = torch.empty((tokens, 1, 256), dtype=torch.int32, device="npu")
    spans = torch.empty((tokens, 1), dtype=torch.int32, device="npu")

    def refresh_visibility():
        build_indices(
            case["swa_bt"],
            case["cu"],
            case["lengths"],
            page_size=32,
            num_cache_blocks=case["swa"].shape[0],
            indices_output=indices,
            lengths_output=spans,
            max_model_len=256,
        )

    refresh_visibility()
    _, candidate = make_metadata(ops, case, indices, spans)

    def run():
        refresh_visibility()
        return run_candidate(ops, case, candidate)

    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual, lse = run()
    pointers = [value.data_ptr() for value in (candidate.schedule, indices, spans)]
    for replay in range(16):
        # Rejection counts include zero and all K for each tested small K.
        # Empty request slots and different active T reuse captured buffers.
        counts = [0 if (r + replay) % 4 == 0 else draft_tokens for r in range(batch)]
        rejected = (0, 1, draft_tokens // 2, draft_tokens, max(0, draft_tokens - 1), draft_tokens)[replay % 6]
        lengths = [256 - rejected + count if count else 0 for count in counts]
        offsets = [0, *torch.tensor(counts).cumsum(0).tolist()]
        host["cu"].copy_(torch.tensor(offsets, dtype=torch.int32))
        host["lengths"].copy_(torch.tensor(lengths, dtype=torch.int32))
        host["q"].neg_()
        host["swa"].neg_()
        host["swa_bt"].copy_(host["swa_bt"].roll(1, 1))
        for name in ("cu", "lengths", "q", "swa", "swa_bt"):
            case[name].copy_(host[name])
        graph.replay()
        torch.npu.synchronize()
        cpu_indices, cpu_spans = indices.cpu(), spans.cpu()
        expected_indices, expected_spans = visibility_oracle(host, max_model_len=256)
        torch.testing.assert_close(cpu_indices, expected_indices, atol=0, rtol=0)
        torch.testing.assert_close(cpu_spans, expected_spans, atol=0, rtol=0)
        expected, expected_lse = oracle(host, expected_indices, expected_spans)
        check(actual, expected)
        torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.015, rtol=0.003)
        assert_schedule(candidate.schedule, offsets)
        assert pointers == [value.data_ptr() for value in (candidate.schedule, indices, spans)]


@pytest.mark.parametrize("batch,tokens", [(0, 0), (0, 8), (4, 0), (32, 160)])
def test_empty_schedule_clears_previous_contents(runtime, batch, tokens):
    schedule = torch.full((1024,), 123, dtype=torch.int32, device="npu")
    torch.ops._C_ascend.v41_dspark_metadata(
        torch.zeros(batch + 1, dtype=torch.int32, device="npu"),
        torch.zeros(batch, dtype=torch.int32, device="npu"),
        torch.zeros((tokens, 1), dtype=torch.int32, device="npu"),
        schedule,
    )
    assert torch.count_nonzero(schedule.cpu()) == 0


@pytest.mark.parametrize("draft_tokens", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("batch", [1, 4, 32])
def test_multik_schedule_native_and_fp32_oracles(runtime, batch, draft_tokens):
    """Changing K only changes query counts and visibility, never planner ABI."""
    ops, _ = runtime
    host, indices, spans = make_inputs(batch, 128 + draft_tokens, ragged=True, draft_tokens=draft_tokens)
    case = device_case(host, gapped=True)
    native, candidate = make_metadata(ops, case, indices.npu(), spans.npu())
    actual, lse = run_candidate(ops, case, candidate)
    baseline, baseline_lse = ops.forward(case["q"], case["swa"], case["sinks"], native, return_softmax_lse=True)
    expected, expected_lse = oracle(host, indices, spans)
    assert_schedule(candidate.schedule, host["cu"].tolist())
    check(actual, expected)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.015, rtol=0.003)
    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
    torch.testing.assert_close(lse, baseline_lse, atol=0, rtol=0)


@pytest.mark.parametrize("draft_tokens", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("batch", [1, 4, 16])
def test_multik_graph_schedule_and_attention(runtime, batch, draft_tokens):
    """Replay tested small K, active requests, zero rows and spans through SMLA."""
    ops, _ = runtime
    host, host_indices, host_spans = make_inputs(batch, 128 + draft_tokens, draft_tokens=draft_tokens)
    case = device_case(host)
    indices, spans = host_indices.npu(), host_spans.npu()
    _, candidate = make_metadata(ops, case, indices, spans)
    for _ in range(3):
        run_candidate(ops, case, candidate)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual, lse = run_candidate(ops, case, candidate)
    pointers = [value.data_ptr() for value in (candidate.schedule, indices, spans)]
    for replay in range(6):
        counts = [0 if (request + replay) % 3 == 0 else draft_tokens for request in range(batch)]
        offsets = [0, *torch.tensor(counts).cumsum(0).tolist()]
        span = (0, 1, 128 + draft_tokens, 256, 128 + draft_tokens, 0)[replay]
        host["cu"].copy_(torch.tensor(offsets, dtype=torch.int32))
        host_indices.fill_(-1)
        host_spans.zero_()
        host_indices[: offsets[-1], 0, :span] = torch.arange(span)
        host_spans[: offsets[-1]] = span
        # Keep some zero-span rows inside active ranges, not only tail padding.
        host_indices[1::7] = -1
        host_spans[1::7] = 0
        host["q"].neg_()
        host["swa"].neg_()
        host["swa_bt"].copy_(host["swa_bt"].roll(1, 1))
        for name in ("cu", "q", "swa", "swa_bt"):
            case[name].copy_(host[name])
        indices.copy_(host_indices)
        spans.copy_(host_spans)
        graph.replay()
        torch.npu.synchronize()
        expected, expected_lse = oracle(host, host_indices, host_spans)
        check(actual, expected)
        torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.015, rtol=0.003)
        assert_schedule(candidate.schedule, offsets)
        assert pointers == [value.data_ptr() for value in (candidate.schedule, indices, spans)]
