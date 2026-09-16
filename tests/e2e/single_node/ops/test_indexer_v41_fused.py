# SPDX-License-Identifier: Apache-2.0
"""Independent gates for native fused CR1 consumer prefill and decode.

Only run after loading the separately built candidate OPP package. The generic
v3 path is selected with a zero output offset, providing a live old-kernel
control without changing global environment or graph state.
"""

import pytest
import torch
from indexer_v41_candidate_reference import (
    assert_candidate_selection,
    candidate_reference,
    candidate_reference_vectorized,
)
from test_indexer_v41 import build_metadata, make_case, runtime  # noqa: F401
from test_indexer_v41_candidate import candidate_device_case, candidates_for


def selector_call(device, metadata, candidates, *, legacy_offset=None, candidate_mode=2):
    """Complete native selector: v3 plus the existing public output contract."""
    indices, _, blocks = torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
        query=device["q"],
        key=device["k"],
        weights=device["w"],
        query_dequant_scale=device["qs"],
        key_dequant_scale=device["ks"],
        topk=512,
        quant_mode=2,
        candidate_topk_index=candidates,
        cu_seqlens_q=metadata.cu_seqlens_q,
        seqused_k=metadata.seqused_k,
        cmp_residual_k=metadata.cmp_residual_k,
        block_table=metadata.block_table,
        output_idx_offset=legacy_offset,
        metadata=metadata.qli_metadata,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=1,
        candidate_mode=candidate_mode,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
    )
    # Include the exact current wrapper postprocessing in whole-selector time.
    bound = metadata.block_table.shape[1] * device["k"].shape[1]
    sentinel = 2147483647
    if bound <= 2**24:
        indices = indices.float()
        sentinel = 2**24
    indices = torch.where(indices >= 0, indices, sentinel).sort(dim=-1).values
    rows = torch.arange(device["q"].shape[0], device=indices.device) < metadata.cu_seqlens_q[-1]
    indices = torch.where((indices != sentinel) & rows[:, None, None], indices, -1).int()
    return indices, blocks


@pytest.mark.parametrize("length", [1, 17, 511, 4097, 32771, 131075])
@pytest.mark.parametrize("adversarial", [False, True])
def test_fused_candidate_eager(runtime, length, adversarial):  # noqa: F811
    case = make_case(1, [1], [length])
    if adversarial:
        case["w"] = -case["w"].abs()
        if length > 32:
            case["bt"][0, 1] = -1
            case["bt"][0, -1] = case["k"].shape[0] + 3
    device = candidate_device_case(case, offset=adversarial)
    ops = runtime(1, "consumer")
    info = build_metadata(ops, device, max_q=1, max_k=length)
    candidates = candidates_for(length, adversarial)
    actual, _ = selector_call(device, info, candidates.to(device["q"].device))
    assert_candidate_selection(actual.cpu(), candidate_reference(case, candidates))


def test_fused_candidate_graph_replay(runtime):  # noqa: F811
    length = 4097
    case = make_case(1, [1], [length])
    device = candidate_device_case(case, offset=True)
    ops = runtime(1, "consumer")
    info = build_metadata(ops, device, max_q=1, max_k=length)
    candidates = candidates_for(length, True)
    candidate_device = candidates.to(device["q"].device)
    for _ in range(3):
        selector_call(device, info, candidate_device)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result, _ = selector_call(device, info, candidate_device)
    for iteration in range(8):
        generator = torch.Generator().manual_seed(88412 + iteration)
        case["q"].random_(-127, 128, generator=generator)
        case["w"].copy_(torch.randn(case["w"].shape, generator=generator).half() / 32)
        case["qs"].fill_(0.25 + iteration / 16)
        case["ks"].fill_(0.125 + iteration / 32)
        case["sk"][0] = (4097, 511, 17, 0)[iteration % 4]
        case["cu"][1] = 0 if iteration == 6 else 1
        case["bt"].copy_(case["bt"].flip(1))
        candidates = candidates_for(max(1, int(case["sk"][0])), True)
        if iteration == 5:
            candidates.fill_(-1)
        for name in ("q", "w", "qs", "ks", "sk", "cu", "bt"):
            device[name].copy_(case[name])
        candidate_device.copy_(candidates)
        graph.replay()
        assert_candidate_selection(result.cpu(), candidate_reference(case, candidates))


def row_candidates(case, adversarial=True):
    """Different candidate sets/order per query; padding also carries real IDs."""
    result = torch.cat([candidates_for(int(case["sk"].max()), adversarial)] * case["q"].shape[0])
    for request, length in enumerate(case["sk"].tolist()):
        begin, end = case["cu"][request : request + 2].tolist()
        for row in range(begin, end):
            visible = max(0, length - (end - begin) + row - begin + 1)
            result[row] = candidates_for(visible, adversarial)[0].roll(row * 17, -1)
            if row % 7 == 3:
                result[row, :, ::3] = -1
    return result


def assert_query_rows(case, candidates, output):
    """Apply frozen set-membership oracle after independent causal row mapping."""
    references = [None] * case["q"].shape[0]
    for request, length in enumerate(case["sk"].tolist()):
        begin, end = case["cu"][request : request + 2].tolist()
        for row in range(begin, end):
            visible = max(0, length - (end - begin) + row - begin + 1)
            single = dict(case)
            for name in ("q", "w", "qs"):
                single[name] = case[name][row : row + 1]
            single["cu"] = torch.tensor([0, 1], dtype=torch.int32)
            single["sk"] = torch.tensor([visible], dtype=torch.int32)
            single["bt"] = case["bt"][request : request + 1]
            references[row] = candidate_reference_vectorized(single, candidates[row : row + 1])
    for row, reference in enumerate(references):
        if reference is None:
            assert torch.all(output[row] == -1)
        else:
            assert_candidate_selection(output[row], reference)


@pytest.mark.parametrize("tokens", [32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("length", [4097, 32771])
def test_fused_prefill(runtime, tokens, length):  # noqa: F811
    case = make_case(1, [tokens], [length])
    device = candidate_device_case(case)
    candidates = row_candidates(case)
    info = build_metadata(runtime(1, "consumer"), device, max_q=tokens, max_k=length)
    actual, _ = selector_call(device, info, candidates.npu())
    assert_query_rows(case, candidates, actual.cpu())


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 20, 32, 64])
def test_fused_multibatch_decode(runtime, tokens):  # noqa: F811
    lengths = [4097, 32771, 17, 511, 131075, 0]
    case = make_case(1, [1] * tokens, [lengths[row % len(lengths)] for row in range(tokens)])
    device = candidate_device_case(case, offset=True)
    candidates = row_candidates(case)
    info = build_metadata(runtime(1, "consumer"), device, max_q=1, max_k=max(lengths))
    actual, _ = selector_call(device, info, candidates.npu())
    assert_query_rows(case, candidates, actual.cpu())


@pytest.mark.parametrize(
    "query_lengths,contexts,padding",
    [
        ([0, 1, 0, 3, 4], [0, 17, 32771, 511, 4097], 3),
        ([0, 32, 1, 0, 64, 1], [0, 32771, 17, 4097, 131075, 511], 5),
        ([1, 256, 0, 128, 1], [4097, 32771, 17, 131075, 511], 7),
    ],
)
def test_fused_ragged_prefill(runtime, query_lengths, contexts, padding):  # noqa: F811
    case = make_case(1, query_lengths, contexts, padding=padding)
    case["bt"][:, 1] = -1
    case["bt"][:, -1] = case["k"].shape[0] + 3
    device = candidate_device_case(case, offset=True)
    candidates = row_candidates(case)
    info = build_metadata(runtime(1, "consumer"), device, max_q=max(query_lengths), max_k=max(contexts))
    actual, _ = selector_call(device, info, candidates.npu())
    assert_query_rows(case, candidates, actual.cpu())


def test_fused_multibatch_graph_replay(runtime):  # noqa: F811
    case = make_case(1, [0, 8, 1, 0, 16], [0, 4097, 17, 511, 32771], padding=7)
    device = candidate_device_case(case, offset=True)
    candidates = row_candidates(case)
    candidate_device = candidates.npu()
    info = build_metadata(runtime(1, "consumer"), device, max_q=32, max_k=32771)
    for _ in range(3):
        selector_call(device, info, candidate_device)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result, _ = selector_call(device, info, candidate_device)
    shapes = ([0, 8, 1, 0, 16], [3, 0, 1, 0, 8], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1])
    for iteration in range(16):
        query_lengths = shapes[iteration % len(shapes)]
        case["cu"].copy_(torch.tensor([0, *torch.tensor(query_lengths).cumsum(0).tolist()]))
        case["sk"].copy_(torch.tensor([17, 511, 4097, 32771, 32]).roll(iteration))
        generator = torch.Generator().manual_seed(88700 + iteration)
        case["q"].random_(-127, 128, generator=generator)
        case["w"].copy_(torch.randn(case["w"].shape, generator=generator).half() / 32)
        case["bt"].copy_(case["bt"].flip(1))
        candidates = row_candidates(case)
        if iteration % 5 == 4:
            candidates.fill_(-1)
        for name in ("q", "w", "sk", "cu", "bt"):
            device[name].copy_(case[name])
        candidate_device.copy_(candidates)
        graph.replay()
        assert_query_rows(case, candidates, result.cpu())


def test_fused_workspace_capacity_fallback(runtime):  # noqa: F811
    # 2048*156704 exceeds the 256 MiB specialization bound. This validates
    # legacy fallback correctness; it is explicitly not optimized coverage.
    tokens, length = 2048, 4097
    case = make_case(1, [tokens], [length])
    device = candidate_device_case(case)
    candidates = row_candidates(case)
    info = build_metadata(runtime(1, "consumer"), device, max_q=tokens, max_k=length)
    actual, _ = selector_call(device, info, candidates.npu())
    assert_query_rows(case, candidates, actual.cpu())


@pytest.mark.parametrize("length", [0, 1, 17, 511, 4097, 32771, 131075])
@pytest.mark.parametrize("active", [False, True])
def test_cpu_vectorized_oracle_matches_frozen(length, active):
    torch.set_num_threads(8)
    case = make_case(1, [1], [length])
    if not active:
        case["cu"][1] = 0
    if case["bt"].shape[1] > 1:
        case["bt"][0, 1] = -1
        case["bt"][0, -1] = case["k"].shape[0] + 3
    candidates = candidates_for(length, adversarial=True)
    expected = candidate_reference(case, candidates)
    actual = candidate_reference_vectorized(case, candidates)
    for name in ("positions", "scores", "errors"):
        assert torch.equal(actual[name], expected[name]), name
