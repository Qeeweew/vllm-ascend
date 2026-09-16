# SPDX-License-Identifier: Apache-2.0
"""Mode4: direct paged Cube QK+WS and one-AIV complete query topk."""

import pytest
import torch
from test_indexer_v41 import build_metadata, make_case, runtime, select  # noqa: F401
from test_indexer_v41_candidate import candidate_device_case
from test_indexer_v41_fused import assert_query_rows, selector_call


def unique_candidates(case, holes=True):
    """Source-like unique IDs, arbitrary order, optional invalid slots/pages."""
    tokens = case["q"].shape[0]
    output = torch.full((tokens, 1, 2048), -1, dtype=torch.int32)
    for request, length in enumerate(case["sk"].tolist()):
        begin, end = case["cu"][request : request + 2].tolist()
        for row in range(begin, end):
            visible = max(0, length - (end - begin) + row - begin + 1)
            blocks = (visible + 7) // 8
            count = min(blocks, 2048)
            if not count:
                continue
            rng = torch.Generator().manual_seed(88123 + row)
            # Newest block is pinned without introducing a duplicate ID.
            output[row, 0, 0] = blocks - 1
            output[row, 0, 1:count] = torch.randperm(blocks - 1, generator=rng)[: count - 1].int()
            if holes:
                output[row] = output[row].roll(row * 17, -1)
                if row % 7 == 3:
                    output[row, 0, ::3] = -1
    return output


@pytest.mark.parametrize(
    "tokens,length",
    [(1, 17), (2, 511), (4, 4097), (32, 4097), (64, 32771), (128, 32771), (512, 131075), (1024, 32771)],
)
def test_paged_unique_prefill(runtime, tokens, length):  # noqa: F811
    case = make_case(1, [tokens], [length])
    device = candidate_device_case(case, offset=True)
    candidates = unique_candidates(case)
    info = build_metadata(runtime(1, "consumer"), device, max_q=tokens, max_k=length)
    actual, _ = selector_call(device, info, candidates.npu(), candidate_mode=4)
    assert_query_rows(case, candidates, actual.cpu())


@pytest.mark.parametrize("tokens", [1, 8, 32, 64])
def test_paged_unique_multibatch_decode(runtime, tokens):  # noqa: F811
    lengths = [4097, 32771, 17, 511, 131075, 0]
    case = make_case(1, [1] * tokens, [lengths[row % len(lengths)] for row in range(tokens)])
    device = candidate_device_case(case, offset=True)
    candidates = unique_candidates(case)
    info = build_metadata(runtime(1, "consumer"), device, max_q=1, max_k=max(lengths))
    actual, _ = selector_call(device, info, candidates.npu(), candidate_mode=4)
    assert_query_rows(case, candidates, actual.cpu())


def test_paged_unique_multibatch_prefill(runtime):  # noqa: F811
    case = make_case(1, [128, 0, 256, 1, 127], [4097, 0, 32771, 17, 131075], padding=17)
    device = candidate_device_case(case, offset=True)
    candidates = unique_candidates(case)
    # Entire internal candidate ranges and the tail become invalid,
    # while valid candidates remain on both sides of the internal holes.
    candidates[:, :, 64:128] = -1
    candidates[:, :, 1024:1152] = -1
    candidates[:, :, 1984:] = -1
    info = build_metadata(runtime(1, "consumer"), device, max_q=256, max_k=131075)
    actual, _ = selector_call(device, info, candidates.npu(), candidate_mode=4)
    assert_query_rows(case, candidates, actual.cpu())


@pytest.mark.parametrize("bucket", [128, 2048])
def test_paged_unique_ragged_graph_replay(runtime, bucket):  # noqa: F811
    # 128 rows / 20 cores forces >2 generations per physical slot.
    case = make_case(1, [0, 64, 1, 0, 32], [0, 32771, 17, 511, 4097], padding=bucket - 97)
    device = candidate_device_case(case, offset=True)
    candidates = unique_candidates(case)
    candidate_device = candidates.npu()
    info = build_metadata(runtime(1, "consumer"), device, max_q=bucket, max_k=32771)
    for _ in range(3):
        selector_call(device, info, candidate_device, candidate_mode=4)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result, _ = selector_call(device, info, candidate_device, candidate_mode=4)
    shapes = ([0, 64, 1, 0, 32], [3, 0, 1, 0, 8], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1])
    for iteration in range(16):
        query_lengths = shapes[iteration % len(shapes)]
        case["cu"].copy_(torch.tensor([0, *torch.tensor(query_lengths).cumsum(0).tolist()]))
        case["sk"].copy_(torch.tensor([17, 511, 4097, 32771, 32]).roll(iteration))
        rng = torch.Generator().manual_seed(90100 + iteration)
        case["q"].random_(-127, 128, generator=rng)
        case["k"].random_(-127, 128, generator=rng)
        case["w"].copy_(torch.randn(case["w"].shape, generator=rng).half() / 32)
        case["qs"].fill_(0.03125 + iteration / 1024)
        case["ks"].fill_(0.015625 + iteration / 1024)
        case["bt"].copy_(case["bt"].flip(1))
        case["bt"][:, 1] = -1
        case["bt"][:, -1] = case["k"].shape[0] + 3
        candidates = unique_candidates(case)
        if iteration % 5 == 4:
            candidates.fill_(-1)
        for name in ("q", "k", "w", "qs", "ks", "sk", "cu", "bt"):
            device[name].copy_(case[name])
        candidate_device.copy_(candidates)
        graph.replay()
        assert_query_rows(case, candidates, result.cpu())


def test_actual_source_unique_consumer(runtime):  # noqa: F811
    case = make_case(1, [33, 1], [4097, 32771], padding=9)
    device = candidate_device_case(case, offset=True)
    source = runtime(1, "source")
    consumer = runtime(1, "consumer", trusted_unique_candidates=True)
    info = build_metadata(source, device, max_q=64, max_k=32771)
    _, candidates = select(source, device, info)
    actual, _ = select(consumer, device, info, candidates)
    candidates_cpu = candidates.cpu()
    for row in candidates_cpu[:, 0]:
        valid = row[row >= 0]
        assert valid.unique().numel() == valid.numel()
    assert_query_rows(case, candidates_cpu, actual.cpu())


def test_paged_unique_underfull_and_negative_scores(runtime):  # noqa: F811
    case = make_case(1, [8], [32771], padding=35)
    case["w"].copy_(-case["w"].abs())
    candidates = torch.full((43, 1, 2048), -1, dtype=torch.int32)
    candidates[:, 0, :3] = torch.tensor([2048, 0, 4], dtype=torch.int32)
    device = candidate_device_case(case, offset=True)
    info = build_metadata(runtime(1, "consumer"), device, max_q=64, max_k=32771)
    actual, _ = selector_call(device, info, candidates.npu(), candidate_mode=4)
    assert_query_rows(case, candidates, actual.cpu())


def test_paged_unique_native_meta_contract():
    query = torch.empty((65, 32, 128), dtype=torch.int8, device="meta")
    key = torch.empty((33, 32, 1, 128), dtype=torch.int8, device="meta")
    weights = torch.empty((65, 32), dtype=torch.float16, device="meta")
    scale = torch.empty((33, 32, 1), dtype=torch.float16, device="meta")
    candidates = torch.empty((65, 1, 2048), dtype=torch.int32, device="meta")
    call = torch.ops._C_ascend.npu_quant_lightning_indexer_v3
    indices, scores, blocks = call(
        query,
        key,
        weights,
        weights,
        scale,
        512,
        2,
        candidate_topk_index=candidates,
        candidate_mode=4,
    )
    assert indices.shape == (65, 1, 512) and indices.dtype == torch.int32
    assert scores.numel() == 0 and blocks.numel() == 0
    with pytest.raises(RuntimeError, match="Consumer requires candidate blocks"):
        call(query, key, weights, weights, scale, 512, 2, candidate_mode=4)
