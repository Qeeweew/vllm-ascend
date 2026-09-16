# SPDX-License-Identifier: Apache-2.0
"""Correctness gates for the opt-in B1 candidate experiment."""

import ast
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from indexer_v41_candidate_reference import assert_candidate_selection, candidate_reference
from test_indexer_v41 import build_metadata, check_outputs, device_case, make_case, runtime, select  # noqa: F401


def metadata(case):
    return SimpleNamespace(cu_seqlens_q=case["cu"], seqused_k=case["sk"], block_table=case["bt"], cmp_residual_k=None)


def candidate_device_case(case, offset=False):
    device = device_case(case, gapped=not offset)
    if offset:
        for name in ("k", "ks"):
            original = device[name]
            storage = torch.full(
                (original.shape[0] * 2 + 1, *original.shape[1:]),
                -1,
                dtype=original.dtype,
                device=original.device,
            )
            device[name] = storage[1::2]
            device[name].copy_(original)
    return device


def candidates_for(length, adversarial=False):
    generator = torch.Generator().manual_seed(719)
    blocks = (length + 7) // 8
    candidates = torch.full((1, 1, 2048), -1, dtype=torch.int32)
    count = min(blocks, 2048)
    candidates[0, 0, :count] = torch.randperm(blocks, generator=generator)[:count].int()
    if count:
        candidates[0, 0, 0] = blocks - 1  # newest partial block, possibly duplicated
    if adversarial:
        candidates[0, 0, 3:9] = candidates[0, 0, 0]
        candidates[0, 0, 10:15] = -19
        candidates[0, 0, 16] = 2**31 - 1
        candidates[0, 0, 17] = -(2**31)
    return candidates


@pytest.mark.parametrize("length", [1, 17, 511, 4097, 32771, 131075])
def test_cpu_oracle_invariants(length):
    case = make_case(1, [1], [length])
    candidates = candidates_for(length, adversarial=True)
    reference = candidate_reference(case, candidates)
    count = min(512, reference["positions"].numel())
    selected = reference["positions"][reference["scores"].topk(count).indices].sort().values
    output = torch.full((512,), -1, dtype=torch.int32)
    output[:count] = selected.int()
    assert_candidate_selection(output, reference)
    assert torch.isfinite(reference["scores"]).all()
    assert reference["positions"].unique().numel() == reference["positions"].numel()


def test_cpu_oracle_rounding_membership_and_causality():
    case = make_case(1, [1], [17])
    case["q"].zero_()
    case["q"][0, 0].fill_(127)
    case["k"].fill_(127)
    case["w"].fill_(-0.1)
    case["qs"].fill_(0.333)
    case["ks"].fill_(1)
    candidates = torch.full((1, 1, 2048), -1, dtype=torch.int32)
    candidates[0, 0, :5] = torch.tensor([0, 0, 2, 2**31 - 1, -7], dtype=torch.int32)
    reference = candidate_reference(case, candidates)
    assert reference["positions"].tolist() == [*range(8), 16]
    # 128*127*127/1024 = 2016.125, rounded to FP16 2016 before weighting.
    head_weight = (case["w"][0, 0] * case["qs"][0, 0]).float()
    assert torch.equal(reference["scores"], torch.full((9,), 2016 * head_weight))
    assert not torch.equal(reference["scores"], torch.full((9,), 2016.125 * head_weight))
    case["bt"][0, 0] = -1
    assert candidate_reference(case, candidates)["positions"].numel() == 0


def test_cpu_final_selection_preserves_large_adjacent_ids():
    # Load this independent experimental module without importing the plugin's
    # NPU registration package; this check is CPU-only, including allocation.
    path = Path(__file__).parents[4] / "vllm_ascend/ops/indexer_v41_candidate.py"
    selector_type = runpy.run_path(str(path))["CandidateIndexerB1"]
    for start, valid in ((0, 17), (2**24 + 1, 512)):
        selector = selector_type(start + valid, "cpu")
        selector.positions.fill_(-1)
        selector.scores.fill_(-torch.inf)
        selector.positions[:valid] = torch.arange(start, start + valid).flip(0)
        # Even the most negative finite score is valid. Replacing -inf by a
        # finite comparison threshold would incorrectly discard this case.
        selector.scores[0, :valid] = torch.finfo(torch.float32).min if start == 0 else -1
        actual = selector.select_scores()[0]
        assert torch.equal(actual[:valid].long(), torch.arange(start, start + valid))
        assert torch.all(actual[valid:] == -1)


@pytest.fixture
def cpu_dispatch_type():
    """Load the actual dispatch class without platform/NPU package imports."""
    root = Path(__file__).parents[4]
    source = root / "vllm_ascend/models/deepseek_v4/indexer.py"
    node = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "AscendIndexerV41Ops"
    )
    candidate = runpy.run_path(str(root / "vllm_ascend/ops/indexer_v41_candidate.py"))["CandidateIndexerB1"]
    namespace = {"torch": torch, "CandidateIndexerB1": candidate, "AscendIndexerV41Metadata": SimpleNamespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["AscendIndexerV41Ops"], candidate


@pytest.mark.parametrize(
    "enabled,prepared,query_lengths,bound,expected",
    [
        (True, True, [1], 4097, "candidate"),
        (False, True, [1], 4097, "native"),
        (True, False, [1], 4097, "native"),
        (True, True, [2], 4097, "native"),
        (True, True, [1, 1], 4097, "native"),
        (True, True, [1], 8192, "native"),
        (True, True, [0], 4097, "empty"),
    ],
)
def test_cpu_optional_dispatch(cpu_dispatch_type, monkeypatch, enabled, prepared, query_lengths, bound, expected):
    ops_type, candidate_type = cpu_dispatch_type
    ops = ops_type(1, "consumer", candidate_max_context=bound if enabled else None)
    calls = []

    def candidate_call(self, *args):
        calls.append("candidate")
        return torch.arange(512).reshape(1, 1, 512).int(), torch.empty(0, dtype=torch.int32)

    def native_call(**kwargs):
        calls.append("native")
        rows = kwargs["query"].shape[0]
        return torch.arange(512).expand(rows, 1, 512).int(), None, torch.empty(0, dtype=torch.int32)

    monkeypatch.setattr(candidate_type, "__call__", candidate_call)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v3", native_call, raising=False)
    if prepared:
        ops.prepare_candidate_workspace("cpu")
    case = make_case(1, query_lengths, [4097] * len(query_lengths))
    info = metadata(case)
    info.qli_metadata = torch.empty(0)
    blocks = torch.zeros((sum(query_lengths), 1, 2048), dtype=torch.int32)
    result, _ = ops.select_topk(case["q"], case["w"], case["qs"], case["k"], case["ks"], info, blocks)
    assert calls == ([] if expected == "empty" else [expected])
    assert result.shape == (sum(query_lengths), 1, 512)


def test_cpu_optional_workspace_contract(cpu_dispatch_type, monkeypatch):
    ops_type, _ = cpu_dispatch_type
    for mode in ("source", "off"):
        with pytest.raises(ValueError, match="consumer"):
            ops_type(1, mode, candidate_max_context=4097)
    for bound in (0, -1, 2**27 + 1, 1.5):
        with pytest.raises(ValueError, match="static bound"):
            ops_type(1, "consumer", candidate_max_context=bound)
    ops = ops_type(1, "consumer", candidate_max_context=4097)
    ops.prepare_candidate_workspace("cpu")
    workspace = ops._candidate_selector
    ops.prepare_candidate_workspace("cpu")
    assert ops._candidate_selector is workspace
    with pytest.raises(ValueError, match="move devices"):
        ops.prepare_candidate_workspace("meta")
    # The model must set this bound from max_model_len: device runtime lengths
    # are never copied to host to validate them during graph replay.
    assert workspace.max_context == 4097
    unprepared = ops_type(1, "consumer", candidate_max_context=4097)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before NPU graph capture"):
        unprepared.prepare_candidate_workspace("npu:0")
    assert unprepared._candidate_selector is None


@pytest.mark.parametrize("query_lengths,prepare", [([1], False), ([1, 1], True), ([2], True)])
def test_optional_native_fallback_graph(runtime, query_lengths, prepare):  # noqa: F811
    length = 4097
    case = make_case(1, query_lengths, [length] * len(query_lengths))
    device = device_case(case)
    source = runtime(1, "source")
    info = build_metadata(source, device, max_q=max(query_lengths), max_k=length)
    _, blocks = select(source, device, info)
    ops = runtime(1, "consumer", candidate_max_context=length)
    if prepare:
        ops.prepare_candidate_workspace(device["q"].device)
        ops._candidate_selector.qk.fill_(17)

    def run():
        return ops.select_topk(device["q"], device["w"], device["qs"], device["k"], device["ks"], info, blocks)

    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        indices, result_blocks = run()
    graph.replay()
    check_outputs(case, 1, "consumer", indices.cpu(), result_blocks.cpu(), blocks.cpu())
    if prepare:
        assert torch.all(ops._candidate_selector.qk.cpu() == 17)
    else:
        assert ops._candidate_selector is None


@pytest.mark.parametrize("length", [1, 17, 511, 4097, 32771, 131075])
@pytest.mark.parametrize("adversarial", [False, True])
def test_candidate_eager(runtime, length, adversarial):  # noqa: F811
    from vllm_ascend.ops.indexer_v41_candidate import CandidateIndexerB1

    case = make_case(1, [1], [length])
    if adversarial:
        case["w"] = -case["w"].abs()
    device = candidate_device_case(case, offset=adversarial)
    candidates = candidates_for(length, adversarial)
    selector = CandidateIndexerB1(length, device["q"].device)
    indices, _ = selector(
        device["q"], device["k"], device["w"], device["qs"], device["ks"], metadata(device), candidates.npu()
    )
    reference = candidate_reference(case, candidates)
    assert_candidate_selection(indices.cpu(), reference)
    positions, scores = selector.positions.cpu(), selector.scores.cpu().flatten()
    assert torch.isneginf(scores[positions < 0]).all()
    valid = positions >= 0
    lookup = torch.searchsorted(reference["positions"], positions[valid].long())
    assert torch.all((scores[valid] - reference["scores"][lookup]).abs() <= reference["errors"][lookup])
    assert selector.count == max(512, min(16384, ((length + 7) // 8) * 8))


@pytest.mark.parametrize("integrated", [False, True])
def test_candidate_graph_dynamic_metadata(runtime, integrated):  # noqa: F811
    from vllm_ascend.ops.indexer_v41_candidate import CandidateIndexerB1

    length = 32771
    case = make_case(1, [1], [length])
    device = candidate_device_case(case, offset=True)
    candidates = candidates_for(length, True)
    candidate_device = candidates.npu()
    if integrated:
        ops = runtime(1, "consumer", candidate_max_context=length)
        ops.prepare_candidate_workspace(device["q"].device)
        selector = ops._candidate_selector

        def run(q, k, w, qs, ks, info, blocks):
            return ops.select_topk(q, w, qs, k, ks, info, blocks)
    else:
        selector = CandidateIndexerB1(length, device["q"].device)
        run = selector
    arguments = (device["q"], device["k"], device["w"], device["qs"], device["ks"], metadata(device), candidate_device)
    for _ in range(3):
        run(*arguments)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        indices, _ = run(*arguments)
    addresses = (selector.key.data_ptr(), selector.qk.data_ptr(), selector.scores.data_ptr())
    for step, visible in enumerate([32769, 17, 0, 4097, length, length]):
        case["sk"][0] = visible
        case["cu"][1] = 0 if step == 4 else 1
        case["bt"] = case["bt"].roll(7, 1)
        if step == 5:
            case["bt"][0, 0] = -1
            case["bt"][0, 2] = case["k"].shape[0]
        case["q"] = case["q"].roll(3, 1)
        case["w"] = -case["w"]
        candidates = candidates_for(max(visible, 1), True).roll(step, -1)
        for key in ("sk", "cu", "bt", "q", "w"):
            device[key].copy_(case[key])
        candidate_device.copy_(candidates)
        graph.replay()
        assert_candidate_selection(indices.cpu(), candidate_reference(case, candidates))
        assert addresses == (selector.key.data_ptr(), selector.qk.data_ptr(), selector.scores.data_ptr())


@pytest.mark.parametrize("trusted,query_lengths", [(False, [1]), (True, [1]), (True, [32, 1]), (True, [0])])
def test_cpu_unique_candidate_dispatch(cpu_dispatch_type, monkeypatch, trusted, query_lengths):
    ops_type, _ = cpu_dispatch_type
    ops = ops_type(1, "consumer", trusted_unique_candidates=trusted)
    modes = []

    def native_call(**kwargs):
        modes.append(kwargs["candidate_mode"])
        rows = kwargs["query"].shape[0]
        indices = torch.full((rows, 1, 512), -1, dtype=torch.int32)
        return indices, None, torch.empty(0, dtype=torch.int32)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v3", native_call, raising=False)
    case = make_case(1, query_lengths, [4097] * len(query_lengths))
    info = metadata(case)
    info.qli_metadata = torch.empty(0)
    blocks = torch.full((sum(query_lengths), 1, 2048), -1, dtype=torch.int32)
    result, _ = ops.select_topk(case["q"], case["w"], case["qs"], case["k"], case["ks"], info, blocks)
    assert modes == ([4 if trusted else 2] if sum(query_lengths) else [])
    assert torch.all(result == -1)
    assert ops._candidate_selector is None


def test_cpu_unique_candidate_contract(cpu_dispatch_type):
    ops_type, _ = cpu_dispatch_type
    for mode in ("off", "source"):
        with pytest.raises(ValueError, match="native CR1 consumer"):
            ops_type(1, mode, trusted_unique_candidates=True)
    with pytest.raises(ValueError, match="native CR1 consumer"):
        ops_type(1, "consumer", candidate_max_context=4097, trusted_unique_candidates=True)
    with pytest.raises(TypeError, match="must be a bool"):
        ops_type(1, "consumer", trusted_unique_candidates=1)
