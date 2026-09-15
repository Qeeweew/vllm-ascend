# SPDX-License-Identifier: Apache-2.0
"""Independent V4.1 CSA selection oracle; device 0 operator bringup tests."""

import pytest
import torch
import torch.nn.functional as F


def make_case(ratio, query_lengths, context_lengths, padding=0, seed=41):
    rng = torch.Generator().manual_seed(seed)
    tokens = sum(query_lengths) + padding
    compressed = [length // ratio for length in context_lengths]
    block_count = max(1, (max(compressed) + 31) // 32)
    batch = len(query_lengths)
    q = torch.randint(-100, 101, (tokens, 32, 128), dtype=torch.int8, generator=rng)
    k = torch.randint(-100, 101, (batch * block_count, 32, 1, 128), dtype=torch.int8, generator=rng)
    w = (torch.randn(tokens, 32, generator=rng) / 64).half()
    qs = (torch.rand(tokens, 32, generator=rng) / 32).half()
    ks = (torch.rand(k.shape[:-1], generator=rng) / 32).half()
    cu = torch.tensor([0, *torch.tensor(query_lengths).cumsum(0).tolist()], dtype=torch.int32)
    sk = torch.tensor(compressed, dtype=torch.int32)
    residual = torch.tensor([length % ratio for length in context_lengths], dtype=torch.int32) if ratio == 2 else None
    bt = torch.arange(batch * block_count, dtype=torch.int32).reshape(batch, block_count)
    return dict(q=q, k=k, w=w, qs=qs, ks=ks, cu=cu, sk=sk, residual=residual, bt=bt)


def reference_scores(case, ratio):
    """Oracle follows hardware's documented intermediate rounding, no native ops."""
    rows = []
    for request in range(case["sk"].numel()):
        lo, hi = case["cu"][request : request + 2].tolist()
        context = int(case["sk"][request]) * ratio
        if case["residual"] is not None:
            context += int(case["residual"][request])
        keys = case["k"][case["bt"][request].long()].reshape(-1, 128).float()
        scales = case["ks"][case["bt"][request].long()].flatten().float()
        for token in range(lo, hi):
            visible = (context - (hi - lo) + token - lo + 1) // ratio
            # INT8 dot fits exactly in FP32 for D128 and is independent of the
            # native INT8 Cube/Fixpipe/FP16 Cube implementation.
            qk = (case["q"][token].float() @ keys[:visible].T / 1024).relu().half().float()
            head_weights = (case["w"][token] * case["qs"][token]).float()
            rows.append((qk * head_weights[:, None]).sum(0) * scales[:visible])
    return rows


def reference_blocks(scores):
    blocks = F.pad(scores, (0, -scores.numel() % 8), value=-torch.inf).reshape(-1, 8).amax(-1)
    if blocks.numel():
        blocks[-1] = torch.inf
    return blocks


def assert_selection(actual, scores, count):
    """Allow only tied cutoff swaps, never a global recall tolerance."""
    selected = actual[actual >= 0].long()
    assert selected.numel() == min(count, scores.numel())
    assert selected.unique().numel() == selected.numel()
    if selected.numel():
        assert selected.max() < scores.numel()
        cutoff = scores.topk(selected.numel()).values[-1]
        tolerance = scores[torch.isfinite(scores)].abs().max() * 2e-6 if torch.isfinite(scores).any() else 0
        assert torch.all(scores[selected] >= cutoff - tolerance)
        required = torch.nonzero(scores > cutoff + tolerance).flatten()
        assert set(required.tolist()) <= set(selected.tolist())


def check_outputs(case, ratio, mode, indices, blocks, candidates=None):
    scores = reference_scores(case, ratio)
    for row, score in enumerate(scores):
        if mode == "consumer":
            allowed = torch.zeros(score.numel(), dtype=torch.bool)
            if score.numel():
                block_ids = torch.arange(score.numel()) // 8
                allowed = torch.isin(block_ids, candidates[row, 0][candidates[row, 0] >= 0])
            score = score.masked_fill(~allowed, -torch.inf)
        assert_selection(indices[row, 0], score, 512)
        valid = indices[row, 0][indices[row, 0] >= 0]
        assert torch.equal(valid, valid.sort().values)
        assert torch.all(indices[row, 0, valid.numel() :] == -1)
        if mode == "source":
            assert_selection(blocks[row, 0], reference_blocks(score), 2048)
            if score.numel():
                assert (score.numel() - 1) // 8 in blocks[row, 0]
    assert torch.all(indices[len(scores) :] == -1)
    if mode == "source":
        assert torch.all(blocks[len(scores) :] == -1)


@pytest.fixture(scope="module")
def runtime():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerV41Ops

    torch.set_num_threads(8)
    torch.npu.set_device(0)
    return AscendIndexerV41Ops


def device_case(case, gapped=False):
    result = {key: tensor.npu() if tensor is not None else None for key, tensor in case.items()}
    if gapped:
        for key in ("k", "ks"):
            original = result[key]
            storage = torch.full((original.shape[0] * 2, *original.shape[1:]), -1, device="npu", dtype=original.dtype)
            result[key] = storage[::2]
            result[key].copy_(original)
    return result


def build_metadata(ops, case, max_q=None, max_k=None):
    return ops.build_metadata(
        case["cu"],
        case["sk"],
        case["bt"],
        max_seqlen_q=max_q or case["q"].shape[0],
        max_seqlen_k=max_k or case["bt"].shape[1] * 32 * ops.compress_ratio,
        cmp_residual_k=case["residual"],
    )


def select(ops, case, metadata, candidates=None):
    return ops.select_topk(case["q"], case["w"], case["qs"], case["k"], case["ks"], metadata, candidates)


@pytest.mark.parametrize(
    "ratio,qlens,contexts",
    [
        (1, [3], [32771]),
        (1, [1, 3, 4], [1, 511, 4097]),
        (1, [9], [9]),
        (2, [4, 3, 1], [4, 1025, 65539]),
        (2, [1], [1]),
    ],
)
def test_native_real_contract(runtime, ratio, qlens, contexts):
    case = make_case(ratio, qlens, contexts, padding=3)
    device = device_case(case, gapped=True)
    for mode in ("off", "source") if ratio == 1 else ("off",):
        ops = runtime(ratio, mode)
        metadata = build_metadata(ops, device)
        indices, blocks = select(ops, device, metadata)
        check_outputs(case, ratio, mode, indices.cpu(), blocks.cpu())
        if mode == "source":
            consumer = runtime(ratio, "consumer")
            changed = dict(case, q=case["q"].roll(3, 1), w=-case["w"])
            changed_device = dict(device, q=changed["q"].npu(), w=changed["w"].npu())
            out, _ = select(consumer, changed_device, metadata, blocks)
            check_outputs(changed, ratio, "consumer", out.cpu(), None, blocks.cpu())


def test_native_graph_replay_changes_lengths_and_candidates(runtime):
    case = make_case(1, [2, 2], [32771, 4097], padding=2)
    device = device_case(case)
    source, consumer = runtime(1, "source"), runtime(1, "consumer")
    metadata = build_metadata(source, device)

    def run():
        indices, candidates = select(source, device, metadata)
        out, _ = select(consumer, device, metadata, candidates)
        return indices, candidates, out

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        indices, candidates, out = run()
    # Keep run closure and all device buffers alive through the final replay.
    for step in range(10):
        case["sk"][:] = torch.tensor([32771 - step * 17, 4097 - step * 9])
        case["q"] = case["q"].roll(1, 1)
        device["sk"].copy_(case["sk"])
        device["q"].copy_(case["q"])
        refreshed = build_metadata(source, device)
        metadata.qli_metadata.copy_(refreshed.qli_metadata)
        graph.replay()
        torch.npu.synchronize()
        check_outputs(case, 1, "source", indices.cpu(), candidates.cpu())
        check_outputs(case, 1, "consumer", out.cpu(), None, candidates.cpu())


def test_quantize_is_explicit_int8_rtn(runtime):
    rng = torch.Generator().manual_seed(41)
    value = torch.randn((8, 32, 128), generator=rng).bfloat16()
    value[0] = 0
    quantized, scales = runtime.quantize(value.npu())
    maxima = value.float().abs().amax(-1)
    scale = maxima / 127
    scaled = torch.where(scale[..., None] == 0, 0, value.float() / scale[..., None])
    expected = scaled.round().clamp(-128, 127).to(torch.int8)
    actual = quantized.cpu()
    # Reciprocal-multiply and divide can land on opposite sides of an exact
    # half-integer. Permit only that one-ULP boundary, not arbitrary code errors.
    mismatch = actual != expected
    assert torch.all((actual.float() - expected.float()).abs() <= 1)
    assert torch.all((scaled[mismatch].abs().frac() - 0.5).abs() <= 1e-5)
    torch.testing.assert_close(scales.cpu(), scale.half(), atol=0, rtol=0)


@pytest.mark.parametrize("ratio,mode", [(0, "off"), (4, "off"), (128, "off"), (1, "typo"), (2, "source")])
def test_rejects_legacy_or_invalid_modes(runtime, ratio, mode):
    with pytest.raises(ValueError):
        runtime(ratio, mode)


def test_empty_and_metadata_contract(runtime):
    from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerV41Metadata

    case = make_case(1, [0], [0])
    meta = AscendIndexerV41Metadata(case["cu"], case["sk"], case["bt"], torch.empty(1024, dtype=torch.int32))
    for mode in ("off", "source"):
        result, blocks = select(runtime(1, mode), case, meta)
        assert result.shape == (0, 1, 512)
        assert blocks.shape == ((0, 1, 2048) if mode == "source" else (0,))
    with pytest.raises(ValueError, match="source candidate"):
        select(runtime(1, "consumer"), case, meta)
    with pytest.raises(ValueError, match="requires it"):
        runtime(2).build_metadata(case["cu"], case["sk"], case["bt"], max_seqlen_q=1, max_seqlen_k=1)
    with pytest.raises(ValueError, match="forbids"):
        runtime(1).build_metadata(
            case["cu"],
            case["sk"],
            case["bt"],
            max_seqlen_q=1,
            max_seqlen_k=1,
            cmp_residual_k=torch.zeros(1, dtype=torch.int32),
        )


@pytest.mark.parametrize("width", [2**24 // 32, 2**24 // 32 + 1])
def test_sort_preserves_large_integer_ids(runtime, monkeypatch, width):
    from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerV41Metadata

    case = make_case(1, [1], [32])
    # Pure wrapper test: static table width decides the exact FP32 safe bound.
    # No enormous NPU cache is needed to verify adjacent integers above 2**24.
    meta = AscendIndexerV41Metadata(
        case["cu"], case["sk"], torch.empty((1, width), dtype=torch.int32), torch.empty(1024, dtype=torch.int32)
    )
    ids = torch.full((1, 1, 512), -1, dtype=torch.int32)
    limit = width * 32
    ids[0, 0, :5] = torch.tensor([limit - 1, limit - 3, 0, 7, limit - 2])
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_quant_lightning_indexer_v3",
        lambda **kwargs: (ids, torch.empty(0), torch.empty(0, dtype=torch.int32)),
    )
    result, _ = select(runtime(1), case, meta)
    torch.testing.assert_close(result[0, 0, :5], ids[0, 0, :5].sort().values, atol=0, rtol=0)
    assert torch.all(result[0, 0, 5:] == -1)
