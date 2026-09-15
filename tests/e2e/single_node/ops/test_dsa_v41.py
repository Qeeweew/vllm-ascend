# SPDX-License-Identifier: Apache-2.0
"""Independent FP32 oracle for V4.1 SWA + selected shared KV on device 2."""

from dataclasses import fields

import pytest
import torch


def make_case(ratio, query_lengths, context_lengths, *, seed=41, sparse=False, sink=0.0):
    generator = torch.Generator().manual_seed(seed)
    batch, tokens = len(query_lengths), sum(query_lengths)
    page_size = 32
    pages = (max(context_lengths) + page_size - 1) // page_size
    cmp_pages = max(1, (max(context_lengths) // max(ratio, 1) + page_size - 1) // page_size)
    # Permuted block tables catch accidental use of logical IDs as physical IDs.
    swa_bt = torch.randperm(batch * pages, generator=generator).int().reshape(batch, pages)
    cmp_bt = torch.randperm(batch * cmp_pages, generator=generator).int().reshape(batch, cmp_pages)
    case = dict(
        q=torch.randn(tokens, 8, 512, generator=generator).bfloat16(),
        swa=torch.randn(batch * pages, page_size, 1, 512, generator=generator).bfloat16(),
        sinks=torch.linspace(sink - 2, sink + 2, 8),
        cu=torch.tensor([0, *torch.tensor(query_lengths).cumsum(0).tolist()], dtype=torch.int32),
        lengths=torch.tensor(context_lengths, dtype=torch.int32),
        swa_bt=swa_bt,
        cmp_bt=cmp_bt if ratio else None,
        cmp=torch.randn(batch * cmp_pages, page_size, 1, 512, generator=generator).bfloat16() if ratio else None,
        indices=torch.full((tokens, 1, 512), -1, dtype=torch.int32) if ratio else None,
    )
    if ratio:
        row = 0
        for query_len, context in zip(query_lengths, context_lengths):
            for offset in range(query_len):
                visible = (context - query_len + offset + 1) // ratio
                ids = torch.arange(visible, dtype=torch.int32)
                if sparse:
                    ids = (
                        ids[torch.randperm(visible, generator=generator)[: min(512, (visible + 1) // 2)]].sort().values
                    )
                ids = ids[-512:]
                case["indices"][row, 0, : ids.numel()] = ids
                row += 1
    return case


def reference(case, ratio):
    """Direct concatenation, one FP32 softmax including a zero-value sink."""
    output = torch.empty_like(case["q"], dtype=torch.float32)
    lse = torch.empty((1, case["q"].shape[0], 8), dtype=torch.float32)
    for request, context in enumerate(case["lengths"].tolist()):
        lo, hi = case["cu"][request : request + 2].tolist()
        swa = case["swa"][case["swa_bt"][request].long()].reshape(-1, 512).float()
        cmp = case["cmp"][case["cmp_bt"][request].long()].reshape(-1, 512).float() if ratio else None
        for row in range(lo, hi):
            end = context - (hi - row) + 1
            kv = swa[max(0, end - 128) : end]
            if ratio:
                ids = case["indices"][row, 0].long()
                ids = ids[(ids >= 0) & (ids < end // ratio)]
                kv = torch.cat((kv, cmp[ids]), dim=0)
            logits = case["q"][row].float() @ kv.T / (512**0.5)
            logits = torch.cat((logits, case["sinks"][:, None]), dim=-1)
            output[row] = logits.softmax(-1)[:, :-1] @ kv
            lse[0, row] = logits.logsumexp(-1)
    return output, lse


@pytest.fixture(scope="module")
def runtime():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    from vllm_ascend.ops.dsa_v41 import AscendDSAV41Ops

    torch.set_num_threads(8)
    torch.npu.set_device(2)
    return AscendDSAV41Ops


def device_case(case, *, gapped=False):
    result = {key: value.npu() if value is not None else None for key, value in case.items()}
    if gapped:
        for key in ("swa", "cmp"):
            original = result[key]
            if original is not None:
                storage = torch.full(
                    (2 * original.shape[0], *original.shape[1:]), 9, dtype=original.dtype, device="npu"
                )
                result[key] = storage[::2]
                result[key].copy_(original)
    return result


def metadata(ops, case):
    return ops.build_metadata(
        case["cu"],
        case["lengths"],
        case["swa_bt"],
        cmp_block_table=case["cmp_bt"],
        max_seqlen_q=case["q"].shape[0],
        max_seqlen_kv=case["swa_bt"].shape[1] * case["swa"].shape[1],
    )


def forward(ops, case, meta, *, lse=True):
    return ops.forward(
        case["q"],
        case["swa"],
        case["sinks"],
        meta,
        cmp_cache=case["cmp"],
        cmp_indices=case["indices"],
        return_softmax_lse=lse,
    )


def check(actual, expected):
    actual = actual.cpu().float()
    error = actual - expected
    assert torch.isfinite(actual).all()
    # BF16 attention rounds probabilities and output; measure global error
    # as well as per-element absolute accuracy, including near-zero outputs.
    assert error.square().mean().sqrt() <= expected.square().mean().sqrt() * 0.006 + 1e-6
    torch.testing.assert_close(actual, expected, atol=0.012, rtol=0.025)


@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize(
    "query_lengths,context_lengths",
    [([1], [1]), ([1, 1], [129, 130]), ([33, 7], [163, 23]), ([1, 1, 1, 1], [2049, 1025, 777, 63])],
)
def test_native_attention(runtime, ratio, query_lengths, context_lengths):
    case = make_case(ratio, query_lengths, context_lengths, sparse=True)
    expected, expected_lse = reference(case, ratio)
    device = device_case(case, gapped=True)
    ops = runtime(ratio)
    actual, lse = forward(ops, device, metadata(ops, device))
    torch.npu.synchronize()
    check(actual, expected)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.015, rtol=0.003)


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_sink_dominates_joint_normalization(runtime, ratio):
    case = make_case(ratio, [3], [134], sink=20)
    expected, _ = reference(case, ratio)
    device, ops = device_case(case), runtime(ratio)
    actual, _ = forward(ops, device, metadata(ops, device))
    check(actual, expected)
    assert actual.float().abs().max().cpu() < 1e-4


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_graph_replay_changes_lengths_and_cache(runtime, ratio):
    case = make_case(ratio, [1, 1], [255, 129])
    device, ops = device_case(case), runtime(ratio)
    meta = metadata(ops, device)
    for _ in range(3):
        forward(ops, device, meta)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual, _ = forward(ops, device, meta)
    graph.replay()
    check(actual, reference(case, ratio)[0])
    updated = make_case(ratio, [1, 1], [256, 130], seed=72, sparse=True)
    changed = device_case(updated)
    for key in device:
        if device[key] is not None:
            device[key].copy_(changed[key])
    fresh = metadata(ops, device)
    for field in fields(meta):
        old, new = getattr(meta, field.name), getattr(fresh, field.name)
        if old is not None and old.data_ptr() != new.data_ptr():
            old.copy_(new)
    graph.replay()
    check(actual, reference(updated, ratio)[0])


@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("query_length", [1, 33])
def test_swa_ring_logical_page_alias(runtime, ratio, query_length):
    case = make_case(ratio, [query_length], [2049], sparse=True)
    expected, _ = reference(case, ratio)
    # Ring retains every row needed by any query in this chunk. Logical
    # pages may alias; absolute position modulo capacity selects the row.
    capacity = ((128 + query_length - 1 + 31) // 32) * 32
    ring = torch.zeros((capacity // 32, 32, 1, 512), dtype=torch.bfloat16)
    full = case["swa"][case["swa_bt"][0].long()].reshape(-1, 512)
    positions = torch.arange(2049 - capacity, 2049)
    ring.reshape(-1, 512)[positions % capacity] = full[positions]
    case["swa"] = ring
    case["swa_bt"] = torch.arange(case["swa_bt"].shape[1], dtype=torch.int32)[None] % ring.shape[0]
    device, ops = device_case(case), runtime(ratio)
    actual, _ = forward(ops, device, metadata(ops, device))
    check(actual, expected)
