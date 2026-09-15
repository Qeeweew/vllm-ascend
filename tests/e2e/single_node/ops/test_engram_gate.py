# SPDX-License-Identifier: Apache-2.0
"""Engram numerical contract, real-width device accuracy, and changing-input graphs."""

import pytest
import torch

from vllm_ascend.ops.engram_gate import engram_gate, engram_gate_reference


def make_case(tokens, seed=41):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(tokens, 4, 5120, generator=generator).bfloat16()
    kv = torch.randn(tokens, 25600, generator=generator).bfloat16()
    q = torch.randn(4, 5120, generator=generator).bfloat16()
    k = torch.randn(4, 5120, generator=generator).bfloat16()
    mask = torch.arange(tokens) % 3 != 2
    return hidden, kv, q, k, mask


def check_close(actual, expected, mask, hidden):
    torch.testing.assert_close(actual[~mask], hidden[~mask], rtol=0, atol=0)
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-20)
    assert error / scale < 2e-4
    # Do not let small aggregate error hide local catastrophes near cancellation.
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.008, atol=0.002)


def test_reference_copy_axis_mask_and_signed_gate():
    h, kv, q, k, mask = make_case(3)
    h.fill_(1)
    q.fill_(1)
    k.fill_(1)
    kv[:, : 4 * 5120] = h.reshape(3, -1)
    kv[:, 5120 : 2 * 5120].neg_()
    kv[:, 2 * 5120 : 3 * 5120].zero_()
    kv[:, 4 * 5120 :].fill_(2)
    result = engram_gate_reference(h, kv, q, k, mask)
    assert torch.equal(result[:2, 0], torch.full_like(result[:2, 0], 3))
    assert torch.equal(result[:2, 1], torch.full_like(result[:2, 1], 1))
    assert torch.equal(result[:2, 2], torch.full_like(result[:2, 2], 2))
    assert torch.equal(result[2], h[2])


def test_reference_invalid_metadata_and_empty():
    case = list(make_case(0))
    assert engram_gate_reference(*case).shape == (0, 4, 5120)
    case[2] = case[2].float()
    with pytest.raises(ValueError, match="q_weight"):
        engram_gate_reference(*case)
    with pytest.raises(ValueError, match="eps"):
        engram_gate_reference(*make_case(1), eps=float("nan"))


def test_reference_masked_nan_padding_is_passthrough():
    case = list(make_case(3))
    case[1][2].fill_(float("nan"))
    output = engram_gate_reference(*case)
    assert torch.equal(output[2], case[0][2])
    assert output.isfinite().all()


@pytest.fixture(scope="module")
def npu_device():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    assert hasattr(torch.ops._C_ascend, "engram_gate"), "full rebuild/install of EngramGate is required"
    torch.npu.set_device(1)
    return torch.device("npu:1")


@pytest.mark.parametrize("tokens", [1, 2, 8, 16, 64, 257, 1024])
def test_npu_real_width(tokens, npu_device):
    case = make_case(tokens)
    expected = engram_gate_reference(*case)
    actual = engram_gate(*(tensor.to(npu_device) for tensor in case)).cpu()
    check_close(actual, expected, case[-1], case[0])


@pytest.mark.parametrize("mode", ["zero", "tiny", "saturated", "masked_nan"])
def test_npu_edge_values(mode, npu_device):
    case = list(make_case(8))
    if mode == "zero":
        case[0].zero_()
        case[1][:, : 4 * 5120].zero_()
    elif mode == "tiny":
        case[0].mul_(1e-12)
        case[1][:, : 4 * 5120].mul_(1e-12)
    elif mode == "saturated":
        case[2].mul_(128)
    else:
        case[1][~case[-1]] = float("nan")
    expected = engram_gate_reference(*case)
    actual = engram_gate(*(tensor.to(npu_device) for tensor in case)).cpu()
    check_close(actual, expected, case[-1], case[0])


def test_npu_graph_changes_inputs_and_mask_without_host_fence(npu_device):
    device_case = tuple(t.to(npu_device) for t in make_case(16))
    output = torch.empty_like(device_case[0])
    for _ in range(3):
        engram_gate(*device_case, output=output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        engram_gate(*device_case, output=output)
    pointer = output.data_ptr()
    results = []
    references = []
    uploaded = []
    for iteration in range(20):
        case = make_case(16, seed=iteration + 91)
        case[-1].copy_(torch.arange(16) % (iteration % 5 + 2) != 0)
        uploaded.append(tuple(t.to(npu_device) for t in case))
        references.append((engram_gate_reference(*case), case[-1], case[0]))
    torch.npu.synchronize()
    for case in uploaded:
        for dst, src in zip(device_case, case):
            dst.copy_(src, non_blocking=True)
        graph.replay()
        results.append(output.clone())
        assert output.data_ptr() == pointer
    # All graph submissions precede the single final host fence.
    torch.npu.synchronize()
    for actual, (expected, mask, hidden) in zip(results, references):
        check_close(actual.cpu(), expected, mask, hidden)


def test_npu_all_masked_and_inplace(npu_device):
    case = list(make_case(9))
    expected = engram_gate_reference(*case)
    device_case = [tensor.to(npu_device) for tensor in case]
    result = engram_gate(*device_case, output=device_case[0])
    assert result.data_ptr() == device_case[0].data_ptr()
    check_close(result.cpu(), expected, case[-1], case[0])
    device_case[-1].zero_()
    device_case[1].fill_(float("nan"))
    passthrough = engram_gate(*device_case).cpu()
    torch.testing.assert_close(passthrough, result.cpu(), rtol=0, atol=0)


def test_npu_unit_norm_nonsaturating_gate(npu_device):
    # Exact unit norms isolate normalization error from reduction rounding.
    # Saturated gates would hide a low-precision reciprocal-square-root.
    case = list(make_case(8))
    case[0] = case[0].sign()
    case[1][:, :20480] = case[0].flatten(1)
    case[2].fill_(1)
    case[3].fill_(1)
    for hc, target_dot in enumerate([1.0, -1.0, 2.0, -2.0]):
        case[2][hc].fill_(target_dot / 5120**0.5)
    case[-1].fill_(True)
    dot = case[2][:, 0].float() * 5120**0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().sqrt(), dot))
    expected = (case[0].float() + gate[None, :, None] * case[1][:, None, 20480:].float()).bfloat16()
    actual = engram_gate(*(tensor.to(npu_device) for tensor in case)).cpu()
    check_close(actual, expected, case[-1], case[0])
