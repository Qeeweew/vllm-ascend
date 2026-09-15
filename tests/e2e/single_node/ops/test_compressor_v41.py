# SPDX-License-Identifier: Apache-2.0
"""Run CPU contract checks with -k reference; NPU tests require the new OPP.

Standalone invocation avoids unrelated serving fixtures during operator bringup:
pytest --confcutdir=tests/e2e/single_node/ops tests/e2e/single_node/ops/test_compressor_v41.py
"""

import pytest
import torch

from vllm_ascend.ops.compressor_v41 import compressor_v41, compressor_v41_reference


def make_case(ratio, lengths, starts=None, capacity=8, padding=0, seed=41):
    rng = torch.Generator().manual_seed(seed)
    starts = [0] * len(lengths) if starts is None else starts
    tokens = sum(lengths) + padding
    raw = torch.randn((tokens, ratio * 512), generator=rng)
    raw = raw.to(torch.bfloat16) if ratio == 1 else raw
    weight = torch.randn(512, generator=rng).to(torch.bfloat16)
    state = torch.randn((len(lengths), capacity, 1024), generator=rng)
    positions = torch.full((tokens,), -1, dtype=torch.int64)
    slots = torch.full_like(positions, -1)
    req_ids = torch.zeros(tokens, dtype=torch.int32)
    boundaries = [0]
    for req, (length, start) in enumerate(zip(lengths, starts)):
        lo, hi = boundaries[-1], boundaries[-1] + length
        positions[lo:hi] = torch.arange(start, start + length)
        slots[lo:hi] = req * capacity + positions[lo:hi] % capacity
        req_ids[lo:hi] = req
        boundaries.append(hi)
    return (raw, positions, slots, torch.tensor(boundaries, dtype=torch.int32), req_ids, weight, state)


def run_reference(case, ratio):
    return compressor_v41_reference(*case, ratio)


def test_reference_cr1_has_bf16_projection_contract():
    case = make_case(1, [3], padding=2)
    output, state = run_reference(case, 1)
    x = case[0][:3].float()
    expected = (x * (x.square().mean(-1, keepdim=True) + 1e-20).rsqrt() * case[5].float()).bfloat16()
    torch.testing.assert_close(output[:3], expected, rtol=0, atol=0)
    assert not output[3:].count_nonzero()
    torch.testing.assert_close(state, case[-1], rtol=0, atol=0)
    with pytest.raises(ValueError, match="kv_score"):
        run_reference((case[0].float(), *case[1:]), 1)


def test_reference_rounds_before_rmsnorm_and_gates_per_channel():
    case = list(make_case(2, [2]))
    # Alternating dominant tokens ensure the softmax axis cannot be confused
    # with the feature axis. Non-power-of-two values expose the BF16 roundtrip.
    d = torch.arange(512)
    case[0][0, :512] = 0.999 + d.float() / 997
    case[0][1, :512] = -0.703 + d.float() / 613
    case[0][0, 512:] = torch.where(d % 2 == 0, 20.0, -20.0)
    case[0][1, 512:] = -case[0][0, 512:]
    output, _ = run_reference(case, 2)
    selected = torch.where(d % 2 == 0, case[0][0, :512], case[0][1, :512])
    rounded = selected.bfloat16().float()
    expected = (rounded * (rounded.square().mean() + 1e-20).rsqrt() * case[5].float()).bfloat16()
    torch.testing.assert_close(output[1], expected, rtol=0, atol=0)
    incorrect = (selected * (selected.square().mean() + 1e-20).rsqrt() * case[5].float()).bfloat16()
    assert (incorrect != expected).count_nonzero() > 32
    assert not output[0].count_nonzero()


def test_reference_chunk_equivalence_and_rollback():
    full = make_case(2, [27])
    expected, expected_state = run_reference(full, 2)
    state = full[-1].clone()
    outputs = []
    offset = 0
    for length in [1, 2, 8, 3, 1, 12]:
        part = list(make_case(2, [length], starts=[offset]))
        part[0] = full[0][offset : offset + length].clone()
        part[5], part[6] = full[5], state
        out, state = run_reference(part, 2)
        outputs.append(out)
        offset += length
    torch.testing.assert_close(torch.cat(outputs), expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    # Propose 5 rows at position 27 but accept only 2. The resumed position 29
    # must pair with retained row 28 rather than the rejected last draft.
    draft = list(make_case(2, [5], starts=[27], seed=71))
    draft[5], draft[6] = full[5], state
    _, drafted_state = run_reference(draft, 2)
    resumed = list(make_case(2, [1], starts=[29], seed=72))
    resumed[5], resumed[6] = full[5], drafted_state
    got, _ = run_reference(resumed, 2)
    direct = list(make_case(2, [2], starts=[28]))
    direct[0] = torch.cat((draft[0][1:2], resumed[0]))
    direct[5] = full[5]
    want, _ = run_reference(direct, 2)
    torch.testing.assert_close(got[0], want[1], rtol=0, atol=0)


def test_reference_metadata_errors_and_empty():
    case = list(make_case(2, [2, 3]))
    case[2][2:] = case[1][2:] % 8
    with pytest.raises(ValueError, match="distinct"):
        run_reference(case, 2)
    for ratio in (1, 2):
        out, state = run_reference(make_case(ratio, [0]), ratio)
        assert out.shape == (0, 512)
        assert state.shape == (1, 8, 1024)


def test_reference_cr1_accepts_no_state_or_request_storage():
    case = list(make_case(1, [4], padding=2))
    original, _ = run_reference(case, 1)
    case[3] = torch.empty(0, dtype=torch.int32)
    case[4] = torch.empty(0, dtype=torch.int32)
    case[6] = torch.empty(0, dtype=torch.float32)
    output, state = run_reference(case, 1)
    torch.testing.assert_close(output, original, rtol=0, atol=0)
    assert state.numel() == 0


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_reference_rejects_legacy_compression_ratios(ratio):
    with pytest.raises(ValueError, match="compress_ratio"):
        run_reference(make_case(1, [2]), ratio)


@pytest.fixture(scope="module")
def npu_device():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    assert hasattr(torch.ops._C_ascend, "compressor_v41"), "rebuild and install the new CompressorV41 OPP/binding"
    torch.npu.set_device(0)
    return torch.device("npu:0")


def check_npu(case, ratio, device):
    expected, expected_state = run_reference(case, ratio)
    tensors = [t.to(device) for t in case]
    output = torch.full(expected.shape, float("nan"), dtype=torch.bfloat16, device=device)
    got = compressor_v41(*tensors, output, ratio)
    assert got.data_ptr() == output.data_ptr()
    torch.testing.assert_close(got.cpu(), expected, rtol=8e-3, atol=2e-3)
    # FP32 state is a bit-preserving copy, with no arithmetic tolerance.
    torch.testing.assert_close(tensors[-1].cpu(), expected_state, rtol=0, atol=0)
    invalid = (case[2] < 0) | ((case[1] + 1) % ratio != 0)
    assert not got.cpu()[invalid].count_nonzero()
    return tensors, output


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("tokens", [0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 512, 2048, 4096])
def test_npu_real_shapes(npu_device, ratio, tokens):
    check_npu(make_case(ratio, [tokens], padding=3), ratio, npu_device)


def test_npu_empty_and_cr1_without_state(npu_device):
    for ratio in (1, 2):
        check_npu(make_case(ratio, [0]), ratio, npu_device)
    case = list(make_case(1, [17], padding=3))
    case[3] = torch.empty(0, dtype=torch.int32)
    case[4] = torch.empty(0, dtype=torch.int32)
    case[6] = torch.empty(0, dtype=torch.float32)
    check_npu(case, 1, npu_device)


@pytest.mark.parametrize("capacity", [8, 16])
@pytest.mark.parametrize("starts", [[0, 1, 8, 31], [1, 2, 15, 64]])
def test_npu_packed_boundaries_ring_wrap(npu_device, capacity, starts):
    check_npu(make_case(2, [1, 3, 41, 0], starts, capacity, padding=11), 2, npu_device)


def test_npu_extreme_scores_and_tiny_norm(npu_device):
    for scale in [0.0, 1e-15, 1.0]:
        case = list(make_case(2, [31], starts=[9]))
        case[0][:, :512] *= scale
        case[-1][:, :, :512] *= scale
        case[0][:, 512:] *= 10000.0
        case[-1][:, :, 512:] *= 10000.0
        check_npu(case, 2, npu_device)


def test_npu_chunk_sequence_and_graph_replay(npu_device):
    case = make_case(2, [3, 2], padding=3)
    tensors = [t.to(npu_device) for t in case]
    output = torch.empty((8, 512), dtype=torch.bfloat16, device=npu_device)
    # Warmup/capture state is deliberately disposable and restored below.
    for _ in range(3):
        compressor_v41(*tensors, output, 2)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        compressor_v41(*tensors, output, 2)
    cpu_state = case[-1].clone()
    positions = [0, 0]
    for step in range(50):
        # Keep bucket/addresses fixed, change active counts, parity, raw data,
        # and even recycle request state on the same block.
        lengths = [3, 2] if step % 2 == 0 else [1, 1]
        if step == 25:
            positions = [0, 0]
            cpu_state.zero_()
        next_case = list(make_case(2, lengths, positions, padding=8 - sum(lengths), seed=step + 50))
        next_case[5], next_case[6] = case[5], cpu_state
        expected, cpu_state = run_reference(next_case, 2)
        for dest, source in zip(tensors, next_case):
            dest.copy_(source)
        graph.replay()
        torch.testing.assert_close(output.cpu(), expected, rtol=8e-3, atol=2e-3)
        torch.testing.assert_close(tensors[-1].cpu(), cpu_state, rtol=0, atol=0)
        positions = [p + length for p, length in zip(positions, lengths)]


def test_npu_rollback(npu_device):
    initial = make_case(2, [20])
    _, state = run_reference(initial, 2)
    draft = list(make_case(2, [5], starts=[20], seed=141))
    draft[6] = state
    tensors, _ = check_npu(draft, 2, npu_device)
    resumed = list(make_case(2, [3], starts=[23], seed=142))
    resumed[6] = tensors[-1].cpu()
    check_npu(resumed, 2, npu_device)


def assert_normalization_accuracy(actual, expected):
    actual, expected = actual.double(), expected.double()
    error = actual - expected
    energy = expected.square().sum()
    nrmse = (error.square().sum() / energy).sqrt().item()
    gain_bias = ((actual * expected).sum() / energy - 1).item()
    assert nrmse < 2e-4, f"normalization NRMSE={nrmse:.8g}"
    assert abs(gain_bias) < 5e-5, f"normalization gain bias={gain_bias:.8g}"
    # One BF16 ULP at most per finite lane, independently of aggregate error.
    next_value = torch.nextafter(expected.bfloat16(), torch.full_like(expected.bfloat16(), float("inf")))
    ulp = (next_value.double() - expected).abs()
    assert torch.all(error.abs() <= ulp)


def normalization_case(ratio, scale):
    generator = torch.Generator().manual_seed(9041)
    # Equal CR2 KV pairs and zero scores make pooling exact, isolating RMSNorm
    # from exponential/softmax approximation and the required BF16 roundtrip.
    values = ((0.25 + 2 * torch.rand(128, 512, generator=generator)) * scale).bfloat16()
    weight = torch.linspace(0.25, 2.25, 512).bfloat16()
    case = list(make_case(ratio, [128 * ratio]))
    case[5] = weight
    if ratio == 1:
        case[0] = values
    else:
        case[0][:, :512] = values.float().repeat_interleave(2, dim=0)
        case[0][:, 512:] = 0
    precise = values.double()
    expected = (precise / (precise.square().mean(-1, keepdim=True) + 1e-20).sqrt() * weight.double()).bfloat16()
    return case, expected


@pytest.mark.parametrize("scale", [1e-15, 1e-5, 1.0, 1e5])
def test_reference_normalization_gate_detects_systematic_rsqrt_bias(scale):
    for ratio in (1, 2):
        case, expected = normalization_case(ratio, scale)
        reference, _ = run_reference(case, ratio)
        assert_normalization_accuracy(reference[ratio - 1 :: ratio], expected)
        values = case[0][::ratio, :512].double()
        unrounded = values / (values.square().mean(-1, keepdim=True) + 1e-20).sqrt() * case[5].double()
        biased = (unrounded * 0.998046875).bfloat16()
        with pytest.raises(AssertionError, match="NRMSE|gain bias"):
            assert_normalization_accuracy(biased, expected)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("scale", [1e-15, 1e-5, 1.0, 1e5])
def test_npu_normalization_has_no_systematic_gain_bias(npu_device, ratio, scale):
    case, expected = normalization_case(ratio, scale)
    tensors = [tensor.to(npu_device) for tensor in case]
    output = torch.empty_like(tensors[0][:, :512], dtype=torch.bfloat16)
    compressor_v41(*tensors, output, ratio)
    assert_normalization_accuracy(output.cpu()[ratio - 1 :: ratio], expected)
