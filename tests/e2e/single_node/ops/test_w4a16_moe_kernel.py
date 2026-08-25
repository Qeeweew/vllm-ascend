import itertools

import pytest
import torch
import torch.nn.functional as F
import torch_npu
import vllm_ascend.vllm_ascend_C  # noqa: F401


GROUP_SIZE = 32


def _pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack [E, N, K] signed INT4 as kernel layout [E, K, N/8]."""
    assert q.dtype == torch.int32 and q.shape[1] % 8 == 0
    q_kn = q.transpose(1, 2).contiguous()
    packed = torch.zeros((*q_kn.shape[:-1], q_kn.shape[-1] // 8),
                         dtype=torch.int32)
    for nibble in range(8):
        packed |= ((q_kn[..., nibble::8] & 0xF) << (4 * nibble))
    return packed.contiguous()


def _linear_ref(x: torch.Tensor, q: torch.Tensor,
                scale: torch.Tensor) -> torch.Tensor:
    """Group-size-32 signed-scale W4A16 reference, returning FP32."""
    groups = q.shape[1] // GROUP_SIZE
    acc = torch.zeros(q.shape[0], dtype=torch.float32)
    for group in range(groups):
        sl = slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)
        # The AscendC kernel accumulates q*x into an FP16 group accumulator.
        group_sum = (q[:, sl].half() * x[sl].half()).sum(dim=1).half()
        acc += group_sum.float() * scale[group].float()
    return acc


def _moe_ref(x: torch.Tensor, q13: torch.Tensor, s13: torch.Tensor,
             q2: torch.Tensor, s2: torch.Tensor, ids: torch.Tensor,
             routing: torch.Tensor, limit: float, dtype: torch.dtype):
    batch, hidden = x.shape
    topk = ids.shape[1]
    inter = q2.shape[-1]
    out = torch.zeros((batch, hidden), dtype=torch.float32)
    for b, route in itertools.product(range(batch), range(topk)):
        expert = int(ids[b, route])
        h13 = _linear_ref(x[b], q13[expert], s13[expert])
        gate, up = h13[:inter], h13[inter:]
        if limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        activated = (F.silu(gate) * up).to(dtype)
        h2 = _linear_ref(activated, q2[expert], s2[expert])
        out[b] += routing[b, route] * h2
    return out.to(dtype)


def _case(batch: int, dtype: torch.dtype, limit: float):
    torch.manual_seed(20260822 + batch)
    experts, topk, hidden, inter = 2, 2, 128, 128

    # Generate on CPU so the reference does not depend on another NPU kernel.
    x = (torch.randn(batch, hidden) * 0.15).to(dtype)
    q13 = torch.randint(-8, 8, (experts, 2 * inter, hidden),
                        dtype=torch.int32)
    q2 = torch.randint(-8, 8, (experts, hidden, inter), dtype=torch.int32)
    s13 = (torch.rand(experts, hidden // GROUP_SIZE, 2 * inter) *
           0.08 + 0.01).to(dtype)
    s2 = (torch.rand(experts, inter // GROUP_SIZE, hidden) *
          0.04 + 0.005).to(dtype)
    # Alternating signs guarantee that both positive and negative signed
    # scales are exercised, including q=-8 with a negative scale.
    s13[..., 1::2].neg_()
    s2[..., ::2].neg_()
    q13[:, 0, 0] = -8
    q2[:, 0, 0] = -8

    ids = torch.arange(batch * topk, dtype=torch.int32).reshape(batch,
                                                                 topk) % experts
    routing = torch.softmax(torch.randn(batch, topk), dim=-1).float()
    expected = _moe_ref(x, q13, s13, q2, s2, ids, routing, limit, dtype)

    actual = torch.ops._C_ascend.npu_w4a16_moe(
        x.npu(), _pack_int4(q13).npu(), s13.npu(), _pack_int4(q2).npu(),
        s2.npu(), ids.npu(), routing.npu(), limit)
    torch_npu.npu.synchronize()
    actual = actual.cpu()
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_ref = expected.float().abs().max().item()
    rel = max_abs / max(max_ref, 1e-6)
    print(f"batch={batch} dtype={dtype} limit={limit:g} "
          f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} rel={rel:.6f}")
    if rel >= 0.08:
        print("actual[:8]  =", actual[0, :8].float().tolist())
        print("expected[:8]=", expected[0, :8].float().tolist())
    # Two quantized GEMVs contain FP16 accumulators and the final output is
    # FP16/BF16. Bound both absolute and scale-relative error.
    assert max_abs < 0.35
    assert rel < 0.08


@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_w4a16_moe_kernel(batch, dtype, limit):
    _case(batch, dtype, limit)


if __name__ == "__main__":
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch_npu.npu.set_device(0)
    for test_limit, test_dtype, test_batch in itertools.product(
            (0.0, 10.0), (torch.float16, torch.bfloat16), (1, 2, 8)):
        _case(test_batch, test_dtype, test_limit)
