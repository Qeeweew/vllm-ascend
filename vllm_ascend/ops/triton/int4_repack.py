"""Fused transpose/repack for compressed-tensors INT4 MoE weights."""

import torch
import triton
import triton.language as tl


@triton.jit
def _int4_repack_kernel(
    src,
    dst,
    n_cols,
    k_rows,
    stride_src_n,
    stride_src_k8,
    stride_dst_k,
    stride_dst_n8,
    NUM_CORES: tl.constexpr,
    BLOCK_N8: tl.constexpr,
):
    pid = tl.program_id(0)
    total_k8 = k_rows // 8
    rows_per_core = (total_k8 + NUM_CORES - 1) // NUM_CORES
    begin = pid * rows_per_core
    end = tl.minimum(begin + rows_per_core, total_k8)

    for k8 in range(begin, end):
        for n8_base in range(0, n_cols // 8, BLOCK_N8):
            n8 = n8_base + tl.arange(0, BLOCK_N8)
            valid = n8 < n_cols // 8
            out0 = tl.zeros([BLOCK_N8], tl.uint32)
            out1 = tl.zeros([BLOCK_N8], tl.uint32)
            out2 = tl.zeros([BLOCK_N8], tl.uint32)
            out3 = tl.zeros([BLOCK_N8], tl.uint32)
            out4 = tl.zeros([BLOCK_N8], tl.uint32)
            out5 = tl.zeros([BLOCK_N8], tl.uint32)
            out6 = tl.zeros([BLOCK_N8], tl.uint32)
            out7 = tl.zeros([BLOCK_N8], tl.uint32)
            for nibble in tl.static_range(8):
                n = n8 * 8 + nibble
                packed = tl.load(src + n * stride_src_n + k8 * stride_src_k8, mask=n < n_cols, other=0).to(
                    tl.uint32
                )
                shift = nibble * 4
                # Checkpoints use offset-binary (q + 8).  Subtracting 8 and
                # masking converts it to the signed two's-complement nibble
                # consumed by the AscendC kernel, including q=-8.
                out0 |= ((((packed >> 0) & 15) - 8) & 15) << shift
                out1 |= ((((packed >> 4) & 15) - 8) & 15) << shift
                out2 |= ((((packed >> 8) & 15) - 8) & 15) << shift
                out3 |= ((((packed >> 12) & 15) - 8) & 15) << shift
                out4 |= ((((packed >> 16) & 15) - 8) & 15) << shift
                out5 |= ((((packed >> 20) & 15) - 8) & 15) << shift
                out6 |= ((((packed >> 24) & 15) - 8) & 15) << shift
                out7 |= ((((packed >> 28) & 15) - 8) & 15) << shift
            dst_base = dst + n8 * stride_dst_n8
            tl.store(dst_base + (k8 * 8 + 0) * stride_dst_k, out0.to(tl.int32), mask=valid)
            tl.store(dst_base + (k8 * 8 + 1) * stride_dst_k, out1.to(tl.int32), mask=valid)
            tl.store(dst_base + (k8 * 8 + 2) * stride_dst_k, out2.to(tl.int32), mask=valid)
            tl.store(dst_base + (k8 * 8 + 3) * stride_dst_k, out3.to(tl.int32), mask=valid)
            tl.store(dst_base + (k8 * 8 + 4) * stride_dst_k, out4.to(tl.int32), mask=valid)
            tl.store(dst_base + (k8 * 8 + 5) * stride_dst_k, out5.to(tl.int32), mask=valid)
            tl.store(dst_base + (k8 * 8 + 6) * stride_dst_k, out6.to(tl.int32), mask=valid)
            tl.store(dst_base + (k8 * 8 + 7) * stride_dst_k, out7.to(tl.int32), mask=valid)


def repack_int4_moe(weight: torch.Tensor) -> torch.Tensor:
    """Convert ``[E,N,K/8]`` offset-binary INT4 to ``[E,K,N/8]`` signed INT4."""
    if weight.ndim != 3 or weight.dtype != torch.int32:
        raise ValueError(f"expected a 3-D int32 tensor, got shape={tuple(weight.shape)}, dtype={weight.dtype}")
    experts, n_cols, k8 = weight.shape
    if n_cols % 8 or k8 == 0:
        raise ValueError(f"INT4 repack requires N divisible by 8, got shape={tuple(weight.shape)}")
    src = weight.transpose(1, 2).contiguous().view(experts * k8, n_cols)
    k_rows = experts * k8 * 8
    dst = torch.empty((k_rows, n_cols // 8), device=weight.device, dtype=torch.int32)
    props = triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())
    num_cores = int(props["num_vectorcore"])
    _int4_repack_kernel[(num_cores,)](
        src,
        dst,
        n_cols,
        k_rows,
        src.stride(1),
        src.stride(0),
        dst.stride(0),
        dst.stride(1),
        NUM_CORES=num_cores,
        BLOCK_N8=256,
    )
    return dst.view(experts, k8 * 8, n_cols // 8)
