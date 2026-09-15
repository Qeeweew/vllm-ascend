# EngramGate integration

Append `engram_gate` to the Ascend 910B ACLNN operator selection in
`csrc/build_aclnn.sh`. The directory is discovered by the attention CMake tree.
Do not add a raw kernel target: this is an ACLNN operator.

Register in `_C_ascend`, behind `VLLM_ENABLE_V41_KERNELS`:

```cpp
void engram_gate(const at::Tensor &hidden, const at::Tensor &kv,
                 const at::Tensor &q_weight, const at::Tensor &k_weight,
                 const at::Tensor &token_mask, at::Tensor &output, double eps)
{
    for (const auto *tensor : {&hidden, &kv, &q_weight, &k_weight, &token_mask, &output}) {
        TORCH_CHECK(tensor->device() == hidden.device() && tensor->is_contiguous(),
                    "EngramGate requires contiguous tensors on the same NPU");
    }
    EXEC_NPU_CMD(aclnnEngramGate, hidden, kv, q_weight, k_weight, token_mask, output, eps);
}
```

Schema:

```text
engram_gate(Tensor hidden, Tensor kv, Tensor q_weight, Tensor k_weight,
            Tensor token_mask, Tensor(a!) output, float eps=1e-20) -> ()
```

The Meta implementation is a no-op with the identical signature. The Python
wrapper allocates output if omitted, or accepts a caller-owned fixed graph
buffer. All inputs are contiguous, hidden/output BF16 `[T,4,5120]`,
kv BF16 `[T,25600]`, weights BF16 `[4,5120]`, mask bool `[T]`.
GEMM and TP reduction must complete before this operator. No aliasing between
the output and the kv/weights/mask is supported.

Numerical contract follows `inference/model.py`: FP32 `q*k` first, then
`(hidden*weight)*key`, per-copy RMS normalization, signed square root with
clamp `1e-6`, sigmoid, separate FP32 value multiplication and residual addition,
and only then BF16 rounding. CUDA's `hidden*q*k*key` association and potentially
contracted multiply-add are deliberately not the reference. Reductions may
differ by normal FP32 reduction-tree rounding. False mask rows copy hidden
exactly, without reading kv; graph padding must set mask false.

A CPU rounding audit with ordinary BF16 operands found equal dot products for
both multiplication associations on 512 copy rows, but simulating the final
FP32 FMA changed 51 of 2621440 BF16 output lanes versus separate multiply/add.
This is why the residual multiply/add remains explicitly separated. The initial
device error was much larger. The source replaces `Reciprocal` with
`Div(1, denominator)`, matching CANN's sigmoid, but r5/r6 pip builds were found
to reuse a stale source copy and binary. They did not validate this change.
The clean r7 Div build reduced errors but did not meet accuracy acceptance.
A subsequent real debug build verified a marker before reading FP32 stages:
for exact unit norms, sums were both 5120 and the raw dot was exact, but basic
`Rsqrt(1)` returned `0.998046875`. Two such estimates reduced the dot by about
0.39%. The production source now uses `Sqrt` followed by `Div(1, sqrt)`, matching
CANN normalization. All temporary DEBUG_STATS instrumentation is removed.
The clean r9 production build passes all 17 correctness tests and all eight
graph accuracy/performance gates; no threshold was relaxed.

Implementation uses 153696 bytes UB per AIV and assigns a whole copy to each
task. Smallest decode (`T=1`) uses four AIVs; this is a performance risk to
measure, not a claim of optimal occupancy. No graph-time host reads, temporary
GM intermediates, inter-core synchronization, or GEMM are inside this operator.

Full editable build/install is required before NPU validation. Do not test an
older OPP binary. Completed NPU correctness, graph, performance results, and
reproduction commands are in the
[acceptance report](../../../benchmarks/deepseek_v41/ENGRAM_GATE_STATUS.md).
Check the copied kernel source and the installed binary fingerprint
after the clean build; a successful pip exit alone is insufficient evidence
that this ACLNN source was recompiled.
