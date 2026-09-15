# W4A16 decode integration

Kernel inputs are contiguous ND tensors on one NPU:

- `x`: `[B,H]`, BF16 or FP16.
- `w13`: `[E,H,2I/8]`, INT32 packing signed INT4 along output N.
- `s13`: `[E,H/32,2I]`, activation dtype; signed scales are preserved.
- `w2`: `[E,I,H/8]`, same packing.
- `s2`: `[E,I/32,H]`, activation dtype.
- `ids`: `[B,topk]`, INT32. Out-of-range routes contribute zero.
- `routing`: same shape, FP32, multiplied exactly once at output.
- `limit`: finite nonnegative scalar; zero disables clamp, V4.1 uses 10.

H must be divisible by 64; I by 32 (including 288). No weight padding.
Output is fresh `[B,H]` in x dtype. Inputs do not mutate. The adapter allocates
bounded workspace `B*topk*I*10+B*H*4` bytes; all addresses are captured during
graph recording and no device-to-host synchronization occurs in its hot path.
Kernel group products and sums are FP32. SwiGLU is rounded to activation dtype
before W2. Gate is clamped above only; up is clamped on both sides.

## Shared integration files (owned by parent task)

Append `csrc/moe/w4a16_moe/op_kernel/w4a16_moe.cpp` to
`VLLM_ASCEND_CUSTOM_OP` and append
`csrc/moe/w4a16_moe/w4a16_moe_torch_adpt.cpp` to `VLLM_ASCEND_SRC` for 910B.
Include `moe/w4a16_moe/w4a16_moe_torch_adpt.h` in torch binding and register:

```cpp
ops.def("npu_w4a16_moe(Tensor x, Tensor w13, Tensor w13_scale, Tensor w2, Tensor w2_scale, Tensor expert_ids, Tensor topk_weights, float swiglu_limit=0.0) -> Tensor");
ops.impl("npu_w4a16_moe", torch::kPrivateUse1, &vllm_ascend::npu_w4a16_moe);
```

Meta takes the same signature and returns `at::empty(x.sizes(), x.options())`.
This direct-launch kernel is built by root `ascendc_library`; the legacy
op_host sources are reference metadata, not the direct launch path. Do not
add W4a16Moe to the ACLNN build list without a proper ACLNN kernel entry.

## Status

CANN GMM has passed real H5120/I288 group32 signed-scale BF16 checks on device
2 (2026-09-15). Raw baseline lives under `benchmarks/deepseek_v41/`.
Native kernel passed full build r4, real-shape correctness, graph replay and
selected B1/2/4 performance gates. See `benchmarks/deepseek_v41/W4A16_STATUS.md`.
`enable_w4a16_decode` remains false by default pending eight-card model acceptance.

## Arch22 launch requirement

Although computation uses only vector cores, this phased kernel requires
`KERNEL_TYPE_MIX_AIV_1_0` for hardware `SyncAll`. Current CANN direct-launch
stubs inject `ffts_addr` only for MIX modes; `AIV_ONLY` omitted that address
and the first real-shape test hung at the global barrier. Restoring the
original branch launch type provides FFTS metadata without launching AIC
compute. This follows the installed CANN generator
`tools/tikcpp/ascendc_kernel_cmake/legacy_modules/util/extract_host_stub.py`
(`add_ffts_addr_func_param_by_mode`, `_generate_ffts_source`).
