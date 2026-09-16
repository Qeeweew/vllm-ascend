# V41 rotary and cache integration

Status: all three entry points implemented; complete native r12 package passes
193 numerical/graph cases, including the batched H32 path. Wrapper CPU tests
pass 33 cases, including FakeTensor aliases. All 64 complete-chain performance
cases pass the unchanged median/P95/stability gates (graph unroll 256).
This document supplies root-owned integration changes;
the agent does not edit shared bindings, build lists, or model dispatch.

## Plain RoPE ABI

Add `v41_rope` to the arch22 operator list. The attention CMake traversal already
discovers its directory. Generated ACLNN name is `aclnnV41Rope`. Its mutable
output is a required input without auto-contiguous conversion or a GE output:

```cpp
void v41_rope(const at::Tensor &x, const at::Tensor &positions,
              const at::Tensor &cos, const at::Tensor &sin,
              at::Tensor &output, bool inverse)
{
    // Apply all validations below before launch, including raw Torch callers.
    if (x.size(0) == 0) return;
    EXEC_NPU_CMD(aclnnV41Rope, x, positions, cos, sin, output, inverse);
}
```

Torch schema and Meta signature:

```text
v41_rope(Tensor x, Tensor positions, Tensor cos, Tensor sin,
         Tensor(a!) output, bool inverse=False) -> ()
```

Meta returns nothing and must not allocate, resize, or mutate metadata. Validate
contiguity/common device, x/output BF16 with identical [T,D] or [T,H,D] shapes,
D128/512, H1/8/32, INT64 positions [T], and FP32 cos/sin with identical nonempty
[P,32] shapes. Reject any shared storage between output and all other arguments
(including distinct views with nonoverlapping ranges, for this initial ABI).
Use tensor storage identity in C++, not tensor object identity. Python wrapper
also checks these contracts using host tensor metadata. No device scalar reads.

The kernel receives pointers adjusted for storage offset by the existing ACLNN
adapter. Do not add `storage_offset()` a second time. Valid positions rotate
interleaved pairs in the last 64 elements; preserve four separate FP32 products
and BF16 rounding. Inverse negates sin before those products. Position <0 or >=P
copies the entire input row unchanged; the output never contains uninitialized
padding. Nonrotary prefix bytes are bit-identical, including signed zeros.

## Native architecture and proof obligations

The general implementation distributes contiguous balanced token/head row
ranges across AIVs and reuses rotary tables across heads. For T>=128 with
H32/D128, `v41_rope_heads.h` moves one complete 8192-byte token per DMA,
rotates all 32 heads using vector repeats, and stores that complete token.
A small shared `op_kernel/v41_rope_core.h` handles UB loads, gather-based pair
layout, FP32 arithmetic, BF16 conversion, and stores. No Cube or user workspace.
The fixed CANN workspace is still requested by host tiling and must be recorded
in performance results. Separate Mul then Sub/Add instructions and explicit
Vector dependency barriers prevent an unintended fused multiply-add rewrite.

Before acceptance, run `test_v41_rope_cache.py` on NPU with the freshly built
artifact. Every valid BF16 output must match CPU reference **bitwise**, including
halfway cases. Graph replay changes x, positions, cos, and sin while retaining
all addresses. Invalid positions copy input exactly. T=0 skips native launch.

Source verification must compare original kernel/core, build-copied source,
isolated-installed source, and record all generated .o SHA256. Use the complete
`build.sh --pkg` script in an isolated source/build directory; never test the
previous installed operator accidentally. Production installation belongs to
root after numerical and performance gates pass.

## Store ABI additions

The public schemas in SMALL_OP_FUSION_PLAN.md remain unchanged. Native ACLNN
adds main cache `cache_stride0` INT attr, or separate key/scale stride0 attrs.
Root binding passes `tensor.stride(0)` and validates contiguous inner axes and
nonoverlapping pages. This allows gapped axis0 and nonzero storage offsets
without materializing a replacement mutable cache.
The cache inputs declare `IgnoreContiguous()` in OpDef; otherwise ACLNN rejects
gapped multi-page views before launching the stride-aware kernel.
Physical slots are already
compressed. CR2 stores publish only odd original positions and rotate at
`floor(position/2)*2`. Invalid slots, table bounds, or nonboundary rows leave
cache bytes untouched. GEMM and RMSNorm remain separate.

Exact generated ACLNN argument order is:

```text
aclnnV41MainCacheStore(x, positions, slots, cos, sin, cache,
                       compress_ratio, cache_stride0)
aclnnV41IndexCacheStore(key, positions, slots, cos, sin, key_cache, scale_cache,
                        compress_ratio, key_stride0, scale_stride0)
```

Both cache APIs support three-dimensional [B,P,D] as well as [B,P,1,D]. Scale
uses D=1. Neither mutable destination may share storage with any input.
Key and scale outputs may share raw storage, as they do in the model runner,
when they have equal positive byte page strides and disjoint periodic byte
regions. With `pitch = key.stride(0)` (INT8),
`delta = (scale.storage_offset()*2 - key.storage_offset()) % pitch`,
require `delta >= P*128` and `delta + P*2 <= pitch`. Python and C++ use the same
nonnegative modulo. This supports raw storage offsets, gaps, scales preceding
keys, and whole-page shifts without permitting within/across-page overlap.
Native tiling reads the logical origin shape and stride
attributes; no implicit contiguous copy is allowed. The Python signatures use
keyword-only `compress_ratio=1`; their native Torch calls pass the integer attr
positionally. Public schemas return nothing; convenience wrappers return the
unchanged caller-owned destination object(s).
