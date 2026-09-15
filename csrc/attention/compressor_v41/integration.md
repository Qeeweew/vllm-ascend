# CompressorV41 integration

This directory implements only the vector compressor. Projection GEMMs and both
cache insertion paths remain separate. CR1 takes BF16 `[T,512]`; CR2 takes FP32
`[T,1024]`. Both produce BF16 `[T,512]`, zeroing invalid/non-boundary rows. CR2
mutates an FP32 `[blocks,capacity,1024]` ring. Its capacity is a power of two at
least 8 and must exceed the speculative step length including its predecessor.

## Central registration changes

Add `compressor_v41` to the Ascend 910B operator list in `csrc/build_aclnn.sh`.
The existing attention CMake traversal discovers this directory automatically.
Do not enable it for arch35 without implementing and testing that target.

If central bindings explicitly include generated ACLNN headers, include
`aclnn_compressor_v41.h`. Like `StoreKVBlock`, this op has mutable input buffers
and no separate GE outputs. Its generated ACLNN inputs follow the order below.

```cpp
void compressor_v41(const at::Tensor &kv_score, const at::Tensor &positions,
                    const at::Tensor &slot_mapping, const at::Tensor &query_start_loc,
                    const at::Tensor &token_to_req_indices, const at::Tensor &norm_weight,
                    at::Tensor &state_cache, at::Tensor &latent_out,
                    int64_t compress_ratio, double eps)
{
    // Reject noncontiguous buffers here as well when exposing the raw operator.
    // In-place inputs must never go through an implicit contiguous copy.
    TORCH_CHECK(kv_score.is_contiguous() && positions.is_contiguous() &&
                slot_mapping.is_contiguous() && query_start_loc.is_contiguous() &&
                token_to_req_indices.is_contiguous() && norm_weight.is_contiguous() &&
                state_cache.is_contiguous() && latent_out.is_contiguous(),
                "CompressorV41 requires contiguous buffers");
    if (kv_score.size(0) == 0) {
        return;
    }
    EXEC_NPU_CMD(aclnnCompressorV41, kv_score, positions, slot_mapping,
                 query_start_loc, token_to_req_indices, norm_weight,
                 state_cache, latent_out, compress_ratio, eps);
}
```

Register the schema in the existing `_C_ascend` library, and its PrivateUse1
implementation using the repository's standard binding convention:

```cpp
ops.def("compressor_v41(Tensor kv_score, Tensor positions, Tensor slot_mapping, "
        "Tensor query_start_loc, Tensor token_to_req_indices, Tensor norm_weight, "
        "Tensor(a!) state_cache, Tensor(b!) latent_out, int compress_ratio, "
        "float eps=1e-20) -> ()");
ops.impl("compressor_v41", &compressor_v41);
```

The Meta implementation has the identical signature and an empty body: it
returns no new tensor and must not resize either aliased argument. The public
Python wrapper validates shape/dtype/device using host tensor metadata only.

## Ownership and scheduling

Each actual request has consecutive positions in a contiguous token segment,
and owns a distinct ring block during a launch. Padding is outside actual
request segments and has slot `-1`. Slot encoding is
`block * capacity + position % capacity`. The scheduler must populate the
predecessor row before starting a request chunk at an odd position.

One boundary task per request first reads the predecessor, then saves at most
`capacity` final input rows. Independent token tasks compute interior pairs
from raw projection outputs only. There is no cross-core synchronization and
no matrix multiplication. A full latent row is written only by one task.

Outputs and inputs may not overlap except the declared state mutation. All
buffers and metadata must retain their addresses during graph replay. Dummy
capture state must be reset or isolated before real execution. The Python
CPU oracle is never a production fallback.

## Validation commands

After the main task performs the complete editable build/install:

```bash
python -m pytest --confcutdir=tests/e2e/single_node/ops \
  tests/e2e/single_node/ops/test_compressor_v41.py -q
python tests/e2e/single_node/ops/benchmark_compressor_v41.py \
  --output /tmp/compressor-v41-eager.json
python tests/e2e/single_node/ops/benchmark_compressor_v41.py \
  --graph --graph-unroll 256 --iterations 20 --output /tmp/compressor-v41-graph.json
```

The benchmark saves all event samples, median/P95 and five-round variability
for the correct PyTorch vector reference and the new AIV op. It excludes the
independent projection and cache-store stages; end-to-end acceptance requires
those measurements separately. It does not claim performance acceptance until
the plan's per-case regression and weighted-speedup gates pass on hardware.

Hardware results and retained raw records are in the
[validation report](../../../benchmarks/deepseek_v41/compressor_v41/report.md).
The graph vector-op gate passed; eager noise and model-level acceptance remain
separate. Existing `torch.mm(..., out_dtype=torch.float32)` was also verified
for independent CR2 projection from **ND BF16** weights; preserve
`skip_weight_nz_conversion=True` on that linear module. CR1 projection remains
BF16. No custom GEMM is required for this verified path.
