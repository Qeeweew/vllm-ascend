# V4.1 small-operation fusion contracts

Status: interface design and delegated implementation, not performance acceptance.
Target: Ascend 910B3, arch22, BF16 model, TP8. These operations are primarily
Vector and memory work. Cube utilization is relevant to QLI/GEMM; adding matrix
work to these operations is not an optimization objective.

## Common rules

- Retain independent projection GEMMs, especially compressor projection.
- Caller-owned output/cache buffers have explicit mutation schemas and stable
  addresses. No `.item()`, device-to-host reads, host-dependent token dispatch,
  or graph-time allocation in the native kernel. Python convenience allocation
  is permitted outside the captured path. Reject unsupported aliasing/layouts.
- Every call reads current device inputs, including positions, slots, masks and
  token IDs. Replay tests must change their contents without changing pointers.
- Test T=0/1/2/4/16/64/128/1024 where applicable. Empty calls skip native launch.
- Preserve operation order and intermediate rounding of the current baseline.
  No precision relaxation to make a performance result pass. Freeze numerical
  gates and tie behavior before measuring the first candidate.
- Measure the complete replaced chain with identical inputs: five alternating
  rounds, at least 20 event samples per round, warmup and graph unroll to reduce
  timer noise. T=1/4/128 median and P95 each must improve at least 10%; other
  tested shapes must not regress over 3%; round-median spread <=3%. Retain failed
  cases. Report explicit scratch plus fixed CANN workspace and allocator peaks.
- Capture `msprof op` evidence for launch, Vector, Scalar, MTE and waits; measure
  Cube only where the operation actually uses it. Full-model `vllm bench` follows
  integrated numerical acceptance and cannot be inferred from microbenchmarks.
- Build with the complete package script, install into an isolated artifact,
  verify source/copied-source/installed-source and kernel binary SHA. Shared
  production installation, Torch bindings/meta and model dispatch belong to root.
  Root schedules NPU windows; independent background CPU builds must use separate
  build directories and bounded parallelism. No concurrent performance runs.

## Task A: modality-aware MoE router

Owner: router agent. Proposed Torch schema (native op `V41MoeRouter`):

```text
v41_moe_router(Tensor logits, Tensor token_ids, Tensor image_mask,
              Tensor? tid2eid, Tensor? text_bias, Tensor image_bias,
              Tensor(a!) weights, Tensor(b!) expert_ids,
              int top_k=6, bool renormalize=True,
              float routed_scaling_factor=1.0) -> ()
```

`logits` is contiguous FP32 [T,E], token IDs INT64 [T], image mask BOOL [T].
Biases are FP32 [E], optional `tid2eid` INT32 [vocabulary,K]. Outputs are FP32
[T,K] and INT32 [T,K]. Initial supported configurations: E384/K6 target and
E128/K3 draft. Reject unsupported shapes without changing other models.

Compute sqrt(softplus(logits)) with the baseline's softplus threshold and FP32
rounding. Dynamic rows select by score plus modality-specific bias; output
weights gather the original unbiased scores, optionally normalize, then scale.
For text rows with `tid2eid`, preserve table order and compute only selected
expert scores; do not perform discarded dynamic top-k. Image rows always use
dynamic routing. The explicit mask, not token ID ranges, decides modality.
Token IDs for image rows must not index the text lookup table. Tests include
token 0, literal image ID as text, mixed rows, extreme finite logits, all-zero
weights, near ties and ties. Resolve the exact existing backend tie behavior
before freezing the native selection rule; do not silently choose a new rule.

Independent baseline: `select_deepseek_v4_vision_experts` and a CPU scalar
oracle. Require identical selected IDs for unique-score cases, exact hash-row
IDs and order, and a documented fixed FP32 weight gate. Adversarial cutoff
cases must follow the frozen tie/numerical contract rather than a recall gate.
Finite logits/biases and valid text/table IDs are input preconditions. Defensive
handling of an invalid text/table ID writes the entire row as zero weights and
-1 IDs, without out-of-bounds reads; this is not a valid row for MoE execution.

Owned files: `csrc/moe/v41_moe_router/`, `vllm_ascend/ops/v41_moe_router.py`,
uniquely named router tests/benchmarks and router acceptance report. Supply an
`integration.md` with exact binding/meta code. Do not edit shared registration,
build scripts, existing router dispatch or other agents' directories.

## Task B: RoPE and paged cache stores

Owner: rope/cache agent. Three explicit entry points share a small device core;
avoid a mode-dependent tensor return interface or projection GEMMs.

```text
v41_rope(Tensor x, Tensor positions, Tensor cos, Tensor sin,
         Tensor(a!) output, bool inverse=False) -> ()

v41_main_cache_store(Tensor x, Tensor positions, Tensor slots,
                     Tensor cos, Tensor sin, Tensor(a!) cache,
                     int compress_ratio=1) -> ()

v41_index_cache_store(Tensor key, Tensor positions, Tensor slots,
                      Tensor cos, Tensor sin, Tensor(a!) key_cache,
                      Tensor(b!) scale_cache, int compress_ratio=1) -> ()
```

RoPE input/output is contiguous BF16 [T,D] or [T,H,D], D=128/512,
H=1/8/32. Cos/sin are FP32 [max_positions,32], rotary width 64. Positions and
slots are INT64 [T]. First implementation requires nonaliasing input/output;
leave the nonrotary prefix bit-identical. Read table rows, cast rotary pairs to
FP32, perform separate multiply/add/subtract, then round to BF16 once. Inverse
negates sin. Do not contract operations into FMA if it changes baseline outputs.
For negative or out-of-table positions, plain RoPE copies the input row exactly;
cache stores leave the destination untouched. Invalid positions never produce
uninitialized output or an out-of-bounds table read.

Main store input is pre-RoPE BF16 [T,512]; index store input is normalized,
pre-RoPE BF16 [T,128]. Projection and RMSNorm remain outside these first stores.
Caches use existing [blocks,page,1,D] layouts, including gapped axis-zero
strides and nonzero storage offsets. Main cache is BF16; index cache INT8 and
scale cache FP16 [blocks,page,1,1]. Reject unsupported inner strides.
The native ACLNN interfaces receive explicit axis-zero element-stride attrs
from the Torch binding; key and scale strides are independent. Their pointers
already include storage offsets, which must not be added a second time.

For CR1/2, rotate at floor(position/CR)*CR. Physical `slots` already address
compressed cache rows and must NOT be divided again. Publish CR2 only at the
group's last token. Negative/out-of-capacity slots and negative positions leave
all cache bytes untouched; valid active destinations must be unique. Invalid
rows must be rejected/skipped before any out-of-bounds table or cache load.

Index store preserves the BF16 RoPE rounding BEFORE dynamic INT8 quantization,
including CANN rounding/saturation and zero-row handling. Scale is rounded to
FP16 only after quantized values are computed. Freeze the actual baseline
`npu_dynamic_quant` contract with dedicated probes; exact INT8 and FP16-scale
agreement is required. No quantization-quality gate relaxation.

Tests include changed-input graph replay, partial CR2 groups, invalid slots,
gapped cache storage with sentinel guards, boundary positions, zero rows,
quantization halfway cases and untouched cache validation. Ordinary RoPE must
match the existing BF16 output including rounding-sensitive constructed cases.

Owned files: `csrc/attention/v41_rope/`, `v41_main_cache_store/`,
`v41_index_cache_store/`, `vllm_ascend/ops/v41_rope_cache.py`, uniquely named
rope/cache tests and report. Supply `integration.md` for root-owned binding,
meta, build list and dispatch changes. First deliver plain RoPE, then stores.

## Existing QLI task and later scope

The QLI agent continues the H32 fused candidate kernel. Its whole-selector gate
already includes wrapper sorting/masking/copy costs. After kernel profiling,
evaluate emitting final ascending unique positions with -1 padding directly,
using caller-owned destinations if that actually removes measured overhead.
Keep the existing public interface and out-of-scope shape behavior stable;
root coordinates any shared wrapper edit. Do not add another split score path.

Query RoPE+quantization, mHC terminal collapse+final norm, and vision-specific
pointwise fusions are follow-ups after these first contracts are validated.
DSpark metadata fusion remains behind its unresolved integration diagnosis.
