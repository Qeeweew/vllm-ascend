# V4.1 TP8 runner integration

## Explicit image-mask text regression

After connecting the typed image mask through history, runtime and the
complete MoE op, the real three-layer E384 CANN graph run passed again with
strict HCCL, prompt lengths 32/40, prefix caching off and the original NUMA
placement. All 12 tokens and selected logprobs are **bitwise identical** to
`graph_real3_strict.json`; each repeated batch also remains exact. This
exercises real expert kernels, shared experts and TP8 communication with an
all-false mask. It does not establish image-prefill correctness.

Evidence: `graph_real3_typed_strict_matched.json`; process exit 0, log
`/tmp/v41-runner-real3-typed-strict-matched.log`. A preceding run with prompt
lengths 40/48 also passed execution/repeat checks and is saved separately as
`graph_real3_typed_strict.json`; differing prompt/batch shapes are not used
for bitwise comparison to the older 32/40 baseline.

## Three-layer execution

On 2026-09-15, the actual vLLM LLM/EngineCore/scheduler and eight Ascend
workers completed both eager and FULL_DECODE_ONLY graph runs. Each run used
production hidden/intermediate widths, three layers, eight routed experts,
and one small BF16 host Engram table. Device parameters are dummy weights;
post-load packed INT4 storage is explicitly initialized to signed ones.

Both runs exercised chunked prefill (40/48 input tokens with a 32-token
budget), two concurrent requests, four generated tokens per request, and
a subsequent request reusing the first prompt prefix. All 12 output tokens
and their selected logprobs match exactly between the two independent runs.
Each worker completed nine Engram preparation steps; device rows and mask
addresses remained stable from capture through the final request, with no
unconsumed preparation left. Graph workers each held two captured graphs;
the execution log also confirms actual `Replaying aclgraph`.

Raw evidence: [eager.json](eager.json), [graph.json](graph.json).
Logs: `/tmp/v41-runner-tp8-eager-r5.log` and
`/tmp/v41-runner-tp8-graph-r1.log`. Both processes exited successfully.

```bash
.venv/bin/python vllm-ascend/benchmarks/deepseek_v41/smoke_runner_tp8.py \
  --output vllm-ascend/benchmarks/deepseek_v41/runner_tp8/eager.json
.venv/bin/python vllm-ascend/benchmarks/deepseek_v41/smoke_runner_tp8.py \
  --graph --output vllm-ascend/benchmarks/deepseek_v41/runner_tp8/graph.json
```

Run from the workspace parent of `vllm-ascend`, without `torchrun`.

## Integration fixes exposed by the runner

- Current upstream `CommonAttentionMetadata` removed three constructor
  fields still used by the Ascend runner. The Ascend subclass now owns
  these fields, and `unpadded()` retains the current upstream metadata
  using `dataclasses.replace`. Nineteen metadata regressions passed.
- The plugin cache binder assigned raw tensors directly, bypassing the
  compressor's layer-specific shape binding. It now invokes a layer's
  custom binder when present and preserves the raw connector view.
  Two CPU regressions and four registered-cache NPU cases passed.

## Limits

These are execution and graph consistency checks. Dummy logits are close
to uniform; token agreement does not establish model quality. Reduced
expert count does not exercise the production E384 native decode dispatch.
The test requests prefix reuse but does not measure the cache hit rate.
It does not establish full checkpoint loading, large-table NUMA placement,
long-context accuracy, vision, speculation, or production throughput.
NPU7 has a separate process holding approximately 34 GiB; no performance
conclusion is drawn under that shared-device condition.

## Forty-layer execution

Both `--layers 40` runs also exited successfully. They preserve all main-KV
and index source/consumer layer identities, the CR2 to CR1 transition, and
both Engram layers. All 12 output tokens and selected logprobs match exactly.
Each of eight workers completed nine preparations using two stable Engram
row buffers; graph workers captured two graphs and actual replay appears
in `/tmp/v41-runner-tp8-graph-40-r1.log`.

Raw evidence: [eager_40.json](eager_40.json), [graph_40.json](graph_40.json).
Add `--layers 40` to the commands above and use these output filenames.
Experts remain reduced to eight and prompts remain shorter than SWA128;
this does not cover long-context eviction or top512 selection pressure.

## Production expert count and native decode dispatch

Three-layer tests with **384 experts**, TP8 and production widths completed
for CANN eager, native decode eager, and native decode graph. All 12 output
tokens and selected logprobs match exactly across the three runs. Weights
are still synthetic. Native eager workers each recorded 18 selected native
calls and 12 CANN fallbacks (including warmup/prefill). Native graph workers
each recorded six native capture calls and no fallback capture calls; real
request execution left those capture counts unchanged and the log confirms
graph replay. Prefill still dispatched through CANN.

Raw evidence: [eager_e384_cann.json](eager_e384_cann.json),
[eager_e384_native.json](eager_e384_native.json),
[graph_e384_native.json](graph_e384_native.json).
Use `--experts 384`, adding `--native-decode` and/or `--graph` as appropriate.

The actual runner exposed a dispatch gap absent from old mocked tests:
V4.1 metadata lacked the host prefill/decode counters read by the MoE guard.
The metadata builder now derives them from CPU request state and boundaries,
including explicit uniform-decode capture metadata. MoE selection skips
hybrid state-cache entries without counters. Sixty-eight CPU regressions
passed. The runner additionally requires fully computed prompts before
replaying a graph containing a decode-only kernel, preventing a cached
single-token prompt tail from entering native decode. Two runner regression
tests passed, including four boundary/configuration cases.

Native decode remains opt-in pending production model quality and TP8
performance acceptance; these integration checks do not establish speedup.

## Longer-context integration

The forty-layer E8 eager and graph runs also passed with 1152/1160-token
prompts (`--layers 40 --prompt-length 1152`). Each worker performed 44 Engram
preparations with stable buffers. All output tokens and selected logprobs
match exactly. Actual graph replay appears in
`/tmp/v41-runner-tp8-graph-40-long-r1.log`.

Raw evidence: [eager_40_long.json](eager_40_long.json),
[graph_40_long.json](graph_40_long.json). This covers SWA128 expired pages
and more than 512 visible entries for both CR1 and CR2. Candidate blocks are
published/reused, but the 2048-block candidate budget is not exhausted;
that requires more than 16384 CR1 tokens.

## Real device-weight subset

`--converted-weights /mnt/models/DeepSeek-V4.1-Flash-W4A16-G32 --experts 384`
selects the real embedding/head/norm and requested layer shards through
temporary symlinks. It preserves checkpoint values, including vision router
bias parameters, and uses the ordinary safetensors loader. Only the temporary
test checkpoint receives a subset index; the incomplete full conversion is
not published or modified. Engram tables remain synthetic and small.

This path completed request execution with three layers and real E384 device
weights. Each rank reported 3.4934 GiB of model loading memory. Its initial attempt cleared vision
config used to construct `gate.bias_vl`, which made the real checkpoint's
router bias fail to load. The test now retains the original vision config
for real device weights. This does not instantiate or enable a vision tower.

Real-weight execution also exposed an invalid write of `mm_prefix_range` to
frozen compressor-state metadata. The runner now writes the legacy field only
to metadata types that declare it; common metadata still carries image ranges
to builders. Eight common-metadata tests passed after this fix.

The registered NUMA graph test verified all eight workers' pages on nodes
`[6,7,4,5,0,1,2,3]` before and after requests, with stable device rows/masks.
A cached 32-token prefix left a single prompt token: the recorded preparation
sizes for that request were `[1,1,1,1]`, and only its prompt step called CANN
once per layer; subsequent steps replayed the native graph.

**Numerical acceptance remains open.** Real-weight native graph, CANN graph
and CANN eager produce the same output tokens, but selected logprobs differ.
The largest observed native-graph versus eager difference is 0.27109; CANN
graph versus CANN eager also differs by 0.27113, so this cannot be attributed
solely to native MoE. Native versus CANN graph differs by up to 0.06075.
With prefix caching disabled, CANN graph/eager still differs by up to 0.05864.
Same-engine repeated requests and activation tracing are being used to
distinguish execution bugs from arithmetic and batch-shape effects.

With prefix caching disabled, repeating the same batch within one eager
engine still changes selected logprobs by up to 0.061671, with identical
tokens. The first 32-token prefill has identical input IDs, positions and
layer-zero input normalization across repetitions. Its first different
activation is layer-zero attention output (NRMSE 0.004062, 61706 of 163840
elements different). This precedes MoE, Engram and the CR2 compressor;
attention projection, sparse attention and TP reduction were then traced
separately. Trace hooks make independent CPU copies
before in-place reduction and are diagnostic only, never performance data.
Raw request evidence: [eager_real3_repeat.json](eager_real3_repeat.json),
[eager_real3_repeat_trace.json](eager_real3_repeat_trace.json).

All-rank tracing identified the first divergence at the HCCL reduction:
each of the eight ranks' first-layer `wo_b` local GEMM results is bit-exact
between repetitions, as is their FP32 sum. The BF16 collective outputs
differ in 93357 of 163840 elements (NRMSE 0.004758, maximum absolute
difference 0.0234375), while ranks agree within each forward. With
`HCCL_DETERMINISTIC=strict`, repeated tokens and selected logprobs are
bit-exact; the default collective run differs by up to 0.0640504 in
selected logprob. This establishes the cause of this same-shape repeated
request discrepancy, not full-model quality or performance acceptance.
Untraced CANN eager/graph confirmation also passed: all twelve generated
tokens and selected logprobs match exactly between modes, and each mode's
same-engine repeated batch is bit-exact. This uses strict HCCL, disabled
prefix caching and the same three-layer/small-Engram fixture. No global
deterministic setting or low-precision workaround was added to the product.

Raw runs: [eager_real3_all_rank_trace.json](eager_real3_all_rank_trace.json),
[eager_real3_all_rank_strict_trace.json](eager_real3_all_rank_strict_trace.json).
All-rank activation comparison:
[attention_trace_all_ranks_comparison.json](../attention_trace_all_ranks_comparison.json).
Untraced confirmation: [eager_real3_strict.json](eager_real3_strict.json),
[graph_real3_strict.json](graph_real3_strict.json). See the
[reduction diagnosis](../ATTENTION_REPEAT_NUMERICS.md) for the precision
distinction between deterministic BF16 reduction and FP32 accumulation.

The [strict native graph run](graph_real3_native_strict.json) also executes
the captured decode kernel on every rank and generates the same tokens as
strict CANN. Selected logprobs differ by up to 0.00390410 between kernels;
repeating the native batch differs by up to 0.00448847. Independent native
same-input replay confirms small numerical variation outside HCCL, with
only two or three BF16 output elements changing per invocation and maximum
NRMSE 0.00034111 against the FP32 contract. See the MoE report below; native
is not described as bit-exact or enabled by default.

See [W4A16_REAL_NUMERICS.md](../W4A16_REAL_NUMERICS.md) for real decode MoE
inputs: isolated CANN exactly reproduces captured outputs, while native is
closer to an independent FP32 arithmetic contract. The two paths differ by
approximately 0.50–0.52% NRMSE due primarily to BF16 rounding boundaries.
Do not treat generated-token agreement as full-model numerical acceptance.
