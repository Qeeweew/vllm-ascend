# B1 candidate selector experiment

Status: optional implementation; 33 correctness tests passed, including
dynamic NPU graph replay. All three B1 shapes have passed frozen latency,
live latency and noise acceptance. Context 4097 required a separately
scheduled repeat after its first dense baseline spread exceeded 3%; both
results are preserved below. The optional dispatch interface passes NPU graph
and fallback tests. A 40-layer synthetic TP8 graph integration also passes;
real full-model quality and end-to-end performance remain unaccepted. Native
dispatch remains the default.
The earlier selector has four
open latency gates, including all three B1 candidate-consumer shapes; see
[the existing report](report.md).

## Frozen contract and acceptance

[candidate_frozen_baseline.json](candidate_frozen_baseline.json) preserves the
existing consumer measurements before this experiment runs on NPU. Each shape
must satisfy median <= dense median, p95 <= dense p95 * 1.05, and maximum
round-median spread <= 3%. The new benchmark compares both the archived limits
and an alternating live dense baseline. Aggregate speedup cannot waive a
failed shape. B8/B32 continue using existing production dispatch.

| B1 compressed context | Archived median limit (us) | Archived p95 limit (us) |
| ---: | ---: | ---: |
| 4097 | 97.5325 | 103.3830 |
| 32771 | 133.4400 | 140.9047 |
| 131075 | 205.6625 | 239.9198 |

The p95 limits above already include the declared 1.05 multiplier; the
benchmark does not apply an additional margin to those limits.

The numerical oracle operates on sets of original positions, independently
of the device gather layout. QK uses exact INT8 dot products, division by 1024,
ReLU and FP16 rounding. Head weights multiply in FP16, then convert to FP32.
The head reduction and final key-scale multiply use FP32. Selection permits
only cutoff exchanges covered by a per-position FP32 reduction error bound;
there is no recall-percentage tolerance. Invalid candidates, duplicate copies,
causally invisible positions and graph padding have score negative infinity.
If fewer than 512 positions are valid, remaining outputs are -1.

## Implementation

The experiment has three independent compute stages:

1. `IndexerV41CandidateGather`, a pure AscendC AIV kernel, resolves candidate
   blocks through the page table, gathers INT8 keys into BF16, and writes
   FP32 scales and original INT32 position IDs. It respects actual cache
   axis-zero strides and rejects out-of-range physical page references.
2. `torch.bmm` independently multiplies BF16 `[1,32,128]` by BF16
   `[1,128,N]`, producing FP32 `[1,32,N]`. BF16 exactly represents INT8
   integers, and every D128 integer dot sum fits exactly in FP32.
3. `IndexerV41CandidateScore`, another pure AIV kernel, applies the specified
   FP16 intermediate rounding, signed head weights, FP32 head reduction and
   explicit invalid mask. Existing topk, index remapping and sorting finish
   the selector.

The wrapper accepts a static maximum context length and preallocates large
workspaces. Runtime lengths must stay within that declared bound. Its gather
width is `max(512, min(16384, ceil(max_context / 8) * 8))`; short contexts avoid
a fixed 16K gather/BMM. Before gathering, the wrapper clips invalid block IDs
to sentinels, converts the bounded IDs exactly to FP32, and sorts descending.
This preprocessing is included in whole-selector timing.

For short contexts the gather enumerates logical blocks and binary-searches
candidate membership, so duplicates cannot displace a later unique candidate.
For full-width gathers, sorted slots define the compact layout and adjacent
duplicates are masked. This keeps output shape fixed through graph replay,
without host reads or a dynamically sized unique operation.

The low-level gather inputs are INT8 keys `[P,page,1,128]`, FP16 scales
`[P,page,1]`, descending FP32 blocks `[2048]`, INT32 page table `[1,pages]`,
INT32 lengths `[1]` and query boundaries `[2]`. The page size is divisible by
eight. Only the key/scale axis-zero stride may contain gaps. The score inputs
are FP32 QK `[1,32,N]`, FP16 weights/scales `[1,32]`, gathered FP32 key scales
`[N]` and INT32 positions `[N]`. Both operators write caller-owned outputs.

## Validation commands

From the vllm-ascend repository root, after the complete operator build and
editable package installation finish:

```bash
../.venv/bin/python -m pytest --confcutdir=tests/e2e/single_node/ops \
  tests/e2e/single_node/ops/test_indexer_v41_candidate.py -q

../.venv/bin/python tests/e2e/single_node/ops/benchmark_indexer_v41_candidate.py \
  --stages --output benchmarks/deepseek_v41/indexer_v41/candidate_graph.json
```

Correctness covers contexts 1/17/511/4097/32771/131075, duplicated and invalid
candidate IDs, negative head weights, newest partial blocks, gapped caches,
zero valid positions, graph padding, and repeated graph execution with changed
page tables, query vectors, weights, lengths and candidate membership. Stage
timings separate preprocessing, gather, BMM, score and final selection; only
whole-selector timing establishes
the latency gate. Neither these synthetic tests nor selector latency establishes
full-model quality or end-to-end throughput.

## Optional dispatch interface

`AscendIndexerV41Ops` accepts the keyword-only `candidate_max_context`, default
`None`. A non-None value requires CR1 consumer mode. Call
`prepare_candidate_workspace(device)` explicitly before capture to allocate
the candidate workspace. Preparation is idempotent on the same device and
rejects moving an existing workspace to another device. First preparation
during NPU graph capture raises before allocating the workspace.

The model exposes this through
`additional_config.enable_indexer_candidate_decode`, default `False`. When
enabled, candidate consumer layers preallocate workspace using the model's
`max_model_len` and their projection-weight device. This is an opt-in production
entry point; the standalone selector is no longer disconnected from the model.

With a prepared workspace, only T=1 with one request, CR1, an eight-aligned
cache page size, sufficient page-table capacity, and the matching device
uses candidate dispatch. Unprepared calls, T>1, multiple requests and
insufficient page-table capacity retain native dispatch. Empty calls preserve
the existing empty-output behavior. No selection path lazily allocates the
large candidate workspace.

The model must pass its static maximum sequence length as
`candidate_max_context` and guarantee that actual CR1 lengths stay within it.
Having a larger page table does not authorize exceeding that bound. The
selector does not read device lengths back to host during capture or replay.

Sixteen CPU tests pass, covering opt-in/default/unprepared dispatch, T>1,
multiple requests, table capacity, empty inputs, argument validation,
idempotent preparation, cross-device rejection and the capture guard.
NPU tests exercise this interface through dynamic graph replay with changing
inputs/metadata and stable workspace addresses, and verify native fallback
graphs for unprepared, B2 and T2 calls. The complete 33-test suite passed in
24.90 seconds; see
[candidate_correctness_r12_optional_dispatch.xml](candidate_correctness_r12_optional_dispatch.xml).
This integration result is separate from earlier standalone wrapper results
below and from subsequent model-level TP8 validation.

## Measured results on Ascend910B3

The complete r12 operator build and editable install succeeded. The current
position-mask implementation passed all 21 tests, with results archived in
[candidate_correctness_r12_position_mask.xml](candidate_correctness_r12_position_mask.xml).
The suite includes CPU reference tests, eager NPU cases, and dynamic graph
replay. Cache tests include both gapped strides and nonzero storage offsets;
replay includes invalid physical page IDs and zero valid length. A CPU case
also verifies exact adjacent position IDs above 2^24.

The three-shape whole-selector measurements are archived in
[candidate_graph_r12_position_mask.json](candidate_graph_r12_position_mask.json).
Times below are median / p95 microseconds. Three alternating rounds each use
12 event samples; candidate graph unroll is 64 and dense unroll is 4, matching
the archived dense measurement. No threshold was changed.

| Context | Candidate | Live dense | Frozen latency | Live latency | Noise |
| ---: | ---: | ---: | :---: | :---: | :---: |
| 4097 | 69.809 / 70.092 | 99.703 / 103.860 | PASS | PASS | FAIL |
| 32771 | 81.312 / 81.710 | 128.675 / 130.730 | PASS | PASS | PASS |
| 131075 | 81.094 / 81.661 | 199.698 / 206.855 | PASS | PASS | PASS |

Noise acceptance includes the round-median spread of every measured stage,
as well as whole selectors. For context 4097, only dense spread fails at
3.2898%; candidate spread is 0.5000% and every other stage is below 3%.
Passing latency does not waive noise acceptance. The separately scheduled
repeat below retained the same thresholds; this failed run remains archived.

| Context | Prepare | Gather | BMM | Score | Topk/remap/sort |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4097 | 10.737 | 12.756 | 10.299 | 7.237 | 27.488 |
| 32771 | 10.656 | 18.658 | 10.205 | 12.829 | 30.509 |
| 131075 | 10.640 | 18.738 | 10.025 | 12.753 | 30.614 |

These independently captured stage medians need not sum to the whole-selector
median. Final selection remains the largest individual stage, but the
position-mask change removed approximately 145–151 us from that stage.

### Context 4097 stability repeat

After the TP8 model test released the devices, context 4097 alone was rerun
with the identical implementation, baseline file, three alternating rounds,
12 event samples, graph unrolls, and all-stage noise gate. No concurrent NPU
performance job ran. The result is archived independently in
[candidate_graph_r12_position_mask_4097_repeat.json](candidate_graph_r12_position_mask_4097_repeat.json).

| Context | Candidate median / p95 (us) | Live dense median / p95 (us) | Frozen latency | Live latency | Noise |
| ---: | ---: | ---: | :---: | :---: | :---: |
| 4097 repeat | 69.472 / 69.789 | 91.120 / 93.050 | PASS | PASS | PASS |

Candidate round-median spread is 0.0396%; live dense spread is 0.8642%.
The maximum spread over every measured stage is 1.7442% for gather, below
the frozen 3% limit. Final topk/remap/sort median is 27.529 us. This establishes
a passing repeat for context 4097, together with the existing passing results
for contexts 32771 and 131075. It does not erase the earlier noise failure or
establish that every future run will meet the stability limit.

Reproduction of the repeat:

```bash
../.venv/bin/python tests/e2e/single_node/ops/benchmark_indexer_v41_candidate.py \
  --lengths 4097 --stages \
  --output benchmarks/deepseek_v41/indexer_v41/candidate_graph_r12_position_mask_4097_repeat.json
```

## Selection profiling and next experiment

The first run, [candidate_graph_r12.json](candidate_graph_r12.json), measured
candidate medians 218.015 / 230.597 / 230.492 us. Its short-context noise gate
also failed. The diagnostic in
[candidate_post_stages_r12.json](candidate_post_stages_r12.json) isolated the
combined expression `torch.where(values > -torch.inf, mapped_int32, sentinel)`
at 151.125 us for context 32771. Topk was 9.942 us, INT32 gather 3.684 us,
FP32 sort including conversion 6.280 us, and restoring output INT32 4.238 us.
Advanced INT32 indexing was slower than gather at 15.796 us.

Moving the mapped-ID conversion before `where` preserves exact IDs when the
static context bound is at most 2^24. However,
[candidate_graph_r12_fp32_select.json](candidate_graph_r12_fp32_select.json)
shows no meaningful latency improvement: medians remained 218.193 / 230.342 /
230.655 us and all three latency gates failed. The original expression profile
does **not** establish that INT32 `where` itself is responsible: it also includes
the comparison against negative infinity. This distinction remains unresolved.
No performance improvement is attributed to that conversion alone.

The replayable profiler now measures the comparison and `where` separately,
including a preallocated negative-infinity tensor comparison. Its finite-bound
comparison is diagnostic only; no weaker validity predicate is used in the
selector. Run when a device is reserved for this experiment:

```bash
../.venv/bin/python tests/e2e/single_node/ops/benchmark_indexer_v41_candidate_post.py \
  --output benchmarks/deepseek_v41/indexer_v41/candidate_post_stages_next.json
```

The current Python-only follow-up obtains validity from `original >= 0`.
Gather encodes every invalid candidate, page, causal lane and padding lane as
position -1. Score adds no other explicit invalidity rule. For finite numerical
inputs in the reference contract, this keeps every legal negative score and
avoids comparing scores against negative infinity. The CPU boundary check uses
the most negative finite FP32 score, which a finite score-threshold shortcut
would incorrectly discard. This follow-up passed all 21 NPU/CPU tests in
26.08 seconds; see
[candidate_correctness_r12_position_mask.xml](candidate_correctness_r12_position_mask.xml).
It reduces whole-selector medians to 69.809 / 81.312 / 81.094 us, as detailed
above. The measured gain belongs to replacing score-based validity with
position-based validity; the standalone comparison/where attribution still
requires the expanded profiler.

This expanded profiler has not yet run. Historical detailed-stage data came
from the preceding diagnostic script. All three experimental B1 shapes now
have a run passing the unchanged correctness, latency and noise requirements.
The experiment does not resolve the existing non-candidate B1 long-context
gate, validate B8/B32 performance, establish full-model quality, or enable the
candidate path by default. Model-level TP8 NPU graph validation is reported
separately from these operator and dispatch tests.

## TP8 opt-in integration

Installed r12 kernel objects, op API/binding libraries, build log and current
integration sources are fingerprinted in
`r12_candidate_integration_manifest.json`. This is a post-integration snapshot;
it does not reconstruct the earlier benchmark's Python revision.

`../runner_tp8/graph_40_candidate_production.json` records a successful
40-layer/E8 synthetic-weight run with 1152/1160-token prompts, prefix caching,
strict HCCL and explicit NUMA `[6,7,4,5,0,1,2,3]`. All eight ranks captured
four candidate consumers and the execution log confirms actual ACL graph
replay. All 12 generated tokens and selected log probabilities are bitwise
identical to the historical native-selector `graph_40_long.json` fixture;
the first two requests also repeat bitwise within the new instance.

All eight Engram shutdown RPCs released their owners and explicit 30-second
client shutdown left EngineCore exitcode 0. Log:
`/tmp/v41-runner-40-candidate-production.log`. The integration covers
production model dispatch and the complete 40-layer source/consumer chain,
but uses synthetic device weights, E8 and small synthetic host tables. It
neither establishes real E384 full-model quality nor passes an end-to-end
latency gate; the option remains disabled by default.

Reproduction from the workspace root:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=30 HCCL_DETERMINISTIC=strict \
.venv/bin/python vllm-ascend/benchmarks/deepseek_v41/smoke_runner_tp8.py \
  --layers 40 --experts 8 --prompt-length 1152 --graph --candidate-decode \
  --engram-numa-nodes 6 7 4 5 0 1 2 3 --repeat-batch \
  --output vllm-ascend/benchmarks/deepseek_v41/runner_tp8/graph_40_candidate_production.json
```
