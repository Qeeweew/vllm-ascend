# V4.1 main attention on 910B

The integration uses the in-tree `SparseFlashMla` AscendC operator through
`vllm_ascend/ops/dsa_v41.py`. Projection GEMMs, normalization, cache writes,
RoPE and inverse RoPE remain separate. It performs one joint SWA + CSA
normalization with a denominator-only FP32 sink for each local query head.

## ABI and limitations

| Input | Contract |
| --- | --- |
| Q | BF16 `[T,8,512]` for TP8; already rotated last64 |
| SWA KV | BF16 `[blocks,block_size,1,512]`; K equals V |
| Compressed KV | Same layout, single KV head, source-layer cache may be shared |
| Block size | Multiple of 16, from 16 through 1024 |
| Block tables | INT32 logical-page to physical-page mapping; original and compressed token units respectively |
| Compressed selection | INT32 `[T,1,512]`, increasing logical row IDs, unique valid prefix and trailing -1 |
| Compression | CR0 disables CSA; CR1 and CR2 use completed compressed groups only |
| SWA mask | Native right-down causal window, left127/right0 |
| Sink | FP32 `[8]`, denominator only, included once across both KV sources |
| Output | BF16 `[T,8,512]`, before inverse query RoPE |

Compressed row `j` must use RoPE position `j * CR` (group first). The main
cache receives compressed latent after its indexer consumer and before
attention reads. The official Q path only normalizes low-rank `qr`; it does
not normalize each final query head.

The older `SparseAttnSharedkv` tiling accepts only CR4/128. The in-tree
`SparseFlashMla` CSA path accepts CR1/2 on arch22, but rejects simultaneous
explicit SWA indices and compressed indices. This wrapper therefore uses
native SWA masking and logical paged tables; it never substitutes V4 CR4
semantics or interprets CUDA physical indices as logical IDs.

A SWA ring can alias logical pages when their live token offsets do not
overlap. It must retain the current query chunk plus the preceding 127
tokens until all queries finish; blindly overwriting a 128-token ring with
a longer prefill chunk loses keys required by early queries. Tests cover
both decode and a 33-token chunk with modulo-addressed ring storage.

## Cache accuracy

This BF16 path is a baseline requiring model quality evaluation. It is
not bit-equivalent to every official implementation. The standalone model
reference (`inference/model.py`) uses:

- SWA applies whole-vector FP8 quantization after RoPE, including last64.
- Main compressed KV applies FP4 group16 with E4M3 scales after group-first
  RoPE. Indexer FP4 group32/E8M0 is a different numerical format.

The checked vLLM CUDA commit differs from that standalone reference: its
`rope_quant_insert` selects MXFP8 all512/group32 for the SM100 528-byte
record, older mixed 448-FP8 + 64-BF16 for other FlashMLA records, or plain
BF16/per-tensor FP8 rows for FlashInfer. It does not apply the standalone
main-KV FP4 group16 rounding in this path.

BF16 storage neither reinterprets packed bytes nor reproduces all these
quantization choices. Kernel correctness versus BF16 inputs is insufficient
to establish quality; comparisons must name the reference backend and format.

## Graph and validation

Metadata has fixed-address device tensors. When requests or lengths change,
refresh the query offsets, both block tables, original lengths, compressed
lengths, CR2 residuals and native scheduling buffer in place. The derived
compressed lengths are not views of the original length tensor. Refreshing
only the latter silently reuses stale causal boundaries during replay.
Direct NPU graph capture is tested. The native op currently has no Meta/Fake
registration, so direct `torch.compile` tracing requires an additional
registration or the normal opaque vLLM attention boundary.

The independent oracle gathers BF16 values on CPU, concatenates visible
SWA and selected compressed rows, then evaluates one FP32 softmax with the
sink. Output normalized RMSE must stay below 0.006, with additional
elementwise bounds (absolute 0.012, relative 0.025); LSE has absolute 0.015
and relative 0.003 bounds. Tests cover full shapes H8/D512, CR0/1/2, partial
CR2 groups, mixed
query lengths, sparse gaps, all-empty compressed selection, sink domination,
permuted pages, noncontiguous physical-block strides, ring page aliasing,
and graph replay with changed lengths, inputs, tables and sparse selections.

The coordinated r7 clean install passed all 24 device tests on 910B3 device 2
(including three graph cases and six ring cases), plus eight CPU ABI tests.
Warm and cold device performance measurements passed the operator gates below. Kernel correctness does not establish model
quality or full-model performance.

## Performance acceptance

The benchmark compares native attention against a batched device gather,
CANN BF16 BMM with FP32 score output, joint softmax and BF16 value BMM.
Both consume identical logical indices and caches; neither includes index
selection, cache write, projection or scheduler construction. It reports
decode B1/4/8, CR0/1/2, eager and graph, median/P95, five alternating rounds,
and the conservative slowest-native-round versus fastest-baseline-round
speedup. Warm graph captures 32 invocations and divides device event time by
32 to amortize host launch gaps; this is not single-request wall latency.
Cold mode forces one invocation per graph so every timed attention sees
the preceding cache flush. The default is warm cache; `--cold` reads/writes
512 MiB outside each timed
interval (larger than twice device L2). Frequency is not locked.

Acceptance requires correctness first, no graph median or P95 regression
against the composed baseline at any production decode shape, and at least
1.25x geometric-mean graph improvement. Missed gates must be reported, not
hidden by averaging with favorable cases. Operator-only results do not
establish 8-card model latency: the final profile must include projection,
cache updates, selector, metadata preparation, inverse RoPE and TP output.

```bash
../.venv/bin/python -m pytest --confcutdir=tests/e2e/single_node/ops \
  tests/e2e/single_node/ops/test_dsa_v41.py -q
../.venv/bin/python tests/e2e/single_node/ops/benchmark_dsa_v41.py \
  --output benchmarks/deepseek_v41/attention_910b3.json
```

## r7 warm results

Raw five-round results are in `attention_910b3.json`; the preliminary
single-replay measurement is retained in
`attention_910b3_single_replay.json` and is not used as pure kernel latency.

| CR | Batch | Native median / P95 (us) | Composed median / P95 (us) | Speedup |
| --- | --- | --- | --- | --- |
| 0 | 1 | 22.33 / 22.62 | 66.62 / 66.84 | 2.98x |
| 0 | 4 | 22.88 / 23.02 | 91.81 / 92.06 | 4.01x |
| 0 | 8 | 23.74 / 23.88 | 109.35 / 109.59 | 4.61x |
| 1 | 1 | 41.14 / 41.40 | 131.32 / 131.60 | 3.19x |
| 1 | 4 | 37.74 / 38.33 | 161.97 / 162.64 | 4.29x |
| 1 | 8 | 41.10 / 41.54 | 190.11 / 190.77 | 4.63x |
| 2 | 1 | 38.75 / 39.52 | 131.31 / 131.97 | 3.39x |
| 2 | 4 | 40.27 / 40.71 | 163.11 / 163.71 | 4.05x |
| 2 | 8 | 39.09 / 39.48 | 189.69 / 190.18 | 4.85x |

Equal-case geometric mean graph speedup is 3.95x; the minimum
conservative round speedup is 2.96x. All nine warm graph median and P95
cases beat the composed baseline. This establishes a useful existing
AscendC path without adding a new attention kernel. Full-model and cache
quantization quality acceptance remain outstanding.

## r7 cold results

Raw five-round results are in `attention_910b3_cold.json`. The 512 MiB
read/write flush runs before each measured invocation. It queues work ahead
of the timing events and can hide host launch gaps; cold eager timings
must not be compared directly with isolated warm eager latency. Neither
timing includes the flush itself.

| CR | Batch | Native median / P95 (us) | Composed median / P95 (us) | Speedup |
| --- | --- | --- | --- | --- |
| 0 | 1 | 25.70 / 26.44 | 92.63 / 94.44 | 3.60x |
| 0 | 4 | 25.98 / 26.76 | 115.18 / 116.68 | 4.43x |
| 0 | 8 | 27.23 / 27.84 | 125.56 / 126.84 | 4.61x |
| 1 | 1 | 41.88 / 42.64 | 154.01 / 156.26 | 3.68x |
| 1 | 4 | 43.18 / 43.86 | 182.10 / 183.88 | 4.22x |
| 1 | 8 | 43.68 / 44.02 | 203.79 / 205.80 | 4.67x |
| 2 | 1 | 41.48 / 42.16 | 155.72 / 157.84 | 3.75x |
| 2 | 4 | 44.98 / 45.62 | 186.04 / 187.88 | 4.14x |
| 2 | 8 | 45.61 / 46.28 | 205.44 / 207.58 | 4.50x |

Equal-case geometric mean cold graph speedup is 4.16x; minimum
conservative round speedup is 3.58x. All nine cold graph cases beat the
composed baseline at median and P95.

## Runner metadata integration

`vllm_ascend/attention/dsa_v41.py` supplies a separate V4.1 metadata
backend for SWA, compressed main KV, and compressed index caches. Cache
layers return `AscendV41CacheBackend`; the runner constructs its builder
with the explicit V4.1 cache spec. Logical block sizes count original
tokens, while physical rows are `block_size / tokens_per_state`.

Before each model invocation or graph replay, the builder copies positions,
device query boundaries, lengths and full logical block tables into fixed
buffers. It derives request IDs and physical write slots from device query
boundaries, ignoring generic compressed slot IDs and CPU query lengths.
CR2 caches publish only completed groups. Native scheduling runs once per
cache-group build. Source and consumer layers must reuse their configured
group metadata rather than rebuilding schedules per layer.

The model resolves source-layer identity, assembles main attention with
`make_v41_attention_metadata(swa_metadata, main_metadata)`, and assembles
selection with `make_v41_indexer_metadata(index_metadata)`. Cache write
slots come from the corresponding metadata object. Source candidate and
top-k buffers belong to the model. The backend's graph-parameter update
hook is a no-op because the builder already refreshes fixed tensor
addresses before replay. Each builder supports one in-flight batch;
overlapping execution needs separate buffer slots.

Full logical-page columns are required even when expired pages are marked
invalid or ring pages alias older pages. A rebased sliding-window table
cannot be indexed by absolute positions. SWA storage must preserve the
current query chunk and its preceding 127 tokens. The current implicit
causal SWA path establishes a text baseline: image-span widened or
bidirectional SWA still needs implementation. Native attention lacks a
Meta/Fake registration, so direct NPUGraph coverage does not establish
`torch.compile` support; compiled runner integration still needs an opaque
attention boundary or equivalent Fake/Meta support.

Validation passed 12 CPU metadata tests and 8 device component tests on
the r7 installation. The device cases cover five cache-role/ratio
combinations, device/CPU boundary mismatch, padding, fixed-address
refresh, and CR0/1/2 attention graph replays against independent FP32
references. These tests do not constitute full-model runner validation.
The performance tables above exclude this builder: its copies, PyTorch
slot arithmetic and AICPU scheduling must be included in model profiling
before making a serving-latency claim.

The registered-cache component suite additionally exercises the actual
32 KiB V4.1 grouping planner, upstream allocation descriptors, runner
allocation/reshape, the actual Ascend plugin cache-binding hook and each backend's metadata
builder. Forward resolves the registered cache dictionary without an
explicit `DeepseekV41AttentionBatch`. Cases cover layers 0/1, 2/3, 2/8/9
and 20/24/25: SWA-only, CR2 compression, main-KV source changes and a
later index-only source consuming source-20 main/index caches and candidate
blocks. Graph replay changes hidden inputs and physical page assignments
while keeping storage addresses fixed. Group page IDs are distinct because
the scheduler's shared pool overlays cache-group allocations.

The binding regression also checks that the runner retains the original
allocated views while each cache layer's binding method runs. In particular,
CR2 circular storage arrives as `[blocks,1,8,1024]` and the compressor consumes
its contiguous `[blocks,8,1024]` view. Bypassing the layer binder with direct
assignment is covered by the device suite.

These cases use reduced dense projection widths with native H8/D512/D128
cache, attention and indexer shapes. Output is checked against independent
FP32 attention/compressor math, with normalized RMSE below 0.012 in addition
to elementwise bounds. TP reduction is replaced by an identity, so this
establishes registered-cache component correctness, not 8-card collectives,
full serving integration or production-shape performance. The test is
`tests/e2e/single_node/ops/test_attention_v41_registered_cache.py`.

The first full-runner request exposed a newer upstream dataclass ABI change:
`CommonAttentionMetadata` no longer declares the legacy CPU cache fields.
`AscendCommonAttentionMetadata` now explicitly owns `_seq_lens_cpu`,
`_num_computed_tokens_cpu` and `dcp_local_seq_lens_cpu` for existing Ascend
consumers. Async mode keeps public CPU lengths unset and device lengths
authoritative; CPU upper bounds remain separate. Unpadding preserves new
upstream fields, slices per-request bounds and clears padded device-derived
caches. Six regression cases execute the real runner constructor and V4.1
builder without mocking the common-metadata class. All six passed, along
with the existing unpadding test and 12 V4.1 metadata tests.
