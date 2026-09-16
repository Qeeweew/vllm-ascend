# V4.1 DSpark NPU graph implementation and acceptance

DSpark is required in the delivered configuration. Ordinary autoregressive
execution is a comparison baseline, not a substitute. Draft graph support is
required; the present eager-only proposer and closed production admission are
unfinished work.

## Verified starting point

The three real draft blocks pass independent stage oracles (contexts 9, 33,
129). Real proposer integration still fails in SparseFlashMlaMetadata. The
AICPU diagnostic channel now passes a device self-test; the guarded TP8 run
must identify the actual failure before any production validation changes.
No serving, acceptance-rate or draft graph result is claimed.

Upstream vLLM at 836bb3839ffefcda8283ea7d41671a89e1a613df captures the
parallel draft backbone and sequential Markov sampling in
`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`. Its DFlash base explicitly
executes context KV projection outside that graph. Therefore copying upstream
query capture alone does not put all DSpark computation into NPU graph.

## Capture boundaries

Use two independently bucketed graph families, ordered on the same stream:

1. Context graph: combine target layers 37/38/39 through main projection and
   normalization, project/store context KV for all three draft layers. Bucket
   by target context rows. Pad positions safely and set unused per-group slots
   to -1, so rejected/padded rows never modify live cache entries.
2. Query graph: K5 query embedding, three E128/top3 draft blocks, final
   normalization and draft vocabulary projection, then all five sequential
   Markov steps. Include confidence computation when enabled. Bucket by padded
   request count, with exactly 5 query rows per request; never reuse the target
   model's uniform decode length as the draft block width.

The number of context rows and the number of query rows vary independently.
The existing `_run_merged_draft` reads Python `_dflash_num_context` and slices
context buffers during capture. A graph keyed only by query rows would freeze
the wrong context extent on later requests. Move context computation to its
own graph before enabling query replay; do not merely delete the forced eager
assignment in `AscendDSparkProposer.__init__`.

## Runner and metadata changes

- Allocate persistent auxiliary, context, seed, sampling, per-group query and
  context-slot buffers before capture. Copy current contents before replay;
  allocation addresses and layout must remain stable across batch changes.
- Build dummy metadata for every draft attention/cache group with K5
  noncausal visibility. The current DSpark dummy path supplies no attention
  metadata and cannot serve as a valid capture template.
- Reuse each V4.1 builder's persistent schedule, page table, SWA candidate and
  length buffers. Refresh current contents before graph replay, including
  rejected-token correction and padded tails. A zero-length padded request
  must neither read invalid pages nor publish cache rows.
- Keep returned proposal storage persistent; slice active requests outside
  capture. Captured LMHead, Markov loops and confidence must use bucket sizes,
  not Python values from the first real batch.
- Keep TP8 collective shapes identical across ranks. Capture and replay the
  target and draft graphs in a defined stream order. Host Engram staging must
  finish before target replay; draft must consume the current target aux.
- Default to graph execution for the final supported configuration. Unsupported
  configurations must fail explicitly, without silently disabling DSpark.

## Required evidence

| Gate | Workload and assertion |
| --- | --- |
| Native metadata | Uninstrumented TP8 proposer; context 9/33/129 and 255/256, rejection 0–5; all ranks clean up |
| Query correctness | Compare eager and graph proposals, per-stage outputs and Markov order using the same weights and inputs |
| Changing inputs | Repeated replay with changed anchors, positions, lengths, page tables and target auxiliary states |
| Independent buckets | Hold query batch fixed while changing context rows, and reverse; test padded multi-request batches |
| Cache rollback | Perturb rejected auxiliary rows, verify unchanged proposals; inspect live and padded cache slots |
| Actual capture | Associate native submissions with the concrete context/query graph objects and verify real-request replay of those objects |
| Full target | Real 40-layer target and host Engram, real draft, target verification, acceptance/rejection and graceful cleanup |
| Performance | `vllm bench` with DSpark enabled; TTFT, TPOT, output throughput, acceptance length and memory; report graph scope |

CPU mocks or capture counters without graph identity and changed-input replay
cannot satisfy the graph gate. Diagnostic synchronization must be absent from
final correctness and performance runs.
