# V4.1 DSpark NPU graph implementation and acceptance

DSpark is required in the delivered configuration. Ordinary autoregressive
execution is a comparison baseline, not a substitute. Draft graph support is
required; the present eager-only proposer and closed production admission are
unfinished work.

## Verified starting point

The three real draft blocks pass independent stage oracles (contexts 9, 33,
129). The original AICPU metadata path fails at context 33 in real proposer
integration; its root cause remains unresolved. The specialized AscendC
replacement passes 58 NPU tests, including multi-request final-attention
comparisons and 96 changed-input graph replays. Uninstrumented TP8 proposer
r9 then passes all 15 cases on eight ranks, including context 33, with exact
CPU Markov selection and clean teardown. This replacement does not change attention
visibility or relax production validation. No serving, acceptance-rate or
complete draft graph result is claimed.

Upstream vLLM at 836bb3839ffefcda8283ea7d41671a89e1a613df captures the
parallel draft backbone and sequential Markov sampling in
`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`. Its DFlash base explicitly
executes context KV projection outside that graph. Therefore copying upstream
query capture alone does not put all DSpark computation into NPU graph.

## Capture boundaries

The working implementation is `spec_decode/dspark_v41_graph.py`, attached by
the V4.1 proposer when FULL graph mode is configured. It does not depend on
`torch.compile`: both families capture native operations directly. Context
row and query request buckets are independent powers of two plus their exact
capacity endpoints. Startup warms and captures every bucket; requests before
capture raise an error. Raw auxiliary states, positions, boundaries, lengths,
page tables and output proposals have persistent storage. Padded context slots
are -1; padded query boundaries repeat the active terminal offset and have
zero lengths. Query capture includes visibility and AscendC scheduling.

Initial execution scope is K5/anchor-first, greedy draft, DP1/CP1 and no LoRA.
TP8 real-weight graph validation is in progress with B1/2/3/4, including B3 in
the B4 bucket. The r3 run captured and replayed both families, but comparing
9-row eager context with a 16-row graph bucket was not bit-exact in BF16 logits
and KV, despite identical proposals. This failure is retained. The next run
adds an eager execution at the identical bucket shape to distinguish padding
numerics from replay errors, retaining the original unpadded proposal check.
This is not yet full graph acceptance or a performance result.

The r4 same-bucket comparison is exact for proposals, logits and all three KV
caches in its first four cases (B1 contexts 9/33/129, then B3 with contexts
9/33/129 and rejections 0/1/2). It stops at that B3 case because one request's
proposals differ from the original unpadded eager execution. The strict gate
remains failed. Per-rank journals and failures are retained in `dspark_graph/`.
A single-NPU real-weight context projection probe independently confirms
shape-sensitive BF16 output: 171 versus 256 rows changes 132 of 875,520 values,
maximum absolute difference 0.0078125. Both paths have nearly identical error
against an FP32 reference. This isolates an initial perturbation, not the
cause of every downstream difference. Per-layer observation is prepared to
locate amplification in attention/MoE. The explicit benchmark-only
`--graph-padding-diagnostic` can continue same-bucket checks while recording
unpadded drift; its result is diagnostic and cannot satisfy graph admission.

R5 completes all 11 TP8 cases in that diagnostic mode: all observed stages,
logits, KV and proposals are exact against same-bucket eager. Each rank runs
22 actual replays per graph family; all 120 tokens match the CPU Markov oracle
and every worker exits cleanly. Original unpadded proposals differ in three
cases. The first large amplification in case 3 occurs in the second block's
MoE; expert-choice analysis and real-target acceptance remain required. See
[graph diagnostic results](DSPARK_GRAPH_RESULT.md) for scope and evidence.

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
