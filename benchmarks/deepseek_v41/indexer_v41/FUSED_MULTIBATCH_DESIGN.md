# CR1 consumer: one Vector owner per complete query

This document describes the earlier generic mode2 implementation. The current
trusted-unique mode4 design is [PAGED_UNIQUE_DESIGN.md](PAGED_UNIQUE_DESIGN.md).
The separate continuous K experiment has been withdrawn and removed.

Status: implementation in progress; supersedes the r14 B1 experiment and the
unimplemented external score / torch.topk proposal. No performance acceptance
is claimed. CR1 source and CR2 remain separate pending workloads.

## Interface and numerical contract

Keep `npu_quant_lightning_indexer_v3` unchanged: TND INT8 query `[T,32,128]`,
paged INT8 key, FP16 query/key scales and weights, INT32 candidate block IDs
`[T,1,2048]`, `cu_seqlens_q`, `seqused_k`, pagetable; return `[T,1,512]` indices.
Use the existing wrapper's ascending-position postprocessing in both benchmarks
and the model. The native result is score ranked. Invalid outputs are -1.

The work unit is a **query row**, not a batch entry. Find request b with
`cu_q[b] <= t < cu_q[b+1]`; repeated boundaries are empty requests. For CR1,
`visible = max(0, seqused_k[b] - (cu_q[b+1]-cu_q[b]) + t-cu_q[b] + 1)`.
Graph padding rows at or beyond `cu_q[B]` produce only -1. Sort/deduplicate
candidate block IDs once per query, validate logical block and physical page
before reading K. The partial newest block is masked position by position.
Candidate source's block8 maxima and independent CR2 dense selection are not
implemented by this specialization and must not be counted as fused coverage.

## Producer and consumer assignment

Let S be split-N width, P the used physical AIC count, and G=P/S the number of
independent worker groups. The first executable baseline chooses the largest power of two S<=8 such that
T*S<=available AICs; if T already fills the device, S=1. P=min(available AICs,
T*S), rounded down to a multiple of S. Sweep S=1/2/4/8 at small T and only
consider 16 if measurements justify it. Freeze thresholds after measurement;
the default is a hypothesis, not an accepted tuning result.

Group g processes rows `t=g, g+G, ...`. AIC `g*S+s` scores a contiguous
16384/S-position range for that row. Within this range QK uses only that query's
Q and candidates; there are no useless cross-request matrix products. Q and
head weights are loaded once per AIC/query and retained while it loops N tiles.
For T>=20, S=1 gives a complete query per AIC; for small T split-N uses otherwise
idle AICs. Each row has exactly one topk owner. With S>1 it is odd AIV paired with
AIC g*S. With S=1 both paired AIVs alternate ownership by row iteration,
using all 40 AIVs for prefill. The owner consumes all 16384 scores and performs the complete topk512 itself. The
implementation may sort 1024-element segments and merge their best512 **on
that same AIV**; there is no cross-AIV local-topk reduction.

## Initial preparation

All AIVs share preparation by striped query rows (row=aiv_id, aiv_id+2P, ...).
Each prepares expanded weights, physical offsets, position IDs and gathered
key scales once per query. Other workers do not repeat its candidate sort. All queries have
distinct GM records for the entire invocation. Each AIV initializes its
own 32-byte IB mailbox to zero on **every graph replay**. One AIV-wide barrier
publishes preparation and clears mailboxes; both paired AIVs then publish READY
using MODE2/PIPE_MTE3. AICs wait READY once before accessing any prepared data.
No AIV accesses AIC L1; all handoffs use GM.

## Per-query completion and bounded overlap

Three MODE2 flag IDs are used: READY, SCORED and ACK. They route only within
one physical AIC/two-AIV pair. Cross-group notification uses the actual arch22
CANN `IBSet<false>` / `IBWait<false>` implementation and 32-byte GM mailboxes,
not an unavailable scalar AtomicAdd/fetch-add or arbitrary MODE2 routing.

For each query assigned to a physical AIC:

1. AIC computes its complete N partition and publishes SCORED on PIPE_FIX,
   after the final reduced-score GM store. It then waits ACK.
2. Both paired AIVs consume that SCORED exactly once. Even AIV relays completion
   to its mailbox with IBSet if S>1, then publishes its MODE2 ACK contribution.
3. With S>1, non-owner odd AIV publishes its ACK immediately. The owner odd
   AIV first consumes all S group mailboxes with IBWait (each resets 1 to 0), then
   publishes ACK **before** starting its full-query topk. With S=1 both
   AIVs ACK after SCORED, and the alternating owner starts its full-query topk.
4. AIC receives ACK only after both paired AIV contributions. It can now score
   the next query while the owner is sorting the preceding query. Owner odd
   AIV next returns to SCORED after finishing topk/output. Consequently an AIC
   can run one query ahead; MODE2 counters and IB mailboxes cannot grow
   unbounded. S=1 needs no IB relay/remote wait.

No per-query SyncAll is used. Unequal loop counts and idle workers are legal.
Even relay AIVs never execute topk, so owner waiting for another producer's
mailbox cannot block that producer's notification. Dependencies either stay
within the same query or point to an earlier query's ACK. Distinct per-query
GM records make early ACK safe: the next query cannot overwrite scores the
owner is still reading. No terminal global barrier is needed; all paired
SCORED/ACK counts are consumed before exit.

For S>1, IBSet may wait for its previous query's mailbox reset; the owner always
resets every partition before beginning topk and before acknowledging its own
AIC. This is the explicit backpressure edge; there is no cyclic dependency on
future producer work. Hardware scheduling uses at most the resident AIC/AIV
counts, never oversubscribed spinning blocks.

## Memory budget and tile study

Initial per-query GM record: expanded weights 1024 B; 2048 uint64 physical
block offsets 16384 B; logical block first-position IDs 8192 B; gathered key
scales, 16 FP16 slots per block, 65536 B; 16384 FP32 reduced scores 65536 B;
aligned validity/length descriptor 32 B. Total 156704 B/query, plus 64 B per
physical AIC for IB mailboxes. QK `[32,N]` never goes to GM. A static 256 MiB user-workspace gate bounds
this first implementation; larger T falls back to the existing native path.
This gate depends only on shapes and is stable under graph replay. T1024
uses about 153 MiB; T4096 would require 612 MiB and is therefore not selected.
The CANN fixed workspace is additional. The initial version
intentionally allocates all records for an invocation; a ring is a later
optimization requiring a separate FREE generation. CANN fixed workspace and
user workspace must be reported separately.

A topk owner needs only a 1024-score segment (4 KiB), corresponding IDs (4 KiB),
interleaved sort output (8 KiB), sort scratch (16 KiB), retained best512 pairs
(4 KiB), merge scratch (8 KiB), padded half scales (4 KiB), float scales (8 KiB)
and block IDs (512 B), comfortably below the 192 KiB UB limit. Preparation's
2048-candidate pair/scratch buffers (16+16 KiB), IDs (8 KiB), expanded weights
(1 KiB) and segment address/scale staging similarly fit. The initial implementation allocates exactly 71,840 B UB per AIV, including
shared preparation and topk buffers, below 192 KiB.

Cube baseline N128: triple key L1 48 KiB; H32 QK needs 4 KiB Q, 16 KiB K,
16 KiB INT32 C per tile. N256: triple key L1 96 KiB; QK B and C each32 KiB,
allowing two L0B slots and four L0C slots; paired FP16 WS N512/H32 needs32 KiB B.
Compact H32 score stages use 64 KiB L1. N512 needs64 KiB L0B for one QK tile,
so it loses naive ping-pong and must be benchmarked with a different lifetime
plan rather than just changing a constant. The current inherited N128 service allocates 384 KiB L1 (64 KiB query,
48 KiB triple keys, 16 KiB weights, 256 KiB score stages), 64 KiB L0A,
64 KiB L0B and 128 KiB L0C. Its allocations still reserve unused M rows; the
compact H32 budgets above describe the required N256 redesign, not the current
allocation. Initial N128 reuses the validated
r14 Cube dataflow, including triple key prefetch and ascending contiguous-run
coalescing. N256/N512 are explicit tuning experiments, not preclaimed wins.

## Performance / correctness acceptance still open

The primary workload is **prefill**, not only multi-batch decode. Required
fused evidence: B1 prefill T32/64/128/256/512/1024 at 4K/32K/128K (capacity
permitting), multi-request ragged prefill, mixed prefill/decode, and decode
T1/2/4/8/16/20/32/64 (and feasible128),
short/long mixed lengths, empty requests, multirow cu_q, graph padding, all
invalid candidates, duplicate IDs, invalid physical pages and partial blocks.
Record actual tiling dispatch for every case. Compare complete selector
median/P95 and prefill tokens/s, with matched candidate density and both
native and split gather/score/topk controls. Preserve correctness oracle and
fresh-package/kernel hashes. Source/CR2 regression tests are separate.

Static striped query assignment has at most one extra row per worker group,
but mixed candidate counts can still create tails. Profile active-core MTE2,
Cube, Vector, waits and task duration; report min/median/max per-group work.
If tails exceed the gate, use the already prepared per-row valid counts to
build a deterministic device-side longest-work-first group list before READY;
this is not a host content read and does not change graph shape. Do not claim
that static striping alone solves mixed-length balance. Benchmark this
scheduling cost as part of the operator. Targets must be frozen against fresh
multi-batch measurements before any acceptance claim; B1 historical numbers
are insufficient.

## Prefill bandwidth follow-up

The clear first baseline remains M32 for one query and split1 when T is large.
It can fill the device but may still be key-bandwidth limited. Profile useful
INT8 QK FLOPs (2*32*128*valid_positions) separately from padded Cube work and
FP16 weight-reduction FLOPs; report traffic and whole-device work imbalance.
A subsequent M64/M128 batching experiment may reuse K only when adjacent
queries of the same request have compatible candidate/page tiles. Different
candidate sets must not be silently combined or expanded into mostly useless
cross products. Measure the compatible fraction and the cost of grouping
before implementing this optimization. CR1 source prefill and CR2 dense long-N
prefill remain independent work and have not passed this consumer's gates.

## Frozen targets for the next measured iteration

Root review fixed these targets before collecting the new baseline: complete
selector prefill median and P95 must not regress against the matching legacy
consumer on T32..1024; T>=128 targets at least 1.2x legacy tokens/s. Measure
peak allocated/reserved HBM too, including the all-query records and CANN
workspace. Decode must retain the r14 B1 package as a real control and target
at most 3% regression; an architecture change does not waive its existing
benefit. If a shape regresses, retaining a measured old specialization behind
T dispatch is an implementation option, not grounds for moving the threshold.

Current shared-device runs may establish correctness and diagnose performance,
but cannot establish final latency acceptance. No report from r14 or a legacy
source/CR2 regression substitutes for actual new fused prefill dispatch.

## First forced split sweep: r17, 32K

The isolated split1/2/4/8 packages now resolve metadata using the common r17
vendor as a fallback. Every timed shape first passes the independent candidate
oracle; complete selector time includes ascending output postprocessing.
Median microseconds, N=32771, one request with T query tokens:

| T | Split 1 | Split 2 | Split 4 | Split 8 |
|---|---:|---:|---:|---:|
| 1 | 311.06 | 275.40 | 256.27 | 250.83 |
| 2 | 320.70 | 284.06 | 266.64 | 260.92 |
| 4 | 329.73 | 293.87 | 281.39 | 376.74 |
| 8 | 346.74 | 313.27 | 404.45 | 610.51 |
| 16 | 377.51 | 446.11 | 606.09 | 1050.79 |
| 32 | 504.14 | 663.72 | 977.18 | 1958.95 |

The best split for these shapes is 8/8/4/2/1/1, consistent with the current
T-driven baseline. Oversplitting reduces independent query groups and causes
large regressions. This is a single-context r17 study, **not** an accepted final
threshold: r18/r19 change the workload balance, and 4K/128K plus ragged contexts
still require comparison. Earlier partial JSONs were preserved; resumed
measurements use new output files. See
`artifacts/qli-fused/r17/split-sweep-summary.json` and each
`r17-split{1,2,4,8}-r2/perf-t*-n32771-r*.json` for raw timing samples,
loaded library fingerprints and correctness status. The partial outputs from
an interrupted earlier sweep are not counted as completed measurements.
