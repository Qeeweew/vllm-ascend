# Historical B1 fused-candidate experiment contract

Status: **superseded scope; not overall user acceptance**. The frozen B1
thresholds below remain the historical experiment contract and its measurements
are retained. The user requires actual fused multi-batch CR1 consumer/source
and CR2 producer coverage; that implementation and overall acceptance are pending.
B8/B32/CR2 running the legacy path are regression controls, not fused coverage.

## Scope and implementation boundary

Preserve `npu_quant_lightning_indexer_v3`. Specialize only Ascend 910B,
one batch and one query, CR1 candidate consumer, 32 INT8 query heads, D128,
2048 candidate blocks of eight positions, and top-k 512. Other shapes retain
the existing native dispatch, including CR2, B8 and B32.

One mixed AscendC kernel performs candidate sorting, duplicate suppression,
page validation, direct paged INT8 loads to L1, INT8 QK, FP16 ReLU/dequant,
FP16 head-weight Cube reduction, key scaling, local top-k and global merging.
QK scores stay in Score L1. No full gathered BF16 key or full QK tensor is
materialized. Caller-side metadata preparation and final index sorting, if
required by the existing wrapper, count toward selector latency.

Reference source: `G_W_E/ops-transformer`, branch `qli_opt`, commit `2ed905f`.
The Cube pipeline uses its corrected Score L1 stage ownership, paired WS
tiles and three-slot key prefetch. Its published 64-head/top-k-2048 timings
are not evidence for this 32-head/candidate workload. In particular its
report retains B1/B2 end-to-end regressions despite faster kernel time.

## Correctness

Use the existing CPU original-position set oracle in
`tests/e2e/single_node/ops/indexer_v41_candidate_reference.py` unchanged:
exact INT8 dot; ReLU(dot/1024) rounded FP16; weights times query scales
rounded FP16; FP32 head reduction and final key-scale multiplication.
Only cutoff exchanges within the existing gamma-38 FP32 error bound are
allowed. No recall-percentage tolerance. Results must be unique, ascending,
with exactly min(512, valid positions) entries and then -1 padding.

Cover signed weights; duplicate/negative/out-of-range candidate IDs;
shuffled and invalid physical pages; actual axis-zero gaps and storage
offsets; partial last blocks; zero query/length; lengths 1, 17, 511, 4097,
32771 and 131075; graph replay with changed Q, weights, scales, lengths,
candidate membership and page tables. Preserve original CR2/B8/B32 tests.

## Performance and memory

For each B1 context 4097, 32771 and 131075 separately:

- Whole-selector median and P95 must each be <= 90% of the alternating live
  r12 split selector, measured on identical inputs after correctness passes.
- Existing archived dense limits in `candidate_frozen_baseline.json` and
  alternating live dense limits remain binding (median <= dense median;
  P95 <= 1.05 times dense P95).
- Three alternating rounds, 12 event samples each, graph unroll 64 for
  candidate selectors and four for dense. All round-median spreads <= 3%.
- CR2/B8/B32 whole-selector median and P95 must not regress more than 3%
  against their unchanged live native baseline.
- User workspace <= 256 KiB, independent of declared context capacity.
  Report the CANN fixed API workspace separately and include it in total
  peak allocated and reserved HBM. Whole-selector peak incremental allocated
  memory must be lower than both the live split and original native paths.

Failures are retained. No averaging away a failed shape, relaxing thresholds
after a run, or using matmul-only latency to claim acceptance. Report kernel
profiling, whole-selector median/P95, explicit workspace, allocated/reserved
HBM and source/build hashes. Full-model performance is a separate gate.
