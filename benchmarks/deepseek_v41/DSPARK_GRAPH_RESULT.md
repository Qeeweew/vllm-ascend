# V4.1 DSpark context/query graph diagnostic

Status on 2026-09-16: **same-bucket graph replay is exact across the tested TP8
matrix; strict comparison with original unpadded eager remains incomplete**.
Full-target validation remains incomplete. This is not
an end-to-end performance or acceptance-rate result.

## Implemented computation

Separate graph families capture context auxiliary projection/normalization and
three context KV stores, followed by K5 query embedding, visibility/schedule,
three draft blocks, LMHead and all five sequential Markov steps. Context rows
and request counts select independent buckets. Persistent raw inputs are
updated on the current stream; padded requests have empty query boundaries,
zero lengths and invalid cache slots. Startup capture is mandatory; a request
before capture fails instead of falling back. Native FULL graph mode works
without torch.compile.

## TP8 r5 evidence

The fixture uses all three real draft blocks and the real shared embedding and
head, with synthetic target auxiliary states. It tests 11 cases covering
B1/2/3/4, contexts 9–256, rejection 0–5, changing physical page assignments,
B3 padding into B4, and independently changing context/query buckets.

- Each rank captures 10 context graphs and three query graphs, then performs
  **22 actual context and 22 actual query replays**. Graph identities are
  associated with replay method calls, not inferred from configuration flags.
- Same-bucket eager and graph outputs are bit-exact at context projection and
  norm, all three attention/MoE/block outputs, final logits, proposals and all
  three KV caches. Perturbing rejected auxiliary rows leaves graph logits and
  proposals unchanged.
- All ranks agree. The independent CPU sequential Markov oracle matches all
  **120 proposed tokens**. Maximum Torch allocation is 6,174,923,264 bytes,
  below the fixed 8 GiB ceiling. All eight workers clean up; launcher exits 0.
- Original unpadded eager proposals differ in cases 3, 4 and 9. The result is
  explicitly `diagnostic_passed`, with `graph_validated=false` and
  `scheduler_validated=false`. The default strict harness still rejects such
  differences; `--graph-padding-diagnostic` enables this additional analysis.

## Padding sensitivity and actual router choices

R1 failed before capture because the generic proposer required torch.compile;
V4.1 now uses its actual FULL graph configuration. R2 reached context capture
but the wrapper rejected its None return; a tensor-container return fixes that
contract. R3 exposed BF16 differences between 9-row eager and 16-row capture.
R4 established same-bucket equality and stopped on the first unpadded B3
proposal mismatch. All four failure histories are retained.

In r5 case 3, context projection changes by at most 0.015625 in 134 values;
context norm changes by at most 0.0009765625. The maximum difference grows to
0.25 in the second attention block, then 6.0234375 in that block's MoE output.
Every observed same-bucket eager/graph stage remains exact. R5 located a
major amplification at MoE without recording which expert choices caused it.

R6 records actual FP32 router logits, top-k IDs and weights with device-only
copies during capture, plus the normalized MoE inputs. All 11 cases complete
on all eight ranks with clean teardown; same-bucket eager and graph are exact
at every observed stage, including actual router choices. The CPU Markov
oracle matches all 120 tokens. Peak allocation is 6,175,574,528 bytes.

In case 3, the second draft block changes its selected experts on query row 7
from `[97, 82, 127]` to `[97, 82, 39]`. The maximum selection-score change on
that row is 0.00994873; the original third/fourth score gap is 0.00917423.
The padded gap is 0.00372231. The following block changes two rows' expert
membership. Case 4 similarly changes the second block's row 7 from
`[32, 97, 12]` to `[32, 97, 60]`; case 9 changes the third block's row 1.
This directly establishes expert-boundary crossings accompanying the large
MoE amplification. Across all recorded cases and ranks, same-bucket routing
remains exact. CPU sqrt-softplus plus checkpoint text bias gives a positive
actual selection margin for every observed row on rank 0 (minimum 0.000148773),
so the recorded choices agree with that score reference.

Different BF16 matrix shapes can perturb the inputs to discontinuous expert
selection. Requiring identical proposals across those shapes is therefore
stronger than graph replay correctness. R6 preserves the original strict
check and explicitly remains diagnostic; it does not force expert IDs or
change numerical tolerances. Next evaluate real target auxiliary states,
target verification and acceptance, with the same-bucket check as the direct
graph equivalence test and unpadded drift reported separately.

An independent one-NPU F.linear probe with real context-projection weights
also reproduces shape sensitivity: 171 versus 256 rows changes 132 of 875,520
values, maximum 0.0078125, relative RMS difference 0.000026814. Both outputs
have nearly identical error versus a CPU FP32 reference. This is evidence of
an initial BF16 perturbation, not proof of the entire downstream cause.

## Remaining acceptance

Test real target auxiliary states and final target verification after the
observed routing-boundary crossings. Complete target/draft cache rollback,
chunked prefill,
serving and DSpark-enabled vllm bench. Profile both graph families and total
proposal time; operator timings cannot establish end-to-end speedup.

Evidence is in `dspark_graph/`: per-rank capture identities, actual replay
counts, stage journals, proposal results, CPU comparison, source hashes and
raw log/tensor hashes. Native artifacts remain r6 Torch binding plus the
isolated AscendC metadata vendor and production r12 fallback vendor. The
production native installation is unchanged.

R6 evidence includes per-rank journals/capture IDs, comparison results and
source hashes. Raw per-case stage tensors are retained under
`/tmp/v41-dspark-proposer-graph-r6/graph_stages_case*.pt`; sizes and SHA256
digests are recorded in `dspark_graph/r6_source_manifest.json`.

## Configurable draft length K1..8

The runtime proposal count supports K1..8, preserving the
checkpoint training block size of five. The same-bucket graph harness now
uses configured K for query rows, target capture widths, rejection cases and
CPU Markov slicing. Every K completes all 11 TP8 cases, with exact same-bucket
stages on all eight ranks, exact CPU Markov selection and clean distributed
teardown. The oracle checks 24*K tokens per run, 864 tokens across the matrix.
K8 peak allocation is 6,185,003,008 bytes. Each rank replays both graph families
22 times per K. Per-rank results, stage journals, capture records, CPU comparisons
and source-file hashes are preserved in `dspark_graph/smallk_r1/`.
These are synthetic-target diagnostics, not full-target or serving acceptance.

Production admission now permits text-only DSpark K1..8 so real target
validation can proceed; that code change is not a claim of completed serving
acceptance. The full-model harness includes draft weights in its capacity
preflight, records speculative acceptance metrics, and requires actual
replays of the target, draft context and draft query graph families. The
complete native installation and real-target checks remain pending.
