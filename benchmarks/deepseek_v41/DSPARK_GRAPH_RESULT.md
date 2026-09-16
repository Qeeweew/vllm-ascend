# V4.1 DSpark context/query graph diagnostic

Status on 2026-09-16: **same-bucket graph replay is exact across the tested TP8
matrix; strict comparison with original unpadded eager remains incomplete**.
Production admission and full-target validation remain closed. This is not
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

## Padding sensitivity still under investigation

R1 failed before capture because the generic proposer required torch.compile;
V4.1 now uses its actual FULL graph configuration. R2 reached context capture
but the wrapper rejected its None return; a tensor-container return fixes that
contract. R3 exposed BF16 differences between 9-row eager and 16-row capture.
R4 established same-bucket equality and stopped on the first unpadded B3
proposal mismatch. All four failure histories are retained.

In r5 case 3, context projection changes by at most 0.015625 in 134 values;
context norm changes by at most 0.0009765625. The maximum difference grows to
0.25 in the second attention block, then 6.0234375 in that block's MoE output.
Every observed same-bucket eager/graph stage remains exact. This locates a
major amplification at MoE, but does not yet establish which router/expert
operation causes it. Inspect expert choices and routing margins next.

An independent one-NPU F.linear probe with real context-projection weights
also reproduces shape sensitivity: 171 versus 256 rows changes 132 of 875,520
values, maximum 0.0078125, relative RMS difference 0.000026814. Both outputs
have nearly identical error versus a CPU FP32 reference. This is evidence of
an initial BF16 perturbation, not proof of the entire downstream cause.

## Remaining acceptance

Explain the padding-induced routing/proposal differences using actual router
inputs and expert choices, then test real target auxiliary states and final
target verification. Complete target/draft cache rollback, chunked prefill,
serving and DSpark-enabled vllm bench. Profile both graph families and total
proposal time; operator timings cannot establish end-to-end speedup.

Evidence is in `dspark_graph/`: per-rank capture identities, actual replay
counts, stage journals, proposal results, CPU comparison, source hashes and
raw log/tensor hashes. Native artifacts remain r6 Torch binding plus the
isolated AscendC metadata vendor and production r12 fallback vendor. The
production native installation is unchanged.
