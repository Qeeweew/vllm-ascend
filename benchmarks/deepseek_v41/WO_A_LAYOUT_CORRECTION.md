# V4.1 output projection layout correction

The real-weight DSpark component oracle exposed a production error shared by
the target and draft attention projections. `AscendColumnParallelLinear` loads
and shards `wo_a`, then transforms its ND storage from
`[local_groups * rank, width]` to `[local_groups, width, rank]`.
V4.1 `project_output` previously viewed that transformed storage as
`[local_groups, rank, width]` and transposed it again, scrambling the matrix.

The correction consumes the loader's three-dimensional layout directly in
the grouped BMM, checking its shape. The two-dimensional layout used by the
generic loader still takes the original explicit transpose. No weight values,
checkpoint format, GEMM accumulation policy or numerical gates change.

## Evidence before the correction

The complete context-9 DSpark capture in `/tmp/v41-dspark-component-c9-r3`
passed 688 of 712 stage checks. All 24 failures were `output_a` across three
layers and eight ranks, with NRMSE 1.4809 to 2.0652. All 208 exact checks passed.
The missing shared-expert capture from an earlier harness attempt was fixed;
the complete comparison includes real local shared-expert outputs.

On rank 0, recomputing the erroneous transposed-storage reinterpretation
matches the observed output with NRMSE below 8.1e-7, while the correct
checkpoint matrices give NRMSE 1.557, 1.671 and 1.712. This isolates the
layout error rather than changing the reference to fit the implementation.
Diagnostic: `/tmp/v41-dspark-output-a-layout-diagnosis.json`.

## Validation scope

Six CPU regression cases use the real Ascend and upstream column weight
loaders on TP ranks 0, 3 and 7, with two local groups and two successive
weight loads. Both stored layouts must agree exactly with independent
per-group linear projections from the checkpoint tensors. Together with
loader and proposer regressions, **80 tests passed**; log
`/tmp/v41-output-layout-input-bounds-cpu.log`.

The corrected real TP8 capture and independent numerical comparison in
`/tmp/v41-dspark-component-c9-r4` passed **712 of 712 stage checks** with the
original frozen gates. Maximum `output_a` NRMSE is now 0.000244; routed MoE
0.000644, shared experts 0.005363, attention 0.002045 and logits 5.71e-5.
Attention results are exactly equal across TP ranks in all three layers.
Peak allocation remains 3.342 GiB per rank and all eight ranks completed
distributed cleanup. These are actual-stage-input comparisons with synthetic
target auxiliary states, not an independently propagated full-model oracle.

Target auxiliary-output graph validation must also be
rerun after the correction. Earlier target repetition and graph equality
tests checked consistency of the same erroneous math, so they do not prove
checkpoint-level correctness or model quality. Historical timing evidence
remains historical; final profiling must use the corrected model.
