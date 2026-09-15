# DSpark initial-input maximum-context audit

Status: the initial input kernel's page-table bounds are corrected and tested;
the full maximum-context speculative contract remains **unresolved**.
Production V4.1 speculative admission remains disabled. The original
reproduction below used CPU only; the correction also has NPU coverage.

The shared `copy_and_expand_dflash_and_dspark_inputs_kernel` reads the query
page-table column using the query-row mask, without checking the table's
logical width. Near maximum context, a K5 query can extend beyond that width.
Later metadata checks cannot protect an earlier device load.

## CPU reproduction

The local probe `/tmp/v41_dspark_maxlen_probe.py` calls the real
`AscendSpecDecodeBaseProposer._propose` and
`AscendDSparkProposer.set_inputs_first_pass`. It intercepts the Triton launch
to inspect its actual arguments. Tensors remain on CPU; model hidden-state
projection and device-grid discovery use test doubles.

With maximum length 256, block size 32, one row of eight page-table columns,
and K5, all 42 combinations of sequence length 251 through 256 and rejection
count `None` or 0 through 5 reach the launch. Twenty combinations calculate
an out-of-range column. For sequence length 254 with no rejection, the query
columns are `[7, 7, 8, 8, 8]`. Column 8 is outside the table.

A second case calls the real `NPUModelRunner.propose_draft_token_ids` with
real `AscendCommonAttentionMetadata` at sequence length 255. It also reaches
the launch with columns `[7, 8, 8, 8, 8]`. This length is below the configured
maximum; it does not depend on admitting an already overlong request.
Raw log: `/tmp/v41_dspark_maxlen_probe.log`.

## Input-kernel correction

Both DSpark and DFlash now pass the logical table width explicitly. The
page-table load itself is masked by row validity, column bounds and cache
ownership; excluded rows and negative page entries emit slot `-1`. Tensor
stride is still used to locate rows, independently of logical width. The
checks remain on device without a host synchronization.

The existing nine NPU input-expansion regressions and 36 new boundary cases
passed: **45 passed** on NPU 0. New cases cover DSpark/DFlash query shapes,
mixed request lengths, rejection absent/zero/five, DCP ownership for both
ranks, contiguous and sliced tables, and negative page entries. A graph
captures interior lengths and replays with a final-page block, requiring
the mask to follow updated device lengths. Outputs match an independent
scalar page/ownership oracle exactly. Log and XML:
`/tmp/v41-draft-input-bounds-npu.log` and
`/tmp/v41-draft-input-bounds-npu.xml`.

The real DSpark caller's CPU test also verifies an eight-column view with
row stride eleven passes both values correctly (one focused test passed).
The combined loader/proposer CPU regression had 80 passing cases before
adding this focused stride assertion.

## Required follow-up

This memory-access correction is only one part of the boundary contract.
Position limits, attention visibility and proposal validity near maximum
context still require scheduler/proposer integration tests, including mixed
request lengths and graph replay. A masked load alone does not establish
correct speculative behavior. Do not enable production admission on the
strength of this audit or the separate component tests.
