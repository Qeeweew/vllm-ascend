# DSpark initial-input maximum-context audit

Status at this checkpoint: **unresolved**. Production V4.1 speculative
admission remains disabled. No NPU access was used for this audit.

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

## Required follow-up

Pass the logical table width explicitly from both DSpark and DFlash callers.
Guard the page-table load itself with row validity, valid column bounds and
cache ownership; use a masked default value and emit the invalid-slot sentinel
for excluded rows. Tensor stride alone does not define logical width.
Keep this check on device without adding a host synchronization.

This memory-access correction is only one part of the boundary contract.
Position limits, attention visibility and proposal validity near maximum
context still require scheduler/proposer integration tests, including mixed
request lengths and graph replay. A masked load alone does not establish
correct speculative behavior. Do not enable production admission on the
strength of this audit or the separate component tests.
