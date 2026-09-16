# Isolated fused-candidate validation

Current trusted-unique mode4 uses direct paged Cube reads; see
[PAGED_UNIQUE_DESIGN.md](PAGED_UNIQUE_DESIGN.md). The continuous K workspace
experiment is withdrawn and has no callable kernel entry. For the current
isolated package use `run_v41_small_ops.py` with the r4 extension and the
`qli_paged_unique_candidate_transformer` vendor, then run
`test_indexer_v41_paged_unique.py`. Native correctness and graph replay must
pass before `benchmark_indexer_v41_prefill.py --candidate-mode 4`.
The current user workspace is `2 * producer_cores * 156704`, containing no K.
The legacy procedures below describe earlier mode2 artifacts and workspace.

## Earlier mode2 validation procedure

Keep the production r12 vendor directory unchanged. Build with the complete
`csrc/build.sh --pkg` workflow and wait for successful package completion.
Install the resulting package with an explicit, new `--install-path` under
`artifacts/qli-fused`; never run the production auto-install wrapper.

Use `tests/e2e/single_node/ops/run_indexer_v41_fused.py --opp-root VENDOR`
with `pytest` or `script` mode. It selects only the isolated candidate vendor
before loading the plugin and suppresses the production bootstrap in that test
process. Loading candidate and r12 tiling libraries together makes operator
registration ambiguous; the candidate package therefore includes the split
experiment's gather and score operators too. No production library, shared
symlink or bootstrap implementation is changed.

Run `test_indexer_v41_fused.py` first, then the whole-selector benchmark.
The benchmark requires the actual `.run` path and expected SHA256, the
candidate vendor directory, and the verified CANN API workspace size. It
checks the package hash, selected vendor ordering and actual mapped ACLNN
library, and records hashes of the installed files and native sources.
An incomplete JSON is written before device work, preserving the active
case when correctness or measurement fails.

The current CANN 9.1 DAV_2201 API workspace is 16,777,216 bytes, independently
of the 163,840-byte candidate user workspace. The combined native allocation
is 16,941,056 bytes before allocator rounding and caller outputs. Evidence
and the exact library hash are in `fused_cann_workspace_evidence.txt`.
Report observed allocated and reserved peaks, including that fixed cost.

Use `benchmark_indexer_v41_fused_regression.py` under the preserved r12
package first, then under the candidate with `--baseline` pointing at the
r12 JSON. Its CR2 B1 and CR1 consumer B8/B32 matrix covers 4K/32K/128K.
A failed noise or latency gate remains a failure. The generic B1 control in
the main benchmark uses an explicit zero output-index offset to select the
original implementation; that extra scalar offset read must be disclosed.

All numerical and performance limits remain those in `FUSED_CONTRACT.md`.
Successful compilation alone is not acceptance. Preserve failed build logs,
correctness failures and benchmark JSON files before another iteration.

For pipeline counters, run `msprof op` with `--kernel-name=QuantLightningIndexerV2`,
`--aic-metrics=PipeUtilization`, `--launch-count=1`, and the application set to
the isolated runner in `script` mode for `profile_indexer_v41_fused.py`.
That script has no source QLI call, so the matched kernel is the consumer.
Use an exclusive device window; profiling numbers collected alongside model
execution cannot establish the performance gate. Keep the complete OPPROF
artifact directory and include task wall time, Cube/Vector/MTE activity and
cross-core waits in the report.

The CMake source-copy and kernel stamps now depend on original source files.
r13 verified a changed nested header automatically updates both copied source
and kernel object through the complete build script, without manual cleanup.
For older, unpatched build trees, invalidate the generated
`csrc/build/binary/ascend910b/gen/quant_lightning_indexer_v2_*.done` stamp,
`bin/quant_lightning_indexer_v2`, and `src/quant_lightning_indexer_v2` before
the full build. Always verify original/copied/installed header hashes and
record the installed `.o` hash. Do not delete unrelated third-party caches.

The isolated QLI package uses this operator selection (the metadata operators
are required by shared AICPU registration):

```sh
CMAKE_BUILD_PARALLEL_LEVEL=12 bash build.sh --pkg --soc=ascend910b \
  --ops='quant_lightning_indexer_v2;quant_lightning_indexer_v2_metadata;sparse_attn_sharedkv_metadata;sparse_flash_mla_metadata;store_kv_block_metadata;vllm_quant_lightning_indexer_metadata;indexer_v41_candidate_gather;indexer_v41_candidate_score' \
  --vendor_name=qli_fused_candidate -j12
```

`CMAKE_BUILD_PARALLEL_LEVEL` also limits nested protobuf compilation, whereas
`-j12` alone does not. A separate vendor install path preserves production r12.
