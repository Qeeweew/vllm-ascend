# DSpark metadata acceptance and performance

The new AscendC metadata kernel passed isolated correctness and graph acceptance
on physical NPU 1 (910B), 2026-09-16. This report covers schedule generation and
its SMLA consumer, not full DSpark decoding throughput or acceptance rate.
DSpark is required to remain enabled; there is no eager fallback in this kernel.

## Correctness

- **58 NPU tests passed in 10.54 seconds** with the isolated r6 Torch extension,
  the complete candidate operator package, and production SMLA/AICPU fallback
  vendor for the reference implementation only.
- 48 cases cover B=1/2/4/8/16/32, candidate spans 0/1/133/256, and dense/ragged
  requests. Final attention output and LSE match original AICPU-generated
  schedules exactly; an independent FP32 attention oracle also passes.
- Six graph cases each run 16 changed-input replays, for **96 replays**. Capture
  includes visibility generation, AscendC schedule generation, and attention.
  Replays vary active query count, empty requests, rejection transitions near
  context limits, Q/K contents, and block tables. Persistent output pointers and
  exact schedule coverage remain valid.
- Four empty cases B/T=(0,0),(0,8),(4,0),(32,160) clear all 1024 schedule words.
- Seven CPU tests execute the production planner source against host stubs,
  including 180 random ragged cases and malformed offsets. This validates
  planner semantics; the NPU tests validate device execution and synchronization.

## Metadata latency

Microbenchmark: context 4096, five queries per request, 133 visible keys,
seven samples, 64 invocations per sample. Graph timings use NPU events around
64 captured calls and divide by 64. Host timings include enqueue and final
synchronization divided by 64. The AICPU reference allocates its output through
its normal public API; the candidate reuses caller-owned output as required by
graph replay. These columns measure different launch modes and must not be
interpreted as an end-to-end graph speedup.

| Batch | Queries | Candidate graph us | Candidate host us | Native AICPU host us |
| --- | --- | --- | --- | --- |
| 1 | 5 | 1.54 | 29.60 | 119.84 |
| 2 | 10 | 1.92 | 28.68 | 121.54 |
| 4 | 20 | 2.64 | 28.39 | 121.10 |
| 8 | 40 | 3.01 | 29.22 | 121.34 |
| 16 | 80 | 3.12 | 29.20 | 123.03 |
| 32 | 160 | 4.34 | 29.80 | 123.26 |
| 64 | 320 | 5.88 | 42.30 | 139.61 |
| 128 | 640 | 9.11 | 44.29 | 202.03 |

The B<=32 graph medians are 1.54–4.34 us. At B128 the median reaches 9.11 us,
consistent with the single-core scalar planner doing more offset validation and
request-boundary traversal. This initial implementation prioritizes a small,
graph-compatible schedule path; it does not claim an optimal large-B planner.

## msprof op analysis

A separate B32 `msprof op --aic-metrics=PipeUtilization` capture measured:

| Metric | Value |
| --- | --- |
| Operator duration | 5.48 us |
| AIV execution time | 4.93 us |
| AIV scalar time / ratio | 4.04 us / 81.94% |
| Vector time / ratio | 0.039 us / 0.80% |
| MTE2 time / ratio | 0.349 us / 7.07% |
| MTE3 time / ratio | 0.140 us / 2.84% |
| Active blocks | 1 AIV |
| Cube utilization | N/A (integer metadata, no matrix multiplication) |

The measured bottleneck is scalar planning, not copying the 4 KiB schedule.
Low transfer utilization is expected for these small transfers. Profiling adds
overhead and used a different invocation mode from graph timing, so 5.48 us is
not substituted for the 4.34 us graph median. If whole-model profiling shows
metadata is still material, the next bounded experiment should parallelize
schedule record construction/request lookup across AIV cores and compare
small-B regression versus large-B benefit. Preserve complete zeroing, invalid
input handling, and the downstream consumer's endpoint rules. No unmeasured
parallel planner is included in this revision.

## Reproducibility

Workspace artifact directory: `artifacts/v41-dspark-metadata/r1/`.

- `build-attempt3.log`: complete `build.sh --pkg --soc=ascend910b
  --vendor_name=v41_dspark_metadata_candidate --ops=v41_dspark_metadata -j4`
  succeeded; isolated `.run` installation succeeded.
- `source_manifest_attempt3.json`, `package.json`: source/object/package audit.
  Source snapshot and installed object correspond to the completed build.
- `native-tests.log`, `cpu_schedule.log`, `native_collection.log`: acceptance.
- `benchmark.json`, `benchmark.log`: all seven timing samples per batch.
- `msprof-b32.log` and `msprof-b32/OPPROF_20260916045446_TDSBOXLHHDIPGTEN/`:
  raw profiling, `OpBasicInfo.csv`, and `PipeUtilization.csv`.
- `source_hooks.log`, `tools_hooks.log`: source and tooling checks.

SHA256 values:

```text
.run package:
47a8c8a95681badf72a614bb6d71604fa2ad6ab1713e051c135829f6a1a4242d
built and installed V41DsparkMetadata kernel object:
3ef1f629a28177018312c56187ff91a4b91c41cfbbcd5eee6abb2517698079b3
r6 vllm_ascend_C.cpython-312-aarch64-linux-gnu.so:
92a84c6b33851c62ea1daac49c364ed7ad6c92653b56a47c62bc6c84c94fa5e5
```

The production installation was not replaced. The tested vendor is
`artifacts/v41-dspark-metadata/r1/opp/vendors/v41_dspark_metadata_candidate_transformer`.
The extension is `artifacts/small-ops-bindings/r6/install/`.
The remaining acceptance gate is full TP8 proposer and target/draft serving,
including graph execution, accepted/rejected token rollback, and `vllm bench`.

## Small draft-length validation

The model adaptation is scoped to **K=1..8**. Metadata kernel capacity is a
separate property: a 256-key candidate buffer and up to 32768 total queries do
not imply support for large model draft lengths. No metadata host/kernel change
was necessary: the planner partitions actual `cu_q` query offsets and has never
contained a K5/133 hard limit. The runtime visibility helper and model admission
checks own the draft-length policy.

The final test matrix uses each K=1/2/3/4/5/6/7/8:

- 24 eager cases (B=1/4/32) compare native AICPU schedules and independent FP32
  attention, including empty requests, graph tail padding and in-range zero-span
  rows.
- 27 visibility + schedule + attention graph cases: every K at B=1/4/16, plus
  K5 at B=2/8/32. Each runs 16 changed-content replays (432 total). A separate
  CPU oracle enumerates prefix-window logical indices and validates visibility
  exactly before comparing attention. Rejection counts include zero and all K.
- 24 direct-index graph cases (every K at B=1/4/16), six changed-content replays
  each (144 total), also exercise spans 0/1/256 as operator capacity boundaries.
- The original 48 fixed-K5 span/ragged cases and four empty clearing cases remain.
  This also exercises the wrapper's explicit-index mask0 path after changing its
  ignored window-distance argument from 132 to 255.
- CPU planner tests cover every small K across B=1/4/32/255. A separate T32768
  structural case uses longer per-request query ranges solely to check planner
  partition bounds; it is not a model draft configuration.

**Final validation passed: 127 NPU tests in 22.52 seconds (576 graph replays),
and 40 CPU planner tests in 1.06 seconds.** Results are recorded in
`smallk-final-native.log`, `smallk-final-cpu.log`,
and `smallk-final-source.json` under the r1 artifact directory. The source
manifest verifies all seven operator sources are byte-identical to the complete
r1 build; that verified package and r6 binding are used without a partial build.
Earlier `multik-*` logs preserve exploratory capacity runs and are not declarations
of supported model draft lengths. These tests establish operator/graph correctness,
not DSpark acceptance rate or full-model speedup; those require TP8 serving tests.
