# V4.1 RoPE and cache fusion

Status: r12 passes all 193 native cases, 33 wrapper CPU cases and all 64
complete-chain performance cases with the original median, P95 and stability
gates. The previously failing T1024/H32/D128 RoPE improves from baseline
243.158 us to 40.879 us (5.95x) with batched-head transfers.
No production installation is changed.

## Frozen numerical and addressing contract

The three explicit entry points are `v41_rope`, `v41_main_cache_store`, and
`v41_index_cache_store`. Projection GEMMs and RMSNorm remain independent.
The shared device core is pure AIV: the last 64 dimensions are interleaved
even/odd pairs, cast to FP32, evaluated with four distinct multiplies followed
by subtract/add, and rounded to BF16 once. The nonrotary prefix is copied.
Plain RoPE copies the full input row for positions outside the table; cache
stores instead skip such rows. Inverse RoPE negates sin before multiplication.

Caches accept [blocks,page,D] or [blocks,page,1,D], including an axis0 gap and
nonzero storage offset. Native attributes carry element stride0 separately for
key and scale caches. Pointer offsets are applied exactly once by ACLNN.
Destinations are physical compressed slots; do not divide them by CR again.
CR2 writes only odd original positions and rotates at the preceding even row.
Negative positions, invalid slots, invalid table rows and incomplete groups
leave every cache byte unchanged. Caller guarantees unique active destinations.

On 2026-09-16 the existing CANN `npu_dynamic_quant` baseline was probed on NPU0
without loading any new operator. The 32 cases cover T=1/2/4/16/64/128/1024 and
zero, negative zero, positive/negative halfway values, six magnitude ranges,
and six seeded random rows. Every nonzero INT8 result and every FP32/FP16 scale
bit pattern agreed with:

```text
rounded = BF16(RoPE_FP32_result)
maximum = max(abs(FP32(rounded)))
quantized = INT8(round_ties_even(FP32(rounded) * Ascend_Vector_Div(127, maximum)))
scale = FP16(FP32(maximum * FP32(1 / 127)))
zero row: quantized is all zero, scale is positive zero
```

The quotient is formed from `127 / maximum`, not by dividing each value by the
rounded scale. All-zero rows have an explicit native branch. INT8 conversion
follows the installed kernel's FP32-to-integer RINT, integer-to-FP16, and
FP16-to-INT8 cast sequence. Scale FP16 conversion follows quantization.

Raw rows, exact bits, source hashes and logs are retained in
[dynamic_quant_contract_r1.json](dynamic_quant_contract_r1.json) and
[dynamic_quant_contract_r1.log.txt](dynamic_quant_contract_r1.log.txt).
These initial 32 baseline observations did not cover every division boundary.
The r9 expanded suite found four values for which a CPU mathematical formula
gives a different INT8 result, while the unchanged fused kernel and the CANN
baseline match exactly: `(value, maximum)` of `(9.8125,19.625)`,
`(-11.875,23.75)`, `(10.6875,21.375)`, and `(9.375,18.75)`. CPU products were
just beyond +/-63.5 and rounded to +/-64; the two NPU paths emitted +/-63.
The production numerical contract therefore retains CANN's actual division
behavior rather than assuming CPU division reproduces it at every boundary.

Native index acceptance now compares both CPU RoPE followed by CANN quant and
the original unfused NPU Torch RoPE followed by CANN quant, requiring bitwise
agreement of the valid BF16 rows, INT8 keys and FP16 scales. Full physical cache
bytes and padding are still checked. The four observed boundaries have explicit
regressions. The CPU quantization helper is labeled a mathematical reference;
this oracle correction changed no device kernel and relaxed no tolerance.

## Implementation and validation progress

- CPU suite: 33 tests pass. Scalar pair oracle, BF16 prefix preservation,
  invalid-position copying, empty dispatch, aliases, gapped storage sentinels,
  compressed physical slots, partial CR2 groups, ties-even and zero quantization.
  Eleven cases validate shared packed key/scale pages, raw offsets,
  gaps, reversed region order, whole-page shifts, and overlap rejection. Eight
  FakeTensor cases additionally cover independent storages and true aliases.
- Plain RoPE complete package build r1 succeeded. Its DataCopy-based core was
  subsequently replaced with DataCopyPad to support unaligned storage offsets;
  r1 is not an acceptance artifact for the final implementation.
- Combined build r2 exposed unsupported INT8 `Duplicate` in the zero branch.
  The implementation now writes zeros through a uint16 view. Combined r5
  completed successfully; r6 rebuilt the final extra Vector barrier
  between reciprocal formation and overwriting the row maximum. No r5 native
  result is used to validate the revised implementation.
- The r6 full build and isolated installation completed on 2026-09-16. SHA
  verification passed for 22 original/snapshot source groups, all copied and
  installed device sources, and three generated/installed kernel objects.
  Package SHA256: `d9408a40f58cb25ea60472ff16328d95d9f0154f76d77109b1c38e53980b38c9`.
  Full evidence: `artifacts/v41-rope/r6/manifest.json` at the workspace root.
- An accidentally overlapping r3 build was stopped; a subsequent r4 build
  exposed its dead-PID compiler lock. The identified stale lock was removed
  after all compiler children exited. Logs and the lock evidence are retained
  under `/tmp/v41-rope-build-r*.log` and
  `/tmp/v41-rope-stale-lock-r4.json`.
- Historical r6-r9 results do not establish final native or whole-model
  performance; the final candidate is evaluated separately below.

The first r6 NPU run, using isolated Torch extension r3 SHA256
`64931ce51efb20c460b64cae5aea4b5eeb24ecb1ac29ec256f3ef62fde36cb13`,
stopped after 91 passing cases. All 84 plain RoPE cases passed, including exact
BF16 rounding, unaligned offsets and graph replay. The first main-cache case
with two gapped pages (T64) was rejected by ACLNN because the OpDef lacked an
explicit noncontiguous policy. This was a registration rejection, not a
numerical mismatch. The cache arguments now specify `IgnoreContiguous()`;
their existing stride attributes and raw pointers remain responsible for
addressing. Complete build r7 rebuilt the changed OpDef but exposed a separate
CMake generation dependency bug: its ACLNN source remained timestamped 02:22
while `libop_host_aclnn.so` was rebuilt at 02:58. The r6/r7 installed opapi
libraries were byte-identical, and r7 again stopped at the same registration
rejection after 91 passes. Source/device-object matching alone was therefore
insufficient for this host-policy change. Evidence is retained in
`artifacts/v41-rope/r7/stale_aclnn_generation.json`, together with the failing
test log. The root-owned CMake command must depend on its host library; a fresh
ACLNN generation and a new complete package build are required before retesting.
The active custom-package generator is `cmake/custom_build.cmake`; the similar
block in the root CMakeLists belongs to another branch. Build r8 initially
received only the latter fix and therefore still retained the stale interface.
The dependency repair must cover the active custom generator and the reusable
`cmake/opbuild.cmake` function as well.
Build r9 received both repairs without deleting the existing autogen directory.
It automatically replaced the cache inputs' `NnopbaseAddInput` calls with
`NnopbaseAddIgnoreContinuesInput`; the main/index generated source SHA256 values
changed to `208480b238ea60b7ece278a08650b0c4f418ed3d04a09447c5b14c40bdefdb34`
and `04939402dab04eebfd6e4ba4d956696bf2e4e6c94ef13d4feb96ad789e244bbd`.
The rebuilt opapi library SHA changed from
`57323594e246eac5921cda617100b5b4996ed208ff3bc8edae73392c3a3362e8` to
`1f29238c0779ea11ab671bcc7680962318acc2b1194ab9a00816f9c95d4ff535`.
All repaired CMake files match the original repository and isolated snapshot.
The failure log is `/tmp/v41-rope-npu-r6.log`. NPU0 had an unrelated namespace
process using approximately 34 GiB, recorded in
`/tmp/v41-rope-npu-r6-device-state.log`; this run supports correctness findings
only, with no exclusive-device performance claim.

Native tests cover T=0/1/2/4/5/10/16/64/128/1024, D128/512, H1/8/32, both rotary
directions, BF16 halfway values, three/four-dimensional caches, odd strides and
storage offsets with untouched sentinel guards. Fixed-address graph replays
change x/positions/slots/tables. DSpark coverage includes H8/D512 and five virtual
query slots near the maximum context boundary; prefix stores use explicit
physical slots through the same ABI. The actual proposer gate remains separate.
Eight extra native cases use the actual runner's shared raw pages with 4D INT8
keys and 3D FP16 scales, including graph replay and full-byte padding sentinels.
Together with four division-boundary regressions, all 187 native cases passed
on physical NPU1 with r9 and isolated Torch extension r3; process exited zero.
Log: `/tmp/v41-rope-npu-r9b.log`.

The accepted package SHA256 is
`4ecacc75c27cc4eac6b6e6e35ae35360e30ca78754b437f1ebd51ff0ca417c67`.
Its `artifacts/v41-rope/r9/manifest.json` records original/snapshot/copied/
installed device sources, three kernel objects, four CMake files, generated
ACLNN sources, opapi and extension hashes. The exact ACLNN phase1 query and
phase2 execution passed on offset/gapped main pages and shared packed index
pages. Each operator requests 16,777,216 bytes of fixed CANN workspace and zero
user workspace; see `artifacts/v41-rope/r9/workspace.json`.

The independent baseline probe `probe_v41_division_boundaries.py` records all
four boundaries at T1 and T64 in `artifacts/v41-rope/r9/division_boundaries.json`.
For example, generic NPU FP32 division gives `127/18.75 = 6.7733330726623535`,
while CPU division gives `6.773333549499512`; CANN quantizes the half-maximum
value to 63. Generic NPU division was measured separately, not read from the
dynamic quantizer's internal UB. Installed quantizer source hashes accompany
the input/output evidence.

## First performance pass and optimization

The r9 complete-chain run measured 64 cases, five alternating rounds and 20
samples per round with graph unroll 32. It passed the latency ratio limits for
63 cases. RoPE T1024/H32/D128 failed: candidate median 295.83 us, original chain
244.15 us (21.2% slower). Another 26 cases failed only the <=3% round-median
spread requirement; several small candidate calls were about 2 us and showed
substantial variability with this short graph workload. No final performance
acceptance is claimed. All raw timing samples and failures are retained in
`artifacts/v41-rope/r9/performance_r1.json`.

The corresponding `msprof_rope_h32` collection reports task duration 355.02 us
(profiling instrumentation, not the benchmark median), Vector utilization
about 18.5%, MTE2 about 32-52% and MTE3 about 24-45%, with long scalar/vector
dependency stalls. The initial row loop rereads both rotary tables for every
head. The next candidate assigns contiguous balanced row ranges and retains
the loaded table across adjacent heads at the same position. Position reads
are similarly reused within a token. Rounding and event ownership remain
unchanged; stores continue refreshing tables per row. Complete build r10 and
fresh numerical/performance acceptance are required for this optimization.

## r10 complete performance matrix and next candidate

The r10 result is terminal (`failed_gate`), with all 64 shapes recorded in
`artifacts/v41-rope/r10/performance_r1.json`. Graph unroll 256 removes the
round-median spread failures from the earlier short benchmark: all baseline
and candidate spreads are within 3%. Exactly one latency gate still fails:
T1024/H32/D128, baseline median 245.936 us versus candidate 273.544 us
(+11.23%). Every other shape passes the existing latency and P95 gates.
The package SHA256 is
`da357fd389062619f25a70a49b0b07416947040265547c00be7bf6d37bbbd7b1`.

The r10 `msprof op` capture for this failing shape reports a 313.30 us task,
40 AIV blocks, mean Vector utilization 20.98%, mean MTE2 utilization 42.65%
and mean MTE3 utilization 30.14%. Active transfer bandwidth averages only
1.55 GB/s per AIV for MTE2 and 2.12 GB/s for MTE3. These pipe counters overlap
and must not be summed. This profiler task time is not the benchmark latency. The row-based
implementation still performs per-head DMA, vector dispatch and synchronization.

The next candidate handles H32/D128 with T>=128 as a complete token: one
8192-byte row load, shared rotary tables, 32-head vector repeats, and one
8192-byte store. It retains separate FP32 products/add/sub and a single BF16
round, and preserves the nonrotary prefix. Other shapes retain the previously
validated row path. Six additional graph tests cover both directions at T127/128/129,
unaligned storage offsets, changed tables/positions and invalid positions
across the row/batched dispatch boundary. The new package passes all 193 native cases on physical NPU1, including these
six cases, in 8.31 seconds (`/tmp/v41-rope-npu-r12.log`). Exact BF16/INT8/FP16
contracts and packed-cache sentinel checks remain unchanged.

Build r11 was interrupted before a package was produced. Its process tree was
confirmed absent before the same sources were submitted to the complete build
script as r12 (`/tmp/v41-rope-build-r12.log`). The incomplete r11 log is retained;
no partial r11 artifact is used for testing. The r12 complete build exits zero
and passes source/snapshot/copied/installed SHA checks for 23 source groups and
three kernel objects. Package SHA256 is
`cfbf3247498ec482ea6c2f73ae3b72bc4742990730015fb9df75dc94ebfb8c71`;
RoPE object SHA256 is
`a52de351edc5b56843707060858113d4ca392049eb106059ddbb3fe5431acebe`.
The manifest is `artifacts/v41-rope/r12/manifest.json`.

## r12 complete-chain performance acceptance

All 64 shapes pass the unchanged median, P95 and <=3% round-spread gates in
`artifacts/v41-rope/r12/performance_r1.json` (`status: passed`). Measurement uses
five alternating rounds, 20 samples per round and graph unroll 256 on physical
NPU1. The pre-run device snapshot records no processes on NPU1. The largest
round-median spread across both implementations and all shapes is 1.588%.

| Replaced chain | Shape | Baseline median (us) | Candidate median (us) | Speedup |
| --- | --- | ---: | ---: | ---: |
| RoPE | T5/H8/D512 (draft query) | 72.089 | 4.292 | 16.80x |
| RoPE | T128/H32/D128 | 155.454 | 17.278 | 9.00x |
| RoPE | T1024/H32/D128 | 243.158 | 40.879 | 5.95x |
| Main cache store | T128/D512/CR1 | 141.917 | 5.521 | 25.70x |
| Main cache store | T128/D512/CR2 | 142.995 | 5.777 | 24.75x |
| Index cache store | T128/D128/CR1 | 180.897 | 8.950 | 20.21x |
| Index cache store | T128/D128/CR2 | 182.170 | 9.609 | 18.96x |

The previously failing H32/T1024 shape now passes: the batched-head candidate
is 40.879 us versus the earlier r10 candidate's 273.544 us. The paired r12
baseline remains 243.158 us. GEMM and RMSNorm are excluded from both compared
chains; this measurement does not predict end-to-end model speedup.

The exact r12 ACLNN query and phase2 launch pass for all three operators,
including gapped main-cache pages and packed index-cache pages with offsets.
Each requests zero user workspace and 16,777,216 bytes of fixed CANN scratch
(`artifacts/v41-rope/r12/workspace.json`). Maximum observed per-case allocated
and reserved memory are 59,266,560 and 140,509,184 bytes, respectively; these
include both benchmark implementations and their captured graphs, not only
the candidate's buffers. Raw samples, build and native/CPU logs and source/
binary manifests remain in `artifacts/v41-rope/r12/`.

## r12 profiling after the optimization

Three independent `msprof op` PipeUtilization captures finish successfully on
NPU1 at 1800 MHz. Raw captures and their SHA256 records are retained under
`artifacts/v41-rope/r12/msprof_{rope,main,index}/` and `profile_summary.json`.
Each capture covers all 40 AIV blocks. These kernels are pure AIV; Cube
utilization is not applicable.

| Kernel shape | Profile task (us) | Vector | Scalar | MTE2 | MTE3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RoPE T1024/H32/D128 | 45.52 | 42.79% | 36.57% | 20.91% | 9.10% |
| Main T128/D512/CR1 | 6.52 | 7.20% | 56.19% | 44.96% | 10.70% |
| Index T128/D128/CR1 | 9.96 | 11.10% | 43.11% | 44.61% | 9.31% |

Counters are arithmetic means across cores, overlap in time and must not be
summed. Profiling task durations include instrumentation and are distinct from
the paired benchmark medians. Compared with the r10 failing RoPE profile,
Vector utilization increases from 20.98% to 42.79%; active per-AIV MTE2
bandwidth rises from 1.55 to 22.52 GB/s and MTE3 from 2.12 to 50.07 GB/s.
This supports the measured improvement from replacing per-head small transfers
and dispatch with token-sized transfers and head repeats.

Remaining RoPE costs include scalar map initialization and vector dependencies.
The small T128 cache-store kernels spend more time on scalar dispatch and
small transfers than arithmetic. They already satisfy their complete-chain
latency gates; these measurements do not establish peak hardware throughput.
Full-model profiling remains necessary to determine their contribution to
TTFT/TPOT and whether further batching materially improves it.

## Wrapper FakeTensor alias validation

Python wrappers now use `torch._C._is_alias_of` to compare storage identity.
Comparing storage data pointers wrongly treats unrelated FakeTensors as aliases
because their pointers are zero. The actual packed-cache byte-stride and region
checks are unchanged. Eight CPU FakeTensorMode wrapper cases cover independent
arguments and rejected input/output aliases for all three operators, plus
allowed packed key/scale pages and rejected overlapping packed regions.
The complete CPU suite passes 33 tests (`/tmp/v41-rope-cpu-fake-r12.log`).
These tests exercise wrapper validation with a mocked native dispatch; compiled
Meta validation and real NPU execution remain separate acceptance layers.

## Acceptance and profiling

Run `tests/e2e/single_node/ops/test_v41_rope_cache.py` with the fresh isolated
package. Require bitwise BF16 RoPE, INT8 key and FP16 scale agreement; retain
failed cases and do not relax the gate. Compare original, copied and installed
kernel/core SHA256 and generated `.o` hashes before using results.

`benchmark_v41_rope_cache.py` measures complete replaced chains, excluding the
independent GEMM/RMSNorm. It uses five alternating rounds, at least 20 event
samples per round and graph unroll. T1/4/128 median and P95 must each improve
10%; all other cases must regress no more than 3%; round-median spread must
remain <=3%. The package path/SHA and separately verified exact ACLNN workspace
size are required arguments. Report user workspace (zero), fixed CANN workspace,
allocated/reserved peaks and every raw timing sample.

Use exclusive-device `msprof op` runs for `V41Rope`, `V41MainCacheStore` and
`V41IndexCacheStore`, with PipeUtilization and launch count 1. Inspect Vector,
Scalar, MTE and wait activity. These operators perform no matrix multiplication;
Cube utilization is not their optimization target. Integrated model profiling
and `vllm bench` follow native and graph acceptance.

The fixed-input workload is `benchmarks/deepseek_v41/profile_v41_rope_cache.py`.
Run it through the same `run_v41_small_ops.py` isolated loader used for native
tests. Select `--kind rope|main|index`, and profile the matching kernel name;
`--heads 8 --width 512` covers the DSpark query layout. Keep raw msprof output
beside the complete-chain benchmark JSON, including failed latency or jitter
gates.

## r12 final native performance acceptance

The complete 64-case run finished with `status=passed` on physical NPU1.
Each comparison uses graph unroll 256, five alternating baseline/candidate
rounds, 20 event samples per round, and exact output checks before timing.
All original median/P95 gates and the <=3% round-median spread requirement
pass. The maximum observed spread across both paths and all shapes is 1.587%.
The smallest median speedup across this matrix is 2.486x (H8/D512 T1024).
The baseline is the actual replaced Torch RoPE plus CANN quantization and
existing cache stores; these numbers do not compare against a hypothetical
optimized baseline or include projection GEMMs, RMSNorm or whole-model work.

| Chain | T / H / D / CR | Baseline median (us) | Native median (us) | Native P95 (us) |
|---|---|---:|---:|---:|
| RoPE | 1 / 32 / 128 / 1 | 68.061 | 3.612 | 3.615 |
| RoPE | 128 / 32 / 128 / 1 | 155.454 | 17.278 | 17.284 |
| RoPE | 1024 / 32 / 128 / 1 | 243.158 | 40.879 | 40.896 |
| DSpark-layout RoPE | 128 / 8 / 512 / 1 | 140.886 | 12.337 | 12.346 |
| Main store | 128 / 1 / 512 / 2 | 142.995 | 5.777 | 5.788 |
| Main store | 1024 / 1 / 512 / 2 | 223.825 | 13.734 | 13.775 |
| Index store | 128 / 1 / 128 / 2 | 182.170 | 9.609 | 9.659 |
| Index store | 1024 / 1 / 128 / 2 | 269.800 | 23.734 | 23.778 |

[performance_r12_summary.json](performance_r12_summary.json) preserves every
shape, median, P95, round median, stability result and acceptance gate, together
with the SHA256 of the complete raw timings at
`artifacts/v41-rope/r12/performance_r1.json`. Original r9/r10 failures remain
available and their gates were not changed. Peak allocated/reserved memory
across the benchmark cases is 59,266,560 / 140,509,184 bytes; these are
benchmark process peaks, including captured baseline/candidate graphs.
User workspace remains zero and exact ACLNN workspace is 16,777,216 bytes
per operator. The native package remains isolated; integrated model checks
and `vllm bench` are separate and still required.

## r12 exclusive operator profiling

Three sequential `msprof op` collections on physical NPU1 completed with
PipeUtilization, one captured launch and the same r12 package/extension.
[msprof_r12/summary.json](msprof_r12/summary.json) records shape, source CSV
paths, hashes and per-AIV means; copied raw CSV files sit beside it.
The profiled task durations include instrumentation and must not replace
the graph benchmark medians above.

| Operator and shape | Task (us) | Vector | Scalar | MTE2 | MTE3 |
|---|---:|---:|---:|---:|---:|
| RoPE T1024/H32/D128 | 45.78 | 42.62% | 36.45% | 20.91% | 9.06% |
| Main T128/D512/CR2 | 6.08 | 7.58% | 62.36% | 40.85% | 10.69% |
| Index T128/D128/CR2 | 10.78 | 10.59% | 42.48% | 45.37% | 10.44% |

For the former failing H32 shape, r10-to-r12 profile task time falls from
313.30 to 45.78 us. Mean active MTE2 bandwidth rises from 1.55 to 22.43 GB/s
per AIV, and MTE3 from 2.12 to 50.06 GB/s per AIV. Together with the measured
40.879 us complete-chain latency, these observations support replacing the
per-head DMA and dispatch loop with whole-token transfers and vector repeats.
The remaining scalar/vector stalls and short cache-row transfers are possible
future optimization targets; further changes require a new measured candidate.
All three kernels are pure AIV, so Cube utilization is not applicable.
Pipe counters overlap; percentages must not be summed.

The first profile attempt overlapped another profiler process whose temporary
script used the same name. Its six collections under `msprof_rope/main/index`
are retained but excluded from acceptance. Both process trees ended, NPU1
was verified idle, and all three cases were rerun sequentially under new
`msprof_exclusive_*` directories with a uniquely named launch script. The
completed 64-case benchmark predates this overlap. NPU1 was released after
the exclusive rerun; no production package or process was modified.
