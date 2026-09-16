# Fused candidate QLI validation status

Acceptance: **overall pending; r14 is a historical B1-only local experiment**.
Its B1 thresholds passed, but the user requires multi-batch CR1/CR2 fusion.
The nine legacy-path regression cases do not establish fused multi-batch
coverage. Original measurements and failures remain intact. No final QLI
acceptance or rollout is claimed.

The implementation and fixed gates are in `FUSED_CONTRACT.md`. The independent
package preserves r12 and includes only the candidate specialization plus build
dependencies. CR2/B8/B32 continue through the original native path.

## Completed evidence

- CPU acceptance-gate tests: 8 passed.
- Ruff and Python compilation: passed for the new test and benchmark scripts.
- Complete r5/r6/r7 package builds and isolated installations: passed.
- r5 did not trigger the fusion specialization: the tiling-info `qkHeadDim`
  field was never populated. Field population was fixed in r6.
- r6/r7 initial test processes still loaded production tiling through a cached
  bootstrap function imported by `platform.py`. The launcher now isolates its
  function globals as well; no production file or installed library was changed.
- r7 isolated dispatch verified by tiling log: `candidate fused=1`, D128,
  `output_idx_offset=nullptr`, 16 groups. The explicit nonempty zero-offset
  control logs `candidate fused=0`.
- r7 original QLI correctness/regression suite: **15 passed**, 6.90 s,
  including CR2, multiple requests and dynamic graph replay.
- r7 fused eager/graph correctness: **13 passed**, 7.21 s. All frozen adversarial
  cases and eight replays with changed device data passed the unchanged oracle.
- Diagnostic length511 returns 503 valid selected positions (exact candidate
  membership), where generic native incorrectly returns all 511. This legacy
  short-context behavior is retained outside the specialization.
- Diagnostic incremental allocated HBM at length511: fused 16,944,128 B;
  generic 103,308,288 B. This is functional evidence, not the final memory gate.
- User workspace: 163,840 bytes (160 KiB).
- CANN fixed API workspace: 16,777,216 bytes (16 MiB) on current CANN 9.1 DAV_2201.
- Combined native workspace: 16,941,056 bytes before allocator rounding and outputs.
  See `fused_cann_workspace_evidence.txt` for the installed library hash and disassembly.

## Live unaffected-path baseline (preserved r12)

All nine cases passed independent correctness and the 3% round-spread gate.
Three rounds of 12 event samples each; graph unroll 64. No concurrent NPU workload.
These numbers are controls, not candidate improvements.

| CR | Batch | Context | Median (us) | P95 (us) | Round spread |
| --- | --- | --- | --- | --- | --- |
| 2 | 1 | 4097 | 51.170 | 51.831 | 0.102% |
| 2 | 1 | 32771 | 81.375 | 81.722 | 0.033% |
| 2 | 1 | 131075 | 181.751 | 181.867 | 0.042% |
| 1 | 8 | 4097 | 116.504 | 116.751 | 0.110% |
| 1 | 8 | 32771 | 296.864 | 297.138 | 0.027% |
| 1 | 8 | 131075 | 497.703 | 498.661 | 0.055% |
| 1 | 32 | 4097 | 202.262 | 203.358 | 0.035% |
| 1 | 32 | 32771 | 561.710 | 562.325 | 0.105% |
| 1 | 32 | 131075 | 1431.305 | 1432.117 | 0.011% |

Raw control: `artifacts/qli-fused/r12-unaffected-baseline-r1.json` at workspace root.
The candidate must pass the same matrix at <= 1.03 times these median/P95 values.

## Build failures retained

- r3: missing `AscendC` namespace qualification in standalone Cube service; fixed.
- r4: unsupported unsigned-integer to float scalar conversion; signed intermediate added.
- Missing-object linker messages in those runs followed the compiler failures.
- Failed build trees and logs are retained; neither package was installed.

## Historical r7 reproduction artifacts

The r7 candidate package SHA256 was:
`9bdeba31e4e8f67bccf30b7d09cbd000e9999202e970280e8c78839b0c1f476e`.
Workspace artifact directory: `artifacts/qli-fused/r7/`.
`correctness-evidence.json` records package, kernel and host-library fingerprints.
The production r12 installation has not been replaced.

The first fully isolated diagnostic is retained as
`/tmp/v41-qli-fused-debug-isolated-r7.log`; earlier diagnostics and failures
remain alongside it. Profiling and performance must use the isolated runner
in `FUSED_RUNBOOK.md`, because platform imports can retain bootstrap function
references before a simple function-name patch is applied.

## Frozen whole-selector results (r7)

Exclusive NPU0, three alternating rounds of 12 event samples, candidate unroll64,
dense unroll4. Every shape passed noise and memory gates. Every shape failed
`fused <= 0.9 * live split` for median/P95; 4K also failed native and dense gates.
The fixed thresholds remain unchanged. Values are microseconds.

| Context | Fused median / P95 | Split median / P95 | Native median / P95 | Dense median / P95 |
| --- | --- | --- | --- | --- |
| 4097 | 126.310 / 126.372 | 69.720 / 70.096 | 101.663 / 101.844 | 97.875 / 99.840 |
| 32771 | 76.952 / 77.033 | 82.091 / 82.644 | 281.634 / 281.751 | 132.130 / 132.600 |
| 131075 | 78.416 / 78.749 | 82.505 / 83.534 | 482.166 / 482.587 | 203.128 / 209.040 |

Raw data: `artifacts/qli-fused/r7/performance-r1.json`. It includes observed
incremental and total allocated/reserved peaks, all event samples, package and
source hashes, and mapped library fingerprints. These complete-selector numbers,
not kernel-only profiler durations, determine acceptance.

## msprof op evidence and next change

All three operator profiles succeeded and passed the selection oracle after
collection. Commands used `msprof op --kernel-name=QuantLightningIndexerV2
--aic-metrics=PipeUtilization --launch-count=1`, with the isolated runner invoking
`profile_indexer_v41_fused.py --length CONTEXT` as `--application`. Outputs are
`artifacts/qli-fused/r7/profile-CONTEXT-r1/OPPROF_*`; a machine-readable summary
is `artifacts/qli-fused/r7/profile-summary.json`. Device frequency was 1800 MHz.

| Context | Task wall (us) | AIC MTE2 mean (us) | AIC Cube mean (us) | Even AIV scalar mean (us) | Even AIV vector mean (us) |
| --- | --- | --- | --- | --- | --- |
| 4097 | 103.920 | 48.378 | 0.610 | 27.252 | 5.071 |
| 32771 | 63.980 | 17.122 | 0.611 | 27.594 | 5.068 |
| 131075 | 65.360 | 16.738 | 0.605 | 29.206 | 5.068 |

At 4K, Cube groups 0–3 show 19.6–20.9 us MTE2, while groups 4–15 show
57.1–58.2 us. Their paired vectors wait about 59 us for SCORED (flag7), versus
about 21 us for active groups. Active vectors then wait about 35–38 us at the
final vector barrier (flag14). Cube arithmetic itself is only 0.57–0.64 us.
This contradicts the initial unverified masking-only hypothesis: the dominant
short-context delay is the dummy INT8 key-load path, propagated through SCORED
and the final barrier. The code maps every invalid candidate to physical offset 0,
so mostly-empty groups issue repeated same-address MTE2 transfers.

r8 adds one validity descriptor per group in the existing score workspace.
Empty groups retain READY/SCORED and final-barrier participation, skip Cube work,
and directly emit sorted (-inf,-1) pairs. No additional workspace is required.
The complete build script packaged a changed header but reused the r7 kernel
object due to a dependency-free `.done` stamp. r8 testing (28 passed), performance
and profiling therefore do not validate the new implementation; they are retained
with `artifacts/qli-fused/r8/INVALIDATED.json`. The unchanged kernel SHA256 is
`0ce6e6a9bf0176fcf90eb706d0b7adf709eed2fbd783f3648f63abdfe669adb8`.
r9 invalidated the kernel stamp but exposed a second stale, dependency-free
source-copy stamp. It was rejected before NPU execution. r10 invalidated both
stamps and exposed use of unsupported `GlobalTensor::ReinterpretCast`; r11
uses a separately typed GlobalTensor descriptor and rebuilds both source and
binary through the complete script. The installed object hash must change. Further vector/scalar optimization will be measured separately.
At 32K/128K, scalar and MTE2 costs remain significant even without the long
empty-group wait; reducing that cost is still necessary to reach the strict
split-relative target.

## Overall gates remain open

Only the historical B1 consumer matrix is complete. Multi-batch actual fused
dispatch, CR1 producer/consumer and CR2 selection, uniform/mixed workloads,
correctness/graph, workspace/HBM, per-shape latency and profiling evidence
remain required. The r14 legacy CR2/B8/B32 controls cannot substitute for
these gates. Full-model integration and profiling also remain separate.

## H32-specific pipeline baseline

The profile shape is H32/D128, one query. QK uses M=32, N=128, K=128;
weighted reduction uses M=16 (the Cube minimum), paired N=256, K=32.
The latter has one useful output row and 15 duplicated rows; there is no
H64 arithmetic hidden in the head dimension. The inherited allocation reserves
more L1 rows than H32 requires (Q L1=256 rows, Score L1=128 rows), which is an
allocation/layout optimization opportunity, not evidence of extra QK Mmad work.

Raw r7 `aic_cube_ratio` means, converted to percent, are 0.592% at 4097,
0.970% at 32771 and 0.931% at 131075. These are msprof pipeline-activity
ratios over its profiling interval, not a percentage of peak theoretical
FLOP/s. The arithmetic active time is only about 0.61 us per Cube core.
All exported activity, ratio, bandwidth and wait fields, including every core,
are retained in `artifacts/qli-fused/r7/profile-summary-full.json`. MTE and
FixPipe activity counters may overlap and include pipeline occupancy; do not
sum them into a serial critical path or call their difference pure idle time.

The first change targets empty-group MTE traffic identified above. For long
contexts, scalar candidate preparation/position masking and many eight-row
GM-to-L1 copies dominate actual H32 arithmetic. Future tile/parallelism changes
must reduce these measured costs and improve complete-selector latency;
raising Cube activity with redundant computation is not a valid improvement.

## Verified empty-group optimization (r11)

Package SHA256 `a6ad2680108fb326ec653d6c612b138dd59ae6055978ec8008e0b512372d196d`;
kernel SHA256 `7adc7cedd4bd1dde7650569b569988cc7cac66d4ac714bc9098259cc4051fbcd`.
Original, copied and installed source headers match. 28 eager/graph/original
QLI cases passed (8.27 s). All three profiles passed the CPU oracle.

| Context | Selector median / P95 (us) | Profile task (us) | Cube ratio mean (%) | Cube active mean (us) | AIC MTE2 mean (us) |
| --- | --- | --- | --- | --- | --- |
| 4097 | 74.825 / 75.090 | 62.720 | 0.314 | 0.195 | 4.713 |
| 32771 | 76.983 / 77.100 | 63.040 | 0.961 | 0.605 | 17.124 |
| 131075 | 77.489 / 78.440 | 63.840 | 0.970 | 0.611 | 16.789 |

4K selector median improves 40.8% versus r7. Its all-core mean Cube activity
decreases because eleven completely empty groups now do no arithmetic; this
is useful-work elimination, not a utilization regression. Active-core Cube
time remains roughly 0.61 us and reported utilization reaches 1.04%. The
long repeated-dummy-load MTE2 tail is gone. At 32K/128K the unchanged scalar
epilogue is still expensive. Every shape passes memory, noise and native/dense
controls, but every shape **still fails** the strict 0.9-times-live-split target.
No acceptance threshold changed.

Raw `artifacts/qli-fused/r11/performance-r1.json` includes samples and HBM;
`profile-summary-full.json` includes all exported counters and per-core rows.
The next isolated change vectorizes the 1024-element local index copy/mask,
preserving integer payload bits and using FP32 conversion only for sign checks.

| Version / context | AIC MTE1 mean (us) | AIC MTE3 mean (us) | AIC FixPipe mean (us) | Even AIV scalar mean (us) | SCORED wait mean (us) | Final barrier wait mean (us) |
| --- | --- | --- | --- | --- | --- | --- |
| r7 / 4097 | 1.190 | 1.988 | 48.184 | 27.252 | 49.717 | 10.079 |
| r11 / 4097 | 0.378 | 1.768 | 4.661 | 14.499 | 5.593 | 26.690 |
| r7 / 32771 | 1.193 | 1.900 | 17.055 | 27.594 | 18.549 | 1.419 |
| r11 / 32771 | 1.193 | 1.912 | 17.085 | 27.106 | 18.459 | 1.307 |
| r7 / 131075 | 1.192 | 1.535 | 16.709 | 29.206 | 18.076 | 2.369 |
| r11 / 131075 | 1.191 | 1.820 | 16.751 | 28.234 | 18.115 | 1.365 |

SCORED is exported `aiv_scalar_wait_id7_time(us)`; final barrier is
`aiv_scalar_wait_id14_time(us)`, averaged over even AIVs. Empty groups now
arrive earlier at the final barrier, so their wait increases while task time
falls. These counters establish where dependencies stall progress; the tool
does not export a single pure pipeline-gap metric in this capture.

## Cube utilization comparison limits

The low activity is real even on active cores. For r11 32K core 0 the original
CSV reports `aic_cube_time(us)=0.618333`, `aic_cube_ratio=0.009815`,
`aic_time(us)=35.337223`, `aic_mte2_time(us)=17.453888`, and
`aic_scalar_wait_id6_time(us)=11.872778`. The ratio implies a roughly 63-us
normalization interval (profile task 63.040001 us), rather than that core's
35.337223-us execution. Normalizing to core execution instead yields only
1.75%, so a denominator change does not explain away the bottleneck. The
installed profiler exports these ratios as compiled-tool output; no exact
source-level formula was available.

The reference `qli_opt` optimization design contains an old 32K prefill example:
Cube active 23.650 ms and task 40.240 ms (about 58.8%). Its current correctness
report explicitly rejects old optimized decode timings due to wrong outputs;
this is not a verified source for the user's previously observed >50% case.
The current reference benchmark uses H64/D128, top-k2048, dense paged attention,
B1..B32 decode at context131072 and prefill lengths2048/8192/32768. Our
profile is one-query B1 H32, sparse candidate blocks2048x8 and top-k512;
large-query dense prefill has much more tile reuse and sustained Cube work.
An apples-to-apples claim requires the original profile and identical shape,
query mode and metric scope.

Synthetic profiling inputs have 4089/16371/16379 effective positions for
4K/32K/128K respectively (one deliberately replaced candidate can duplicate
a block). Useful QK work is 33,497,088 / 134,111,232 / 134,176,768 integer
operations, counting multiply and add separately; Q and K are INT8 with INT32
accumulation. Useful FP16 WS multiply-add work is 261,696 / 1,047,744 /
1,048,256 operations. Each active group actually processes 1024 slots:
8,388,608 QK integer operations plus 1,048,576 FP16 WS operations because
WS uses the minimum 16-row M tile (only one output row is useful).
These are calculated operation counts, not profiler hardware FLOP counters.

Beyond scalar masking, `KeyNd2NzForPA` currently issues 128 separate 1-KiB
GM-to-L1 copies per full group. Consecutive physical candidate runs can be
merged without a candidate-sized GM staging tensor; the next transfer-strategy
experiment will preserve page/tile boundaries and validate the same oracle.

## Local mask/copy vectorization (r13)

Package `8ee57990e69d1fd9327ccf359c220d6de9bb5c7cdb799acfe521a4d7ed5f2ddf`,
kernel `e5bfea9ca9bfae813e40e679b42a2b8a799260feccd219c2ad3ac79a7f2ec3a2`.
28 correctness/graph/original-QLI tests passed (8.56 s). The source-copy/kernel
CMake dependency fix automatically rebuilt this version without manual cache
cleaning; original/copied/installed headers match.

| Context | Selector median / P95 (us) | Split-relative gate | Other gates |
| --- | --- | --- | --- |
| 4097 | 62.421 / 62.847 | Failed | Passed |
| 32771 | 63.421 / 63.512 | Passed | Passed |
| 131075 | 64.450 / 64.885 | Passed | Passed |

The r13 comparison remains unaccepted because the 4K split-relative gate
fails. r14 changes the transfer strategy: traverse candidate blocks in ascending
order, merge physically adjacent valid blocks within the same page and L1
tile, and skip invalid-column key loads. It does not allocate a gathered GM
key tensor or change Cube arithmetic. The r13 profile and r14 build run
concurrently on separate CPU/NPU resources.

All three r13 profiles passed the oracle. Task durations are 48.820, 50.440
and 51.060 us; reported mean Cube activity ratios are 0.390%, 1.229% and
1.201% for 4K/32K/128K. At 32K even-AIV scalar activity falls from r11's
27.106 us to 13.384 us, while AIC MTE2 remains 17.175 us. This directly
confirms the scalar-mask bottleneck and motivates the separate MTE coalescing
change. r13 raw counters remain in its `profile-summary-full.json`.

A refreshed exclusive-NPU0 r12 unaffected baseline passed all nine correctness
and noise cases: `artifacts/qli-fused/r12-unaffected-baseline-r2.json`. The
next candidate regression must compare against this refreshed control.

## Physical-run coalescing (r14)

Package SHA256 `cdf878ffe0ea3097e249ce03632b77fc915321d829b7764ead0c48b921c71460`;
kernel SHA256 `386f79de85038684a10191e0bbc9a20c1b7e22787e322680192555117d85736f`.
Both kernel headers match across original, copied and installed source.
28 correctness/graph/original-path tests passed (11.20 s).

| Context | Fused median / P95 (us) | Live split median / P95 (us) | Native median / P95 (us) | Dense median / P95 (us) |
| --- | --- | --- | --- | --- |
| 4097 | 58.256 / 58.523 | 69.630 / 70.100 | 103.204 / 103.521 | 95.138 / 97.470 |
| 32771 | 61.408 / 61.511 | 81.789 / 82.103 | 281.069 / 281.341 | 129.695 / 131.610 |
| 131075 | 64.929 / 65.193 | 81.965 / 83.278 | 481.956 / 482.900 | 204.342 / 210.625 |

All original per-shape gates pass, including median/P95 <=0.9 times live split,
live/frozen dense and native controls, <=3% round spread, and allocated HBM.
Whole-selector improvements versus live split are 16.3%, 24.9%, and 20.8%
for median latency. Compared with r13, coalescing improves 4K by 6.7% and
32K by 3.2%, but costs 0.7% at 128K where candidate runs are less dense;
this small regression is retained rather than hidden by reporting averages.

Fused incremental allocated HBM is 16,944,128 B at every length, versus
103,308,288 B for generic native and 22,615,136 / 27,478,016 / 27,478,016 B
for split. User workspace remains 163,840 B; the CANN fixed 16-MiB workspace
is included. Raw total allocated/reserved peaks are in `performance-r1.json`.
The full public wrapper (float conversion, sentinel mapping, ascending sort,
row masking and INT32 conversion) remains included in these timings.

All r14 `msprof op` runs finished successfully and passed the frozen CPU
oracle after collection. The following times are per-core means in us;
Cube ratios show all launched AICs / only AICs with nonzero Cube activity.

| Context | Task us | Cube ratio all / active (%) | Cube us | MTE1 us | MTE2 us | MTE3 us | FixPipe us | Even AIV scalar us | SCORED wait us | Barrier wait us |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4097 | 38.720 | 0.505 / 1.615 | 0.192 | 0.383 | 1.635 | 2.140 | 1.607 | 10.480 | 2.628 | 10.608 |
| 32771 | 45.480 | 1.368 / 1.368 | 0.615 | 1.192 | 11.899 | 2.062 | 11.409 | 14.135 | 13.061 | 1.709 |
| 131075 | 49.460 | 1.229 / 1.229 | 0.602 | 1.195 | 15.784 | 0.935 | 15.273 | 14.282 | 16.898 | 2.121 |

4K MTE2 maximum drops from r13's 17.576 us to 6.343 us. At 32K its mean
drops from 17.175 to 11.899 us; at 128K from 16.761 to 15.784 us. The
Cube arithmetic remains approximately 0.6 us per active core. More effective
feeding and shorter non-Cube phases improve active-core Cube ratio, but
small B1 workloads still spend most time in candidate preparation, transfers,
local/global top-k and synchronization. No redundant arithmetic was added.

Profiler inputs use synthetic candidates; selector acceptance uses candidates
from the live source selector. Do not subtract profiler task time from the
selector timing to infer exact wrapper cost. The 128K coalescing benefit in
the synthetic profile does not erase the small measured live-selector
regression versus r13.

## Unaffected-path final regression

All nine cases pass the frozen <=3% median/P95 regression requirement and
<=3% round spread against refreshed preserved r12. Positive values below
mean slower than r12; none are hidden by cross-case averaging.

| CR | Batch | Context | Median change (%) | P95 change (%) |
| --- | --- | --- | --- | --- |
| 2 | 1 | 4097 | +0.846 | +0.826 |
| 2 | 1 | 32771 | +0.745 | +0.960 |
| 2 | 1 | 131075 | +2.473 | +2.523 |
| 1 | 8 | 4097 | +0.656 | +0.776 |
| 1 | 8 | 32771 | +0.649 | +0.673 |
| 1 | 8 | 131075 | +0.349 | +0.348 |
| 1 | 32 | 4097 | -0.329 | -0.353 |
| 1 | 32 | 32771 | +0.292 | +0.313 |
| 1 | 32 | 131075 | +0.398 | +0.376 |

Raw result: `artifacts/qli-fused/r14/unaffected-regression-r1.json`. The worst
regression is CR2/B1/128K, +2.473% median and +2.523% P95, still within the
unchanged 3% limit. NPU0 was released after every process exited naturally.

Remaining optimization opportunities include the existing public wrapper's
output sorting/masking, H32-specific L1 reservation/tile simplification,
and further small-transfer reduction. Any output-contract fusion should use
an explicit opt-in attribute with the original v3 behavior as default, and
requires its own full correctness/graph/performance matrix. Current r14
acceptance does not depend on any of these unimplemented changes.
