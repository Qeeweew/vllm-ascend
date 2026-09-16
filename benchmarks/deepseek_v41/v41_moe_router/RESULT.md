# V4.1 fused router status

Native r3 numerical, graph and performance gates passed. All **48 exclusive
NPU2 performance cases passed**, using the r3 kernel and corrected r3 binding,
actual scaling=1.5 and graph unroll=256. Native accuracy and changed-input graph
replay passed all **84 tests**. Production dispatch stays disabled pending
full-model integration validation.

The earlier r2 timing and profiling evidence below is retained as diagnostic
history. It is not used for r3 performance acceptance or r3 pipe-utilization claims.

## Implemented

- AscendC AIV-only operation for E384/K6 and E128/K3, explicit image mask,
  optional text hash lookup and bias, caller-owned FP32 weights / INT32 IDs.
- Hash rows preserve lookup order and skip all dynamic sorting and bias work.
  Only the K selected logits participate in score computation (hardware vector
  work is padded to one 64-lane repeat). Initial implementation loads the small
  full logit row into UB; this avoids K separate unaligned DMA requests.
- Invalid text token or selected expert IDs produce whole-row zero/-1 sentinels
  without out-of-bounds loads. Image rows never load their token or hash table.
- Host tiling validation, Python metadata/alias checks, explicit integration
  handoff, independent scalar oracle, NPU accuracy/graph tests and full-chain
  alternating graph-event benchmark are supplied.

CPU validation: `tests/ut/ops/test_v41_moe_router.py`: **22 passed**.
Python Ruff and applicable pre-commit checks pass. Complete native package build
r2 exited 0; installed only into `artifacts/v41-router/r2/opp`, never production.
`package_r2.json` records package/source/object SHA256 and proves raw, snapshot,
copied and installed source equality plus built/installed object equality.
Native validation: **84 passed in 18.42 s**, process exit 0, raw log
`native_r2.log.txt`. Both E128/K3 draft and E384/K6 target pass T0/1/2/4/5/16/64/
128/1024 across hash, mixed and dynamic routes; eight changed-input graph replays
per E/K; invalid-token/table whole-row sentinels; exact tie IDs, negative-tail and
subnormal scores; and baseline-score bias cancellation at the cutoff. Outputs
retain the unchanged rtol=2e-6 / atol=2e-7 weight and exact-ID gates.

Tests loaded the newly rebuilt isolated Torch extension with SHA256
`22d1d373655ecb9d9c7e0b40b58b03b47b8471c7fd57fd84777c1f1665cf841f`;
the production extension/package were untouched.

## Frozen acceptance gates

Selected expert IDs and hash ordering must be exact, including constructed
cutoff cases. Weights use rtol=2e-6 / atol=2e-7. No selected-set recall gate.
Softplus threshold is 20, beta is 1; normalization clamps to FP32 smallest
positive normal, and scaling follows normalization as a separate FP32 operation.

`baseline_contract_r1.json` records the installed NPU baseline independently:
all 24 E128/E384, T1/4/128 tie patterns select equal scores in ascending expert
index order, repeated three times identically. Softplus preserves negative tails
(-20 -> 2.0611537e-9, -100 -> 3.7835e-44); naive Exp/Adds/Ln gives zero and is
incorrect. The candidate now uses compensated log1p with a tiny-input branch.
The candidate passed native numerical validation including
subnormal values and near-cutoff bias cancellation. The independent scalar
oracle preserves the baseline's negative tails.

Complete replaced-chain latency is measured for five alternating rounds with
20 samples per round and equal graph unroll (256 in the final r3 run). T1/4/128 median and P95 must each be
at least 10% lower; other tested rows must not regress by over 3%; both paths'
round-median spread must be at most 3%. Failed cases remain in JSON output.

Run profiling only in the root-approved exclusive NPU window, for example:

```bash
msprof op --kernel-name=V41MoeRouter --output=/tmp/v41-router-op-profile \
  --application="../.venv/bin/python benchmarks/deepseek_v41/v41_moe_router/benchmark.py --rows 1 --experts 384 --modes hash --profile fused --output /tmp/v41-router-profile.json"
```

This is Vector/Scalar/MTE work; Cube utilization is not an optimization target.
Report task wall, launch count, Vector/Scalar/MTE utilization and waits, alongside
whole-chain timings. Explicit user workspace is zero; CANN fixed workspace and
actual allocator peaks must be measured, not reported as zero total memory.

## Remaining steps

1. Shared binding/meta/model integration and full-model verification by root.
2. Further optimization can target the measured r3 Scalar/wait and MTE2 costs;
   keep complete-chain numerical and performance gates unchanged.

## First whole-chain measurements: speed passes, stability does not

`performance_r2_failed_spread.json` retains all 48 cases and every sample from
five alternating rounds (20 event samples/round, equal graph unroll 32). All
median/P95 speed gates passed, but 21 short-kernel cases exceeded the unchanged
3% round-median spread limit (approximately 3–8%). **Overall acceptance is false.**
This first diagnostic used scaling=1.0. The converted checkpoint's
`text_config.routed_scaling_factor` is **1.5**, so the final benchmark now defaults
to 1.5 and records the factor explicitly. Final runs must use the actual factor.
Root subsequently observed external processes outside our PID namespace on all
NPUs; their exact start time is unknown. Exclusivity during these first timing
and profiling runs is therefore uncertain, another reason they are not final
performance acceptance. No external process was touched.

The following are diagnostic medians, not final performance acceptance:

| E/K | T / mode | Baseline us | Fused us |
| --- | --- | ---: | ---: |
| 384/6 | 1 / hash | 48.62 | 3.39 |
| 384/6 | 1 / dynamic | 32.92 | 3.88 |
| 384/6 | 128 / dynamic | 120.22 | 11.99 |
| 128/3 | 5 / hash | 54.87 | 3.14 |
| 128/3 | 5 / dynamic | 37.11 | 3.02 |
| 128/3 | 128 / dynamic | 99.21 | 11.01 |

Single-call incremental allocation was 16,777,728 bytes for the fused operator,
including the fixed CANN workspace; explicit user scratch is zero. Root found a
separate queued-callback workspace ownership issue in this Torch extension, so
final memory/performance acceptance must use the corrected extension. Increase
equal graph unroll after that fix to reduce short-kernel timer/launch noise;
the 3% stability threshold and numerical gates stay unchanged.

## msprof op evidence and next optimization

Raw `OpBasicInfo.csv` / `PipeUtilization.csv` are archived under `profiles/` with
SHA256 and original artifact paths in `profile_r2_manifest.json`. These are
actual `msprof op --aic-metrics=PipeUtilization` runs at rated/current 1800 MHz.
Ratios below are weighted by active AIV cycles; overlapping pipe counters must
not be summed. Profiling replay task duration is separate from graph timing.

| Shape | Task us | Vector | Scalar | MTE2 | MTE3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E384 T1 hash | 5.76 | 6.59% | 52.54% | 21.99% | 2.24% |
| E384 T128 dynamic | 12.14 | 25.18% | 45.06% | 33.74% | 5.39% |
| E128 T5 dynamic | 3.08 | 19.65% | 60.15% | 28.27% | 12.19% |

Cube counters are inapplicable: this operator performs no matrix multiplication.
The measured constraints are Scalar/control overhead and small dependent DMA
requests. Candidate r3 specializes E/K at compile time (128/3, 384/6) to remove dynamic
tiny-loop/address overhead while preserving FP32 operation order. Its complete
isolated package build and source/binary provenance checks passed. The same
**84 native tests passed** against the rebuilt binding with the workspace
ownership fix; raw `native_r3_binding_r3.log.txt`, process exit 0. This test-suite
runtime is not a kernel-performance comparison. The validated r2 package is
retained.

Final r3 package manifest is `package_r3.json`, kernel SHA256
`f475af9b3a86d9a1b2e070c8c5503d8ae02e8207ab84ab59a20add0fa531b072`.
The new extension SHA256 is
`64931ce51efb20c460b64cae5aea4b5eeb24ecb1ac29ec256f3ef62fde36cb13`.

## Final r3 exclusive performance acceptance

Root ran the complete matrix on exclusive NPU2 with process exit 0:
`performance_r3_exclusive.json` records **accepted=true**, all 48 cases, all
samples and unchanged numerical/latency/stability gates. Both paths used five
alternating rounds, 20 samples each, actual scaling=1.5 and equal unroll=256.
`performance_r3_manifest.json` records JSON/log SHA256 and exact kernel/binding
provenance. The 3% round-spread threshold was unchanged.

| E/K | T / mode | Baseline median us | Fused median us |
| --- | --- | ---: | ---: |
| 384/6 | 1 / hash | 48.21 | 2.48 |
| 384/6 | 128 / dynamic | 120.11 | 11.30 |
| 128/3 | 5 / dynamic | 37.65 | 2.36 |
| 384/6 | 1024 / dynamic | 168.33 | 48.96 |

Fused single-call incremental allocation remained 16,777,728 bytes, including
fixed CANN workspace. User scratch is zero. These measurements cover the entire
replaced routing chain; they do not establish full-model throughput or validate
production integration. The r2 msprof counters above remain explicitly r2;
no r3 Cube/Vector/Scalar utilization improvement is inferred from latency alone.

## r3 msprof op follow-up

Root collected three additional r3 `msprof op` profiles on NPU2, all process
exit0. Raw CSVs are archived under `profiles/profile-r3-*`, with SHA256 and
provenance in `profile_r3_manifest.json`. The fixture JSONs say accepted=false
because profile-only mode does not execute the performance matrix; the separate
48-case timing matrix above remains accepted. Current/rated frequency is1800MHz.

Ratios are weighted by active AIV cycles. Wait ratios use summed per-core wait
time divided by summed AIV time; counters overlap and must not be summed.

| Shape | Task us | Vector | Scalar | MTE2 | MTE3 | Wait | Wait IB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E384 T1 hash | 5.22 | 7.23% | 48.28% | 23.61% | 2.18% | 30.76% | 29.88% |
| E384 T128 dynamic | 11.32 | 27.36% | 41.83% | 32.60% | 6.62% | 65.66% | 1.42% |
| E128 T5 dynamic | 3.06 | 19.09% | 55.45% | 28.43% | 7.56% | 49.01% | 6.06% |

Cube remains inapplicable for this pure-AIV router. Scalar activity and waits
remain substantial; T128 dynamic also shows significant MTE2 activity. These
profiles identify remaining costs. Profiling task durations include profiler
replay conditions and must not replace graph-event complete-chain latencies.
The r2 runs had uncertain exclusivity and different scaling, so differences
between r2/r3 counters do not isolate the effect of compile-time specialization.
