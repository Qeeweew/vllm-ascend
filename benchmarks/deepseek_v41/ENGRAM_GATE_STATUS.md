# Engram gate acceptance

The clean r9 build passes **17 correctness tests and all eight graph
accuracy/performance gates** on one Ascend 910B3. These results cover only the
post-wkv normalization, gate, and residual update. Host gather, H2D, TP
collectives, GEMM, and full-model latency require separate profiling.

| Tokens | Graph median (us) | p95 (us) | Composed baseline (us) | Speedup | NRMSE |
| --- | --- | --- | --- | --- | --- |
| 1 | 3.551 | 3.566 | 106.024 | 29.86x | 0 |
| 2 | 3.846 | 3.861 | 128.730 | 33.47x | 8.33e-9 |
| 8 | 5.870 | 5.886 | 147.690 | 25.16x | 1.50e-6 |
| 16 | 8.320 | 8.332 | 160.079 | 19.24x | 8.68e-6 |
| 32 | 12.296 | 12.311 | 182.394 | 14.83x | 2.24e-6 |
| 64 | 19.345 | 19.373 | 218.711 | 11.31x | 4.32e-6 |
| 256 | 61.608 | 61.632 | 377.481 | 6.13x | 1.24e-5 |
| 1024 | 231.874 | 231.896 | 1487.593 | 6.42x | 1.06e-5 |

[Measured results and eager timings](engram_gate_910b.json) retain all samples
and acceptance outcomes. Timing uses all-active masks, 32 captured invocations,
10 replays per device-event interval, and 50 samples. Times are divided by the
number of invocations; they measure amortized device work, not single-request
wall latency. The baseline is a device-only composition of the same equations.
Its separate CPU-oracle audit achieved NRMSE `3.35e-8`.

Acceptance requires NRMSE below `2e-4`, elementwise `rtol=0.008`/`atol=0.002`,
and exact masked-row pass-through. Graph median limits are 25 us for T=1–16,
40 us for T=32–64, and 250 us for T=256–1024, with at least 2x baseline speedup.
No tolerance was relaxed. The 17 tests include three CPU contract tests and
14 NPU tests: T=1/2/8/16/64/257/1024, zero and tiny norms, saturated gates,
masked NaN padding, in-place output, exact-unit-norm regression, and 20
changed-input/mask graph replays without intervening host result reads.

The kernel uses complete-row reductions and 153696 bytes UB per AIV, with no
algorithmic GM intermediates or inter-core synchronization. T=1 uses four AIVs.
GEMM remains a separate operation.

Reproduce from the workspace root (torch 2.10.0+cpu, torch-npu 2.10.0.post4,
CANN 9.1.0, aarch64; measurements used NPU 1):

```bash
OMP_NUM_THREADS=8 .venv/bin/python -m pytest \
  --confcutdir=vllm-ascend/tests/e2e/single_node/ops \
  vllm-ascend/tests/e2e/single_node/ops/test_engram_gate.py -q
OMP_NUM_THREADS=8 .venv/bin/python \
  vllm-ascend/benchmarks/deepseek_v41/bench_engram_gate.py \
  --device 1 --repeats 50 \
  --output vllm-ascend/benchmarks/deepseek_v41/engram_gate_910b.json --enforce
```

The full clean build/install log is
`/tmp/deepseek-v41-kernels-build-r9-production.log`. Repository source matched
the generated compiler source. Installed object
`EngramGate_5acf5f4e33f5a76ff66b73d02067b13f.o` has SHA256:

```text
e0f3a72fc3c25008891835e0e01028349c5616017cea28b95d7fcd52d1805eba
```

Two reproducibility findings affected earlier runs. First, generated `.done`
targets reused stale kernel source and objects despite a successful pip build;
the full-build script now clears generated trees. Check the copied source and
installed binary fingerprint when validating kernel changes. Second, arch22
basic `Rsqrt(1)` returned `0.998046875` in a verified diagnostic build, causing
unacceptable gate error even with exact square sums and raw dots. Production
uses CANN normalization's `Sqrt` plus `Div(1, sqrt)` and precise sigmoid division.
[Raw FP32 diagnostics](engram_gate_diagnostics_910b.json) preserve that evidence;
all diagnostic code and markers were removed from the accepted kernel.
