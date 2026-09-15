# V4.1 W4A16 decode verification and profiling

Measured 2026-09-15 on Ascend 910B3 device 2, CANN 9.1.0,
PyTorch 2.10.0 / torch-npu 2.10.0.post4. Prefill keeps CANN GMM.
The existing personal-branch decode kernel has been migrated and verified;
eight-card model acceptance is still pending, so its runtime switch defaults off.

## Implementation and correctness

The source was migrated from personal-branch commit
`274dd30c3230804e35b7985e288dacbde3326ece`. Its split-K algorithm and single
fused launch are retained. Unused batch GEMM code was removed.

- I288 uses K blocks 128,128,32 and nine scales, without padding weights.
- BF16 activations, products, group sums and signed scale accumulation use FP32.
  INT4 first casts exactly to FP16 integers, then to FP32; activation values
  are never narrowed to FP16. SwiGLU rounds to BF16 before W2.
- Clamp remains `gate <= 10`, `-10 <= up <= 10`; routing applies once after W2.
- Native and CANN share one packed layout. Repacking expands one expert at a
  time, removing the old full-layer INT32 temporary.
- Symmetric CANN GMM omits optional zero offsets. All six tested K/N/M cases
  are bit-identical to explicit zero offsets, including negative scales and
  q=-8. This saves **101.25 MiB/layer, 3.955 GiB/rank across 40 layers**.

Two synchronization bugs were caught before enabling dispatch. AIV_ONLY direct
launch omitted the FFTS address needed by SyncAll; restoring the original
MIX_AIV_1_0 launch mode fixed the hang while retaining vector-only computation.
The old final-cast loop also lacked a backward MTE2 dependency: B32 exposed
an overwrite of earlier tiles. MTE3_MTE2 now protects the next tile load.
Full build r4 includes both fixes; no kernel edits followed that build.

Validation:

- Native H5120/I288/top6 B1/2/8/32/64, limits 0/10, BF16 range regression,
  changed-input/route graph replay, invalid input rejection: **13 passed**.
- Exact repacking plus CANN omitted-offset equivalence: **9 passed**.
- An extra B32 precheck made the combined run **23 passed**.
- Real method dispatch with an E384 bank, expert IDs through 383 and graph
  replay: **1 passed**.
- Existing W4A16, dispatch boundary/fallback and full AscendConfig UT:
  **150 passed**.
- E384 benchmark reference checks: native NRMSE at most **0.000435**;
  production CANN NRMSE at most **0.005826** against the independent reference.

The native precision gate is NRMSE < 0.006 and peak-relative error < 0.015.
The reference forms effective weights in FP32 and does not imitate the old
FP16 group accumulator. Full-model activation/logit quality remains a separate
integration requirement.

## Production-path timing and selected threshold

The CANN comparison includes the actual AllGather route initialization,
INT4 grouped matmuls, clamp plus npu_swiglu, and token unpermute, including its
BF16 routing-weight cast. TP communication and shared experts are outside
this operator measurement and must be included in eight-card acceptance.

Each initial case used 5 rounds × 20 individual NPU event measurements after
warmup. Spread routing uses up to 384 distinct experts. Hot routing uses the
same six distinct experts for every token. Cold runs touch 512 MiB outside
the timing interval, exceeding twice the platform-declared 192 MiB L2.

The hot graph workload determines the safe shape threshold:

| B | Native median µs | Native P95 µs | CANN median µs | CANN P95 µs | Native speedup |
| --- | --- | --- | --- | --- | --- |
| 1 | 114.91 | 118.76 | 342.27 | 345.64 | 2.98× |
| 2 | 135.03 | 149.06 | 335.30 | 347.18 | 2.48× |
| 4 | 236.58 | 249.00 | 342.16 | 356.48 | 1.45× |
| 8 | 416.82 | 429.86 | 352.77 | 359.10 | 0.85× |
| 16 | 796.01 | 808.04 | 354.54 | 366.14 | 0.45× |
| 32 | 1483.86 | 1502.56 | 359.76 | 381.34 | 0.24× |
| 64 | 2921.58 | 2942.50 | 379.91 | 397.54 | 0.13× |

**Select native only for decode B ≤ 4.** B8 regresses 18% on hot graph traffic;
larger hot batches regress substantially, even though spread/cold traffic
shows approximately 3× speedup through B64. A batch-only threshold above 4
would therefore violate the regression gate.

Across the four initial production comparisons (eager spread, graph cold
spread, graph hot, eager cold hot), selected B1/2/4 cases have an equal-case
geometric mean speedup of **2.74×**. The narrowest margin is eager cold hot B4:
261.63/264.22 µs native median/P95 versus 310.50/312.26 µs CANN. These are
operator measurements, not a claim of end-to-end model speedup.

A 5×100-sample hot-graph repeat retained the B4 improvement (238.46/242.62 µs
versus 339.22/345.76 µs). A second repeat increased warmup to 1000 calls and
measurement to 5×300 calls:

| B | Native median/P95 µs | CANN median/P95 µs | Worst-round speedup |
| --- | --- | --- | --- |
| 1 | 98.19 / 111.64 | 316.06 / 331.64 | 2.87× |
| 2 | 128.98 / 148.22 | 332.32 / 339.58 | 2.31× |
| 4 | 240.96 / 250.98 | 340.32 / 351.02 | 1.37× |

Worst-round speedup pairs the slowest native round median with the fastest
CANN round median. Native round medians still vary by 3–12%; frequency was
not locked, so this experiment cannot certify small percentage improvements
or identify the cause of that variation. The conservative comparison still
exceeds 1.10× for every selected shape, supporting the opt-in threshold. All
individual samples and round medians are retained. Final eight-card acceptance
must recheck timing stability under a sustained model workload.

Raw results:

- `w4a16_decode_eager_spread_910b3.json`
- `w4a16_decode_graph_cold_spread_910b3.json`
- `w4a16_decode_graph_hot_910b3.json`
- `w4a16_decode_eager_cold_hot_910b3.json`
- `w4a16_decode_graph_hot_recheck_910b3.json`
- `w4a16_decode_graph_hot_steady_910b3.json`

Files ending `.provisional.json` used a slower Torch composition baseline and
are retained only as investigation history; they do not set the threshold.
`cann_w4a16_baseline_910b3.json` records the earlier six-expert GMM-only checks.

Reproduce the production comparison from the repository root:

```bash
python benchmarks/deepseek_v41/benchmark_w4a16_decode.py \
  --device 2 --experts 384 --routing hot --graph \
  --output hot-graph.json
```

Add `--cold` for the cache sweep, or omit `--graph` for eager execution.
Use `--batches 1 2 4 --iterations 300` to repeat the selected shape gate.

## Opt-in integration and remaining gate

```text
--additional-config '{"enable_w4a16_decode": true}'
```

The default is false until eight-card model acceptance. The guarded path
requires BF16, group32, H5120/I288/E384/top6, B≤4, TP without EP, SILU,
output routing weights and explicit host decode metadata. Prefill, mixed
batches, missing metadata, other shapes, LoRA and EPLB keep CANN. In particular,
missing metadata during a graph capture does not guess that a batch is decode.

The V4.1 cache builder now publishes host execution counts from the runner's
CPU prefill flags and query boundaries. A one-token prefill (including a
cached prompt tail) stays prefill; an unavailable host flag keeps the fallback.
These counters do not control device cache addressing. A uniform single-token
capture explicitly declares decode, since dummy capture can inherit stale
request flags; later ordinary builds restore actual request classification.
Hybrid metadata dictionaries can start with compressor state, so the MoE guard
selects a cache metadata entry carrying execution counts instead of relying on
dictionary insertion order.

CPU regression uses real `AscendCommonAttentionMetadata` and
`AscendV41CacheMetadata`, covering pure decode, one-token prefill, mixed batches,
padding, missing host state and capture/rebuild. The metadata, decode-dispatch
and common-metadata ABI suites jointly pass 68 cases. This validates branch
selection, not native graph execution or full-model performance. The runner
must also exclude incomplete prompts from uniform decode graph selection;
otherwise a captured decode branch can bypass the eager prefill guard.

The method returns FusedExpertsResult. Existing TP finalize/allreduce and
shared-expert stream scheduling retain their ownership; the fused kernel
exposes no internal GMM stage events. No new environment variables are added.

The actual TP8 runner now selects native decode and replays its captured
graph with E384 synthetic weights; all tokens and selected logprobs match
CANN. Three real device-weight layers also load and execute through the
ordinary runner. Isolated real MoE inputs match the FP32 native contract
more closely than CANN's additional BF16 rounding boundaries; see
[W4A16_REAL_NUMERICS.md](W4A16_REAL_NUMERICS.md).

Real-run numerical acceptance remains open: same-engine CANN repetitions
already differ at first-layer attention, before MoE. This is being localized
with all-rank projection/reduction traces, rather than changing native MoE
arithmetic. Remaining acceptance includes full-model loading and quality,
full-size host Engram, stable operator timing, TPOT/TTFT including
communication, and final model memory/profile checks. Enable by default
only after those gates pass.
