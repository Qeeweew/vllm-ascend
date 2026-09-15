# DeepSeek V4.1 CompressorV41 on Ascend 910B3

2026-09-15. Earlier compressor builds passed the original BF16 tolerance checks and graph microbenchmark gates. The r9 gate investigation exposed a systematic normalization precision risk that those checks could miss. The minimal Sqrt+Div compressor fix was installed by the unified r10 full build; its strict NPU correctness suite now passes, and the repeated 38-case graph performance matrix passes its strict latency and noise gates. The additional 12-case closing-group decode matrix passes as well. The initial timing tables below describe the earlier kernel; the final r10 section contains its updated accepted graph results. Eager measurements also remain outside the strict noise gate. These results do not establish eight-card model performance.

## Scope and numerical contract

The kernel contains no GEMM. It accepts an independent projection result:
BF16 `[T,512]` for CR1, FP32 `[T,1024]` laid out as `[kv,score]` for CR2.
CR2 performs a separate two-token softmax in each feature channel, pools in
FP32, rounds to BF16, promotes back to FP32 for RMSNorm with epsilon `1e-20`,
and returns BF16. This intermediate BF16 rounding is intentionally preserved.
RoPE, indexer projection/norm, and main/index cache insertion are excluded.

The output has the full static token-bucket shape, with invalid/non-boundary
rows zeroed. One task per request owns its ring reads and tail writes;
independent interior tasks read raw input only. The op uses 22 KiB UB per AIV,
no user GM workspace, and no cross-core synchronization. The framework may
reserve its standard ACLNN workspace separately.

## Environment and correctness

- Ascend910B3 device 0, CANN 9.1.0, driver 25.5.0.
- PyTorch `2.10.0+cpu`, torch-npu `2.10.0.post4`.
- Fork base `b49962987e89b850586f1819ce8f85daa85a0f81`, with the new sources
  rebuilt through a complete editable install; source hashes are in the JSON.
- Device 0 was reserved for these measurements; stock power/frequency settings
  were used. Other agents worked on separate devices.
- **42 tests passed**, including full H512 shapes through T4096, official
  BF16 rounding, per-channel gating, tiny norms/extreme scores, odd/even chunk
  boundaries, ring wrap, speculative rollback, zero-token/empty-state handling,
  and 50 graph replays changing positions, active counts, inputs and slot reuse.
- The FP32 ring is compared exactly. BF16 latent checks use `rtol=0.008` and
  `atol=0.002`; state copying has no arithmetic tolerance.
- The batched PyTorch benchmark baseline was separately compared against the
  independent CPU oracle on six mixed/chunked cases, exactly.

The JUnit record is [correctness.xml](correctness.xml).

## Measurement method and frozen gates

The baseline uses batched PyTorch gather, pair softmax, BF16 roundtrip,
RMSNorm, output scatter, and ring scatter across all requests. Its metadata
indices are precomputed outside timing. It is a correct simple reference, not
a claim about the fastest possible fused CANN implementation.

Graph results use **256 operator calls captured in each graph**, five rounds
of 20 event samples, and divide each graph interval by 256. This prevents CPU
replay submission and event overhead from dominating 1–3 microsecond kernels.
Each captured path is checked again after replay. Baseline and candidate order
alternate between rounds. All capture-external input tensors, including
baseline gather indices, are kept alive until graph use ends.

The main matrix fixes equal weight for 38 cases: both ratios, T values
`1,2,4,8,16,32,64,128,512,2048,4096`; single-request prefill and, through T128,
one-token-per-request mixed-parity decode. Buckets include three padding rows.
Supplemental all-odd decode covers T1, T8 and T128, so the single-request
group-closing path is measured separately from the pending-state path.

Gates were not relaxed: candidate median no more than 1.03 times baseline,
P95 no more than 1.05 times baseline, equal-weight geometric speedup at least
1.10, and each implementation's round-median spread below 3%.

## Graph results

All 38 main cases pass. Their geometric mean speedup is **13.4138 times**.
The maximum round-median spread, over both implementations, is **0.2843%**.

| Path | T | AscendC median (us) | AscendC P95 (us) | PyTorch median (us) | Speedup |
|---|---:|---:|---:|---:|---:|
| CR1 prefill | 1 | 1.4299 | 1.4330 | 29.1276 | 20.37 |
| CR1 prefill | 128 | 7.5327 | 7.5380 | 83.9712 | 11.15 |
| CR1 prefill | 4096 | 53.2102 | 53.2332 | 138.9630 | 2.61 |
| CR2 closing prefill | 2 | 2.2075 | 2.2152 | 58.9746 | 26.72 |
| CR2 prefill | 128 | 8.3644 | 8.3770 | 141.5995 | 16.93 |
| CR2 prefill | 4096 | 90.3707 | 90.5450 | 323.3856 | 3.58 |

Supplemental all-odd CR2 decode medians are **2.3235 us** for one request,
**3.6862 us** for eight, and **12.3643 us** for 128. Supplemental gates also
pass. Its raw data includes CR1/prefill control cases; they are not folded into
the 38-case main geometric mean.

Large prefill remains a tuning opportunity: the simple row-wise AIV kernel
does not overlap adjacent rows' DMA with Vector work. The current data supports
its advantage over the stated baseline; it does not identify the limiting
pipeline without a dedicated profiler trace.

## Eager results and limits

All 38 eager cases meet the median/P95 comparison thresholds, and geometric
mean speedup is **3.9600 times**. However, 25 cases exceed the 3% round-median
noise limit. **Eager strict performance acceptance remains open.** These event
intervals include CPU dispatch gaps and are not pure device-kernel durations.
Increasing graph unroll fixed the microkernel measurement issue without
changing the operator, but does not prove an eager scheduling SLO.

Eight-card end-to-end validation, independent GEMM/cache-store timings, complete
model precision, and model-level TTFT/TPOT remain separate integration gates.
These microbenchmarks repeatedly use stable tensors and do not establish
cold-cache or changing-request end-to-end performance.

## Independent projection API evaluation

The installed torch-npu registers `aten::mm.dtype` and `aten::mm.dtype_out` in
`torch_npu/csrc/aten/npu_native_functions_by_codegen.yaml`, lines 2500/2503,
with the `op_api` implementation. Both of these calls were verified on device 0:

```python
y = torch.mm(hidden_bf16, weight_bf16.T, out_dtype=torch.float32)
y = torch.ops.aten.mm.dtype(hidden_bf16, weight_bf16.T, torch.float32)
```

No new GEMM kernel is needed. The CANN installation also exposes
`aclnnMmGetWorkspaceSize(self, mat2, out, cubeMathType, ...)` in
`include/aclnnop/aclnn_mm.h` and the general `aclnnMatmul` API in
`include/aclnnop/aclnn_matmul.h`. The verified production-facing interface is
the torch `out_dtype` overload; header presence alone is not treated as proof
of every low-level dtype combination.

The inspected upstream
[MmKernelNpuOpApi.cpp](https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/MmKernelNpuOpApi.cpp)
allocates the requested output dtype and calls `aclnnMm` for ND inputs (lines
44–53 in the source retrieved on 2026-09-15). Its NZ branch calls
`aclnnMatmulWeightNz`, whose installed header documents BF16/FP16 output only.
The timings below therefore apply to **ND BF16 weights**. Set the existing
`skip_weight_nz_conversion=True` flag on the compressor projection module so
the normal unquantized-linear post-load hook does not change that contract.

Projection tests use the actual BF16 checkpoint weights
`layers.2.attn.compressor.wkv.weight` and `wgate.weight`, concatenated into
`[1024,5120]`, with deterministic synthetic BF16 activations. The concatenated
weight SHA256 is
`d0a1fa565d8d787af9b86933816e58db903b656295795cf378d599d675e14f57`.
`torch.npu.matmul.allow_hf32` is explicitly false. Each graph captures 16
independent GEMMs and is checked for identical outputs after replay; timing
uses five rounds of 20 samples. The largest round-median spread is 0.199%.

| T | FP32 inputs/weights (us) | FP32 with activation cast (us) | BF16 inputs, FP32 output (us) | Speedup vs FP32 |
|---:|---:|---:|---:|---:|
| 1 | 35.735 | 36.966 | 10.886 | 3.28 |
| 2 | 35.517 | 36.896 | 10.900 | 3.26 |
| 4 | 35.519 | 37.223 | 10.955 | 3.24 |
| 8 | 35.514 | 37.893 | 11.118 | 3.19 |
| 16 | 15.605 | 17.996 | 11.416 | 1.37 |
| 32 | 23.701 | 26.209 | 11.852 | 2.00 |
| 64 | 23.646 | 26.326 | 13.560 | 1.74 |
| 128 | 37.911 | 40.868 | 17.944 | 2.11 |
| 512 | 87.071 | 92.269 | 29.103 | 2.99 |
| 2048 | 326.044 | 339.927 | 101.686 | 3.21 |
| 4096 | 606.468 | 634.016 | 173.108 | 3.50 |

BF16-input/FP32-output NRMSE relative to NPU FP32 GEMM is `6.29e-7`–`6.60e-7`.
Against independent FP64 reference samples (64 output columns at stride 16),
its NRMSE is `2.93e-7`–`3.43e-7`; the FP32-input path is `5.46e-7`–`7.23e-7`.
These are different accumulation orders, not bitwise-identical projections.
The incorrect control of BF16 GEMM output followed by `.float()` has NRMSE
`1.64e-3`–`1.71e-3`. At most 0.0061% of the correct FP32 outputs land on BF16
grid points, independently confirming that output precision is not truncated
to BF16.

For the BF16 source checkpoint, the recommended CR2 projection is therefore
the existing `torch.mm(..., out_dtype=torch.float32)` with BF16 weights. It
avoids a permanently promoted weight copy and a full activation cast while
keeping the GEMM completely separate from compression. Model-level numerical
validation remains required before treating the accumulation-order difference
as accepted for the complete model. CR1 still requires BF16 projection output.

Reproduce with
`python tests/e2e/single_node/ops/benchmark_compressor_v41_projection.py --output /tmp/projection.json`.
Full errors and event samples are in [projection.json](projection.json), SHA256
`9ed63a28f5e3f0ee1b72425a15493d5aae412ed421685a8bf3e05a56a9dd03e2`;
[projection_summary.csv](projection_summary.csv) provides the compact table.

## Reproduction and raw records

Run after the complete editable build/install:

```bash
python -m pytest --confcutdir=tests/e2e/single_node/ops \
  tests/e2e/single_node/ops/test_compressor_v41.py -q
python tests/e2e/single_node/ops/benchmark_compressor_v41.py \
  --graph --graph-unroll 256 --iterations 20 --output /tmp/compressor-graph.json
python tests/e2e/single_node/ops/benchmark_compressor_v41.py \
  --graph --graph-unroll 256 --iterations 20 --tokens 1 8 128 \
  --decode-parity odd --output /tmp/compressor-closed.json
python tests/e2e/single_node/ops/benchmark_compressor_v41.py \
  --output /tmp/compressor-eager.json
```

Set `--source-sha` and `--hardware-notes` for an attributable run. JSON files
include the source fingerprints, all event samples, medians/P95, round
variability, and gate decisions; [summary.csv](summary.csv) is a compact view.

| Record | SHA256 |
|---|---|
| [graph.json](graph.json) | `8c6140d488069446d2a1dba5c953fdaaa9c045a66dfd072f20e937959d3117b2` |
| [graph_closed.json](graph_closed.json) | `35cdb753575f29c2b6d916c76dd197be59769a4fc10a586d3d799243f0942fa7` |
| [eager.json](eager.json) | `b137896ce93351a848d52bc5624aa11e510925999e6d261e5a851929635fa0b8` |
| [correctness.xml](correctness.xml) | `6b79f301fc06a883e8eb50b818e6c0852dc523046f8de65ea70dc6663149e313` |

## r10 normalization correction and stricter acceptance

The previous 42 tests established storage, request ownership, CR1/CR2 shapes, pooling and graph/state behavior under a BF16 output tolerance of `rtol=0.008, atol=0.002`. Those limits allow roughly a 0.2% multiplicative normalization error; small output values are additionally protected by absolute tolerance. Testing many shapes does not compensate for an insufficient numerical acceptance criterion. An all-ones output is also inadequate: a biased pre-round value can round back to the correct BF16 value.

The separate gate investigation measured basic arch22 `Rsqrt(1.0)` as `0.998046875`, a -0.1953125% reciprocal bias at that input. This is a measured primitive point, not a claim that every reciprocal input has exactly that bias. Compressor used the same basic primitive. Its prior accuracy label is therefore superseded until stricter validation completes.

The r10 source change is confined to `Normalize`: `Duplicate(squared, 1, 1); Sqrt(sum, sum, 1); PipeBarrier<PIPE_V>(); Div(sum, squared, sum, 1);`, followed by the existing Vector-to-Scalar synchronization. The square buffer is dead after the completed reduction and is reused for the numerator. No additional allocation, tiling field, ABI argument, scalar divide, GEMM, request-state logic or cross-core synchronization was added. CANN 9.1.0 arch22 interface and `dav_c220` implementation were checked: these FP32 count APIs emit `vsqrt` and `vdiv` respectively. All source changes were rebuilt through the parent's full editable r10 build and installed successfully. Source, both installed OPP kernel objects, metadata and the torch binding were fingerprinted in `r10_binary_manifest.json`; the updated objects are newer than the source change. No new NPU tests were run against the previous installed binary.

New numerical tests isolate normalization with 128 independent H512 rows and BF16 weights. CR2 repeats each KV row twice with zero scores, making its pairwise pooling exact before the mandated BF16 roundtrip. Scales `1e-15`, `1e-5`, `1`, and `1e5` cover epsilon-dominated, small, ordinary and large norms. An independent FP64 RMSNorm oracle rounds only the final weighted result to BF16. Each CR/scale combination must satisfy all three conditions:

- NRMSE relative to the rounded FP64 oracle below `2e-4`.
- Absolute multiplicative gain bias `abs(dot(actual,expected)/dot(expected,expected)-1)` below `5e-5`.
- Every finite lane differs by at most one BF16 ULP.

The CPU control deliberately injects `0.998046875` gain into the **unrounded** normalized output, then rounds to BF16. It must fail the strict gate. Applying that multiplier after BF16 rounding would often round back to the same representable value and is not a valid bias-sensitivity control. All four scales and both ratios pass the CPU reference checks and reject the biased control: **12 CPU reference tests passed** overall. NPU acceptance adds eight strict normalization cases to the existing storage/pooling/graph suite. The installed r10 run passes **54 tests**: 12 CPU reference and 42 NPU tests, including all eight strict CR/scale cases, original pooling/state cases and fifty dynamic graph replays. Results are in `correctness_r10.xml`; the complete run took 24.06 seconds. After an initial run was stopped before any samples to make room for the parent runner, the subsequent isolated performance window completed both unchanged graph matrices. Their strict acceptance and raw records are documented in the final r10 section.

`CompressorV41Backend` also now supplies `get_impl_cls()` with a no-op `update_graph_params` hook. The runner enumerates all backends during full graph replay, including storage-only compressor state; inherited `get_impl_cls()` would raise. Metadata builder calls already refresh the fixed-address tensors, so no mutable graph task parameters are needed. The metadata and graph-hook CPU suite passes **6 tests**. This is not a substitute for full graph device replay after installation.

## Checkpoint projection and normalization weight audit

Read-only safetensor inspection found 11 main compressor tensors across owners 2, 8, 14 and 20 in `/mnt/models/DeepSeek-V4.1-Flash`. All are **BF16**: every wkv/wgate is `[512,5120]`; each norm is `[512]`. Owners 2, 8 and 14 have wgate, while owner 20 has only wkv, matching CR2/CR1 projection structure. These dense compressor weights need neither FP8 dequantization nor expert INT4 conversion.

All 512 elements of each norm were read. Projection statistics use only 4096 elements per matrix (16 evenly spaced rows, first 256 columns); they are samples, not full-weight extrema. Raw observations are in `checkpoint_weight_audit.json`.

| Owner | Norm min | Norm max | Norm RMS | Sample wkv RMS | Sample wgate RMS |
|---|---:|---:|---:|---:|---:|
| 2 | 0.0022583 | 0.503906 | 0.393895 | 0.0247553 | 0.0225584 |
| 8 | 0.0000691414 | 0.855469 | 0.539992 | 0.0252118 | 0.0316221 |
| 14 | -0.000930786 | 0.867188 | 0.672731 | 0.0245393 | 0.0332709 |
| 20 | -0.00299072 | 1.11719 | 0.822885 | 0.0284424 | absent |

Near-zero and negative norm weights are real; implementations must not clamp or assume positive gamma. The kernel multiplies the stored BF16 gamma directly. These checkpoint statistics do not measure hidden-state, projected-KV or score activation distributions, which require a model run.

The independent projection path remains ND BF16 inputs/weights. CR1 returns BF16 directly; CR2 uses `torch.mm(hidden_states, weight.t(), out_dtype=torch.float32)` and must not replace it with BF16 output followed by `.float()`. Earlier independent FP64 sampled validation already established that the NPU path retains true FP32 output precision. The kernel normalization fix changes neither GEMM nor the checkpoint loading/packing order `[wkv, wgate]`.

## r10 graph performance acceptance

After the parent runner exited on a CPU metadata ABI error, device 0 was reserved exclusively for this benchmark; other agents continued CPU-only fixes. The completed primary matrix contains the same 38 cases as the earlier result, CR1/CR2, prefill/decode, T1–4096, with 256 operations per graph replay, five alternating-order rounds and twenty timed replays per round. All raw event samples, source fingerprints and device notes are retained in `graph_r10.json`; `summary_r10.csv` compares each shape with the historical kernel.

**38/38 cases pass**: median no slower than 1.03× the current torch baseline, P95 no slower than 1.05×, and both candidate and baseline round-median spread below 3%. Equal-weight geometric speedup is **12.9719×**, above the 1.10× aggregate threshold. Maximum round-median spread across all main candidate/baseline series is **0.7891%**. These are graph microbenchmarks of compression/state/RMSNorm only; independent GEMM, cache insertion and runner costs are excluded.

| CR | Mode | T | Median µs | P95 µs | Speedup vs torch |
|---|---|---:|---:|---:|---:|
| 1 | decode | 1 | 1.483 | 1.488 | 19.25× |
| 1 | decode | 8 | 2.431 | 2.435 | 16.08× |
| 1 | decode | 128 | 7.981 | 7.987 | 10.17× |
| 1 | prefill | 4096 | 55.663 | 55.677 | 2.40× |
| 2 | mixed-parity decode | 1 | 1.615 | 1.619 | 2.32× |
| 2 | mixed-parity decode | 8 | 3.609 | 3.615 | 22.99× |
| 2 | mixed-parity decode | 128 | 10.565 | 10.577 | 18.39× |
| 2 | prefill | 4096 | 93.286 | 93.557 | 3.42× |

The geometric ratio of r10 latency to historical candidate latency across the same 38 cases is **1.00683** (approximately +0.68%). This is a comparison across separate runs, not a paired old/new binary A/B experiment; it must not be interpreted as an isolated instruction-cost measurement. Each r10 acceptance decision instead compares against its contemporaneous torch baseline.

CR2 mixed-parity T1 decode opens a group and does not normalize. A separate odd-position matrix explicitly closes every decode group; its results are recorded separately below. Historical eager measurements did not pass their noise gate and have not been reaccepted here. Eight-card runner throughput remains outside this microbenchmark's evidence.

The separate `graph_closed_r10.json` matrix passes **12/12 cases**, with geometric speedup **14.7328×** and maximum candidate/baseline round-median spread **0.1102%**. It uses T1/8/128 for both CRs and prefill/decode, with all decode starts odd. Thus every CR2 decode request actually executes pool, the intermediate BF16 roundtrip and precise normalization.

| CR2 closing decode T | Median µs | P95 µs | Speedup vs torch |
|---:|---:|---:|---:|
| 1 | 2.358 | 2.362 | 26.44× |
| 8 | 3.844 | 3.851 | 27.78× |
| 128 | 13.247 | 13.252 | 16.50× |

Device 0 was released immediately after both processes exited; no performance samples overlap the resumed eight-card runner. The accepted scope is the compressor graph microbenchmark. No updated eager noise acceptance or model-level throughput is claimed.

| r10 record | SHA256 |
|---|---|
| [graph_r10.json](graph_r10.json) | `10dde08693871b85d9be84d4fac7eb1e9c4a7c3d76d6f5d69e775f59c72114d6` |
| [graph_closed_r10.json](graph_closed_r10.json) | `a95477a1d501125815feccabb128dcd96bfcc56d2a1a60cefbca9a83edf531d9` |
| [summary_r10.csv](summary_r10.csv) | `1041d3c04606129114c6ae2e3be0776a4c4c9932198447c9457eff901b3f8b06` |
| [correctness_r10.xml](correctness_r10.xml) | `3ff1aade655ecb52ca620b6318a59c192f002b713dcd0c4d09d742b3f3469aaf` |
| [r10_binary_manifest.json](r10_binary_manifest.json) | `6af06949266c13febb20210c8ffd13787d32199641d4e64556c3d34e5f03cfef` |
