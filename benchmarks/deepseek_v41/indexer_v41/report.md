# DeepSeek V4.1 CSA indexer on 910B

Status: the independent selector is functionally validated. Performance acceptance is **open**: 23/27 latency gates pass; all 27 measurement stability gates pass. This is not full-model or eight-card acceptance, and no INT8 model-quality claim is made.

## Implemented interface and verified semantics

`vllm_ascend/models/deepseek_v4/indexer.py` appends `AscendIndexerV41Metadata` and `AscendIndexerV41Ops`; existing V4 classes are unchanged. Projections, RMSNorm, RoPE and K-cache ownership remain separate from this selector.

- Query INT8 `[T,32,128]`, weights/query scales FP16 `[T,32]`, paged key INT8 `[blocks,block_size,1,128]`, key scales FP16 `[blocks,block_size,1]`.
- CR1/CR2 are explicit. `seqused_k` contains compressed lengths. `cmp_residual_k` contains original context remainder for CR2; **CR1 requires None**, even an all-zero tensor is rejected by native tiling. Metadata `max_seqlen_k` uses original-token units.
- Query visibility is `floor((original_context - query_count + local_query_index + 1)/ratio)`.
- Source layer 20 computes per-eight-position block maxima, pins the newest reachable partial block, and selects 2048 blocks. Later index source layers use those block IDs but their own Q and signed weights to select 512 positions. Candidate source/consumers in this checkpoint are CR1.
- Final position IDs are increasing, with invalid slots -1 at the end; candidate block IDs retain score order. Padding rows are masked by device `cu_seqlens_q[-1]`; no `.item()` or CPU length synchronization is used.
- Query heads are replicated across ranks, matching upstream CUDA V4.1. Reusing V4's CR4 compressor, 64-head assumptions or Hadamard rotation would be incorrect.
- Graph replay requires stable input buffers and refreshed QLI scheduling data. Tests update lengths/Q, build fresh schedule and copy it into the captured schedule buffer before replay. Candidate buffers produced in the same graph feed consumers directly.

The native operator is `npu_quant_lightning_indexer_v3`; no new C++ binding or kernel was added for this component. Its numerical score is proportional to:

```python
qk = relu((q_int8 @ k_int8.T) / 1024).half()
head_weight = (weights_half * query_scale_half).half()
score = sum(qk.float() * head_weight.float(), head_dim) * key_scale_half.float()
```

The omitted global factor 1024 does not affect ranking. Both intermediate FP16 roundings matter and are present in the independent oracle. Signed weights are preserved.

## Correctness evidence

`test_indexer_v41.py` exercises real 32x128 query heads, top512, true 2048x8 candidate filtering beyond 16K positions, signed weights, newest partial-block pinning, CR2 odd/even visibility, zero completed groups, packed requests, graph padding, physically gapped K/scale caches, source-to-consumer Q/weight changes, and ten graph metadata/candidate replays. Selection checks permit only a numerically tied cutoff swap, not a blanket recall tolerance. CPU wrapper tests cover rejected V4 ratios/modes, empty shapes and FP32 sorting boundaries. Original test results are in `correctness.xml`.

Dynamic INT8 quantization is per last-dimension 128 values, with FP16 output scales. Comparing against CPU divide+RNE found 11/32768 one-code differences solely at exact half-integer boundaries; the test only permits these reciprocal-versus-division rounding boundaries (distance <= 1e-5), and scales match exactly including zero rows.

## Numerical audit: INT8 is a different approximation

Seed 41 synthetic BF16 Gaussian Q `[8,32,128]`, K `[32771,128]`, signed BF16 weights. The common baseline uses the same unquantized BF16 inputs with FP32 QK/head reduction. MXFP4 reference follows the official group32, ceil-power-of-two E8M0 scale, E2M1 RNE, BF16 fakequant roundtrip. INT8 scores use actual 910B quantized codes/scales and the native FP16 intermediate rounding.

| Approximation | Score NRMSE vs common baseline | top512 recall vs common baseline |
|---|---:|---:|
| Official-style MXFP4 fakequant | 0.158854 | 81.4941% |
| INT8 native pipeline | 0.00905446 | 98.9014% |

INT8 versus MXFP4 score NRMSE is 0.159957; top512 overlap is 81.3721%. This comparison separates two approximations: lower mutual overlap does not itself imply INT8 is worse. These are synthetic inputs, not checkpoint activations, language benchmarks or model-quality evidence. FP8/MXFP4 bytes and scales must first be interpreted using their real format; they cannot be reinterpreted as INT8. The wrapper accepts already projected/rotated BF16 or FP16 vectors for fresh INT8 quantization.

## Profiling result and optimization

910B3 device 0, 20 AIC/40 AIV, CANN 9.1.0, torch 2.10.0 + torch-npu 2.10.0.post4. Three alternating measurement rounds, 12 event samples per round. Native graph unroll 64; dense/gather baselines unroll 4 to limit dense intermediate memory. All captured tensor-owning closures stay alive. Reported units are microseconds per selector invocation; metadata construction, projection, RoPE, quantization/cache writes and cross-rank communication are excluded.

The stage audit found CANN INT32 sorting was the dominant extra cost: B1/B8/B32 sort latency was 69.76/399.22/1520.77 us, versus exact FP32 sort plus INT32 cast 6.46/11.30/16.05 us. The wrapper now selects FP32 sorting only when the static cache addressing bound is <= 2**24. Every valid position is exactly representable and sentinel 2**24 is exact. Larger spaces retain INT32 sorting. CPU tests verify adjacent IDs around this boundary.

The dense baseline uses the same faster FP32 ordering, a fully batched FP32 QK, the native FP16 intermediate rounding, causal/candidate masks and source block selection. Its quantized-input conversion and candidate membership setup are precomputed outside timing, so it is deliberately competitive. No Python per-query loop or host topk occurs inside the timed baseline.

Acceptance thresholds declared in `graph.json`: native median <= dense median, native p95 <= dense p95*1.05, and maximum round-median spread <= 3%. Final geometric mean speedup is 2.30835x; **four latency gates fail**, all with batch 1: consumer at S4097/32771/131075 and off at S131075. All 27 stability gates pass. Do not describe aggregate speedup as universal acceptance.

Consumer measurements, including an experimental path that really gathers only candidate K:

| Batch | Compressed context | Native wrapper us | Dense baseline us | CANN candidate gather + bmm us |
|---:|---:|---:|---:|---:|
| 1 | 4097 | 104.15 | 97.53 | 632.20 |
| 1 | 32771 | 284.19 | 133.44 | 422.16 |
| 1 | 131075 | 484.70 | 205.66 | 429.22 |
| 8 | 4097 | 120.88 | 143.38 | 1069.45 |
| 8 | 32771 | 300.13 | 337.16 | 887.34 |
| 8 | 131075 | 500.91 | 1812.89 | 892.59 |
| 32 | 4097 | 203.81 | 250.74 | 4497.05 |
| 32 | 32771 | 567.63 | 1741.39 | 1977.93 |
| 32 | 131075 | 1436.55 | 7688.29 | 1976.00 |

Original pre-optimization data is retained in `graph_int32_sort.json`; final data is `graph.json`, and `summary.csv` contains compact comparisons. The full native stage audit is retained in `stages.json` (captured before wrapper sorting optimization).

## Why candidate mode currently does not reduce QK work

In `quant_lightning_indexer_v2_kernel_arch22.h`, the S2 loop spans the full visible sequence. `quant_lightning_indexer_v2_service_cube_arch22.h::KeyNd2NzForPA` loads contiguous logical S2 through the page table and does not consume candidate IDs. Candidate membership is applied later in `quant_lightning_indexer_v2_service_vector_arch22.h::ProcessVec1`. Thus consumer mode adds membership work to full QK; it is not sparse matmul. This explains the single-query consumer regression.

The installed `aclnnQuantLightningIndexer` ABI has no candidate-ID input. A chain of existing CANN operations can implement actual pruning: gather candidate blocks through the page table, gather scales, run BF16 `torch.bmm(..., out_dtype=torch.float32)`, round QK/1024 to FP16, reduce signed weights, then topk/remap/sort. BF16 represents every INT8 integer exactly and D128 dot sums fit exactly in FP32; independent oracle comparisons passed for all nine consumer benchmark shapes. However, the measured gather/materialization chain is slower in eight of nine cases and is **not enabled in production**. It also uses temporary per-query gathered K, so replacing the existing path based only on nominal FLOPs would be misleading.

Minimal future AscendC change for real pruning:

1. Add a consumer-specific schedule over each query's 2048 candidate blocks rather than the full context. Candidate lists differ per query, so a tile cannot silently reuse another query's list.
2. Extend the AIC key-load address calculation to map `candidate_id*8 + offset` through the existing paged block table and actual key/scale axis-0 strides. An eight-position INT8x128 block is a contiguous 1024-byte transfer. Preserve independent matrix multiplication and vector reduction/topk responsibilities.
3. Carry original position IDs into topk. Invalid candidate IDs, latest-block positions beyond that query's causal limit, and graph padding must score -inf before topk; zero-filled dummy K is insufficient because valid signed-weight scores may be negative.
4. Preserve FP16 QK and weight-scale rounding unless a separately measured numerical change is intended. Keep source block maxima and latest-block pinning unchanged. Short contexts should avoid a fixed 16K gather overhead.
5. Preallocate graph workspaces and retain per-query candidate buffers; test dynamic candidate membership, page remaps, gapped caches, negative weights and zero-valid rows. Require both median and p95 improvement on batch 1 plus no regression on batch 8/32 before enabling.

No new complicated kernel or unmeasured automatic fallback was introduced.

## Reproduction

From the workspace root:

```bash
.venv/bin/python -m pytest --confcutdir=vllm-ascend/tests/e2e/single_node/ops vllm-ascend/tests/e2e/single_node/ops/test_indexer_v41.py -q
.venv/bin/python vllm-ascend/tests/e2e/single_node/ops/benchmark_indexer_v41.py --output vllm-ascend/benchmarks/deepseek_v41/indexer_v41/graph.json
.venv/bin/python vllm-ascend/tests/e2e/single_node/ops/benchmark_indexer_v41_stages.py --output /tmp/indexer_v41_stages_current.json
```

References: official `/mnt/models/DeepSeek-V4.1-Flash/inference/model.py` Indexer/select_candidate_blocks; official `inference/kernel.py` fp4_quant_kernel; upstream `sources/vllm/vllm/models/deepseek_v41/attention.py` V4.1 indexer; existing vllm-ascend QLI v2 torch adapter, arch22 Cube/Vector implementation and golden numerical pipeline.
