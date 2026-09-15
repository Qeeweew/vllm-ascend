# DeepSeek V4.1 host offload and component checks

These checks cover components on Ascend 910B3, not a running eight-card model.
The target branch is `deepseek-v41-910b-w4a16-engram` in the personal fork.
Software: CANN 9.1.0, driver 25.5.0, torch 2.10.0, torch-npu 2.10.0.post4.

## Checkpoint and hashing

The first three converted shards were sampled with an independent scalar
signed-scale RTN calculation. Packed codes and BF16 group32 scales matched
exactly; dense BF16 dequantization matched exactly. The comparison explicitly
unpacks checkpoint offset-binary codes, rather than device two's-complement
codes. Results: [conversion_samples.json](conversion_samples.json).

The real tokenizer has 129280 tokens and produces exactly 99092 compressed
IDs. All primes and multipliers match the model's `inference/engram.py`.
Twelve comparisons covering two requests, chunk boundaries, image masks and
recomputation matched the reference exactly. Results and TP8 head ranges:
[real_hash.json](real_hash.json).

`HostEngramHasher` is stateless. Each invocation requires real current and
lookback token IDs. Missing generated history is rejected, rather than hashed
as a placeholder. Getting these IDs from the actual asynchronous runner is
still pending; this component does not solve that scheduling dependency.

## Pinned storage and graph ordering

`EngramTableShard` loads only the rank's selected head buckets directly into
final host storage. `EngramOffloadManager` owns two pinned staging slots, one
stable device input per layer, and copy/compute events. The required outer
runner protocol is `prepare -> wait_ready -> model/replay -> mark_consumed`.
Host lookup is outside capture and must run on every replay invocation.

Eight CPU table tests and three hardware offload tests passed. The hardware
stress test runs 64 changing inputs across buckets 4/8, including zero actual
tokens, padding, dead IDs, two layers and stable device addresses. Output
snapshots are checked after the loop; there is no per-step result read or
global NPU fence. A reused host staging slot only waits for its previous DMA.

A separate allocation test successfully allocated and touched 1, 8 and 46 GiB
of pinned BF16 host storage. It wrote different values into the first, middle
and final rows, gathered into pinned staging and verified H2D results.
46 GiB covers the estimated 45.78 GiB of table storage per TP8 rank.
This is a single-process capacity check, not simultaneous eight-rank loading
or a NUMA/random-access throughput benchmark. Driver-backed allocations are
not fully reflected in `ru_maxrss`; those readings must not be used as the
physical pinned-memory budget. Raw data: [pinned_capacity.jsonl](pinned_capacity.jsonl).

## Compressor and delayed mHC integration

Three NPU component tests passed for the independent projection, CR1/CR2
vector compression and device-built ring metadata. CR2 uses native BF16 ND
GEMM operands with FP32 output; the loader disables automatic NZ conversion.
See the [compressor performance report](../compressor_v41/report.md).

The runner now allocates a circular compressor state as one FP32 tensor and
binds `[blocks,1,capacity,1024]`, rather than splitting it into K and V. The
new allocation/reshape test passed, alongside 19 existing cache tests. Ring
capacity and device-boundary metadata tests also passed. Full scheduler
allocation, prefix-cache recomputation and draft integration remain pending.

Five delayed-mHC NPU tests passed, including external pre-mix, post-mix
orientation, terminal collapse and changing pre-mix under graph replay.
The existing AscendC `HcPre v3` accepts the incoming pre-mix and returns the
new one for the next sublayer. FP32 control parameters stay FP32 in storage,
but its Cube projection uses HF32 internally. Strict FP32 comparisons on
actual control weights from layers 0/1/2/20/39 are recorded in
[mhc_real_weights.json](mhc_real_weights.json), without truncating the reference
weights to match the kernel. These component errors do not establish model
quality after forty layers.

## Reproduction

Run from the workspace root with the active editable environment. Use a free
physical NPU when setting `ASCEND_RT_VISIBLE_DEVICES`.

```bash
.venv/bin/python vllm-ascend/benchmarks/deepseek_v41/host/verify_quantized_samples.py
.venv/bin/python vllm-ascend/benchmarks/deepseek_v41/host/verify_hash_reference.py
ASCEND_RT_VISIBLE_DEVICES=3 .venv/bin/python vllm-ascend/benchmarks/deepseek_v41/host/probe_pinned_capacity.py
ASCEND_RT_VISIBLE_DEVICES=3 .venv/bin/python vllm-ascend/benchmarks/deepseek_v41/host/verify_mhc_reference.py
ASCEND_RT_VISIBLE_DEVICES=3 .venv/bin/python -m pytest vllm-ascend/tests/e2e/single_node/ops/test_engram_offload.py vllm-ascend/tests/e2e/single_node/ops/test_compressor_v41_component.py vllm-ascend/tests/e2e/single_node/ops/test_mhc_v41.py -q
```

The verification scripts accept `--source`; the conversion comparison also
accepts `--output` and `--max-shards`. Full-model TTFT, TPOT, HCCL, Engram
NUMA access and combined memory peaks remain separate final profiling work.

## Eight-rank offload and HCCL graph smoke

`../smoke_engram_tp8.py` completed with eight 910B ranks, two synthetic host tables, 24 total hash heads, head dimension 256, and 32 changing-ID/padding steps. Pinned double staging, fixed device inputs, HCCL head all-gather and graph replay matched the CPU reference bitwise. No per-step output readback was used. Result: `engram_tp8_smoke.json`. This does not load the full Engram tables or run wkv/gating, and is not a throughput benchmark.
