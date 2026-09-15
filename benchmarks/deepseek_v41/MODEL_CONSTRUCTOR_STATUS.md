# V4.1 TP8 model constructor smoke

The registered `AscendDeepseekV41ForCausalLM` constructed successfully on
all eight 910B3 ranks using real HCCL groups, the Ascend model registry,
`EngineArgs`, `initialize_model`, and the production MoE factory.
`model_tp8_constructor/result.json` contains eight successful rank reports;
`model_tp8_constructor/torchrun.log` records a zero-exit run.

## Scope

The synthetic configuration uses three layers, eight routed experts and
top-6 routing. It retains H5120, I2304, 64 attention heads, D512, Q rank 1280,
output rank 1024, one shared expert, vocabulary 129280, and an Engram layer
with its real projection dimensions. Compression ratios are `[0,0,2]`.
Context capacity is 2048; the token bucket is 128. Cache layout is LBNHC
with logical block size 32. No 40-layer dense model is allocated.

This is constructor validation. Parameters are allocated but no checkpoint
is loaded, no weight postprocessing or model forward executes, and no Engram
host table is loaded. It does not establish generation correctness, graph
compatibility, loading accuracy, throughput or latency. NPU7 had an unrelated
process using about 34 GiB before the run; it was left untouched. No performance
comparison uses this run.

## Validated contracts

- All three layers select `AscendMoERunner` and `AscendRoutedExperts`, with
  `AscendFusedMoEMethod` wrapping `AscendW4A16FusedMoEMethod`.
- W4 uses group 32, TP8 and logical EP1. Every rank retains all eight routed
  experts. Shared experts stay BF16. Native decode remains disabled for this
  E8 test; its E384 production guard is not expanded.
- Replicated fused query/KV weights, TP-sharded query/output projections,
  BF16 shared projections, FP32 HC weights, Engram q/k/wkv, embeddings and
  LM head all have the expected production-width shapes and NPU placement.
- The model exposes correct local-expert metadata and registers six cache
  layers: three SWA caches, CR2 main KV, CR2 index keys/scales and a compressor
  circular state cache. MoE registry entries are counted separately.
- Engram creates no device embedding-table parameter. Its projection is
  `[25600,6144]`, q/k parameters are `[4,5120]`, and each TP rank owns three
  hash heads for staging.

The following are checkpoint-layout parameters before postprocessing:

| Parameter | Per-rank shape | Dtype |
| --- | --- | --- |
| w13_weight_packed | [8,576,640] | INT32 |
| w2_weight_packed | [8,5120,36] | INT32 |
| w13_weight_scale | [8,576,160] | BF16 |
| w2_weight_scale | [8,5120,9] | BF16 |
| w13_weight_shape / w2_weight_shape | [8,2] | INT32 |

Each rank contains 908,561,448 parameter bytes. PyTorch reports 914,720,768
allocated bytes and the same peak, about 0.852 GiB/rank. These allocator figures
exclude unrelated processes and driver-managed memory.

## Defect found and fixed

The first real TP8 run found that the outer MoE metadata divided eight experts
by the eight-rank EP communication group even when expert parallelism was
disabled. It incorrectly reported one local expert per rank. Communication
group existence does not imply EP weight sharding. `DeepseekV4MoE` now uses
logical EP size 1/rank 0 when EP is disabled; all eight ranks subsequently
reported eight local experts. Original failure evidence is retained in
`model_tp8_constructor/failed_ep_metadata/`.

## Reproduction

From the vllm-ascend repository root:

```bash
../.venv/bin/python benchmarks/deepseek_v41/smoke_model_tp8.py \
  --prepare --output benchmarks/deepseek_v41/model_tp8_constructor
../.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 \
  benchmarks/deepseek_v41/smoke_model_tp8.py \
  --output benchmarks/deepseek_v41/model_tp8_constructor
```

Full synthetic runner execution and real checkpoint/model profiling remain
separate acceptance steps.
