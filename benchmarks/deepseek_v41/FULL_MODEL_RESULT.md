# Complete target with real Engram: integration results

The first full 40-layer TP8 eager run passed on 2026-09-16. It loaded all
48 converted shards and both real host tables through the production model
registry and loader. This result includes the corrected `wo_a` layout.
It validates execution, repeatability and resource cleanup; the raw-token
prompts do not establish language quality, long-context correctness or speed.

## Eager r1

- Source checkout: personal branch `deepseek-v41-910b-w4a16-engram`,
  HEAD `47b43226b`; installed native package r12. Concurrent fusion sources
  were being developed but were not installed or used by this run.
- Converted checkpoint: `/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32`,
  complete 48-shard INT4 group32 RTN MoE / BF16 dense checkpoint.
  Conversion manifest SHA256:
  `4d48bba36c91edca7cae890cd6186a7728f67603c8efc4ae358d1447ddab2bed`.
- Eight 910B3 devices, TP8, strict HCCL, CANN W4A16; native decoding and
  split candidate indexer disabled. Existing complete QLI was used.
- All 40 target layers, 384 experts per layer, real Engram layers 1 and 14;
  no vision allocation or speculative decoding. Maximum context 512,
  chunk size 128, one sequence, 256 MiB KV cache per rank.
- Prompt lengths 40, 3, 129 and 384; the three-token case includes token 0
  and literal image sentinel 129264. Four generated tokens per prompt,
  two rounds. All 16 token IDs and selected logprobs match exactly across
  rounds; all selected logprobs are finite.
- Sixteen real pinned host owners hold 366.221833 GiB in total. All 144
  source FP8/group32-scale → converted BF16 → loaded-row samples pass.
  NUMA placement samples match `[6, 7, 4, 5, 0, 1, 2, 3]` by rank.
- Stable device staging pointers across generation; each rank reaches
  offload step 38 with no pending prepared input. No graph replay in eager.
- All 16 owners explicitly unregister; all eight workers exit gracefully,
  and EngineCore exits 0. Post-run `npu-smi` showed no device processes.

| Memory observation | GiB / rank |
| --- | ---: |
| End-of-run Torch allocated | 39.553790 |
| End-of-run Torch reserved | 42.218750 |
| Process peak RSS, including loading | 71.3100–71.3929 |
| End-of-run resident PSS | 48.1919–48.3342 |

Torch values above are current allocator observations, **not device peak
HBM measurements**. Process peak RSS includes temporary loading/file-backed
pages and cannot be equated with permanent pinned storage. Weight loading
reported 81.64 seconds; subsequent bounded expert repacking, host loading,
initialization and shutdown are separate. No TTFT/TPOT claim is made.

Command, from the repository root:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HCCL_DETERMINISTIC=strict OMP_NUM_THREADS=4 \
  ../.venv/bin/python -u benchmarks/deepseek_v41/validate_full_model_tp8.py \
  --run --output /tmp/v41-full-eager-run-r1.json
```

Unmodified evidence:

| File | SHA256 |
| --- | --- |
| `full_model_eager_r1.json` | `82ecf7706e56f958e90378d80a3e45033c9549b94df6038c85877eb5bb44d05f` |
| `full_model_eager_r1.log.txt` | `a2beae1ef5ee31033f97e00341a0a9e34fd75377e1379fa4089096f208b6569a` |

## Remaining acceptance

Full-model graph is being tested against this exact eager reference. The
mandatory fused lightning indexer must still pass numerical, memory and
performance gates and full-model integration. Natural-language quality,
long context, corrected full-model vision and speculative decoding,
native W4A16 comparisons and final profiling remain pending.
