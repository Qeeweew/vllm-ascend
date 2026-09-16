# Complete target with real Engram: integration results

The first full 40-layer TP8 eager and graph runs passed on 2026-09-16. They loaded all
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

The mandatory fused lightning indexer must still pass memory and performance
gates and full-model integration. Broad language/quantization quality, context
lengths beyond the 4243-token smoke below, corrected full-model vision and
speculative decoding, native W4A16 comparisons and final profiling remain pending.

## Graph r1

The same complete checkpoint and CANN backend passed `FULL_DECODE_ONLY`
against the exact eager r1 reference above. Prefill stays eager. Each rank
executed **26 observed graph replays**, with 38 offload steps and unchanged
device staging pointers. All generated token IDs and selected logprobs were
identical both across two graph rounds and against eager. All 144 sampled
real-table rows passed; all 16 owners unregistered and all eight workers
exited gracefully. EngineCore exited 0; devices were idle after shutdown.

Current Torch allocated was 39.553791 GiB / rank, reserved 44.097656 GiB /
rank. Process peak RSS was 71.3055–71.4025 GiB and resident PSS at the final
observation was 48.3111–48.3441 GiB. These are the same observation types as
the eager table, not peak HBM measurements. Source HEAD at launch was
`47b43226b`, native r12; the later eager-evidence commit changed only reports.
The mandatory new fused QLI had not been installed.

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HCCL_DETERMINISTIC=strict OMP_NUM_THREADS=4 \
  ../.venv/bin/python -u benchmarks/deepseek_v41/validate_full_model_tp8.py \
  --run --graph --reference /tmp/v41-full-eager-run-r1.json \
  --output /tmp/v41-full-graph-run-r1.json
```

Unmodified evidence:

| File | SHA256 |
| --- | --- |
| `full_model_graph_r1.json` | `15b06267fa1ee7779bd0dcc2564ee575d450c4cf4ee7f51400ded698417aebbe` |
| `full_model_graph_r1.log.txt` | `6722a258018f13098e0b0a4d8c9239422248f6997e87dea68224bbc143591679` |

## Natural-language and retrieval graph r1

All five exact-answer smoke cases passed using the complete checkpoint, real
Engram and CANN W4A16 on 2026-09-16. The official V4.1 chat encoder used
`thinking=False`; decoding was greedy, with a 32-token limit. This is a small
functional smoke, not a general quality or quantization-accuracy benchmark.

| Case | Prompt tokens | Answer |
| --- | ---: | --- |
| Arithmetic | 17 | `42` |
| Chinese capital | 16 | `北京` |
| English extraction | 22 | `Q7M2` |
| JSON sorting | 25 | `[1, 4, 9]` |
| Long-context retrieval | 4243 | `NPU-7319` |

All selected logprobs were finite. Each rank executed 20 observed graph
replays and 58 offload steps, with stable device staging pointers. All 144
source/converted/host row checks passed; all 16 owners unregistered, eight
workers exited gracefully and EngineCore exited 0. No NPU process remained.
End-of-run Torch allocated/reserved were 42,743,025,664 / 47,414,509,568 bytes
per rank; these are current allocator observations, not peak HBM.

The launch used source HEAD `2c2b877fb`, installed production r12, maximum
context 8192, chunk size 128 and 512 MiB KV per rank. New fusion, vision and
DSpark were disabled. During this run, the benchmark sources were extended
for a later image smoke; those later edits were not loaded into this run.
Raw per-case elapsed times include instrumentation and are diagnostic only.

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HCCL_DETERMINISTIC=strict OMP_NUM_THREADS=4 \
  ../.venv/bin/python -u benchmarks/deepseek_v41/check_full_text_tp8.py \
  --run --graph --output /tmp/v41-full-text-graph-r1.json
```

Unmodified evidence: `full_text_graph_r1.json` and
`full_text_graph_r1.log.txt`; fingerprints are recorded in
`full_text_graph_r1_manifest.json`.
