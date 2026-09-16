# Real DSpark proposer integration

Status on 2026-09-16: **first TP8 attempt failed**. This result does not
invalidate the passed independent draft stage or maximum-context component
tests, but demonstrates that those tests did not cover the real proposer
integration. Production DSpark admission remains disabled.

`check_dspark_proposer_tp8.py` constructs the real `NPUModelRunner`, takes
its actual drafter and invokes `proposer.load_model` and `_propose`. The
fixture loads all three real E128/top3/H5120 draft blocks and shares the real
target embedding/head through the actual proposer logic. Target auxiliary
states are deterministic synthetic BF16; no target layers or Engram tables
are loaded. The fixed allocator ceiling is 8 GiB per rank.

The intended 15 cases cover contexts 9/33/129, plus maximum 256 with contexts
255/256 and rejection counts 0–5. Independent checks cover initial anchor /
noise inputs, logical positions, context and query cache slots, noncausal
visibility, finite logits and sequential Markov greedy selection. Rejected
auxiliary rows are perturbed to check that they cannot change logits or
proposals. Every rank must agree and clean up its distributed state.

## Evidence

- CPU preparation passed with the complete conversion manifest; NPU remained
  uninitialized. `dspark_proposer/r2_prepare.json` records the exact scope.
- Two CPU oracle regressions passed: each selected token conditions the next
  Markov step, and actual BF16 addition rounding can change argmax compared
  with a promoted FP32 addition. Oracle inputs must retain actual logits dtype.
- Real TP8 runner construction, three-shard loading and actual shared
  embedding/head setup completed. The first context-9 request then failed
  with `SparseFlashMlaMetadata` AICPU error `0x2a` / invalid parameters.
  The stack reports a later model operation because native calls are async.
- Rank 0 recorded `failed_cleanup` after the device error also prevented a
  normal synchronization. Torch distributed launcher terminated its other
  seven workers; driver exited 1. Post-exit `npu-smi` showed no processes.
  No proposal case passed, no rank agreement or CPU Markov comparison ran.

Raw logs, preparation, rank-0 failure and SHA256 are preserved under
`dspark_proposer/`. The other ranks were terminated before writing results;
their absence is not interpreted as success. Root-cause diagnosis must retain
the failure evidence and distinguish diagnostic synchronization from the
unmodified execution path. No production correctness gate is relaxed.

Diagnostic r3 adds `--observe-metadata`, recording the small metadata tensors
on CPU before the native call and synchronizing after it. All eight ranks
then passed case 0 (context 9), but case 1 (context 33) failed at the native
metadata synchronization. Every rank recorded valid-looking query boundaries
`[0, 5]`, sequence length `[38]` and five top-k lengths of 38. The preceding
length-14 call completed. Per-rank input snapshots and failures are preserved
as `r3_*`; this is a diagnostic result with changed synchronization, not a
passing integration result. A separate physical-device-1 builder probe passed
20 repeated length-14 calls without synchronizing before the first call.
Root cause remains open; changing lengths and optional-input handling are
being investigated without relaxing the native parameter checks.

The venv does not provide a `torchrun` executable; use its Python module:

```bash
OMP_NUM_THREADS=4 ../.venv/bin/python benchmarks/deepseek_v41/check_dspark_proposer_tp8.py \
  --output /tmp/v41-dspark-proposer-next
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HCCL_DETERMINISTIC=strict OMP_NUM_THREADS=4 \
  ../.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 \
  benchmarks/deepseek_v41/check_dspark_proposer_tp8.py --run \
  --output /tmp/v41-dspark-proposer-next
```

Only after all eight ranks pass execution and cleanup, run `--compare` on
that output directory. Even a passing result does not establish scheduler
acceptance/rejection, rollback, target/draft graph coordination or serving.

## Separate full-target text preparation

`check_full_text_tp8.py` is prepared for five real-checkpoint chat smoke
cases: arithmetic, Chinese fact, English extraction, JSON sorting and a
4243-token retrieval prompt. It uses the official V4.1 tokenizer with
`thinking=False`, CANN W4A16 by default, real host tables and explicit cleanup.
`full_text_prepared_r2.json` records successful CPU tokenization without NPU
initialization. Its preflight snapshot was taken during the graph run and
therefore records occupied HBM; admission is repeated by an actual `--run`.
No text case has executed yet. These small exact-answer checks are not a
general quality or quantization benchmark, and elapsed times are diagnostic.
