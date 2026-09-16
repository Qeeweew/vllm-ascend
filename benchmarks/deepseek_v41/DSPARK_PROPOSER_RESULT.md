# Real DSpark proposer integration

Status on 2026-09-16: **real TP8 proposer passes all 15 cases with AscendC
metadata**, including context 33. All eight ranks agree, the independent CPU
Markov selection oracle passes exactly, and every worker exits cleanly. The
original AICPU failure's root cause remains unresolved. Full target/draft graph,
scheduler and serving acceptance remain outstanding; production DSpark
admission remains disabled until those gates pass.

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

## AscendC replacement: r9 acceptance

The specialized `V41DsparkMetadata` operator writes the caller-owned 1024-word
INT32 schedule directly on NPU. Attention visibility, sparse indices, cache
layout and the SMLA computation are unchanged. Its independent 58-case NPU
suite covers B1/2/4/8/16/32 and 96 changed-input graph replays; see
[metadata results](v41_dspark_metadata/RESULTS.md).

The uninstrumented r9 proposer run uses the r6 Torch extension, the isolated
metadata vendor and unchanged production vendor. No diagnostic guard,
metadata observation synchronization or task-queue override is enabled.
Contexts 9/33/129 and 255/256 with rejection counts 0–5 all pass on eight
ranks. Perturbing rejected auxiliary rows leaves logits and proposals exactly
unchanged. The independent CPU sequential Markov oracle matches all 75
proposed tokens, and rank proposals agree exactly. Maximum Torch allocation
is 4,490,209,280 bytes per rank, below the fixed 8 GiB ceiling. Launcher exit
is zero, all ranks record distributed cleanup, and no NPU worker remains.

This harness uses one request per case, three real draft blocks, real shared
embedding/head and synthetic target auxiliary states. It executes the proposer
eagerly to isolate the metadata repair. It does not establish full DSpark
graph, multi-request proposer, target verification, scheduler rollback,
serving or end-to-end performance. Those remain required; eager execution is
an intermediate correctness result. The production installation is unchanged.

Preserved evidence: `dspark_proposer/r9_comparison.json`, eight `r9_rank*.json`
files, `r9_prepared.json`, and `r9_manifest.json` with log, tensor and native
artifact hashes. Raw execution and CPU comparison logs remain at
`/tmp/v41-dspark-proposer-r9-ascendc.log` and
`/tmp/v41-dspark-proposer-r9-ascendc-compare.log`.

## Original AICPU failure evidence

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

Diagnostic r4 repeats the real TP8 proposer with the revised scratch-ownership
extension (`64931ce51efb20c460b64cae5aea4b5eeb24ecb1ac29ec256f3ef62fde36cb13`)
and unchanged production r12 OPP. All eight ranks pass context 9 and fail the
next context 33 at the metadata synchronization, as in r3. This separates the
confirmed shared scratch bug from the still-unresolved metadata failure.
The new `r4_metadata_rank*.json` also records shape, stride, storage offsets,
storage bytes, data pointers, NPU format, stream and mapped operator libraries.
Observed inputs remain contiguous ND INT32 `[0,5]`, `[38]` and five `[38]`
top-k lengths. The attempted run used shared devices and is correctness
diagnosis only; no timing result is inferred. Production DSpark stays gated.

Diagnostic r5 additionally supplies explicit `seqused_q = cu_q[1:] - cu_q[:-1]`
to each native metadata call, using the same r3 extension and r12 OPP. All
eight ranks again complete context 9 and fail context 33 with invalid
parameters. Per-rank records are retained as `r5_rank*.json`; the temporary
probe and raw log hashes are recorded in the manifest. This tests one optional
input only and does not establish that all optional-input handling is correct.
The rank-0 teardown aborts after the device error; launcher cleanup terminates
the remaining workers. No test workers remained in the process namespace.
The next diagnostic will capture the actual AICPU validation branch and input
values in an isolated operator package, with a guard that prevents diagnostic
output from reaching attention. Production validation remains unchanged.

The two raw failure logs retain the vendor's original spelling and SHA256.
Only these exact log paths are excluded from spelling hooks; source, reports,
JSON evidence and all other applicable checks remain enabled.

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
The subsequent complete-model graph run passed all five exact-answer cases;
see `FULL_MODEL_RESULT.md` and the unchanged `full_text_graph_r1` evidence.
These checks are not a general quality or quantization benchmark, and elapsed
times are diagnostic. This target success does not resolve the DSpark failure.

## Guarded AICPU diagnostic and graph requirement

The native build was missing file dependencies for AICPU object/archive
relinking. Commit `815a8bf02` fixes this, with four actual Ninja/Make fixtures;
restoring the old rule causes all four regressions to fail. A successful
build exit alone did not prove a changed diagnostic library was linked.

After complete rebuild and source/library hash checks, isolated diagnostic
r3 passes a real NPU self-test: the mandatory Python guard reads the device
marker and stops before attention. Output has 1024 INT32 elements / 4096
bytes. The runtime unknown-rank flag can coexist with concrete dimensions;
the diagnostic now records the flag without discarding valid descriptors.

Guarded proposer r7 uses the rebuilt failure-only r5 vendor and r3 Torch
extension. All eight ranks pass context 9, then context 33 fails before a
diagnostic record is readable. The runtime reports errcode 1 / invalid
execution parameters; no Prepare validation branch is established. All
workers exited, and a post-run device check found no remaining processes.
A separate NPU probe alternates lengths 14/38/134/260/261 for 21 metadata
calls successfully, so changing lengths alone does not reproduce the error.
Neither probe is final correctness or performance acceptance. The queue and
AICPU entry/parameter lifecycle remain under investigation.

`r7_diagnostic_manifest.json`, `r7_aicpu_diagnostic_rank*.json` and
`channel_selftest_r3.json` preserve the new evidence and exact raw-log hashes.
Production DSpark admission is still closed while this is fixed; this is
unfinished adaptation, not an accepted autoregressive fallback. DSpark must
be enabled in the final configuration, with actual draft NPU graph replay.
See [the graph plan](DSPARK_GRAPH_PLAN.md) for context/query capture boundaries
and the remaining full-target verification, rollback and benchmark gates.

Diagnostic r8 repeats r7 with `TASK_QUEUE_ENABLE=0`: all eight ranks again
pass context 9 and fail context 33 with errcode 1 and no device diagnostic
marker. Disabling the Torch launch queue does not resolve this failure.
`r8_queue0_summary.json` preserves each rank record hash and the raw log hash.
All eight devices were idle after launcher cleanup.
