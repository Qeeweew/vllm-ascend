# Real target auxiliary-state graph probe

The probe enables target auxiliary outputs through the production runner and
model registry on eight 910B cards, without loading a drafter. The fixture is
`/tmp/v41-mm-production-numa-graph-r1`: three real target layers and a small
synthetic Engram table. CANN decode, strict HCCL, image limit zero and explicit
NUMA placement are used. This is a correctness diagnostic, not a timing run.

Worker hooks independently average the four post-FFN HC streams in FP32 and
round to BF16 before the next layer's Engram injection. Device-side counters
and maximum errors survive graph replay; capture/warmup counters are reset
before real requests. No observation copies to CPU during capture.

## Results

After correcting the target/draft `wo_a` layout (`da6da081a`), the probe was
rerun at source commit `1e10173ae` and passed with the same exact gates.
`target_aux_real3_graph_woa_fixed.json` records two eager calls and six graph
replays per rank, zero auxiliary-state error, identical repeated outputs,
eight released owners and EngineCore exit zero. The log
`/tmp/v41-target-aux-real3-graph-woa-fixed.log` confirms graceful worker exits
without forced termination or resource-tracker leak warnings.

Corrected output IDs are `[5774, 4798, 34105, 52103]`, whereas the historical
probe produced `[43172, 29514, 93589, 74413]`. Both runs were internally
repeatable; only the new run includes the independently validated projection
layout correction. Neither is a full-model language-quality evaluation.

The historical results below are preserved to retain the original startup
and lifecycle evidence; their repeatability did not detect the layout error.

`target_aux_real3_graph_r2.json` passed on every rank:

- All three auxiliary tensors match the independent HC average exactly.
- Each rank executed two eager requests and six decode graph replays.
- Two repetitions produced identical token IDs and selected logprobs.
- All eight Engram owners released their shard weights; EngineCore exited zero.
- The log records graceful worker exits with no forced termination or
  resource-tracker leak warnings.

The first attempt, retained in `target_aux_real3_graph.json`, failed during
startup with `BaseRouter._select_experts()` rejecting `image_token_mask`.
The driver had imported the worker before plugin setup. Moving that import to
a separate worker-only module resolved the failure in the second attempt;
no production router change was made.

The first attempt was **not a clean shutdown**. Its executor exhausted the
worker grace period, sent SIGTERM and then SIGKILL to one worker, and reported
one leaked semaphore and nine leaked shared-memory objects. After its NPU
workers had exited, the stuck owned EngineCore process 3287421 was terminated
to let the parent constructor return. The unrelated NPU 7 process was untouched.
This failed lifecycle result remains part of the evidence.

Logs: `/tmp/v41-target-aux-real3-graph.log` and
`/tmp/v41-target-aux-real3-graph-r2.log`.

## Reproduction

Run from `vllm-ascend`, using a new output path for each attempt:

```bash
OMP_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn \
HCCL_DETERMINISTIC=strict VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=60 \
../.venv/bin/python -u benchmarks/deepseek_v41/probe_target_aux_tp8.py \
  --checkpoint /tmp/v41-mm-production-numa-graph-r1 \
  --output /tmp/target_aux_graph_new.json
```

This validates the bounded three-layer auxiliary-output protocol. It does not
validate the real 40-layer auxiliary indices, real full Engram tables, DSpark
proposal execution, acceptance/rollback, or speculative scheduling. Production
DSpark admission remains disabled pending those integration checks.
