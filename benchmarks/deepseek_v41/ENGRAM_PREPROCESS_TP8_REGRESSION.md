# Production TP8 regression after Engram preprocessing change

Baseline: `9182fe1a89a8fb525987a1f9cd68eb4813a04d3e`.
This regression exercises the moved Engram staging call through the production
model and processor registries. It reuses the existing
`/tmp/v41-mm-production-numa-graph-r1` fixture: three real language layers,
384 real W4A16 experts per layer, real vision weights, and one synthetic
4096-row Engram table. It is not full-model quality, full-table loading or
performance acceptance.

`regress_engram_preprocess_tp8.py` uses the existing observation-only
`V41MMSmokeWorker` without test model registration. New router observations
count legitimate text token `0`; existing observations validate unchanged raw
IDs, typed image masks, real graph replays, native W4A16
capture and CANN prefill dispatch. These synchronous diagnostic reads are not
suitable for timing comparisons.

## Checks

Each vision configuration repeats the literal-image-ID text prompt and a text
prompt beginning with valid token `0`. The enabled-vision run also repeats the
real photo prompt. Each request generates four tokens; repeated IDs must match
exactly and selected logprobs must be finite with maximum absolute difference
at most `1e-4`. Actual differences are retained in JSON.

The `1e-4` repeated whole-model logprob threshold is a **new diagnostic
repeatability gate**, not the established native W4A16 numerical-correctness
criterion. The existing W4A16 report documents small nondeterministic FP32
AtomicAdd reductions. Failing this new gate does not by itself establish a
numerical-correctness regression in Engram or native W4A16, nor does it establish
model quality degradation. The observed magnitude and propagation still need
controlled investigation; no root cause is inferred from this experiment.

Every rank must retain stable row/mask addresses, finish without a pending
runtime step, replay the captured graph, and route real token `0` as text in
all three layers. Enabled vision must expose the 189-token typed image span;
disabled vision must allocate no tower/aligner parameters and invoke no encoder.
Both modes retain literal `129264` as text when no image payload is present.

NUMA placement is checked against `[6,7,4,5,0,1,2,3]`. Terminal checked shutdown
must release all eight registered table owners before explicit 30-second
EngineCore shutdown; EngineCore must exit zero with no live process. Logs are
audited separately for forced exits and resource-tracker leaks.

The pre-existing NPU 7 process PID `3836133` used 34364 MiB at the start.
It is unrelated to this regression and is neither stopped nor modified.

## Reproduction

From `vllm-ascend`, use new output/log filenames for every run:

```bash
OMP_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn \
HCCL_DETERMINISTIC=strict VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=60 ../.venv/bin/python -u \
  benchmarks/deepseek_v41/regress_engram_preprocess_tp8.py --run \
  --image ../sources/vllm/tests/v1/ec_connector/integration/hato.jpg \
  --image-limit 1 \
  --output benchmarks/deepseek_v41/engram_preprocess_mm_graph_strict_9182fe1a8.json
```

Repeat with `--image-limit 0` and a fresh output filename after the first engine
has exited. Omitting `--run` only validates fixture metadata and image inputs;
it does not launch TP8. Existing checkpoint weights and historical evidence are
never overwritten.

## Results

The first default-HCCL run completed with **failed** numerical-repeat status.
All three prompt classes returned identical generated IDs across repeats, but
selected-logprob differences exceeded `1e-4`:

| Prompt | Maximum selected-logprob absolute difference |
| --- | ---: |
| Literal image ID as text | 0.0507781506 |
| Valid token zero as text | 0.0544831753 |
| Typed image | 0.0475189686 |

Both `HCCL_DETERMINISTIC` and `ASCEND_LAUNCH_BLOCKING` were unset; the original
MM acceptance report used strict HCCL. This missing experimental control must
be corrected before attributing the observed difference to staging or native
decode. The original failed JSON is preserved. The full comparison and
per-rank observations are in `engram_preprocess_default_mm_audit.json`, with
the launch environment audit in
`engram_preprocess_mm_graph_9182fe1a8_environment_audit.json`.

Each rank completed 18 graph replays, observed legitimate zero text tokens in
all three layers and two typed image prefills, retained stable Engram row/mask
addresses, and had no pending runtime after requests. These are address and
typed-input observations; the first run did **not** capture Engram hash inputs
or row contents. All eight owners were released and EngineCore PID 3255082
exited zero. No claim of row-content equality is inferred from stable pointers.

The fixture's routers report `text_hash_checked=false`: no `tid2eid` table is
present in these V4.1 layers. An erroneous new diagnostic assertion requiring
that table was removed. It was not reached before the first numerical failure
and removing it does not alter that run's failed status or numerical gate.

The strict-HCCL native-MM control also completed with **failed** numerical
repeat status, while **all structural checks passed**. It used the same
synchronous observations as the first run, with no new row-content D2H reads.

| Prompt | Strict-HCCL maximum selected-logprob absolute difference |
| --- | ---: |
| Literal image ID as text | 0.0579013824 |
| Valid token zero as text | 0.0010704994 |
| Typed image | 0.0022363663 |

Generated IDs remain identical. The first generated token's logprob, produced
by prefill, is identical between repetitions for all three prompt classes.
Differences appear in the subsequent decode tokens. For literal text, the
first round's second-token logprob is `-2.7317228317`, versus
`-2.6738214493` on repetition; the historical production-MM graph result records
`-2.6738214493`. The first strict-control image response's four logprobs match
the historical production-MM response exactly. These are localization
observations, not proof that staging or a particular kernel causes the change.

All eight owners were released and EngineCore PID 3266790 exited zero. The
strict control is `engram_preprocess_mm_graph_strict_9182fe1a8.json`, log
`/tmp/v41-preprocess-mm-graph-strict-9182fe1a8.log`.

A strict-HCCL, image-limit-zero CANN-decode control **passed**. Both literal-ID
and valid-zero text requests returned exactly equal repeated token IDs and
selected logprobs: maximum absolute difference **0**. Every rank replayed
12 graphs, captured the CANN fallback with zero native captures, allocated no
MM parameters and made no encoder calls. All eight owners were released and
EngineCore PID 3277445 exited zero. `--cann-decode` selects the existing fallback
without changing production code.

All three runs' logs contain no internal forced-worker/engine termination or
resource-tracker leak warnings. The client explicitly waited up to 30 seconds;
the offline LLM engine itself logged its default `mode=abort timeout=0s`, but
terminal owner release had completed and all workers exited gracefully. This
is distinct from the earlier HTTP server's immediate force-kill case.

The new aggregate `engram_preprocess_tp8_regression_summary.json` preserves
SHA256s of all three original result JSONs, every repeat comparison, per-rank
replay counts and cleanup audits. Historical production-MM results contain the
same literal-ID prompt but use native decode, so cross-result differences are
observations rather than a same-backend precision gate. No matching earlier
CANN result for these exact short prompts was identified.

This control changes both decode backend and vision admission relative to the
strict-native MM run. It narrows the investigation but cannot isolate its root
cause. The new native-MM strict-repeatability gate remains **failed**; its
threshold has not been relaxed. This is not a failed native numerical-accuracy
acceptance claim. No production model code was changed in this regression.

The optional `--rows-diagnostic` worker is prepared for a separately scheduled
investigation. It adds blocking D2H snapshots and row/mask SHA256s and has not
been enabled in the controls reported above.

The TP8 window was released after the final successful control. `npu-smi`
showed no processes on NPUs 0–6 and only the unchanged external PID 3836133
(34364 MiB) on NPU 7. No additional NPU diagnosis was launched.

- Enabled vision output: `engram_preprocess_mm_graph_9182fe1a8.json`.
  Log: `/tmp/v41-preprocess-mm-graph-9182fe1a8.log`.
- Disabled vision CANN output: `engram_preprocess_text_cann_strict_9182fe1a8.json`.
  Log: `/tmp/v41-preprocess-text-cann-strict-9182fe1a8.log`.
