# Engram placeholder validation before upstream preprocessing

## Problem and change

The inherited `GPUModelRunner._preprocess` clamps speculative input IDs with
`clamp_(min=0)` before multimodal embedding lookup. Previously the Ascend
sanitizer preserved Engram `-1` IDs, but Engram preparation happened after
this inherited clamp. An unresolved active placeholder could therefore become
token `0` before history validation and hashing. Testing only sanitizer then
Engram preparation missed the intervening inherited method.

`NPUModelRunner._preprocess` now performs the existing Engram preparation once,
before calling the inherited preprocessing method. Its existing packed D2H
snapshot contains final device IDs, positions and real-request query bounds.
Existing history validation rejects negative active IDs before hashing, table
staging, inherited clamping or embedding. Valid token `0` remains valid. Query
bounds exclude graph padding, so a padding `-1` is harmless.

The resulting Engram kwargs join the inherited model kwargs. The original
forward-boundary preparation is replaced by `wait_ready`; `mark_consumed`
remains after the model call. This retains one snapshot, one prepare, one wait
and one consume per successful forward, while permitting row DMA to overlap
encoder/embedding work. If preprocessing fails after staging, the runtime
retains its pending state for checked shutdown and all-stream synchronization.

Only `worker/model_runner_v1.py` changes in production. The model and
speculative-decoding admission guard remain unchanged. This fix does not admit
or establish end-to-end DeepSeek V4.1 speculative decoding.

## Input ordering and capture

The existing `_prepare_inputs` completes its enqueue phase before the new
snapshot. It invokes `_prepare_input_ids`, including the existing asynchronous
PP receive fence and device sampled-token scatter, writes real query bounds,
and enqueues device position corrections. The existing
`synchronize_input_prep` context and its event synchronization/record are
unchanged. Snapshot concatenation follows these operations on the compute
stream; its single blocking `.cpu()` resolves queued corrections.

For V4.1's flat positions, inherited preprocessing subsequently changes only
padding positions to zero. Those positions are excluded by the final real
query boundary. The prompt's processor-owned image masks were already seeded
in history and do not depend on encoder execution.

Non-Engram models immediately delegate to inherited preprocessing. Dummy and
capture paths use their existing direct static buffers and do not call this
runtime preprocessing hook. Runtime preparation remains outside graph capture;
the graph consumes stable staged row/mask addresses.

## Validation

- CPU runner tests: **29 passed, 0.52 seconds**. New cases call the real
  inherited speculative clamp and multimodal embedding path. They prove active
  `-1` rejection before embedding/hash, unchanged valid `0`, excluded padding,
  exactly one packed `Tensor.cpu` snapshot, deferred single wait/consume,
  unchanged legacy clamping, and pending-state cleanup after embedding failure.
  Log: `/tmp/v41-placeholder-cpu.log`.
- Real **NPU 1** test: **1 passed, 6.02 seconds**. It rejects active `-1` before
  actual embedding, then queues corrected device IDs `0,3,0,4` while scheduler
  placeholders remain unresolved. Four graph replays consume real pinned-host
  Engram rows at stable addresses and match an independent full-history hash
  oracle exactly. Actual embeddings match the corresponding token rows;
  padding remains masked. No TP/HCCL or other NPU was used.
  Log: `/tmp/v41-placeholder-npu1.log`.
- Broader existing runner/history CPU regression: **108 passed, 1.78 seconds**,
  including `test_model_runner_v1.py`, `test_engram_runner.py` and
  `test_engram_history.py`. Log: `/tmp/v41-placeholder-runner-regression.log`.

Test files: `tests/ut/worker/test_engram_runner.py` and
`tests/e2e/single_node/ops/test_engram_runner_placeholder.py`. These tests use a
constructed runner to exercise the low-level call chain, not production model
admission. They do not measure end-to-end performance; the structural gate is
that no second D2H synchronization was introduced.
