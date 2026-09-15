# Engram runner integration review

Scope: CPU tests of the real `NPUModelRunner` helper methods through `__new__`, mock runtime, and real CPU request history where lifecycle semantics matter. The review subsequently made one scoped runner change: skip legacy placeholder-to-zero sanitization when an Engram runtime is active. No C++ source was changed by the Engram review. No NPU resources were used.

## Validated helper behavior

`tests/ut/worker/test_engram_runner.py` passes 16 cases. The final combined run with `test_engram_history.py` passes all 33 tests; `runner_results.xml` records that combined run. Finished request IDs are dropped before a reused ID is seeded. New requests receive actual int64 CPU prompt IDs; absent actual prompts fail before reset. Preemption retains executed history; finish/reuse discards the old generated tail. Multimodal requests without a mask provider fail explicitly, provider masks are passed unchanged, invalid masks fail through real history, and text requests never call the provider.

Preparation uses final `.gpu` input buffers, original request slot order, and only `num_reqs+1` boundaries. Tests deliberately supply incorrect CPU mirrors, reversed request order, padded request entries, different graph buckets and a zero-length query. The runtime receives real boundaries and bucket-sized input-ID views; positions and optional mask are passed unchanged. Returned row/mask buffers are installed before `wait_ready`; unrelated model kwargs survive. Preparation failures do not publish buffers or wait. Missing/None runtime leaves old-model state untouched without reading input/scheduler objects.

## Reviewed integration sites

- Request update runs before the persistent input batch update. Preempted requests are not dropped; completed IDs are.
- Preparation is directly before `_model_forward`, outside its graph execution. Successful preparation/wait is followed by a `finally` that calls `mark_consumed`, including model exceptions.
- Dummy/profile forwards slice stable preallocated device rows and zero the token mask; they do not call CPU history, gather or offload preparation. Offload rows are initialized with zeros by their manager.
- `load_model` creates the runtime immediately after model loading and before graph wrapping. The model exposes the factory; old models leave runtime unset/None.

These integration-site statements are source review, not full `execute_model`, graph capture, load or device-event tests. Real runtime/NPU smoke evidence belongs to the parent task.

## Findings requiring parent handling

1. **Fixed: placeholder sanitization hid missing actual tokens.** Before Engram preparation, legacy `_sanitize_placeholder_input_ids_for_forward` changes remaining `PLACEHOLDER_TOKEN_ID` values to token 0 when asynchronous speculative placeholders exist. The history would then hash a synthetic token instead of rejecting unknown input. The scoped fix now skips this legacy conversion when an Engram runtime exists. Three regressions verify unchanged legacy behavior, preserved actual/padding IDs, rejection of unresolved real `-1` without history commit, and success after the real device ID arrives despite a stale scheduler placeholder. Graph-padding `-1` remains outside real query boundaries.
2. Preparation and `wait_ready` occur before the model's try/finally. If waiting fails after preparation, runtime remains prepared; its current `mark_consumed` also requires a successful wait. This concerns recovery after a device/wait error, not ordinary successful inference. A future abort/close path would need an explicit offload state protocol rather than blindly calling `mark_consumed` on an unready step.
3. MM mask-provider rejection is intentional. No claim of working image-span Engram masking is made. A future provider must supply both full prompt masks and current packed token masks consistently.

CPU test command:

```bash
.venv/bin/python -m pytest --confcutdir=vllm-ascend/tests/ut/worker \
  vllm-ascend/tests/ut/worker/test_engram_runner.py -q \
  --junitxml=vllm-ascend/benchmarks/deepseek_v41/engram_history/runner_results.xml
```
