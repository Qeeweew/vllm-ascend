# Engram request history component

`worker/engram_history.py` provides CPU-only `EngramRequestHistory` and `EngramHistoryBatch`. It changes no model or runner code and allocates no NPU resources.

## Contract

- `reset_request(request_id, full_prompt_ids, prompt_mask=None, executed_tail=None, tail_mask=None)` explicitly seeds/replaces a request incarnation. IDs are CPU INT64. Masks are CPU BOOL; false means a real masked token such as an image token, never a missing token ID. An optional restore tail must contain contiguous actual executed inputs after the complete prompt.
- `prepare(request_ids, actual_input_ids, actual_positions, query_start_loc, token_mask=None)` accepts only unpadded packed rows. Every nonempty request has contiguous real positions. It builds newest-first lookback before hashing, returns every layer's CPU hash matrix plus the current row mask, and records the actual inputs.
- Re-executing earlier positions overwrites and truncates the generated tail; the complete known prompt remains available. Executed prompt values and masks must match the seed. No `-1` async placeholder is accepted, including masked rows. An unknown request or gap beyond known history fails closed.
- Validation and hashing complete before any request is updated. One failing request or a hash exception does not partially overwrite/truncate other requests. Seed and update buffers are copied so caller buffer reuse cannot mutate persistent history or returned masks.
- Normal preemption retains the entry. Prefix hits anywhere inside the known prompt can construct lookback without executing preceding chunks. Restoration after host-state loss needs an explicit reset with known executed tail; a generated prefix cannot be invented from a scheduler position. `drop_request` is idempotent and reserved for finished requests. Request IDs, not batch row slots, key all state.

Persistent payload is one `array('q')` plus a `bytearray` per request: 9 bytes per known token, excluding small containers and allocator capacity. Python integers are not retained per token. Normal decode modifies only the executed chunk and rejected tail; it does not clone the full prompt/history. Hashing and temporary lookback scale with current tokens and request count, not prompt length. Full prompt validation/copy occurs on explicit reset.

## Runner integration recommendations

1. Create one history owner alongside the model's `HostEngramHasher`, after tokenizer/config validation. In `_update_states`, reset only genuinely new/reused request incarnations with the full prompt and true prompt mask. Drop `finished_req_ids`; do not drop merely preempted or absent rows.
2. In `execute_model`, obtain final model input IDs **after** `_prepare_inputs` asynchronous reconstruction, `_sanitize_placeholder_input_ids_for_forward`, and `_preprocess`. Current relevant source areas are `_prepare_inputs` around line 1464 and the `_preprocess` call around line 2420; hook before `_model_forward` around line 2503. Use the actual one-dimensional token positions after speculative acceptance correction, not optimistic CPU positions.
3. Outside graph capture/replay, copy the final unpadded IDs and final positions to pinned CPU buffers in one batched transfer per tensor (or one packed transfer if convenient). Wait for that D2H completion before calling history. Use the pre-padding CPU query boundaries and matching request order. Do not hash `requests.output_token_ids`, which may contain async placeholders, or use padded query boundaries.
4. Call `history.prepare(...)`; feed `batch.hash_ids` to the offload manager and propagate `batch.token_mask` to the gate/mask staging path. Preserve manager `prepare -> wait_ready -> model/replay -> mark_consumed` ordering. The history helper itself performs no D2H, NPU lookup, graph capture or manager calls.
5. If runtime input representations do not contain true token IDs (for example prompt-embedding-only input without IDs), fail explicitly instead of substituting zero tokens. If a failed invocation is retried, replay the actual positions/inputs; no speculative tail is authoritative merely because it exists in history.

## Validation

The exclusive test file is `tests/ut/worker/test_engram_history.py`. It compares all layer hashes against an independent scalar n-gram/XOR implementation and covers chunking, masked lookback, complete prompt prefix hits, draft rollback, prompt preservation, generated-mask replacement, preemption/restore, row reorder, finished cleanup, explicit live-ID reset, unknown history, caller buffer reuse, batch atomicity, empty requests, malformed metadata and non-CPU input rejection. No NPU is used.

```bash
.venv/bin/python -m pytest --confcutdir=vllm-ascend/tests/ut/worker vllm-ascend/tests/ut/worker/test_engram_history.py -q
```

This component does not establish end-to-end Engram correctness or throughput by itself; actual D2H/hash/offload/graph integration must be measured in the runner.
