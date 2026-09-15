# V4.1 Engram image-mask bridge

The CPU component, request history, runtime and runner are connected. The
multimodal wrapper remains unregistered pending actual multimodal request
and graph integration checks.

## Interface and ownership

`vllm_ascend/worker/engram_image_mask.py` provides:

- `V41EngramImageSpans.from_prompt(prompt_ids, image_ranges, image_roles=...)`:
  validates processor-owned `PlaceholderRange` entries and optional roles,
  then stores immutable per-image half-open ranges and prompt length.
- `V41EngramImageSpans.from_request(request)`: adapts scheduler new-request
  data, including cached `MultiModalFeatureSpec.data=None`.
- `prompt_keep_mask()`: returns the full CPU bool mask needed by existing
  `EngramRequestHistory.reset_request`.
- `pack_v41_engram_token_mask(request_ids, positions, query_start_loc,
  request_spans, input_ids=..., out=...)`: builds the current CPU mask from
  final positions, real query boundaries and final request order. Optional
  final raw IDs cross-check that all declared image positions still contain
  the configured image token. Optional CPU output storage retains its
  address across calls and stays unchanged when validation fails.

Every image position is dead, including START, NEW_LINE and END. Adjacent
image ranges remain separate. A literal sentinel ID outside a declared image
range remains text, as do generated tokens beyond the prompt, regardless of
their ID. Padding is false and its token IDs/positions are not interpreted.
This component supports image features only and rejects other modalities.

Validation covers span length, bounds, nonoverlap, all-image-ID contents,
the requirement that every image-span position receives an embedding, and
the optional role grammar: START, equally wide IMAGE rows terminated by
NEW_LINE, then END. Cached features without role tensors remain valid because
their full processor-owned range still includes every delimiter.

The wrapper validates each request's immutable spans. The existing history
owns the resulting prompt masks, retaining them on preemption, removing
finished IDs and replacing them on request-ID reuse. There is no second
request map. History also owns DEAD lookback, actual tokens, rollback and
generated-tail validation.

## Runtime connection

After the runtime's existing final-token CPU snapshot, history selects
seeded prompt masks at actual positions in its validation loop. The pack
helper remains a standalone reference; the production path reuses history
instead of maintaining duplicate request state. No second D2H snapshot is
introduced. Hashing and host staging remain outside capture and replay.

## Verification

```bash
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest \
  tests/ut/worker/test_engram_image_mask.py \
  tests/ut/worker/test_engram_history.py -q
```

**43 passed**, 0.35 seconds test time. Log:
`/tmp/v41-engram-image-mask-tests.log`.

Tests cover all delimiter roles, adjacent images, cached features, reordered
mixed batches, empty requests, graph padding, stable output storage, prefix
hits, image-interior chunks, text immediately after an image, more than three
later text tokens, generated/literal ID 129264, neighboring IDs, preemption,
finish and ID reuse. Hash outputs from the actual `EngramRequestHistory`
match an independent scalar n-gram/XOR oracle exactly for both Engram layers.
Malformed ranges/roles/final metadata and incorrectly retained delimiter
embedding masks fail closed.

Those initial 43 tests exercised only the CPU bridge. Subsequent runtime
and NPU results are recorded below; full multimodal serving remains open.

## Runtime integration follow-up

The runtime now reuses the validated prompt masks already stored in
`EngramRequestHistory` after its existing final-token CPU snapshot. An
explicit `use_seeded_prompt_mask` option selects masks by request and actual
position within the history's existing validation loop. Generated positions
remain text, including repeated occurrences of a masked prompt token ID.
This avoids another request-state map and another D2H synchronization.
Explicit step masks retain the existing mismatch checks; the history's
default direct-call behavior is unchanged.

The history/image-mask suites now pass **47 CPU tests**, runner regressions
pass **16 CPU tests**, and the runtime passes **2 NPU tests** for explicit
and seeded masks. Both modes cover an actual masked prompt row, prefix
lookback, rollback, fixed addresses and graph replay. Logs are
`/tmp/v41-engram-seeded-mask-{cpu,runner,npu}.log`.
The multimodal wrapper and typed image-mask MoE routing are separate gates;
no public vision entry is registered by this change.

## Explicit image routing mask

History now accepts an independent `prompt_image_mask` and returns the
corresponding packed `image_token_mask`. An Engram DEAD position is not
necessarily an image. Only the validated wrapper prompt hook supplies image
positions; generated positions and graph padding are always false, even
when their raw token ID equals 129264. Reset clones prompt classification.

The runtime copies both masks into stable device buffers after the same CPU
snapshot. The runner forwards image identity through backbone, decoder,
MoE and the opaque complete-forward op, preserving raw IDs for hash routing.
Dummy capture clears both masks. Raw-token multimodal capture now uses both
IDs and the same embedding buffer as actual decode. V4.1 image spans have no
V4-style leading padding. No registry entry is enabled by these changes.

Validation: 76 CPU history/mask/runner/backbone/hash checks, 93 CPU
runner/metadata/V4-MoE regressions, and two NPU runtime tests passed.
The NPU tests replay changed image masks, prefix hits and generated DEAD
tokens, verify false padding and stable addresses, and compare output
exactly. CPU factories also pass under a non-CPU default-device context.
Logs: `/tmp/v41-typed-mask-cpu.log`,
`/tmp/v41-typed-mask-runner-regressions.log`,
`/tmp/v41-typed-mask-runtime-npu.log`.

Router/complete-op validation is documented separately in
[TYPED_IMAGE_MOE_INTERFACE.md](TYPED_IMAGE_MOE_INTERFACE.md).
