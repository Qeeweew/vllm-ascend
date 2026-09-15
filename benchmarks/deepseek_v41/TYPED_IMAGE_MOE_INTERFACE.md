# Typed V4.1 image-mask transport through MoE

Status: source audit followed by implementation in the four plugin MoE/router
files on 2026-09-15. CPU validation passed: 137 MoE/router tests and 27 image
mask tests, including the meta-default CPU allocation regression. Two bounded
NPU graph transport tests also passed. This is not end-to-end model validation.

## Required semantics

Pass an explicit `image_token_mask: bool[T]` alongside unchanged raw
`input_ids`. True identifies an image-span position, including delimiters.
Literal prompt or generated token ID 129264 outside a processor-owned image
span remains text and uses the original `tid2eid` lookup. Do not encode image
identity by changing IDs or store a per-call mask on the router, module,
forward-context globals, or other process-global mutable state.

The CPU Engram keep mask has the opposite meaning and excludes padding.
Therefore `~engram_keep_mask` alone is not a valid image mask: padded rows
would become images. Construct the image mask from the same immutable spans
and real request boundaries, with false padding, then copy into stable device
storage before graph execution. Every graph replay updates that storage.

## Existing call path and missing extension points

1. `DeepseekV4MoE.forward` calls the runner returned by `FusedMoEFactory`.
2. The OOT registration makes this an `AscendMoERunner`.
3. Its `forward` invokes `vllm::ascend_moe_forward_complete`.
4. That outer opaque op calls unbound upstream `MoERunner.forward`.
5. Upstream invokes `_forward_entry`, which selects `_moe_forward`,
   `_moe_forward_shared`, or the plugin shared-SP entry.
6. Those entries resolve the layer and call plugin `_forward_impl`.
7. `AscendRoutedExperts.forward_impl` prepares communication, then selects
   experts through the router's inherited `BaseRouter._select_experts`.
8. `AscendFusedTopKRouter._compute_routing` selects vision/text routes.

There is no per-invocation routing-kwargs hook along this path. Factory
`runner_args` and `routed_experts_args` only configure construction.
`CustomRoutingRouter` receives hidden states, gate output, top-k and
renormalization arguments, without request modality. `MoeRouterInput` is
constructed after selection and cannot deliver a mask to the earlier router.

## Minimum plugin-only interface

Append `image_token_mask: Tensor | None = None` to these plugin interfaces:

| Interface | Responsibility |
| --- | --- |
| `DeepseekV4MoE.forward` | Forward unchanged IDs and explicit modality |
| `AscendMoERunner.forward` | Pass mask into complete opaque op |
| Complete opaque op and fake | Declare mask as an explicit graph input |
| Plugin complete-forward helper | Preserve upstream forward orchestration |
| `AscendMoERunner._forward_impl` | Forward mask with routed input |
| `AscendRoutedExperts.forward_impl` / `_select_experts` | Preserve existing dispatch and mapping |
| `AscendFusedTopKRouter._select_experts` / `_compute_routing` | Align rows and select modality route |
| `select_deepseek_v4_vision_experts` | Use explicit mask in selection |

Retain the exact existing upstream `MoERunner.forward` path when the optional
mask is absent. For typed calls, use a plugin-local helper mirroring its small
orchestration body. Its inner entry can be an ordinary Python helper inside
the existing complete opaque op. The outer op already returns the final
combined Tensor and encloses communication-dependent reduction decisions;
three additional inner custom ops are not necessary for this interface.

The inner helper must resolve `self._encode_layer_name()` through upstream
`get_layer_from_name(_resolve_layer_name(...))` before `_forward_impl`.
When layer names are represented by `from_forward_context`, resolution
advances `moe_layer_index`. Directly calling `self._forward_impl` would bypass
that contract. The ordinary inner helper need not implement fake tuple shape
logic because only the complete op's final Tensor is visible to tracing.

An ephemeral delegation object could override `_forward_entry` and call
unbound upstream `MoERunner.forward` without changing the real module. That
avoids a small mirror but adds fragile attribute delegation around module
properties and future upstream changes. Prefer the explicit helper, pinned
to the installed upstream API with regression tests.

## Preserve the complete-forward contract

The typed helper must preserve the following upstream order:

1. Apply routed input transform unless shared input was provided.
2. Pad hidden states and retain both original width values.
3. Resolve the layer and invoke `_forward_impl` with the explicit mask.
4. Unpack routed-only or shared/routed result and trim routed output.
5. Query the plugin's communication-dependent reduced-output property.
6. Reduce routed output before a potentially nonlinear output transform.
7. Reduce shared output when needed to match the routed output.
8. Apply routed scaling, then the routed output transform.
9. Add shared and routed output.
10. Apply final reduction/width restoration, then zero-expert contribution.

Return the final Tensor to `DeepseekV4MoE.forward`. Returning an intermediate
shared/routed tuple would enter its legacy model-level scale/reduction path
and could count contributions twice. Preserve the plugin's shared-expert
overlap and sequence-parallel execution in `_forward_impl`; the new mask
belongs only to routed selection.

## Router and communication details

Override `_select_experts` only in the plugin router. The `None` path delegates
to upstream. The typed path retains its complete template: validate EPLB,
compute routing, capture logical IDs, apply EPLB mapping, then convert index
dtype. Keep the routed-expert layer's later `log2phy` map, mix-placement shared
IDs, force-EPLB and profile load balancing in their existing order.

Communication preparation happens before selection. The implemented typed
path explicitly requires DP=1, EP=1, SP disabled and ALLGATHER. In this mode
the existing ID all-gather helper is an identity and the mask retains its
rows, avoiding unnecessary bool/integer casts or collectives. A future
extension must apply the same row alignment to IDs and mask independently:

- ALLGATHER uses `all_gather_input_id_with_dp_group`.
- Other modes use `pad_and_split_input_ids`.
- The existing conditional `sequence_parallel_chunk` must affect both.
- Padding for the mask is false. Use an integer representation for the
  collective if bool is unsupported, then convert back to bool.

The current one-dimensional alignment helpers zero-pad their last dimension.
Do not stack IDs and mask as two columns and reuse them: that would pad
columns rather than token rows. TP8/DP1 without SP normally needs no added
collective, but tests must cover the other supported alignment paths.

In vision selection, use the explicit mask for text-versus-vision selection
bias and for dynamic-versus-hash expert IDs. Expert weights still come from
unbiased scores, with normalization and routed scaling in the existing order.
Actual image rows can use a safe temporary hash lookup index before
`torch.where`; raw IDs remain unchanged. Text rows must use their original ID,
including 129264. Validate dtype, rank, device and aligned row count using
metadata, without `.item()` or a host branch on `mask.any()`.

For V4.1 vision-enabled routing, absence of a typed mask must fail explicitly;
text-only V4.1 calls provide an all-false mask. Retain the historical five-ID
fallback only for V4. The present V4.1 `image_sentinel_count=1` correction
avoids classifying adjacent IDs as images but still misclassifies literal or
generated 129264 and is insufficient by itself.

## Acceptance checks

- A mixed batch with identical raw sentinel IDs routes image-span positions
  dynamically and literal/generated positions through the text hash table.
- Raw IDs are unchanged after eager, compiled and graph replay calls.
- Changed masks with stable shape/storage affect successive graph replays;
  no recapture, host synchronization or mutable router state is required.
- Shared/no-shared, ALLGATHER/MC2/ALLTOALL, routed transforms, scaling and SP
  retain the existing reduction count and final Tensor shape.
- Capture receives logical IDs; EPLB, `log2phy`, mixed placement and profile
  load balancing remain ordered correctly.
- DP/SP padding, interleaved request rows, chunked prefill and text decode
  keep mask rows aligned with router logits; all padded rows are false.
- Legacy models passing no mask continue through the existing complete op
  behavior; V4.1 missing-mask failure is explicit and actionable.

Existing starting points are `tests/ut/ops/test_fused_moe.py`,
`tests/ut/ops/test_deepseek_v41_vision_router.py`, and
`tests/ut/worker/test_engram_image_mask.py`. The new
`tests/ut/ops/test_v41_typed_moe.py` compares typed complete-forward output and
ordering against the installed upstream forward, tests strict parallelism
guards and layer index advancement, and checks changed masks through one CPU
FX graph with the complete op opaque. Model-level integration remains separate
validation work.

Validation commands:

```bash
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest \
  tests/ut/ops/test_v41_typed_moe.py \
  tests/ut/ops/test_deepseek_v41_vision_router.py \
  tests/ut/ops/test_fused_moe.py -q
../.venv/bin/python -m pytest tests/ut/worker/test_engram_image_mask.py -q
```

The complete MoE suite passed 137 tests in 0.70 seconds after environment
startup; the mask suite passed 27 in 0.24 seconds. Ruff, Markdown lint and
`git diff --check` passed for the owned changes. The CPU default-device fix
explicitly sets `device="cpu"` on all tensor factories in the mask helper;
the regression executes the request adapter and packer under
`with torch.device("meta")` and verifies concrete CPU outputs.

## Bounded NPU graph result

`tests/e2e/single_node/ops/test_v41_typed_moe.py` passed both shared/no-shared
cases on NPU 2 in 5.42 seconds after startup. The process exited zero and
released the device. Log: `/tmp/v41-typed-moe-npu.log`.

```bash
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest \
  tests/e2e/single_node/ops/test_v41_typed_moe.py -q
```

The test uses the actual registered complete opaque op and actual
`AscendFusedTopKRouter`. Routed computation is deliberately replaced with
small FP32 expert matrix multiplications (8 tokens, width 16, four experts,
top-k two) to isolate transport. Shared output uses a separate small matrix;
the complete runner combines it with routed output after a 1.5 scale.
Collective reductions are identity stubs. No HCCL, production W4A16 expert
kernel, model checkpoint, new r12 operator, or TP8 execution is exercised.

Each case replays one graph 12 times across three changing masks. All token
IDs are 129264: true image rows select vision experts [2,3], while false
literal/generated rows preserve text hash experts [1,0]. Captured expert IDs
match exactly. Results match an independent CPU numerical oracle with
`rtol=2e-4, atol=2e-5`, including shared addition. Raw IDs remain unchanged;
input/mask/capture buffers and graph output retain their addresses. Layer
resolution advances the context index once during capture and graph replay
does not execute or increment the Python resolver again. Device snapshots
are collected without intervening result reads during the replay sequence.

## Read-only integration audit

The current history stores independent immutable-prompt image positions and
returns false for generated positions; this avoids treating generated DEAD
tokens as images. Runtime copies both masks from the same final CPU snapshot
workflow and clears padded image positions in a stable device buffer. The
runner seeds the image mask only from the validated multimodal request hook,
then passes the stable slice through LM/model/layer/MoE. No additional D2H
snapshot is introduced by modality tracking.

One issue was reported to integration owners: the conditional-generation
wrapper initially accepted merged `inputs_embeds` without an explicit mask,
allowing the model's text-only all-false fallback to classify an image as
text. The image entry must require explicit typed modality. Fix and tests
are owned by the wrapper/integration agents, not this report. Actual TP8
complete-runner and model graph execution are also owned by integration.

The initial `smoke_mm_runner_tp8.py`/`smoke_mm_runner_worker.py` audit found
that its wrapper pre-hook proves arrival of typed masks at the wrapper, but
does not independently observe every actual router or verify selected Engram
rows. Suggested additions were sent to the smoke owner: eager-only router
observation of raw IDs and masks, exact text `tid2eid` assertions, boundary
Engram row checks, a text-only literal-129264 request, and actual replay
counting. A nonzero captured-graph count alone proves capture, not replay.
These are acceptance gaps in the initial smoke, not evidence that the
production routing path failed. Subsequent smoke-owner changes require their
own validation record.
