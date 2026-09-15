# V4.1 bounded multimodal wrapper

`AscendDeepseekV41ForConditionalGeneration`, appended to the existing
`vllm_ascend/models/deepseek_v4/model.py`, composes the Ascend text model with
the independently validated vision tower/aligner and three BF16 delimiter
parameters. It is **not registered** as the serving model in this change.
Registry, configuration and runner changes remain separate acceptance work.

## Protocol

| Method / field | Contract |
| --- | --- |
| `requires_raw_input_tokens` | `True`; router and hash consumers retain actual token IDs |
| `supports_encoder_tp_data` | `False`; vision parameters/computation are replicated |
| `get_language_model()` | Returns the existing Ascend V4.1 text child |
| `embed_multimodal(patches, vit_grid, llm_grid, types)` | Eager encode; one complete tensor per image span, including every delimiter |
| `embed_input_ids(ids, mm_embeddings, is_multimodal=mask)` | Text embedding followed by explicit-mask replacement in `[T,D]`, before child HC expansion |
| `forward(ids, positions, ..., inputs_embeds=..., **kwargs)` | Requires explicit bool `image_token_mask` with the same shape/device as raw IDs; forwards IDs, merged embeddings and masks to the child |
| `engram_prompt_mask(request)` | CPU keep mask from `V41EngramImageSpans.from_request(...).prompt_keep_mask()` |
| `create_engram_runtime()` | Delegates host tables and runtime initialization to the text child |
| `compute_logits`, `get_expert_mapping` | Delegate to the text child |
| `load_weights(iterator)` | Consume once; load MM tensors locally and invoke the language loader exactly once |

The existing processor classes in
`patch/worker/patch_deepseek_v41_mm.py` already produce the four input fields.
The wrapper consumes their protocol without importing their package during
model import or adding a processor registry decorator. Registration can bind
`DeepseekV41VLMultiModalProcessor`, `DeepseekV41VLProcessingInfo` and
`DeepseekV41VLDummyInputsBuilder` after integration acceptance.

Grids and roles must remain CPU integer tensors. Validation checks positive
patch geometry, exact ceil-downsampled aligner geometry, total patch count,
the configured per-image token budget and the exact pad-free role layout:
`[START] + ([IMAGE]*W + [NEW_LINE])*H + [END]`. Encoder spans use simple
row slices to place the aligner rows and learned delimiters. No CR2 alignment
padding or image SWA visibility change is introduced.

The embedding merge requires an explicit boolean mask; it does not infer
image positions from raw token IDs. Every selected raw ID is validated
against the configured image ID. This value validation can synchronize when
IDs are on NPU, but it runs only during eager image prefill. Encoder execution
and merging reject NPU graph capture. Text-only embedding and decoder forward
do not execute image validation or encode an image.

## Bounded construction and loading

The wrapper requires BF16 dense/vision dtype, PP1, no speculation and no
encoder data-parallel or encoder-only mode. Image limit must explicitly be
0 or 1 **per request**; an encoder batch may contain one image from each of
several requests. `engram_prompt_mask` rejects a request exceeding this limit.
The child retains its existing TP8, no-EP and no-sequence-parallel guards.

With image limit 0, no tower, aligner or delimiter parameters are allocated,
and MM checkpoint tensors are deliberately skipped. The existing text class
and its loading logic are unchanged. With image limit 1, all 266 MM tensors
must be loaded exactly once with BF16 checkpoint dtype and exact shapes;
unknown, duplicate and missing MM keys fail. Norm gamma is copied losslessly
from BF16 checkpoint values into its FP32 parameter storage.

The streaming loader never sorts or materializes the full checkpoint. A
generator intercepts MM weights while the child consumes language weights
through one loader call. Child loaded names receive `language_model.` so
the returned set matches the wrapper's actual parameter namespace. The
child continues to own expert packing/finalization and host Engram skipping.

## Validation and remaining registration gates

CPU wrapper tests: **35 passed**. Combined with the existing vision component
tests: **57 passed**. Evidence:
`vision_wrapper_and_components_cpu.xml`. Tests cover processor BatchFeature
compatibility, odd grids, multiple images from an encoder batch, all delimiter
positions and row order, merge before HC expansion, simultaneous raw IDs and
embeddings, typed-mask passthrough, prompt-owned Engram masks/cache hits,
literal image IDs outside spans, one-image request admission, image-limit-zero
allocation/loading, loader failures and initial configuration bounds.

A complete meta-device wrapper checks **all 266 MM parameter names/shapes and
BF16 checkpoint headers**, then executes streaming dispatch without allocating
the real weights. This complements, rather than replaces, the separate full
32-layer real-photo NPU numerical test documented in
`VISION_COMPONENTS_REPORT.md`. Ruff and `git diff --check` pass.

Required before serving registration:

1. Bridge the CPU prompt keep mask to final packed runtime positions, including
   generated text and graph padding, without another image-span lifecycle.
2. Pass an explicit `image_token_mask: bool[T]` into the V4.1 LM, decoder and
   MoE router. It must be true only for processor-owned image positions and
   false for literal/generated image-token IDs and graph padding. Preserve raw
   IDs for hash routing; do not substitute fake token IDs. The wrapper already
   passes typed kwargs unchanged, but downstream routing is separate work.
3. Keep checkpoint-reference causal LLM SWA. Do not silently activate upstream
   image-bidirectional SWA through configuration or metadata.
4. Validate complete-image eager prefill followed by changing text graph
   decode, including raw-ID routing, DEAD Engram history and request-slot reuse.
5. Retain the synthetic-gradient numerical stress failure and single-photo
   scope from the component report. Neither the wrapper tests nor the one
   passing photo establish universal image quality, MM serving or performance.

Typed image identity and Engram history validity are independent. A text
position can have invalid historical n-grams; therefore even
`(~engram_keep) & real_token_mask` is not a valid general image classifier.
The runtime now stores an independent processor-owned `prompt_image_mask`
at request reset and packs it by current positions, with generated positions
and graph padding false. Every wrapper forward, including text-only,
profiling and capture calls, must receive an explicit typed mask; absence,
wrong dtype, shape or device fails before the language child executes.
Initial scope keeps image prefix reuse and encoder graph execution disabled
and requires admitted complete image spans.

```bash
.venv/bin/python -m pytest --confcutdir=vllm-ascend/tests/ut/models \
  vllm-ascend/tests/ut/models/test_v41_multimodal_wrapper.py \
  vllm-ascend/tests/ut/models/test_v41_vision_components.py -q
```
