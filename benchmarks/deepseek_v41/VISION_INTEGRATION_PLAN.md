# DeepSeek V4.1 vision integration on 8 × Ascend 910B

Date: 2026-09-15. This is a read-only source audit and executable integration plan. No model, runner, attention or operator source was modified by this task; no NPU was used. The existing 40-layer text runner work continues independently.

## Decision and source of truth

Implement a V4.1 multimodal wrapper around the current Ascend text backbone, with the official V4.1 processor/embedding semantics, a BF16 vision tower, exact image masks and the existing pinned-host Engram runtime. Initial support should be TP8/PP1, one image request at a time, a complete prompt prefill in eager mode, then graph-compatible text decode. Broader batching/chunking follows separate tests.

**Do not silently inherit image-span SWA widening from V4.** Two available official sources disagree:

- The released checkpoint's `inference/model.py` is the numerical reference for this plan. Its LLM SWA remains causal with width 128 even inside images. Its ViT attention, separately, is fully bidirectional within one image.
- The checked-out vLLM V4.1 implementation enables image-internal bidirectional SWA through shared `sparse_swa.py`. This changes the model computation relative to the checkpoint reference. It is an explicit compatibility discrepancy, not evidence that the Ascend causal path is missing an image kernel.

Use the released inference behavior for initial correctness acceptance. Keep a small differential test documenting the vLLM behavior. If a newer authoritative V4.1 reference establishes the widened behavior, update the declared target and implement the separate attention extension described below; do not mix these two oracles when reporting accuracy.

### Audited sources

Primary reference files are under `/mnt/models/DeepSeek-V4.1-Flash/`:

| File / lines | Contract |
|---|---|
| `inference/image_processor.py` | Resize, patch order, all-one-ID image spans and role IDs |
| `inference/vision.py` | ViT, 2D RoPE, norms, attention and aligner |
| `inference/model.py:410` | `get_window_topk_idxs`: causal prefill window |
| `inference/model.py:700` | `_window_kv`: no image input or visibility widening |
| `inference/model.py:765` | Attention forward takes hidden states/start position only |
| `inference/model.py:809` | Image routing bias replaces text bias for expert selection |
| `inference/model.py:973` | `image_mask` reaches FFN, not attention |
| `inference/model.py:1228` | Merge image embeddings before hyper-connection expansion |
| `inference/model.py:1243` | Image mask, Engram mask and atomic initial image prefill |
| `inference/engram.py:130` | DEAD image tokens stop all longer n-gram lookback |
| `inference/generate.py:56` | Images must fit wholly inside initial prefill |

Comparison source: `sources/vllm`, HEAD `836bb3839ffefcda8283ea7d41671a89e1a613df`. Relevant files are `vllm/models/deepseek_v41/nvidia/vl_model.py`, `common/mm_preprocess.py`, `attention.py`, `nvidia/model.py`, `vllm/transformers_utils/configs/deepseek_v41.py` and `vllm/v1/attention/backends/mla/sparse_swa.py`. The vLLM V4.1 wrapper explicitly imports the V4 common vision tower; that import is a reuse choice, not the reference used to establish V4.1 behavior here. The tower's math was compared with the actual V4.1 `inference/vision.py`.

Source hashes are appended at the end so this audit remains attributable if those files change.

## Exact V4.1 image and embedding contract

Released vision geometry is 32 layers, hidden width 1024, 16 heads of width 64, SwiGLU intermediate width 2816, patch size 14, 2D RoPE theta 10000, and a 3 × 3 aligner downsample. Maximum image span is 1024 LLM positions; minimum image area is 295936 pixels; the checkpoint has no maximum width/height ratio.

1. Decode RGB. Apply the official minimum-area and aspect/patch-aligned resize plan, then `ImageOps.pad` with `(127,127,127)` unless the configured wide-image special case applies. Normalize pixels as `(pixel/255 - 0.5)/0.5`, convert to BF16, and produce row-major patches `[n_vit_h*n_vit_w,3,14,14]`.
2. ViT patch projection maps 588 features to 1024. Every block has FP32-compute RMSNorm (`eps=1e-6`), QKV with bias, noncausal per-image attention, output projection with bias, another RMSNorm, and a bias-free SwiGLU MLP. Q/K use 2D half-split RoPE; this is not the text model's 64-wide tail rotation or its epsilon `1e-20`.
3. Aligner pads the patch grid on bottom/right to multiples of 3, converts to CHW and uses `unfold(kernel=3,stride=3)`. The flattening order is channel first within each 3 × 3 region, not a token-major reshape. Its 9216→5120→5120 projections have bias and use ordinary `F.gelu` between them, not SwiGLU or an approximate GELU substituted without validation.
4. Let `H=ceil(n_vit_h/3)`, `W=ceil(n_vit_w/3)`. The full span is `[START] + ([IMAGE]*W + [NEW_LINE])*H + [END]`, length `H*(W+1)+2`. Aligner rows fill only IMAGE roles; learned `image_start`, `image_newline`, `image_end` fill every delimiter. All span roles, including delimiters, occupy embedding positions.
5. **Every span position has raw token ID 129264.** Roles are separate integers `START=0, IMAGE=1, NEW_LINE=2, END=3`; text role is -1. There are no five consecutive V4 sentinel IDs, no extra padding role and no alignment pad inserted to make CR2 groups even.
6. Replace the text embedding at image positions in `[T,5120]` before expansion to `[T,4,5120]`. Raw IDs still reach the MoE router and Engram preparation. Accepting only `inputs_embeds` while dropping IDs is incorrect.

A read-only safetensor-header audit found **266 vision/aligner/delimiter tensors, all BF16, total 970536960 stored bytes**. Examples: patch weight `[1024,588]`; aligner weights `[5120,9216]` and `[5120,5120]`; delimiter vectors `[5120]`. These weights do not need FP8 dequantization or expert INT4 conversion. Preserve their values and load all of them; account for tower/aligner memory before the cache budget is chosen. Header bytes describe checkpoint storage, not TP8 runtime allocation or workspace usage.

## Image-span SWA: specify the difference before implementation

For a query at original-token position `p`, reference SWA admits precisely:

```text
max(0,p-127) <= k <= p
```

This applies to image patches, image delimiters and text equally. Source/consumer compressed-cache topology also remains unchanged. Compressors pool consecutive original positions across text/image boundaries; do not insert image-boundary pads, skip image KVs, reset ring state or align to a new CR2 group. An incomplete CR2 group still publishes only at its real group end and uses group-first RoPE.

In contrast, current vLLM's shared vision SWA path computes for an inclusive image span `[a,b]`:

```text
left  = min(p-a, max_image_tokens-1)
right = min(b-p, max_image_tokens)
start = max(0, p-127-max(left-127,0))
end   = p+right                         # inclusive
```

Outside spans it is ordinary causal SWA. `deepseek_v41/attention.py:248` enables this and `sparse_swa.py:909,1008` implements it with Triton. `DeepseekV41Config` also sets `is_mm_prefix_lm` and `mm_prefix_clamp_sliding_window` from the existence of the vision tower. Those configuration flags conflate atomic image scheduling with widened attention; review their effects explicitly in the Ascend adapter.

A decisive test needs only an image span `[10,309]`: at query 10, reference sees keys `[0,10]`, whereas the vLLM image path sees `[0,309]`; at query 250, reference sees `[123,250]`, whereas the widened path reaches the image start and end. Query 310 is outside the span and returns to `[183,310]`. Also test two adjacent image spans: raw IDs alone cannot distinguish their boundary, so widening must use per-image ranges, never one coalesced run of sentinel IDs.

For the initial reference-compatible implementation, keep the current causal Ascend DSA metadata/kernel path. Ensure MM scheduling/configuration does not silently turn on noncausal attention. Do not alter SWA admission to a fixed ring: the paged cache must still retain the current chunk plus 127 previous positions.

If widened vLLM semantics becomes the selected target, it requires its own implementation gate:

- Carry unambiguous per-request image ranges and convert `[offset,length)` placeholders to inclusive `[offset,offset+length-1]` where required. Build token visibility from final request order/positions and real query boundaries.
- Prepare fixed-address range/left/right or explicit physical-index buffers outside graph replay. Ignore padded requests/positions. Current `AscendV41CacheMetadata` has no such fields.
- Use an attention primitive supporting per-query left/right bounds or explicit selected SWA keys. The current causal original-cache mask (`ori_mask_mode=4`) cannot represent future image keys merely by increasing `pre_tokens`.
- If adding an AscendC kernel, keep projection GEMMs independent; integrate under vllm-ascend and validate sink contribution plus compressed-cache joint softmax, not two separately normalized outputs added together.
- Width must cover ordinary window plus image expansion (up to 128+1024 in upstream's allocation). Do not set global `causal=False`, which would leak text or other images across requests.
- Keep each image span atomic until cache retention and incomplete-span semantics have been validated. No bidirectional read may target an uncomputed/freed page.

## Engram masks and routing

The exact reference image mask is `token_types >= 0`, including START/NEW_LINE/END. Engram keep mask is its complement. Images both receive zero Engram contribution and become DEAD positions in history: once a lookback encounters a dead position, all older n-gram slots use the compressed pad token. Merely zeroing the final gate on image rows leaves the next text token's hash wrong.

The existing host history/hash and pinned offload runtime already accept masks, preserve them across prefill/decode and stop lookback at dead positions. Missing pieces are the model/runner bridge:

- Add `engram_prompt_mask(request)` to the multimodal wrapper. Construct the full CPU mask from processor-owned per-image placeholder ranges and roles, validating bounds, lengths, nonoverlap and that every image-span raw ID equals the configured image ID. Do not infer five-ID ranges. All generated positions begin as text unless real multimodal content is explicitly inserted.
- Supply the packed current-step keep mask after request reordering and final token/position corrections, before `runtime.prepare`. Prefer membership in persisted processor-owned image ranges using final request IDs/positions. `input_ids != image_token_id` is a valid fast path only when the serving contract reserves that ID exclusively for actual image spans; a literal/generated occurrence without image metadata must not silently become an image or a dead hash token. Cross-check the fast path against stored prompt metadata in tests. A CPU prompt mask alone is insufficient: current history rejects a later image row incorrectly presented as unmasked.
- Use the same false mask for image rows in hashing and the gate, and false for graph padding. Future text decode must retain the dead history boundary even when no image encoder work occurs in that step.
- Delegate `create_engram_runtime` through the wrapper so pinned tables are created exactly once after language weights load. Reuse the existing fixed device row buffers and prepare→wait→model/replay→consumed protocol; never load/hash/gather image Engram tables inside a captured model.

Two current hazards require explicit regressions:

1. The local sanitizer now preserves -1 for Engram, but inherited `GPUModelRunner._preprocess` still does `input_ids.clamp_(min=0)` whenever speculation is configured. Therefore future Engram+spec support must preserve or validate real IDs before that parent clamp; do not claim the local guard fixes the whole path. Initial vision support keeps speculation disabled.
2. `select_deepseek_v4_vision_experts` uses `image_sentinel_lo + 5`. V4.1 passes 129264 as the lower bound but that still misclassifies 129265–129268. Add an explicit V4.1 equality predicate or parameterized sentinel count without changing V4's own five-ID behavior. Official image bias **replaces** text correction bias for expert selection; selected weights come from unbiased transformed scores and are normalized/scaled afterward. Test neighboring IDs and score/weight separation.

## Existing Ascend gaps and minimum code boundaries

| Area | Current state | Minimum future change |
|---|---|---|
| Model registration | `DeepseekV41ForCausalLM` maps to text-only `AscendDeepseekV41ForCausalLM` | Register a V4.1-aware wrapper; preserve a text path when image limit is zero |
| Multimodal protocol | No `SupportsMultiModal`, processor registration or `embed_multimodal` | Implement protocol and bind V4.1 processor/dummy/input schemas |
| Tower / delimiters | Not instantiated; loader skips `vision.*`, `aligner.*`, `image_*` | Build/load tower and all three delimiter vectors under an explicit MM owner |
| Embedding | Backbone accepts `inputs_embeds` before HC expansion | Wrapper merges complete image spans; mark `requires_raw_input_tokens=True` |
| Engram | Runtime supports masks; MM new requests currently fail without provider | Full-prompt provider plus packed live mask and lifecycle delegation |
| MoE image mask | V4 range-of-five helper reused | V4.1 single-ID predicate and unbiased-weight regression |
| SWA | Current Ascend path is causal, no image visibility buffers | Keep for reference contract; separate project if widening selected |
| Vision attention | NPU `AscendMMEncoderAttention` exists | Validate actual OOT dispatch and noncausal TP2-head execution |
| Graph | Text runtime stages Engram outside replay | Image encoder/merge eager initially; reuse decoder static rows/masks for text decode |
| Prefix cache | MM prefix machinery exists upstream | Verify image content identities enter prefix hashes; initially disable vision prefix reuse |

Suggested future files: `vllm_ascend/models/deepseek_v41/multimodal.py` (or a clearly isolated V4.1 wrapper within the repository's existing model organization), V4.1-specific registration changes, and narrow runner mask hooks. Reuse the current text model by composition. Do **not** subclass the NVIDIA V4.1 wrapper unchanged: its constructor instantiates NVIDIA text modules, its model state uses CUDA Engram, and its loader maps into a different namespace.

The V4.1 `common/mm_preprocess.py` is a useful direct processor candidate after pure-CPU parity tests. Despite V4 names on its classes, it explicitly describes the V4.1 pad-free format. The common V4 vision tower is reusable only after component parity with the V4.1 reference; its major math matches (same patch projection, 2D RoPE, norm epsilon, SwiGLU and unfold/GELU aligner). Preserve NPU-specific attention dispatch rather than importing a CUDA execution path.

Weight loading should dispatch a streaming iterator by prefix: consume vision/aligner/delimiters in the wrapper and language tensors in the existing child loader. Preserve the child's fused expert finalization lifecycle and original dense `[wkv,wgate]` packing. Avoid copying the NVIDIA wrapper's `sorted(weights)` wholesale: retaining all tensors to sort can create an unacceptable peak for this large checkpoint. When delegating names, prevent duplicate prefix/suffix remapping and assert each expected vision parameter was loaded exactly once. Verify converted checkpoints still contain these 266 tensors before model construction.

## CUDA dependencies: what actually needs replacement

| Source | Actual dependency | Ascend action |
|---|---|---|
| Released `inference/vision.py` | Pure torch modules, `F.scaled_dot_product_attention`, `F.unfold`, no explicit CUDA or TileLang import | Treat as CPU numerical oracle; port attention to existing NPU FIA and retain its math |
| Released RoPE cache | Factory tensors rely on global default device; LRU key omits device | Keep frequency cache on CPU then transfer/cache by NPU/device/dtype, or register buffers explicitly |
| Released `generate.py` | NCCL, `torch.cuda.set_device`, CUDA allocator settings and global CUDA default | Use vLLM Ascend worker/HCCL initialization; do not import this launcher into production |
| Released `inference/model.py` | Imports `kernel.py` at module import | Do not import whole Transformer just to get a vision utility |
| Released `kernel.py` | TileLang GPU kernels, shared memory/fragments, warp/TMA pass settings, FP8/FP4 intrinsics | Existing Ascend text adaptations replace these; no direct TileLang path on 910B |
| vLLM NVIDIA V4.1 wrapper | NVIDIA text child/model-state/Engram/attention imports | Implement Ascend wrapper composition and mask staging instead |
| Common vLLM vision tower | `MMEncoderAttention` generic dispatch, not intrinsically CUDA | Assert it resolves to `AscendMMEncoderAttention` |
| Generic `MMEncoderAttention` module | Imports Triton/FlashInfer/FlashAttention wrappers and FP8 helpers | Audit import-time availability with current dependencies; do not invoke CUDA/FP8 encoder branches |
| Existing Ascend MM attention | `npu_fused_infer_attention_score`, no mask, sparse_mode 0, effectively unlimited pre/next tokens | Candidate for full bidirectional per-image attention; validate H64, TP8 (2 local heads), variable image sizes and output accuracy |

Initial vision tower weights remain BF16 with no FP8 encoder mode. TP8 divides 16 vision heads exactly, so tensor parallel vision is the simplest first implementation. Image-data-parallel tower mode is a later optimization: it needs variable-image ownership, empty-rank collectives and stable output reordering tests before use.

## Ordered implementation and acceptance matrix

Do not broaden production support until the corresponding gate passes.

| Stage | Work and tests | Required result |
|---|---|---|
| 0: lock reference | Record source hashes; CPU key-visibility differential for causal/widened image spans | Explicit selected contract; discrepancy cannot silently disappear |
| 1: processor | Square/tall/wide/tiny images, non-multiple-of-14 shapes, patch grids non-multiple-of-3, alpha/RGB conversion, minimum area, 1024-token budget, bad placeholder counts, two adjacent images | Pixel/patch tensors and role/token counts match released functions exactly on identical PIL inputs |
| 2: embeddings | Coordinate-coded 2D RoPE, unfold ordering, aligner odd-grid pad, learned delimiter insertion, row-major IMAGE count, HC expansion location | Exact indexing/layout; independent FP32/FP64 numerical oracle with BF16 rounding separated from math |
| 3: weights | Header inventory, streaming wrapper dispatch, TP8 QKV/MLP shards and row biases, missing/duplicate keys, image-limit zero | All 266 MM tensors accounted for; no silent skip, double mapping or full-checkpoint materialization |
| 4: masks/routing CPU | Whole image span and delimiters dead; text immediately after image; >3 text tokens later; neighboring IDs 129263/129265–129268; all-padding; request reorder/preemption/finish/reuse | Exact scalar n-gram hash IDs, zero image gate, correct VL bias selection and unbiased routing weights |
| 5: NPU tower | 1 and several images, TP1 oracle vs TP8, H64 attention, non-square grids, near-zero norms; component captures only if enabled | Finite outputs and component error report; no CUDA calls; preserve official norm/GELU semantics |
| 6: runner eager | One complete image prompt, text-only through MM wrapper, image then text decode, image at CR2 odd/even boundaries, prompt length around 128/32 boundaries | Image rows replaced before HC; valid raw IDs retained; caches/indexer/compressor and Engram hashes match selected oracle |
| 7: decoder graph | Capture after eager image prefill, 50 changing token/mask/position replays, request-slot reuse, all-padding, two batch buckets | Stable row/mask/cache addresses; no CPU hash/gather or image encode in replay; eager/graph outputs agree |
| 8: broader requests | Unequal text/image batches, multiple/adjacent images, image-span atomicity rejection, MM cache hits with identical vs changed image bytes, cancellation | No cross-request spans, embeddings, cache pages or history; chunking/cache policy tested before enabling |
| 9: performance | Isolated TP8 runs at small/medium/max image grid, prompt+decode matrix; stage timers and msprof | Report image decode/resize, H2D, ViT, aligner, merge, Engram D2H/hash/gather/H2D, attention/MoE, TTFT and steady-state decode separately |

Recommended numerical controls: exact CPU image preprocessing and integer masks/hash/routing selection when scores are not tied; component-level high-precision references with RMSNorm gain-bias checks rather than only a loose BF16 max error. Compare full converted-model image logits with a **converted-weight** reference to distinguish INT4/BF16 adaptation error from vision integration error. The original FP8/MXFP4 model remains a separate model-quality comparison; do not describe both as bit-exact targets.

For performance acceptance, retain at least five alternating-order rounds and raw medians/P95/noise statistics. Set per-stage baselines before kernel optimization, then require median <=1.03× and P95 <=1.05× the same-stage baseline with round-median spread <3%; do not hide TTFT regressions behind a single throughput average. Evaluate decoder throughput after image prefill against equivalent text-decode bucket sizes. CPU image preprocessing and pinned Engram transfers are part of user latency even when outside the graph.

## Concrete initial serving envelope

Start with image limit 1/request, TP8/PP1, no sequence-parallel MoE, no expert parallel, no speculation, no vision prefix reuse, no encoder graph capture, and a prompt that fits in one admitted prefill chunk. Image loading and encoder execution stay eager; text decode graph support is enabled only after stage 7. Configuration should reject a span that cannot fit, rather than silently truncate, split or reshape it. This deliberately bounded envelope is a first integration target, not a claim that the current text adapter already serves images.

The first implementation can proceed without a new AscendC vision kernel because an NPU noncausal encoder attention operator is already available. Add or optimize an AscendC operator only after profiling identifies a material bottleneck or the selected image-SWA semantics requires functionality absent from existing CANN operators.

## Source fingerprints

| Source | SHA256 |
|---|---|
| `/mnt/models/DeepSeek-V4.1-Flash/inference/model.py` | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| `/mnt/models/DeepSeek-V4.1-Flash/inference/vision.py` | `5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c` |
| `/mnt/models/DeepSeek-V4.1-Flash/inference/image_processor.py` | `482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272` |
| `/mnt/models/DeepSeek-V4.1-Flash/inference/engram.py` | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` |
| `/mnt/models/DeepSeek-V4.1-Flash/inference/generate.py` | `8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0` |
| `sources/vllm/vllm/models/deepseek_v41/attention.py` | `fcc7019a88175007b555414e9f58a1fb5d63ef3c4d5f08d8e067256728a6b6e8` |
| `sources/vllm/vllm/models/deepseek_v41/common/mm_preprocess.py` | `91f66f0c84f55709c10f16127b486a8af0e5f291efe08b46d617830eed3e8cab` |
| `sources/vllm/vllm/v1/attention/backends/mla/sparse_swa.py` | `a47853ad8a3fe40f24704592581ea323a18041b84610df010a5534da634e311e` |
