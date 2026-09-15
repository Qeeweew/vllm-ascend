# V4.1 vision tower and aligner components

The new components are appended to `vllm_ascend/models/deepseek_v4/model.py`; no existing text model, runner, image mask, routing or attention class was changed by this component task. They implement the released `/mnt/models/DeepSeek-V4.1-Flash/inference/vision.py` contract directly and do not import the NVIDIA model or TileLang reference Transformer.

Status: 22 CPU component tests, six NPU component tests and one full 32-layer
real-photo test pass. The synthetic-gradient full-depth stress case remains
outside its fixed numerical gate. Components are replicated, and no
multimodal serving, TP vision, encoder graph or performance readiness is claimed.
The later unregistered wrapper and its 29 CPU tests are documented separately
in `VISION_WRAPPER_REPORT.md`; runtime and routing integration remain gated.

## Minimum wrapper interface

```python
from vllm_ascend.models.deepseek_v4.model import (
    AscendV41VisionTower,
    AscendV41VisionAligner,
)
from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention

# Construct under the normal vLLM model config context and BF16 default dtype.
# The factory is called at construction, never lazily during graph replay.
vision = AscendV41VisionTower(
    config,
    attention_factory=lambda heads, dim: AscendMMEncoderAttention(
        num_heads=heads, head_size=dim,
    ),
)
aligner = AscendV41VisionAligner(config)
features = vision(patches, n_vit_h, n_vit_w)
image_rows = aligner(features, n_vit_h, n_vit_w)
```

- Config uses flattened V4.1 `vision_dim`, `vision_n_heads`, `vision_n_layers`, `vision_inter_dim`, `vision_patch_size`, `vision_rope_theta`, `vision_downsample_ratio`, and language `hidden_size`.
- Input patches must already be preprocessed, on the same device/dtype as weights, with exact shape `[n_vit_h*n_vit_w,3,patch_size,patch_size]`.
- Tower returns `[n_vit_h*n_vit_w,vision_dim]`; aligner returns `[ceil(n_vit_h/r)*ceil(n_vit_w/r),hidden_size]`.
- Weights default to BF16; `dtype=torch.float32` selects the CPU numerical path. Norm gamma is stored initially in FP32 and all norm arithmetic is FP32 with vision epsilon `1e-6`.
- Without an attention factory, the tower uses the exact standalone reference SDPA invocation over `[H,N,D]`. This is a correctness path. Production NPU integration should inject the existing encoder FIA implementation; the default does not claim an efficient full-size NPU SDPA allocation strategy.
- The current components have **replicated weights and no TP collectives**. They are not a completed TP-sharded vision tower. A future wrapper must budget this replicated memory or add separately tested TP/image-DP ownership. The current text TP8 configuration is not implicitly inherited by these `nn.Linear` modules.
- A wrapper owns `vision`, `aligner`, the three learned delimiter vectors, multimodal scheduling and masks. Local state dict names exactly match the checkpoint `vision.` and `aligner.` subtrees. Use strict loading of those subtrees or stream each parameter through the wrapper's existing loader; do not send unrelated text/delimiter weights into a component and silently skip them.

The standalone encoder does not add START/NEW_LINE/END embeddings. It does not replace language embeddings, expand HC streams, classify image token IDs, construct Engram masks, widen LLM SWA or register a multimodal entry point. Those remain the integration steps in `VISION_INTEGRATION_PLAN.md`.

## Numerical and layout implementation

ViT uses the official patch projection, pre-norm residual blocks, half-split 2D RoPE, fully bidirectional image attention, biased QKV/output projections and bias-free SwiGLU MLP. Cos/sin are built on the explicit input device once per image and shared across all blocks; there is no device-ambiguous process-global cache. Aligner pads only bottom/right, uses channel-major `unfold` ordering, then biased linear → exact GELU → biased linear. All GEMMs are ordinary existing PyTorch calls; no new kernel or kernel build was introduced.

CPU BF16 comparison initially exposed that a superficially equivalent `[1,H,N,D]` SDPA call can choose a different implementation from the reference `[H,N,D]` call. The default reference path now preserves the original dispatch shape. CPU tests compare both FP32 and BF16 outputs **bit-for-bit**, without relaxing tolerances to hide that difference.

## Validation

`tests/ut/models/test_v41_vision_components.py`: **22 passed**. Covers four grids `(1,1)/(2,5)/(3,3)/(4,7)` in FP32 and BF16 against the actual released module, coordinate-based 2D RoPE, scalar channel-major unfold including odd grids, tiny norms/negative and zero gamma, preconstructed attention injection, strict missing/extra/shape-error loading, invalid geometry, and input dtype/grid checks.

The final CPU test constructs the complete 32-layer released geometry on the **meta** device and checks all **263 vision+aligner checkpoint names/shapes/BF16 dtypes** against safetensor headers. It allocates no 970 MB weight copy. The separate three delimiter tensors belong to the future wrapper and are therefore excluded from this component's 263 tensors.

NPU functional suite `tests/e2e/single_node/ops/test_v41_vision_components.py`: six cases using the real `AscendMMEncoderAttention` FIA path on reserved NPU1. Tests include H64 at small widths, one real-width 1024/16-head block with 5120-wide aligner, odd grids, a uniform-score proof of future-key visibility, and repeated images of changing dimensions without changing parameter storage. This suite does not test all 32 real-weight blocks or TP8 sharding. The final run with FP64 metrics passed **6/6**, with peak torch allocation **289637888 bytes** and reservation **379584512 bytes**, both below the allowed 2 GiB. Maximum tower NRMSE was **0.003849440100232081**; maximum aligner NRMSE was **0.005609695274571765** and maximum cosine error was **0.000015730125692003938**. No performance sampling was performed while the parent exercised the TP8 runner.

For NPU BF16-vs-CPU-reference comparisons, each tower/aligner output must be finite, have NRMSE <0.01 and cosine error <5e-5, and satisfy elementwise `rtol=0.03, atol=0.02`. Raw XML records per-output NRMSE, cosine error and maximum absolute difference. CPU bit-exact reference tests remain separate from hardware BF16 accumulation-order acceptance. Metrics are evaluated on CPU in FP64 to avoid a small negative cosine error from FP32 reduction rounding.

Artifacts: `vision_components_cpu.xml`, `vision_components_npu.xml`. These functional tests do not establish image TTFT, throughput, vision encoder graph support, full checkpoint image logits or multimodal serving readiness. Those require the wrapper and isolated profiling stages.

## Full real-weight synthetic stress case: currently failing

The additional `check_v41_vision_checkpoint.py` run loaded all 263 tensors and
executed all 32 vision blocks plus the aligner on a 32 × 32 patch grid. Its
input was a deterministic BF16 RGB gradient over `[-1,1]`, shaped as 1024
14 × 14 patches; it was **not** a real image processed by `image_processor.py`.
The CPU oracle was the unmodified released `vision.py`, with BF16 linear
weights and FP32 norm gamma. All comparison metrics use FP64 CPU reductions.

The threshold was fixed before execution: finite outputs, NRMSE < 0.03,
cosine error < 0.0005 and peak allocated NPU memory below 2 GiB. This test
**failed** the numerical threshold and does not approve production use:

| Output | Shape | NRMSE | Cosine error | Maximum absolute error |
| --- | --- | --- | --- | --- |
| Tower | 1024 × 1024 | 0.1166869156072316 | 0.006817904253358487 | 0.10400390625 |
| Aligner | 121 × 5120 | 0.06937203149750022 | 0.0024060693398131328 | 0.0543212890625 |

Both outputs were finite. Peak allocation was 1144144384 bytes and peak
reservation was 1268776960 bytes. Raw evidence is `vision_checkpoint_full.json`.
Checkpoint tensor SHA256 is
`d4b58607455c3c80b5ce0db15f9c869fb2d9e328219321c4ca211a3e3a9c50cc`;
released reference SHA256 is
`5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c`.

Diagnosis now captures CPU layer activations once and compares each native
norm, QKV projection, attention result, output projection and MLP using the
same CPU inputs, separating local operator errors from accumulated errors.
The cache is checked against checkpoint and reference hashes, grid, input
construction and PyTorch version before reuse. This diagnostic mode performs
synchronous transfers and is not a performance implementation or benchmark.

The completed isolated checks found the following maximum local NRMSE when
each module received the same CPU-reference input:

| Operation | Maximum local NRMSE |
| --- | --- |
| RMSNorm | 0.000024203590174611985 |
| QKV projection | 0.000034308837955945474 |
| Attention output projection | 0.000039751072995310154 |
| MLP | 0.0002077993752843767 |
| FIA BF16 | 0.001254205141665949 |
| SDPA attention | 0.00007659178643417707 |
| 2D RoPE | 0.000021352111829958056 |
| Complete block using SDPA | 0.0011101084275559443 |
| Either residual addition | 0, bit-exact |

For the initial FIA path, accumulated block-output NRMSE rose from 0.00134
at block 0 to 0.01320 at block 11, jumped to 0.03970 at block 12 and reached
0.10774 at block 31. The final norm produced the 0.11669 tower error. Neither
block 12 nor block 31 showed a large local error when fed its exact CPU input.
These observations identify depth-dependent amplification of small BF16
differences, rather than establishing a single incorrect norm or residual
operator. FIA is a larger local source, but replacing it alone did not meet
the full-depth threshold:

| Diagnostic attention path | Full tower NRMSE | Full aligner NRMSE | Peak NPU allocated bytes | Accepted |
| --- | --- | --- | --- | --- |
| Existing BF16 FIA | 0.1166869156072316 | 0.06937203149750022 | 1144144384 | No |
| Original-shape SDPA on NPU | 0.08906257086897519 | 0.05323159745877062 | 1209288704 | No |
| FIA with FP16 input/output intermediate | 0.13679297893699807 | 0.0745136540741865 | 1144144384 | No |

The FP16 variant is confined to the diagnostic harness and was not adopted
by the product components. Raw records are
`vision_checkpoint_diagnosis.json`,
`vision_checkpoint_sdpa_diagnosis.json`,
`vision_checkpoint_fp16_diagnosis.json` and
`vision_checkpoint_block_diagnosis.json`.

A CPU-only control retained all original BF16 weights, inputs and operations,
changing only SDPA's layout from the released `[H,N,D]` call to the equivalent
`[1,H,N,D]` call. It produced tower NRMSE **0.10931110866516136** and aligner
NRMSE **0.06680216709597231** versus the released CPU dispatch, without any
NPU execution. Raw evidence: `vision_checkpoint_cpu_dispatch_probe.json`.
This independently demonstrates substantial BF16 dispatch sensitivity for
the synthetic gradient input; it does not waive the numerical gate or prove
which implementation is closer to a higher-precision result.

A subsequent full CPU FP32 reference retained the exact BF16 checkpoint
weight values but performed the entire tower/aligner computation in FP32,
using the same quantized input patch values. Relative to this reference:

| BF16 execution | Tower NRMSE | Aligner NRMSE |
| --- | --- | --- |
| Released CPU 3D SDPA | 0.21236442062333272 | 0.1281155606773711 |
| NPU BF16 FIA | 0.22385422232520355 | 0.13656450823007263 |

Thus both BF16 executions deviate substantially from FP32 on this input;
the NPU result is slightly farther away. The CPU BF16 result should not be
presented as a high-precision oracle. The original cross-backend threshold
remains unmet, and this comparison is not a model-quality or production
acceptance test. Raw records:
`vision_checkpoint_cpu_fp32_probe.json` and
`vision_checkpoint_npu_vs_fp32.json`.

The error jumps coincide with unusually large MLP outputs for this input:
block 12 MLP RMS is 0.985862 (block 11: 0.084778), and block 31 MLP RMS is
9.281951 (block 30: 0.123604). Higher-precision reference comparison and
validation with a real image passed through the released processor are
tracked separately to avoid treating this gradient input as a quality test.

An independent torch NPU FP32 rsqrt check over 32768 logarithmically spaced
values from 1e-12 to 1e12 found maximum relative error 1.113877857181573e-7
against FP64; `1 / sqrt` had the same measured error. This does not reproduce
the earlier low-level AscendC Rsqrt issue. See `vision_rsqrt_diagnostic.json`.
Installed torch-npu's FIA contract defaults `inner_precise=0` and states that
BF16 does not distinguish the high-precision/high-performance flag; changing
that flag is not an established correction for this failure.

Reproduce full check and retain the expensive CPU reference for diagnosis:

```bash
.venv/bin/python vllm-ascend/tests/e2e/single_node/ops/check_v41_vision_checkpoint.py \
  --device 1 --grid-height 32 --grid-width 32 --diagnose \
  --reference-cache /tmp/v41_vision_reference_grid32.pt \
  --output vllm-ascend/benchmarks/deepseek_v41/vision_checkpoint_diagnosis.json
```

## Full real-weight photo through the released processor: passing

The natural-photo input comes from the upstream vLLM fixture
`sources/vllm/tests/v1/ec_connector/integration/hato.jpg`, depicting pigeons
on a street. To bound reference computation, the original 3082 × 2048 RGB
photo was explicitly converted to a 512 × 340 LANCZOS thumbnail. That image
was passed through the **unmodified released `image_processor.load_image`**
using the checkpoint's original `min_pixels=295936` and
`max_image_tokens=1024`, producing a valid 32 × 48 patch grid. The aligner
grid is 11 × 16 (176 feature rows); the full image span including delimiters
and newlines would be 189 tokens. This test does not execute that LLM span.

The same predeclared gate used for the synthetic case passed without any
threshold adjustment: finite outputs, NRMSE < 0.03, cosine error < 0.0005,
and peak NPU allocation below 2 GiB.

| Output | Shape | NRMSE | Cosine error | Maximum absolute error |
| --- | --- | --- | --- | --- |
| Tower | 1536 × 1024 | 0.021500502918028543 | 0.00023085055435467794 | 0.01318359375 |
| Aligner | 176 × 5120 | 0.014077014343809757 | 0.00009881744989570507 | 0.00830078125 |

Peak allocated memory was **1161278976 bytes**; peak reserved memory was
**1251999744 bytes**. The test used BF16 FIA, all 32 blocks and all 263 real
vision/aligner tensors. No NPU performance sampling was performed.
Raw full and isolated-module metrics: `vision_checkpoint_hato512.json`.

Provenance (SHA256):

- Source JPEG bytes: `8f7e776cf614298af55cb64b7116a513c37f8710959fb90a5e2babedece489b4`.
- Source decoded RGB pixels: `a3a049ab5163ee469f2d6f1d171a11ea0f22ea920dc51c835d98c1b61c536ac7`.
- Input thumbnail RGB pixels: `c118d9acb613f5a321ac804983d1e7a09e091a9b36c90b87095c01eb01f5e293`.
- Released processor source: `482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272`.
- BF16 patch tensor bytes: `0142bcc07e3a136f458cb16b7a9f4405b4f35c69b6a870c9d7d8e24b58ce4f38`.

This is one natural-photo acceptance case, not a claim that all images or the
synthetic stress case pass. The latter remains explicitly failed above.

The same photo was also evaluated with full FP32 CPU arithmetic, retaining
the identical BF16 checkpoint weight and input values. CPU BF16 versus FP32
tower/aligner NRMSE was **0.03308684893161416 / 0.02031539161114225**;
NPU BF16 FIA versus FP32 was **0.03536541014281076 / 0.021973850330129833**.
Both BF16 paths are comparably close to this higher-precision result on the
photo, unlike the highly sensitive gradient case. These diagnostics use a
different reference from the predeclared cross-backend gate; they do not
redefine or replace it. Raw evidence: `vision_checkpoint_hato512_fp32.json`
and `vision_checkpoint_hato512_npu_vs_fp32.json`.

```bash
.venv/bin/python vllm-ascend/tests/e2e/single_node/ops/check_v41_vision_checkpoint.py \
  --device 1 \
  --image sources/vllm/tests/v1/ec_connector/integration/hato.jpg \
  --image-thumbnail-max-edge 512 --diagnose \
  --reference-cache /tmp/v41_vision_reference_hato512.pt \
  --output-tensors /tmp/v41_vision_hato512_fia.pt \
  --output vllm-ascend/benchmarks/deepseek_v41/vision_checkpoint_hato512.json
```

Reproduce CPU tests without unrelated fixtures:

```bash
.venv/bin/python -m pytest --confcutdir=vllm-ascend/tests/ut/models \
  vllm-ascend/tests/ut/models/test_v41_vision_components.py -q
```

Select an explicitly allocated NPU before `pytest.main`; the device fixture honors the caller's current device instead of forcing device 0. Constructing an injected vLLM CustomOp also requires `set_current_vllm_config`; the NPU fixture supplies that context. Use `-o junit_family=xunit1` when collecting numeric XML properties.
