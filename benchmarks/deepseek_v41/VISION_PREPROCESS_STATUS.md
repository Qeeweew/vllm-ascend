# V4.1 vision preprocessing components

**43 CPU tests pass** against the local release's `inference/image_processor.py`
and real tokenizer. An isolated import also succeeds with imports of the
upstream GPU-selecting `vllm.models.deepseek_v41` package explicitly blocked.
No fake package entries, GPU kernels, or model registration are used.

Components are in
`vllm_ascend/patch/worker/patch_deepseek_v41_mm.py`, adapted with the upstream
Apache-2.0 license from `common/mm_preprocess.py`. They reuse public vLLM
processing interfaces and expose V4.1-specific class names:

- `DeepseekV41VLImageProcessor`: decoded PIL image to BF16 ViT patches and grids.
- `DeepseekV41VLProcessor`: concatenate images and their span-role tensors.
- `DeepseekV41VLProcessingInfo` and `DeepseekV41VLDummyInputsBuilder`: processor
  configuration, modality limits, and profiling inputs.
- `DeepseekV41VLMultiModalProcessor`: vLLM field splitting and prompt replacement.

Every image-span token is **129264**. Roles are carried separately as int64
`types`: start, image, newline, and end. Span length is `h * (w + 1) + 2`; each
position, including delimiters, receives a multimodal embedding. Patches and
aligner rows use reading order. There is no V4 compressor alignment padding,
role-offset token ID, or permutation tensor.

| Output | Shape | Dtype |
| --- | --- | --- |
| `patches` | `[sum(vit_h * vit_w), 3, 14, 14]` | BF16 |
| `vit_grid` | `[images, 2]` | int64, retained on CPU |
| `llm_grid` | `[images, 2]` | int64, retained on CPU |
| `types` | `[sum(llm_h * (llm_w + 1) + 2)]` | int64, retained on CPU |

The exact-value tests cover 21 RGB/RGBA/grayscale and size combinations,
including tiny, tall, wide, and over-budget images. Other cases verify odd
patch/LLM grids, aspect-cap boundary behavior, adjacent and separated multiple
images, changed text-prefix lengths, field splitting, all-position embedding
masks, text-only prompts, and missing vision towers. Prompt expansion is checked
through the actual `BaseMultiModalProcessor` with the release tokenizer and
independently constructed spans from the release image/type functions.

Two upstream issues are corrected in this component. Placeholder/image counts
are checked in both directions before image processing, preventing excess
placeholders from remaining silently in the prompt. For the release's uncapped
aspect ratio, the dummy image uses the actual maximum of 9189 ViT patches and
1024 span tokens; the upstream square generated only 8649 patches and 994 tokens.
The upstream bounded square fallback remains for aspect-capped configurations.

Reproduce from the repository root:

```bash
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest \
  tests/ut/patch/worker/test_deepseek_v41_mm.py -q
```

Result: `43 passed` in 3.45 seconds of test execution. Ruff and diff checks pass.
Tests requiring release fixtures skip if the local model tokenizer/reference is
absent; the successful run used `/mnt/models/DeepSeek-V4.1-Flash`.
Reference fingerprints:

```text
vLLM checkout: 836bb3839ffefcda8283ea7d41671a89e1a613df
image_processor.py SHA256:
482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272
tokenizer.json SHA256:
c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b # gitleaks:allow (public file SHA256)
```

This is a component-level CPU result. The top-level vision wrapper, registry
connection, raw-token/Engram masking, image prefill attention, and end-to-end
multimodal generation require their own integration validation. This component
does not claim that multimodal serving is enabled or performance accepted.
