# V4.1 maximum declared image encoder capacity

## Fixed scope and gate

The processor permits an uncapped aspect ratio and up to 1024 image-span
tokens. Its maximum dummy image is 42882×42 pixels, producing a ViT grid
3×3063 (**9189 patches**), aligner grid 1×1021 and a complete 1024-token span.
This exceeds the 1536-patch photo tested in the numerical component report.

The component acceptance gate, fixed before the first NPU run, is finite
tower/aligner/span outputs, exact shapes and **peak PyTorch allocated memory
below 4 GiB**. Reserved memory and device free/total memory are also reported.
This is an encoder component budget, not a full 40-layer model HBM admission
test. No CPU quadratic attention oracle, numerical-equivalence threshold or
image-quality claim is involved. The earlier gradient numerical failure is
not superseded by a capacity test.

The fixture constructs only the encoder modules and delimiter parameters;
it calls the actual wrapper's `embed_multimodal` implementation. All 266
checkpoint header names, shapes and BF16 dtypes must match before a run.
The NPU path uses the full 32-layer real-weight tower and production Ascend
encoder attention in inference/eager mode, without a language model or TP.

## Preparation

CPU preparation passed; evidence: `vision_capacity_prepared.json`.
The real local `DeepseekV41VLDummyInputsBuilder` and image processor generated
the input. No manually shortened sequence or alternative dummy geometry was
used. The script defaults to CPU preparation and selects no NPU unless
`--run` is present.

```bash
.venv/bin/python \
  vllm-ascend/tests/e2e/single_node/ops/check_v41_vision_capacity.py \
  --output vllm-ascend/benchmarks/deepseek_v41/vision_capacity_prepared.json
```

## NPU probe

After explicit device coordination, run exclusively on NPU1:

```bash
.venv/bin/python \
  vllm-ascend/tests/e2e/single_node/ops/check_v41_vision_capacity.py \
  --run --device 1 --repeats 3 \
  --output vllm-ascend/benchmarks/deepseek_v41/vision_capacity_npu.json
```

One cold call, one warmup and three steady calls record synchronized wall and
NPU-event times. Timings cover already preprocessed device patches through
the complete span; CPU resize, H2D, weight loading and post-run finite checks
are excluded. Hooks retain only the current tower/aligner outputs to validate
their shapes and finiteness. No intermediate layers are copied to the host.
Memory peaks conservatively include those validation outputs/checks.

The probe completed on **Ascend910B3, NPU1**, with exit code zero and the
device released. All five calls passed finite and exact-shape checks:
tower `[9189, 1024]`, aligner `[1021, 5120]`, complete span `[1024, 5120]`.
Evidence: `vision_capacity_npu.json`, `/tmp/v41-vision-capacity-npu.log`.

| Measurement | Observed |
| --- | ---: |
| Baseline allocated (weights + preprocessed patches) | 1,029,785,600 B |
| Peak allocated | 1,349,122,560 B (about 1.256 GiB) |
| Peak reserved | 1,549,795,328 B (about 1.443 GiB) |
| Fixed allocated-memory budget | 4,294,967,296 B |
| Cold synchronized wall time | 1351.613 ms |
| Warmup synchronized wall time | 233.113 ms |
| Steady wall times (three samples) | 232.126 / 233.057 / 233.268 ms |
| Steady wall median | 233.057 ms |
| Steady NPU-event median | 232.699 ms |

The fixed component capacity gate **passed**. The cold call includes first-use
operator/runtime overhead; it is reported separately from the warm samples.
Three steady samples provide an initial component baseline, not a latency
SLA or statistical tail-latency estimate. Timing excludes preprocessing and
host transfer, so it must not be presented as multimodal TTFT.

This validates the largest declared single-image shape in isolation. It does
not establish production MM profiling safety alongside the full language
model, multiple admitted images, encoder-cache occupancy or concurrent
requests. The numerical oracle and unresolved stress case in
`VISION_COMPONENTS_REPORT.md` remain separate acceptance evidence.
