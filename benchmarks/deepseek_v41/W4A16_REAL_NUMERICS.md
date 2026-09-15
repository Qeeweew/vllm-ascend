# Real-checkpoint W4A16 arithmetic diagnosis

## Finding

The observed single-layer native/CANN difference is approximately 0.5% NRMSE
on captured decode inputs from layers 0–2. Native closely matches the FP32
arithmetic contract. CANN closely matches a contract with BF16 dequantized
weights and BF16 intermediates/routing probabilities. CANN reruns reproduce
the captured production output bit-for-bit on all three inputs.

These results do not identify the cause of complete model log-probability
changes. The separate runner comparison already shows a roughly 0.27 maximum
log-probability change between CANN eager and CANN graph/NUMA runs, without
native decode. The runner's prefix/cache behavior remains a separate
investigation. No kernel, precision threshold or default switch was changed.

## Inputs and isolation

The converted checkpoint is
`/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32`. Layer 0 uses the completed
`model-00003-of-00048.safetensors`; the script reads finished shard headers when
the converter has not yet written its final index. Gate/up projections select
TP0 output rows 0–287; down projection selects TP0 input columns 0–287. Group32
BF16 scales and offset-binary checkpoint INT4 are converted to the same signed
N-packed runtime representation consumed by both kernels.

First, synthetic normalized hidden states were multiplied by the real layer-0
FFN norm and routed using the real gate projection and text correction bias.
Input RMS was 0.12655; four tokens selected 22 experts. This establishes a
bounded reproducible arithmetic fixture, not a real activation distribution.

The stronger checks use captured `x`, `ids`, `routing`, and local CANN `output`
from `/tmp/v41-real3-cann-moe-inputs/layer{0,1,2}_tp0.pt`. They came from the
three-layer real-device-weight runner with small synthetic Engram tables.
Selected experts are remapped to a dense small bank without modifying weights
or routing probabilities. The exact CANN-output reproduction validates this
isolation. It excludes TP reduction, shared experts, attention, graph scheduling
and full-model quality.

Only device 2 was used for individual correctness calls, without timing loops.
Peak PyTorch NPU allocation was 193.2 MiB for the synthetic fixture and at most
104.8 MiB per captured layer, below the assigned 8 GiB bound. Processes exited
and released these allocations.

## Arithmetic contracts

Native performs signed INT4 group sums and signed BF16 scale application in
FP32. W13 stays FP32 into clipped SwiGLU; the activation is rounded to BF16.
W2 stays FP32, is multiplied once by FP32 routing weights, and six routes are
accumulated in FP32 before one final BF16 cast.

The CANN pipeline has additional BF16 boundaries: dequantized effective
weights, W13 output before clipped SwiGLU, W2 output before combine, and
routing probabilities supplied to token unpermute. Both use gate upper clamp
10 and up clamp [-10, 10]. None of the tested inputs activated either clamp.

Independent CPU references form effective dense weights, perform linear
algebra in FP32, and apply the declared BF16 boundaries. Intermediate reference
variants progressively enable GMM1, GMM2 and router rounding; an additional
variant rounds dequantized weights to BF16. Their errors are not additive.
FP32 accumulation order and activation approximation can still differ from
device execution at BF16 rounding boundaries.

## Captured decode results

NRMSE is `norm(actual - reference) / norm(reference)`, not a percentage.

| Layer | Tokens | Loaded experts | Native vs CANN | Native vs FP32 reference | CANN vs BF16 contract | CANN rerun vs capture |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2 | 11 | 0.0050195544 | 8.3445263e-07 | 0 | 0 |
| 1 | 2 | 11 | 0.0049552312 | 0.00013824984 | 0.001264951 | 0 |
| 2 | 2 | 10 | 0.0051725432 | 0.00034110335 | 0 | 0 |

Native/CANN maximum absolute differences are 0.00024414, 0.00024414 and
0.00048828 respectively. Native/FP32-reference maximum absolute differences
are 9.54e-7, 0.00012207 and 0.00012207. Layers 0 and 2 CANN outputs match the
BF16-contract reference bit-for-bit; layer 1 retains 0.1265% NRMSE residual.
This residual is reported rather than hidden by a tolerance change.

Using exactly CANN's own activation as down-GMM input further isolates weight
dequantization. The BF16-dequant reference has NRMSE about 4.35e-6 on layer 1,
versus 0.00184 for unrounded effective weights. On layers 0/2 the corresponding
BF16-dequant residual is also tiny. GMM1 has some BF16 output rounding
mismatches (layer 1 NRMSE 0.000201), so these full references need not be
bitwise equal even when their main arithmetic boundaries agree.

The evidence supports different rounding contracts as the main single-layer
difference. It does not indicate clipping, signed-scale layout or duplicate
routing-weight application errors on these inputs. This is insufficient to
claim identical model logits or select a full-model acceptance threshold.

## Same-input native repeatability

After strict HCCL eliminated CANN runner repeat differences, the native
three-layer graph run retained a maximum repeated selected-logprob difference
of 0.00448847. Generated tokens remained identical; native versus strict
CANN graph differed by at most 0.00390410 in selected logprob. These runs use
small synthetic Engram tables and do not validate full-model quality.

Each captured layer input was therefore independently replayed 100 times
in eager mode and 100 times through a single NPU graph, without HCCL or
model state. At most two or three BF16 output elements differ from the
first invocation; all outputs are finite. The custom kernel uses FP32
atomic accumulation for split-K W13 and routed W2, so HCCL deterministic
mode does not make this separate path bit-exact. The experiment observes
its variation without changing accumulation precision or acceptance gates.

| Layer | Mode | Max changed elements | Max repeat NRMSE | Max NRMSE vs FP32 contract |
| --- | --- | ---: | ---: | ---: |
| 0 | Eager | 2 | 2.669e-5 | 2.670e-5 |
| 0 | Graph | 2 | 1.668e-6 | 1.865e-6 |
| 1 | Eager | 3 | 4.310e-6 | 1.3825e-4 |
| 1 | Graph | 2 | 4.310e-6 | 1.3825e-4 |
| 2 | Eager | 2 | 3.800e-7 | 3.4111e-4 |
| 2 | Graph | 2 | 2.061e-7 | 3.4111e-4 |

These remain within the pre-existing operator error gate. They establish
bounded numerical variation on these inputs, not deterministic native
execution or a full-model logprob threshold. Full-model quality remains
separate from operator precision and CANN/native arithmetic parity.

## Artifacts and reproduction

- `diagnose_w4a16_checkpoint.py`: bounded diagnostic, no product changes.
- `w4_real_weights_cpu_diagnosis.json`: synthetic-hidden CPU boundary variants.
- `w4_real_weights_npu_diagnosis.json` and `.pt`: same fixture on NPU.
- `w4_real_decode_layer{0,1,2}_diagnosis.json` and `.pt`: actual captured inputs,
  native/CANN outputs, and CPU references.
- `w4_real_decode_layer{0,1,2}_repeats.json` and `.pt`: 100 eager and 100
  graph outputs per layer, plus independent references. Add
  `--native-repeats 100` to reproduce this numerical diagnostic; CPU
  snapshots synchronize deliberately, so its runtime is not performance data.

```bash
python benchmarks/deepseek_v41/diagnose_w4a16_checkpoint.py \
  --layer 0 --tp-rank 0 --device 2 \
  --input /tmp/v41-real3-cann-moe-inputs/layer0_tp0.pt \
  --output benchmarks/deepseek_v41/w4_real_decode_layer0_diagnosis.json
```

Omit `--input` for the deterministic real-router/synthetic-hidden fixture.
`--cpu-only` runs the independent boundary references without initializing an
NPU. Captured inputs are diagnostic activations, not a model-quality dataset.
