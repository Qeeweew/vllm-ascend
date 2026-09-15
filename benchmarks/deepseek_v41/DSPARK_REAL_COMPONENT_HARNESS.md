# V4.1 real-weight DSpark component harness

CPU preparation and seven independent reference tests passed on 2026-09-15.
The coordinating agent released the eight-card window for context-9 capture.
The first attempt loaded real draft weights and shared vocabulary, then failed
before forward execution because this harness omitted `enable_custom_op()`.
All eight rank records have `distributed_cleanup=true`; torchrun exited 1.
Evidence remains in `/tmp/v41-dspark-component-c9/rank*.json` and
`/tmp/v41-dspark-component-c9-run.log`. That attempt produced no numerical result.

The harness now explicitly enables the compiled custom operators and logs
construction, loading and capture phases. Attempt r2 completed device capture
but its CPU comparison exposed a missing shared-expert capture. Ascend splits
the shared MLP into linear/activation stages, bypassing the module forward
hook. The harness now observes the actual `_run_shared_mlp` input/output before
its collective and rejects incomplete captures before saving them.

## Context-9 numerical result

Attempt r3 captured every required stage and failed the unchanged numerical
gates: 688 of 712 checks passed, with all 24 failures isolated to `wo_a`
(three layers across eight ranks). The maximum NRMSE was 2.0652. Independent
CPU diagnosis reproduced the actual output using the loaded transposed weight
storage incorrectly reinterpreted as the original layout. The production
projection was corrected to consume the loaded layout.

Attempt r4 passed all **712 of 712** independent stage checks, with identical
TP attention outputs across ranks. Both attempts exited torchrun with code 0,
and every rank recorded successful distributed cleanup. Peak allocated NPU
memory was 3,588,575,744 bytes per rank (3.342 GiB), below the fixed 8 GiB gate.
All numerical thresholds remained unchanged.

| Context-9 r4 stage | Maximum NRMSE across ranks/layers |
| --- | --- |
| Grouped output projection `wo_a` | 0.00024408 |
| Routed W4A16 experts | 0.00064369 |
| Shared BF16 experts | 0.00536281 |
| Noncausal attention | 0.00204518 |
| LM-head logits | 0.00005708 |
| Markov bias | 0.00002199 |
| Confidence | 0.000000091 |

Tracked evidence is in [dspark_real_components](dspark_real_components/manifest.json):
r3/r4 comparisons, per-rank status/memory records, layout diagnosis, source
fingerprints and a manifest of original log paths and SHA256 values. Large
capture tensors remain in the corresponding `/tmp/v41-dspark-component-c9-r*`
directories. Contexts 33 and 129 are prepared; their execution is pending.

## Scope and provenance

`check_dspark_v41_tp8.py` constructs the registered `DSparkV41DraftModel`
through real vLLM engine configuration and loads all three released draft
layers: 128 experts per layer, top three routing, hidden size 5120, TP8.
The converted stage inventories contain 1176, 1174 and 1178 tensors.
All draft, expert, embedding, vocabulary, Markov and confidence weights come
from `/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32`. The loader consumes converted
BF16 dense tensors and signed-scale INT4 group32 experts.

Target auxiliary inputs are deterministic synthetic BF16 values, explicitly
recorded in `prepared.json`; target layers and host Engram tables are not
constructed. The fixture keeps all three draft layers and all 128 experts.
It constructs small target vocabulary holders and uses the production
proposer's sharing methods, checking both embedding and head object identity.
This does not validate a complete target model or speculative acceptance.

Context lengths 9, 33 and 129 exercise short, cross-page and sliding-window
cases. Each case uses five noncausal query tokens, real paged caches and a
reversed block table. Logical attention indices and physical cache-write
slots remain distinct. Peak allocated NPU memory must remain below 8 GiB
per rank. Actual peak and allocator reservation are recorded separately.

## Independent numerical checks

`dspark_v41_reference.py` imports no production model or operator math.
Each CPU comparison consumes the actual captured input to that stage.
This is a stage oracle, not independently propagated full-model output.
The reference streams one selected expert at a time and chunks vocabulary
projections; it never expands the complete expert bank to BF16.

Checks cover context projection and norm, context KV/RoPE, query and query
KV projection/RoPE, independent dense noncausal attention with denominator-only
sinks, inverse RoPE and grouped output projections, delayed HC controls and
stream orientation, attention/FFN norms, exact interstage delayed mix wiring,
exact initial query embeddings, top-three expert identity, routing weights,
local W4A16 experts, local shared experts, final weighted collapse, final norm,
logits, Markov embeddings/bias and confidence.

Router normalization is followed by the configured factor 1.5. The current
factory uses `apply_routed_scale_to_output=False`, so this scaling appears in
captured router weights. The CPU expert reference respects BF16 boundaries
for effective weights, gate/up outputs, clipped SwiGLU, down output and routing
weights. TP attention output agreement across ranks is required. Deviation
from an FP32 sum of local BF16 outputs is reported as a separate diagnostic.

Fixed gates are inherited from accepted component checks:

| Component | Gate |
| --- | --- |
| Dense, MoE and head projections | NRMSE < 0.006 and peak-relative error < 0.015 |
| Attention | RMS error <= 0.006 * reference RMS + 1e-6; rtol 0.025, atol 0.012 |
| HC control tensors | NRMSE < 2e-4 |
| HC post | rtol 0.01, atol 0.01 |
| Collapse, delayed wiring, IDs and embedding lookup | Exact equality |
| Every result | Finite values |

Capture instrumentation copies tensors to CPU for diagnosis. Its durations
are not model performance measurements and must not appear as latency or
throughput claims. Each rank must finish checked distributed cleanup before
CPU comparison admits its capture.

## Commands

Run from the repository root. Use a new output directory for every capture.
Preparation validates headers and the real engine configuration without
initializing NPU. The verified preparation used
`/tmp/v41-dspark-component-c9`, with `npu_initialized=false`.

```bash
OMP_NUM_THREADS=4 ../.venv/bin/python benchmarks/deepseek_v41/check_dspark_v41_tp8.py \
  --prepare --context-tokens 9 --output /tmp/v41-dspark-component-c9

# Only after the coordinating agent releases all eight cards:
HCCL_DETERMINISTIC=strict OMP_NUM_THREADS=4 ../.venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=8 benchmarks/deepseek_v41/check_dspark_v41_tp8.py \
  --run --context-tokens 9 --output /tmp/v41-dspark-component-c9

# After every NPU worker exits; this stage uses CPU only:
OMP_NUM_THREADS=8 ../.venv/bin/python benchmarks/deepseek_v41/check_dspark_v41_tp8.py \
  --compare --context-tokens 9 --output /tmp/v41-dspark-component-c9

OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest \
  --confcutdir=tests/e2e/single_node/ops \
  tests/e2e/single_node/ops/test_dspark_v41_model.py -q
```

Repeat preparation/capture/comparison in separate directories with
`--context-tokens 33` and `--context-tokens 129` once the first case is resolved.
Retain `prepared.json`, `rank*.json`, `rank*.pt`, `comparison.json` and process
logs for each attempt, including failures. Never relax the gates to make a
capture pass.
