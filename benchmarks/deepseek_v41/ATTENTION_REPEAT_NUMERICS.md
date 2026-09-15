# First-layer attention repeatability diagnosis

The measured first divergence is the BF16 tensor-parallel reduction after
`layer0.attention.wo_b`. All eight rank-local matmul outputs are identical
between the two repeated requests, but their HCCL-reduced outputs differ.
Repeating the experiment with `HCCL_DETERMINISTIC=strict` removes this
divergence in the traced sample.

## Experiment and evidence

Both runs use eight 910B NPUs, eager execution, three layers of converted
checkpoint device weights, small synthetic pinned host Engram tables, and
the CANN W4A16 path. Native W4 decode and prefix caching are disabled.
Forwards 0 and 11 contain the same complete 32-token input, positions,
query boundaries, and sequence lengths; request IDs differ.

The diagnostic captures each rank's local `wo_b` output immediately before
the caller performs its in-place all-reduce. CPU snapshots are cloned so
later mutation cannot overwrite this evidence. All stages preceding the
reduction, including Q projection, sparse attention and `wo_b` input,
are identical on every rank within each run.

| Measurement, forward 0 versus 11 | Default HCCL | Strict HCCL |
| --- | ---: | ---: |
| All eight local matmul outputs identical | Yes | Yes |
| FP32 sum of eight local outputs identical | Yes | Yes |
| Reduced output consistent across ranks within each forward | Yes | Yes |
| Changed reduced BF16 elements / 163840 | 93357 | 0 |
| Maximum absolute reduced-output difference | 0.0234375 | 0 |
| Reduced-output NRMSE | 0.004757636 | 0 |
| Repeated generated token IDs identical | Yes | Yes |
| Maximum repeated selected-logprob difference | 0.064050436 | 0 |

The complete comparisons retain every rank's stage differences and metadata:

- [Default all-rank comparison](attention_trace_all_ranks_comparison.json).
- [Strict all-rank comparison](attention_trace_all_ranks_strict_comparison.json).
- [Default smoke result](runner_tp8/eager_real3_all_rank_trace.json).
- [Strict smoke result](runner_tp8/eager_real3_all_rank_strict_trace.json).
- [Earlier TP0 detailed attention comparison](attention_trace_repeat_comparison.json).
- [Earlier three-layer comparison](layer_trace_repeat_comparison.json).

The first-layer attention divergence occurs before the first MoE, Engram
(layer 1), or CR2 compressor (layer 2). In the earlier TP0 trace, physical
SWA pages changed between requests while sparse-attention output remained
identical. Those page changes therefore do not explain the first divergence.
Downstream routing can change after the initial reduction perturbation.

## Interpretation and limits

The eight-rank evidence localizes the observed repeatability failure to
the reduction boundary, and the strict-mode control supports HCCL
reduction ordering as its source. This evidence does not expose HCCL's
internal accumulation schedule or identify a library implementation bug.

Strict BF16 reduction still differs from a sum accumulated in FP32 and
rounded to BF16 once: 80178 elements differ, maximum absolute difference
0.015625, NRMSE 0.003512628. Deterministic reduction is not an FP32 accuracy
mode. No product precision contract, numerical threshold, or default
switch was changed by this diagnosis.

These synchronous activation traces are correctness diagnostics and cannot
measure performance. Their result does not establish graph/eager equivalence,
batch invariance, native/CANN parity, or full-model quality. In particular,
three layers and synthetic Engram tables cannot validate checkpoint quality.
Untraced eager/graph repeats and separate native/CANN checks remain necessary.

## Replaying the comparison

The raw tensors currently reside in local temporary directories; the JSON
artifacts above retain the numerical comparison if those directories are
removed. From the repository root:

```bash
../.venv/bin/python benchmarks/deepseek_v41/compare_layer_trace.py \
  --directory /tmp/v41-real3-tp-all-trace --all-ranks \
  --first 0 --second 11 \
  --output benchmarks/deepseek_v41/attention_trace_all_ranks_comparison.json

../.venv/bin/python benchmarks/deepseek_v41/compare_layer_trace.py \
  --directory /tmp/v41-real3-tp-strict-trace --all-ranks \
  --first 0 --second 11 \
  --output benchmarks/deepseek_v41/attention_trace_all_ranks_strict_comparison.json
```

The comparator rejects mismatched or truncated request identity/boundary
tensors. The test-only worker rejects graph-mode tracing and skips capture
and profile forwards. Its CPU harness verified snapshot independence under
in-place reduction, metadata capture, hook reinstallation, and nonzero-rank
attention-only tracing.
