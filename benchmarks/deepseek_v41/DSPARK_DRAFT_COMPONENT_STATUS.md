# V4.1 DSpark draft component status

Status: CPU component implementation and contract tests pass. Production
speculation admission remains disabled. No draft NPU execution, real full
draft loading, acceptance-rate measurement or performance result is claimed.

## Implemented in the existing draft module

`models/deepseek_v4/dspark.py` now contains separate
`DeepseekV41DSparkModel` and `DSparkDeepseekV41ForCausalLM` classes. The
existing V4 draft math is unchanged; the new classes do not inherit it.

- Clone the draft config without changing the target's 384/top6 expert
  layout. The clone uses the released 128/top3 layout, no hash routing,
  vision router or Engram, and initially validates TP8/PP1, H5120/HC4 and K5.
- Construct three actual `DeepseekV41DecoderLayer` blocks and call V4.1
  attention with `is_draft_layer=True`. Each block requires CR0. Noncausal
  attention metadata is supplied by the separate attention integration.
- Use replicated BF16 target-aux projection, layer-specific context KV
  projection/normalization/RoPE, and the existing physical-slot cache writer.
  Context slot mappings are per layer. Profiling without slots exercises
  projections without writing unbound caches.
- Carry delayed pre-mix between sublayers and collapse using the final FFN
  pre-mix. Forward returns pre-norm hidden; logits apply the final norm.
  There are no learned V4 `hc_head` parameters.
- Expose the existing proposer hooks and explicit shared embedding/head
  flags. Every draft query has a false typed image mask, including a literal
  image-token ID.
- Load only converted `mtp.*` tensors, map Markov `embed/head` to
  `markov_w1/markov_w2`, fuse query/KV and shared gate/up with their real TP
  callbacks, and preserve packed expert tensors/scales/shape metadata.
  Required coverage is checked per fusion slice and per expert ID; missing,
  duplicate, unknown, unconverted-dtype or unconsumed weights raise errors.
  The released vision-only draft router bias is explicitly skipped.

The draft class is not registered or admitted by this component change.
Target auxiliary output, noncausal attention, runner/proposer cache metadata,
placeholder handling and scheduler rollback belong to separate integration
changes. See [the audit](DSPARK_INTEGRATION_AUDIT.md) for their contracts.
The older audit is a pre-implementation snapshot, not a current checklist of
which surrounding files have since been modified.

## CPU validation

`tests/ut/models/test_deepseek_v41_dspark.py`: **26 passed**, in 0.58 seconds
after import/bootstrap. Log: `/tmp/v41-dspark-cpu.log`; JUnit:
[dspark_cpu.xml](dspark_cpu.xml). NPU remained uninitialized.

Coverage includes config isolation, stage placement, exact released name
mapping, nonzero TP rank with genuine linear weight loaders, all 128 expert
IDs and all packed/scale/shape suffixes, negative scales passed unchanged,
missing expert/fusion rejection, sink slicing, per-layer context slots and
profiling behavior. Header-only inspection of converted shards 44–46 checks
1176/1174/1178 draft keys and the released 128-expert shapes without loading
their payloads.

The math test executes three real V4.1 decoder objects with CPU replacements
for device HC primitives and simple attention/MoE operations. Analytic
nonuniform mixes and an independent six-sublayer recurrence match BF16
terminal output exactly (`rtol=0, atol=0`); this catches use of the current
mix instead of the delayed mix and terminal normalization mistakes. It does
not validate the Ascend HC kernel, attention kernel or packed MoE arithmetic.
Those need component NPU numerical tests before integration admission.

Ruff format/check and whitespace checks pass. Reproduce from the repository:

```bash
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest \
  tests/ut/models/test_deepseek_v41_dspark.py -q \
  --junitxml=benchmarks/deepseek_v41/dspark_cpu.xml
```

## Remaining acceptance

Construct/load the real E128 draft under the normal worker, confirm target
embedding/head object identity after proposer loading, and compare complete
context/query/head numerics to an independent reference on a scheduled NPU.
Then verify actual noncausal paged attention, target auxiliary boundaries,
corrected input IDs, all acceptance lengths and graph buffer lifetimes.
Current native W4A16 dispatch does not cover draft E128/top3; functional
fallback execution and its latency remain unmeasured for this draft.

This work paused after the CPU/lint checkpoint for the requested code review
and staged commits. No commit was made by the component agent.
