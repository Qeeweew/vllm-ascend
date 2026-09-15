# V4.1 40-layer source-chain audit

## Reference and supported topology

Audited the downloaded `DeepSeek-V4.1-Flash/config.json` text configuration,
standalone `inference/model.py` (`Attention`, `Indexer`,
`SharedAttentionRuntime`) and upstream vLLM
`vllm/models/deepseek_v41/attention.py` and `nvidia/model.py`.

Config SHA256:
`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`.

| Backbone layers | CR | Shared main/index K owner | Top512 publisher | Candidate publisher |
| --- | --- | --- | --- | --- |
| 0–1 | 0 | None | None | None |
| 2–7 | 2 | 2 | 2 | None |
| 8–13 | 2 | 8 | 8 | None |
| 14–19 | 2 | 14 | 14 | None |
| 20–23 | 1 | 20 | 20 | 20 |
| 24–27 | 1 | 20 | 24 | 20 |
| 28–31 | 1 | 20 | 28 | 20 |
| 32–35 | 1 | 20 | 32 | 20 |
| 36–39 | 1 | 20 | 36 | 20 |

Every backbone layer owns its own causal SWA128 cache. Only KV sources own
compressors, main KV and index K caches. Index-only sources 24/28/32/36 own
query/weight projections; they must not own an index-key projection or norm.
Only CR2 sources 2/8/14 own persistent compressor rings. CR1 source 20 is
stateless. The three CR0 entries following the 40 backbone ratios belong to
draft layers and are not instantiated by this backbone audit.

The resulting storage registrations are 40 SWA + 4 main + 4 index + 3 circular
state = 51. MoE entries in the static forward context are not cache layers.
All cache groups use the planner's common 32768-byte page while retaining
original-token block units and the per-role compressed physical row count.

## Publication and lifetime

The current Ascend source wiring matches the reference topology. Each KV
source computes a pre-RoPE latent; its index-key projection consumes that
latent before main KV receives RoPE. Group-first RoPE and group-end-only
publication are separate contracts. Consumers resolve the configured KV source
by name and consume the most recently published shared top512. Layer 20
publishes candidate blocks once; indexers 24/28/32/36 replace top512 while
retaining layer 20's candidates.

The standalone reference explicitly states that shared slots need no reset
between forwards: sources execute before consumers on every pass. The same
reasoning applies to the Ascend shared buffers with one in-flight batch and
sequential execution of all 40 layers. Overlapping invocations require their
own metadata/buffer slots; this audit does not establish concurrent execution.

## CPU regression

`tests/ut/models/test_deepseek_v41_source_chain.py` constructs all 40 real
attention modules with reduced dense widths, native H8/D512/index32×128 shapes,
real cache registration/specs, real planner grouping, cache binding, real
Common/V4.1 metadata, `_resolve_batch` and the actual attention forward branch.
The tests retain cache-write validation and group-end masks, replacing only
the native scatter execution with a CPU scatter.

Trace operations replace projection, compressor, quantization, index selection
and attention mathematics. Layer- and generation-specific values detect a
wrong cache owner, stale top512, publication after consumption, post-RoPE index
projection, accidental candidate replacement and writes outside active rows.
All 512 selected IDs are checked at every compressed layer.

Three consecutive steps use positions 1150–1153, 1154 and 1155–1157, shrinking
and growing the active batch. SWA retains absolute logical table columns with
expired pages set to -1 and recycles eight physical pages. The first step
straddles a block boundary; the second retires another old page and leaves CR2
incomplete. Active shared-buffer rows refresh while inactive rows remain
unchanged. Four compressor and eight index calls are required on each pass.

CPU cache allocations are independent per layer: scheduler shared-pool
aliasing is covered separately by the registered-cache NPU integration test.
This regression validates control flow and metadata addressing, not numerical
attention, checkpoint quality, NPU graph replay or performance.

Validation: the two new tests plus V4.1 metadata and cache planner suites pass
64 cases together. Ruff, markdownlint and `git diff --check` pass.

Run:

```bash
python -m pytest -q tests/ut/models/test_deepseek_v41_source_chain.py
```

## Long-context acceptance boundaries

A 1152-token prompt exceeds SWA128 and fills top512 for both CR2 (576
compressed positions) and CR1 (1152 positions). It is a useful native runner
stress case for those boundaries, but candidate selection has 2048 blocks of
8 positions: actual candidate pruning starts only beyond 16384 CR1 tokens.
At 1152 tokens, candidate publication/reuse can be validated without testing
the two-level pruning limit.

Native long-context acceptance must independently check:

- Correct access after scheduler SWA eviction, preserving absolute page columns.
- Joint SWA/CSA softmax with 512 valid compressed indices and causal partial groups.
- Candidate pruning above 16384 CR1 positions, including the pinned newest block.
- Cold-cache behavior and complete TP8 timing with communication and host Engram.

INT8 indexer quantization and BF16 main/SWA cache remain numerical differences
from the standalone quantized reference; correct source wiring does not
establish checkpoint quality. Use a named reference backend for quality
comparisons because upstream CUDA cache precision also varies by backend.
