# DeepSeek V4.1 DSpark integration audit

## Implementation checkpoint after this audit

The audit below records the initial gap analysis. Subsequent guarded changes
now export target HC auxiliary means, provide the V4.1 draft model and loader,
and add explicit noncausal K5 draft metadata/attention. The Engram staging
boundary now checks active placeholders before inherited embedding clamps.
Component results are recorded in
[DSPARK_DRAFT_COMPONENT_STATUS.md](DSPARK_DRAFT_COMPONENT_STATUS.md),
[DSPARK_ATTENTION_COMPONENT.md](DSPARK_ATTENTION_COMPONENT.md), and
[ENGRAM_PLACEHOLDER_CLAMP_FIX.md](ENGRAM_PLACEHOLDER_CLAMP_FIX.md).
The production registry now maps `DSparkV41DraftModel` to the Ascend draft.
A CPU test executes the real proposer `load_model` sharing branches with
text and multimodal target shells and verifies embedding/head object identity
(including valid token 0). Registry, loader-component and admission regressions:
**57 passed**; log `/tmp/v41-dspark-loading-cpu.log`. Allocations and checkpoint
IO are replaced in this sharing test, so real-worker payload loading remains
unverified. Speculative admission stays disabled pending real-weight integrated
execution, acceptance/rollback and end-to-end graph validation.
The historical blockers below must not be read as the current implementation
status or as evidence that those integration gates have passed.

## Outcome and scope

The current Ascend V4.1 target deliberately rejects speculative decoding.
Removing that guard is insufficient: the V4.1 draft architecture is not
registered, the target does not export the required auxiliary states, the
draft needs different hyper-connection math and expert counts from V4, and
the current V4.1 attention backend does not implement noncausal draft blocks.
There is also a concrete input-ID sanitization hazard before Engram staging.

This is a source and checkpoint-header audit of the shared working tree on
2026-09-15. It makes no production changes and runs no NPU workload. Paths
beginning with `sources/vllm` are relative to the workspace; other source
paths are relative to `vllm-ascend`. Line numbers describe this snapshot and
may move as the main adaptation proceeds. Existing component test evidence
is distinguished below from missing DSpark integration evidence.

## Released model contract

The local `/mnt/models/DeepSeek-V4.1-Flash/config.json` specifies:

| Property | Target | DSpark draft |
| --- | --- | --- |
| Decoder layers | 40 | 3, checkpoint `mtp.0` through `mtp.2` |
| Routed experts / selected experts | 384 / 6 | 128 / 3 |
| Compression | CR0 at 0–1, CR2 at 2–19, CR1 at 20–39 | CR0 at 40–42 |
| Hidden / HC streams | 5120 / 4 | 5120 / 4 |
| Target auxiliary layer IDs | 37, 38, 39, zero based | Concatenated input, width 15360 |
| Prediction block | — | `dspark_block_size=5` |
| Noise token | — | 128799 |
| Markov rank | — | 256 |

Use `method="dspark"`. Official
`sources/vllm/vllm/config/speculative.py` rejects V4.1's legacy `mtp` method
and selects architecture `DSparkV41DraftModel`; the presence of `mtp.*`
checkpoint names does not imply the MTP proposer is appropriate.

The original index contains 2401 `mtp.*` tensor keys in shards 44–46 of 48.
The converted `DeepSeek-V4.1-Flash-W4A16-G32` directory already contains
1176, 1174 and 1178 draft keys in those three shards respectively. Counts
change because conversion changes tensor representation and scale keys.
Draft conversion is therefore not wholly missing. Loading the converted
draft into an Ascend V4.1 draft model remains unverified.

Examples from converted shard 46 headers:

| Tensor | Dtype | Shape |
| --- | --- | --- |
| `mtp.2.confidence_head.proj.weight` | BF16 | `[1, 5376]` |
| `mtp.2.markov_head.embed.weight` | BF16 | `[129280, 256]` |
| `mtp.2.markov_head.head.weight` | BF16 | `[129280, 256]` |
| `mtp.2.norm.weight` | BF16 | `[5120]` |

Dense draft weights are converted to BF16 and draft experts to packed INT4
group32. The target loader intentionally skips `mtp.*`; a separate draft
loader must consume them. There are no learned `hc_head` tensors or Engram
tables in the three released draft stages.

## Concrete interface blockers

| Interface | Current evidence | Minimum required change |
| --- | --- | --- |
| Admission | `patch/platform/patch_engram_config.py:45` rejects speculation; the MM wrapper constructor also rejects it. | Keep rejection until an explicitly bounded DSpark route passes the gates below. |
| Draft registry | `models/__init__.py:51` overrides `DSparkDraftModel`, not `DSparkV41DraftModel`; official registry resolves the latter into the CUDA/AMD V4.1 package. | Register an Ascend V4.1 draft implementation under the exact official architecture. |
| Target auxiliary output | `DeepseekV41Model` returns only final normalized logits hidden; V4.1 causal-LM and MM wrappers lack `SupportsEagle3` and aux setters. Runner requests this interface for DSpark and raises without it. | Export the selected post-layer HC means, relay setters through wrappers, and return the aux list when requested. |
| Draft HC semantics | `models/deepseek_v4/dspark.py` constructs V4 decoder blocks and learned `hc_head_fn/base/scale`. V4.1 uses delayed pre-mix and no learned HC head. | Compose V4.1 decoder blocks and collapse using the last FFN pre-mix. |
| Draft expert layout | Existing draft allocates/maps `config.n_routed_experts`, i.e. target 384; released draft is 128/top3. | Use `dspark_n_routed_experts` and `dspark_num_experts_per_tok` in construction and packed-weight mapping. |
| Head weight mapping | Old `_remap_dspark_name` retains `markov_head.embed/head`; shared head modules register `markov_w1/markov_w2`. | Follow official V4.1 remapping and verify every required converted tensor is consumed. |
| Draft attention | `ops/dsa_v41.py` hardcodes causal SWA (`ori_mask_mode=4`, left127/right0), without `ori_sparse_indices`. Builder ignores `common.causal`. | Add a CR0 V4.1 draft attention path with explicit paged visible indices for the entire noncausal query block. |
| Context cache population | Official draft derives context KV independently for each layer from projected target aux states. Existing V4 draft accesses old `dsa_attn.swa_cache_layer`. | Expose V4.1 cache names and per-layer context slot mappings; apply each layer's KV projection, normalization and V4.1 RoPE. |
| Placeholder validity | Ascend sanitizer preserves `-1` for Engram validation, but inherited `_preprocess` clamps it to zero first. | Resolve or reject active placeholder IDs before embedding and Engram hashing; padding outside real query bounds remains excluded. |
| Cache planning | V4.1 planner accepts its own cache spec classes and a fixed capacity 8/1024 FP32 ring at page 32768. | Draft must expose the V4.1 CR0 SWA spec, not legacy V4 cache classes; initially fix K5. |
| Draft graph metadata | `AscendDSparkProposer.use_cuda_graph=False`; device metadata setup recognizes the old `AscendDSAMetadataBuilder`. | Keep draft eager initially; introduce V4.1 metadata and stable graph buffers before enabling draft capture. |

### Target auxiliary states are not final logits hidden

Official `sources/vllm/vllm/models/deepseek_v41/nvidia/model.py:695–737`
captures each selected layer's **post-FFN, post-HC hidden state**, averaged
over the four HC streams. It is captured before the following layer's
Engram injection. The official fused implementation may materialize this
as `previous_aux` in the following block; that is an implementation detail,
not a different semantic boundary.

The Ascend decoder already materializes its HC post result before returning.
Thus `hidden.mean(dim=1)` immediately after each selected layer is the
minimal matching output. Each aux tensor is `[T,5120]`; the runner combines
three into `[T,15360]`. Do not substitute final RMSNorm output, the
pre-mix-weighted `mhc_collapse`, flattened `[T,20480]` streams, or states
after the next Engram injection.

`worker/model_runner_v1.py` already translates DSpark's zero-based
`[37,38,39]` to one-based `(38,39,40)` for the EAGLE interface. Preserve this
convention. `get_mtp_target_hidden_states` is only used for method `mtp`;
adding a second full-HC transport buffer is unnecessary for DSpark.

### Minimum draft model composition

Use the math and weight contract in
`sources/vllm/vllm/models/deepseek_v41/nvidia/dspark.py`, while keeping
Ascend attention, packed MoE and HC implementations inside the plugin:

1. Project and normalize the concatenated target aux states once.
2. For each of three CR0 draft layers, project this same context through
   that layer's KV projection and normalization, apply V4.1 RoPE at actual
   context positions, and write its own context slots.
3. Expand query token embeddings to four HC streams and execute three
   V4.1 decoder blocks with delayed pre-mix. The draft has no Engram and
   uses text routing with an explicit all-false typed image mask.
4. Collapse with the final FFN pre-mix. Return pre-norm head hidden;
   `compute_logits` applies the final norm. Retain the Markov and confidence
   head contracts used by the existing proposer.
5. Declare `has_own_embed_tokens=False` and `has_own_lm_head=False`.

Embedding/head aliasing already exists in
`spec_decode/llm_base_proposer.py:438–576`, including extraction of the
target language model from an MM wrapper. Do not implement a duplicate
sharing mechanism. Assert shared object identity after loading so absent
draft embed/head tensors cannot leave randomly initialized parameters.

The existing V4 `main_proj` uses a gathered column-parallel linear whereas
the official V4.1 version is replicated. That may be a valid sharding
choice after matching shapes and numerical behavior; it is not a reason
to import the old V4 HC decoder or its weight mapper.

### Noncausal paged attention can reuse a small existing component

`attention/dsa_v1.py:416` provides `build_dspark_swa_indices`. It derives
`query_lens` from query offsets, `prefix_lens=seq_lens-query_lens`, and
enumerates `[max(0,prefix_lens-window_size), seq_lens)` through each
request's block table. Every query in a request receives the same trailing
context plus complete query block; unused columns contain `-1`. It also
accepts caller-owned output buffers for stable graph addresses.

This is reusable only after checking the V4.1 logical block size, storage
stride, context-window convention and sink/RoPE semantics. Existing DSA
passes such indices to `npu_sparse_flash_mla` as `ori_sparse_indices`.
The limitation on combining explicit SWA with compressed indices does not
block CR0 draft layers, which have no compressed cache. Reuse this bounded
component rather than transplanting the complete legacy V4 attention stack.

For default `sample_from_anchor=True`, the proposer executes five query
rows per request for K5. Target verification executes six rows: anchor plus
five proposed tokens. These counts must not be interchanged in attention,
scratch allocation or graph bucket selection.

## Engram and compressor rollback

### Existing support and its assumptions

`worker/engram_history.py` already maintains request-keyed token/mask
history. `prepare` hashes using actual corrected positions and preceding
tokens, then transactionally overwrites the executed span and truncates a
generated tail beyond the new stop, preserving the immutable full prompt.
Consequently, a subsequent execution beginning earlier than a rejected
suffix does not include that suffix in its hash lookback.

This is valid while one target preparation is in flight and the next
actual start position is authoritative. Host history may temporarily retain
verified-but-rejected inputs after sampling; it must not be read by another
consumer before corrected preparation. Draft noise/query tokens must never
be sent through target Engram staging. Draft layers have no Engram.

`worker/engram_runtime.py` already has one post-correction D2H input
snapshot, host hashing/gathering outside the graph, stable staging buffers,
and prepare/wait/consume lifecycle. No blanket per-round copy of the full
host history is required. Preserve the independent prompt image mask:
generated literal token 129264 is text, even though the same ID marks actual
image spans in the prompt.

However, the snapshot currently occurs too late to catch clamped active
placeholders. The confirmed call order is:

```text
Ascend _sanitize_placeholder_input_ids_for_forward  (Engram: preserve -1)
  -> inherited GPUModelRunner._preprocess           (speculation: clamp min0)
  -> Ascend _prepare_engram_model_kwargs             (snapshot/hash IDs)
  -> model forward
```

The unconditional clamp is in
`sources/vllm/vllm/v1/worker/gpu_model_runner.py:3532`; Ascend calls
`_preprocess` around 2497 and stages Engram around 2571. It also applies with
`image_limit=0`. Fix within the plugin's integration boundary and add a
call-chain regression: an unresolved **active** `-1` must never become a
valid zero-token history entry. Checking only the standalone sanitizer is
insufficient. Real resolved zero tokens must remain valid.

CR2 compressor state already uses a position-indexed FP32 ring. In
`models/deepseek_v4/compressor.py`, capacity is
`max(8, next_power_of_two(K+2))`: anchor/draft verification plus a preceding
row. For K5, six overwritten positions leave the necessary preceding row
available in capacity 8. CR1 has no historical pair state.

### Required rejection trace

Let a target round execute positions `p` through `p+5`, with the anchor at
`p`. If `a` proposed tokens are accepted, the next correction/bonus input
starts at `q=p+1+a`:

| Accepted draft tokens `a` | Next input position `q` | Required preceding CR2 row |
| --- | --- | --- |
| 0 | `p+1` | `p` |
| 1 | `p+2` | `p+1` |
| 2 | `p+3` | `p+2` |
| 3 | `p+4` | `p+3` |
| 4 | `p+5` | `p+4` |
| 5 | `p+6` | `p+5` |

The row is needed only for the appropriate CR2 pair parity. Cover both
parities of `p`, ring wraparound and request boundaries. Corrected rows
overwrite rejected rows; compressor metadata must expose only completed
pairs valid at each query position. Main/index compressed cache entries
from a rejected suffix must be hidden by corrected lengths and compression
boundaries, then overwritten when execution reaches those slots again.
Retained SWA lookback must remain available; the target currently requests
`extra_retained_tokens=num_speculative_tokens`.

The planner at `patch/platform/patch_kv_cache_utils.py:610` fixes a 32768-byte
common page and requires an unpadded capacity 8/1024 FP32 ring. K5 fits;
K7 requires capacity 16 and is rejected. Operator support for larger rings
does not mean the allocator supports them.

Existing tests include CPU history rollback, prompt/mask preservation and
compressor chunk equivalence, plus NPU compressor rollback and graph
replay with changed chunks in `test_compressor_v41.py`. These validate
components. They do **not** validate target verification, acceptance
sampling, scheduler correction, Engram and all cache roles together.

## Graph and performance requirements

The first functional milestone should use fixed K5, batch1, greedy
sampling, eager draft and eager target verification, TP8/PP1/DP1/CP1,
no EP/SP/DBO, image limit0, prefix caching off and asynchronous scheduling
off. This is a bounded acceptance target, not a claim of present support.

Next enable target graph verification at the six-row bucket, keeping
Engram preparation outside capture. Stable graph inputs include token IDs,
positions, typed image/Engram masks, gathered Engram rows, query offsets,
sequence lengths, block tables, slot mappings, compressor positions/state,
candidate/index buffers and the aux outputs consumed by the proposer.
Replay must change token values, positions, rejection counts and request
slot ownership without stale captured values. No per-layer host sync or
pageable-memory copy may enter the captured model path.

Draft graph is a separate milestone: it needs stable context/query slot
mappings, explicit noncausal indices, per-group cache metadata and seed/
confidence/Markov buffers. Simply removing `use_cuda_graph=False` is not
sufficient. The existing V4.1 metadata builder also labels multi-token
queries as prefill and only forces decode during capture for query length 1.

Current native W4A16 dispatch must not be advertised as DSpark acceleration:

- `quantization/methods/wna16/w4a16.py:39` limits the native path to at most
  four token rows and 384 experts/top6/H5120/local intermediate288.
- K5 target verification has six rows per request and uses the fallback.
- The draft has 128 experts/top3 and also uses the fallback.

The existing CANN W4A16 fallback is a candidate correctness baseline, whose
draft packed-weight loading and execution still need validation. A native
draft or verification kernel requires measured shape-specific improvement
and independent numerical checks before its dispatch predicate is widened.

Profile both accepted tokens per round and each latency contributor:
draft context projection/cache insertion, three draft layers, Markov/head
sampling, target verification, Engram D2H/hash/gather/H2D, acceptance and
scheduler overhead. Compare against greedy target-only decoding using the
same converted weights, prompts, output lengths, graph mode and capacity.
Report TTFT, median/P95 inter-token latency, delivered tokens/s, acceptance
histogram, peak HBM and pinned host bytes. Report warm steady state
separately from load, graph compilation and first-touch costs.

Proposed performance gates must be fixed before measurement: at least 30
timed rounds per workload after warmup; all correctness gates pass; median
and P95 latency are both no worse than the matched fallback for any native
kernel enablement; representative serving throughput improves over the
target-only baseline before DSpark is enabled by default. Report a neutral
or negative result explicitly; a faster draft alone is not a serving win.

## Ordered implementation and acceptance gates

1. **CPU contract:** register the correct draft architecture, export exact
   auxiliary states, validate 128/top3 packed tensor mapping and absent
   learned HC head, and prove embedding/head identity. Exercise actual
   converted checkpoint headers and reject missing required parameters.
2. **Attention and state components:** compare noncausal paged draft SWA
   against a dense CPU oracle for short contexts, window edges, disjoint
   requests and changed block tables. Verify context KV projection/RoPE
   independently. Cover the full rejection trace for Engram, CR2 rings and
   main/index cache visibility, including immutable prompt image spans.
3. **Eager TP8 integration:** complete at least two consecutive draft and
   verification rounds with forced acceptance counts 0–5; compare target
   logits/caches to sequential target execution. Include EOS, preemption,
   cancellation and reused batch slots. Ensure no placeholder is silently
   converted to token 0 and no noise token enters host target history.
4. **Target graph TP8:** replay fixed six-row verification with changed
   acceptance/positions and repeat the state/logit checks. Keep draft eager
   and compare generated greedy tokens to eager target-only execution.
   Check host shard release and clean worker shutdown.
5. **Performance report:** measure the decomposition above before writing
   a new native MoE specialization or draft graph path. Add each optimized
   route only with numerical, median and P95 evidence and a safe fallback.
6. **Broader admission:** validate batch>1, image-bearing target requests,
   variable K and additional parallel modes independently. Until then,
   reject those combinations early rather than inheriting untested V4
   behavior. Non-greedy sampling also needs an explicit distributional
   correctness gate; greedy equivalence does not establish it.

No DSpark integration, acceptance-rate result or end-to-end speedup is
claimed by this audit. Its purpose is to define the smallest correct path
and identify the concrete interfaces that currently prevent that path.
