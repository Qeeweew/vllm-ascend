# V4.1 complete-image prefill and text graph integration audit

## Scope and evidence

This audit follows the production runner and local processor interfaces. The
independent smoke fixture was prepared on CPU and root ran the controlled
TP8 eager and graph integration successfully (`mm_eager.json`, eager r3;
`mm_graph.json`, graph r1). This is a
three-layer execution fixture, not full-model serving acceptance. Wrapper and
vision component CPU tests pass **57 / 57**, including six new forward-mask
boundary cases. The earlier real 32-layer vision photo result remains in
`VISION_COMPONENTS_REPORT.md`; the synthetic-gradient stress failure remains
an unresolved numerical limitation.

## Interface findings

| Boundary | Finding / required behavior | Status |
| --- | --- | --- |
| Model registry | `DeepseekV41ForCausalLM` resolves to the local MM composition | Production TP8 graph passed |
| Processor registry | Local processor, processing info and dummy builder are registered on the wrapper | Production TP8 graph passed |
| MM encoder input | Exactly `patches`, `vit_grid`, `llm_grid`, `types`; patch values move to NPU while grids/roles stay on CPU | Implemented and CPU-tested |
| Encoder output/cache | One tensor for every full span, including START/NEW_LINE/END; upstream caches by image identifier | Production TP8 eager encoder + decode graph passed |
| Gather after reordering | Upstream uses `mm_position.offset`, `.length`, `.is_embed` to slice spans and fill `is_mm_embed` | Processor must select every span position |
| Raw IDs | `_prepare_mm_inputs` retains IDs when `requires_raw_input_tokens=True`; merged embeddings are supplied simultaneously | Existing upstream interface supports this |
| Typed image routing | Explicit bool tensor per packed token; processor-owned positions only, generated text and graph padding false | Runtime/LM/decoder/MoE implementation by root |
| Wrapper forward | Missing/wrong dtype/shape/device image mask fails before entering the language child, even for text-only inputs | Implemented; six new CPU cases |
| Capture inputs | Original `_dummy_run` captured raw IDs with `inputs_embeds=None`, although runtime supplies both | Root fixed through shared `_prepare_mm_inputs` helper |
| Span offsets | Original runner defaults to V4 leading alignment modulo 4; V4.1 has no such padding | Root fixed V4.1 default to zero |
| Image limit 0 | Allocates no tower/aligner/delimiters, skips MM weights, but still requires typed all-false mask | Wrapper tested; registry text regression remains required |

The ACL graph wrapper does not copy runtime arguments into persistent inputs.
It keys capture by batch descriptor and only debug-checks positional tensor
addresses; it does not detect the `inputs_embeds=None` versus Tensor keyword
signature change. Capture and actual decode must therefore use the same
static embedding buffer and raw-ID buffer.

Engram history validity is independent of typed image identity. In
particular, `(~engram_keep) & real_token_mask` is not a general image
classifier: a text position may have unusable historical n-grams. The runtime
stores a separate `prompt_image_mask` at request reset and gathers that mask
by current positions. No raw-ID comparison may classify literal/generated
129264 as an image. The embedding merge may validate IDs after selecting an
already typed span.

Upstream `_extract_mm_kwargs` returns additional forward inputs only for
`is_multimodal_raw_input_only_model`; `requires_raw_input_tokens` alone does
not activate it. This wrapper follows the ordinary encoder/embedding path,
so the four image processor tensors are not intended as extra LM kwargs.

## Admission and graph constraints

1. Production processing info advertises one image, and the wrapper defaults
   to one when no limit is given. Explicit `limit_mm_per_prompt={"image": 0}`
   disables the tower. More than one image is rejected.
2. Keep `cudagraph_mm_encoder=False` and `compile_mm_encoder=False`.
   `_execute_mm_encoder` attempts the encoder graph manager before calling
   `embed_multimodal`; the wrapper's own capture guard alone cannot enforce
   this policy. Initial image prefill remains eager, text decode uses the
   full decode graph after correct static-buffer capture.
3. Use `disable_chunked_mm_input=True` and a token batch large enough to admit
   the complete actual image span plus surrounding text. The photo fixture
   produces 189 image tokens, 191 total prompt tokens. Frontend admission
   compares the processor's declared maximum (1024), so the harness uses a
   1024-token batch limit even for this smaller fixture. A first frontend-only
   attempt with 512 failed that validation before launching workers.
   Prefix caching is disabled for this initial run.
4. Preserve reference causal language-model SWA. Upstream config enables
   MM-prefix plumbing when vision layers exist, but that must not silently
   enable bidirectional image attention in the V4.1 DSA implementation.
5. Vision dummy profiling can choose a very wide one-row image to maximize
   the token budget, reaching approximately 9189 patch tokens. The 1536-patch
   passing photo does not validate that memory peak. The later independent
   full 32-layer capacity probe does pass that single-image shape at
   1,349,122,560 B peak allocated; see `VISION_CAPACITY_REPORT.md`. This
   bounded TP8 smoke uses
   `skip_mm_profiling=True` and explicit KV memory; production worst-case
   admission/memory profiling remains separate work.

## Ready-to-run fixture and harness

Files: `smoke_mm_runner_tp8.py`, `smoke_mm_runner_worker.py`,
`smoke_mm_registration.py`.
The script defaults to CPU-only preparation; `--run` explicitly launches all
eight NPUs. The current harness uses the production model and processor
registries without test-only registration. `smoke_mm_registration.py` is a
historical artifact used by the earlier eager/graph results below. Spawn
remains necessary to avoid the frontend OpenMP/fork conflict. The worker
rejects the old text class and any encoder graph manager.

Production-entry run `mm_production_numa_graph.json` passed with strict HCCL,
three real language layers/E384, full real 32-layer vision, small synthetic
Engram, and NUMA nodes `[6,7,4,5,0,1,2,3]`. Each rank encoded once (189-token
span), replayed six graphs, captured three native W4A16 calls with zero CANN
fallback captures, and retained CANN prefill. All host table pages matched
the requested nodes. All eight terminal shutdown RPCs released their owners;
explicit 30-second client shutdown left EngineCore exitcode 0 and no live
process. Log: `/tmp/v41-mm-production-numa-graph-r1.log`.

The production image-limit-zero run also passed, recorded in
`mm_production_limit0_graph.json` and
`/tmp/v41-mm-production-limit0-graph-r1.log`: all eight ranks allocated zero
MM parameters, made zero encoder calls and replayed six graphs. Raw token
IDs were used with `inputs_embeds=None`, image masks stayed false, all host
owners were released, and EngineCore exited with code 0. Thus both enabled
and disabled vision routes have now run through the real production registry.

Production registry and processor CPU suite: **87 passed**, log
`/tmp/v41-mm-production-entry-cpu-r3.log`. This result supersedes the earlier
registration-pending status; it is still a bounded integration fixture, not
full-model quality or performance acceptance.

Preparation includes real converted language weights for layers 0–2 with
all 384 experts and the real BF16 vision tower, aligner and three delimiters.
Small synthetic Engram tables make this an execution fixture, not a model
quality test. Converted shard 00001 contains all 263 tower/aligner tensors;
shard 00002 contains the language embedding and the three delimiters.
All 266 MM header dtypes are checked before running. A mixed unselected shard,
duplicate MM weight or incomplete set fails preparation.

CPU preparation passed at `/tmp/v41-mm-smoke-prepared`; evidence is
`mm_smoke_prepared.json`. The source image SHA256 is
`8f7e776cf614298af55cb64b7116a513c37f8710959fb90a5e2babedece489b4`;
its 512×340 LANCZOS thumbnail pixel SHA256 is
`c118d9acb613f5a321ac804983d1e7a09e091a9b36c90b87095c01eb01f5e293`.
The official processor is then used unchanged.

From the workspace root, after typed-mask wiring lands and after
coordinating exclusive TP8 access:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn .venv/bin/python \
  vllm-ascend/benchmarks/deepseek_v41/smoke_mm_runner_tp8.py \
  --checkpoint /tmp/v41-mm-eager \
  --image sources/vllm/tests/v1/ec_connector/integration/hato.jpg \
  --run --output vllm-ascend/benchmarks/deepseek_v41/mm_eager.json

VLLM_WORKER_MULTIPROC_METHOD=spawn .venv/bin/python \
  vllm-ascend/benchmarks/deepseek_v41/smoke_mm_runner_tp8.py \
  --checkpoint /tmp/v41-mm-graph \
  --image sources/vllm/tests/v1/ec_connector/integration/hato.jpg \
  --run --graph --output vllm-ascend/benchmarks/deepseek_v41/mm_graph.json
```

The script records four generated tokens/logprobs for the photo request and
a later text request containing literal 129264 without image data. The local
processor now permits reserved IDs when there are zero image payloads while
retaining strict placeholder counts for requests with images; all **43**
processor CPU tests pass, including that regression.
The harness asserts one encoder invocation per worker, complete
189-token image prefill, raw image IDs, two surrounding text positions,
nonoverlap of typed image/Engram keep masks, pinned tables, stable row/mask
addresses and all-false typed image masks for the final active text bucket.
Each layer's eager router is observed to receive unchanged raw IDs and the
runtime's typed mask; hash-router text selections must exactly equal
`tid2eid[raw_id]`, including the later literal image ID. Graph runs require
captured graph entries and successful invocations of existing graph entries
after both image and text prompts. Buffer capacity beyond the
last active bucket is deliberately excluded from mask-value checks.

Compare eager/graph token IDs and selected logprobs in the two JSON files;
both runs passed their functional gates. Their generated IDs match but their
selected logprobs are not bit-exact, as recorded below. Instrumentation
reads eager masks back to CPU and is installed after capture. These runs are
not valid timing data. A later profiling run must remove these hooks, report
TTFT and steady decode latency separately, account for Engram D2H/hash/H2D,
and retain the correctness gates before accepting any optimization.

The second startup attempt used fork after MM frontend processing initialized
CPU thread pools. Workers aborted in `ParallelOpenMP.cpp:64` with `Invalid
thread pool!`, before checkpoint loading. Subsequent runs use the existing
vLLM spawn option above; thread-pool checks are not bypassed.

The third attempt passed all eager functional gates on eight ranks. Each
rank encoded one 189-token span, and all three routers received 189 typed
image positions followed by ordinary text decode. The later literal 129264
remained text. Image output IDs were `[60190, 60190, 60190, 60190]`; text
output IDs were `[91488, 55063, 44099, 34954]`, with finite selected logprobs.
All router observations report `text_hash_checked=false`: this Flash config
has no `num_hash_layers`, so these three layers have no `tid2eid`. No hash
expert-selection numerical coverage is claimed for this run. Teardown still
logged force termination/shared-memory cleanup warnings after writing the
passing result; lifecycle cleanup remains a separate gate.

The r3 result predates native/fallback MoE dispatch counters and checked
terminal Engram shutdown. It must not be cited as evidence of either.
Subsequent runs install dispatch observation in `worker.load_model`, before
warmup/capture, and require native decode (capture for graph runs) plus
additional eager fallback calls from actual prefills. After all outputs and
mask/router assertions, a terminal `finish_mm_smoke` RPC calls
`runtime.shutdown()`, verifies `_closed`, no pending prepare and every
`shard.weight is None`, then calls it a second time to check idempotency.
Failures propagate rather than producing a passing JSON. This separately
records checked resource unregistration even if upstream multiprocessing
later exceeds its five-second teardown grace period. No requests or
inspection RPCs may follow the terminal release.

The graph r1 result passes those dispatch/release checks on all eight ranks:
each rank has one captured graph, six actual replays, `native_capture=3`,
`native_eager=3`, `fallback_capture=0`, `fallback_eager=9`, and a closed runtime
with its shard weight released. Both image and later text token IDs match
eager r3 exactly. Maximum absolute selected-logprob differences are
**0.004553079605102539** for the image request and
**0.00006556510925292969** for the later text request. First-token prefill
logprobs match exactly; differences occur during decode. Known native FP32
atomic accumulation is not bit deterministic, but these observed values
are reported without changing an acceptance tolerance or claiming numerical
equivalence. Machine-readable comparison: `mm_eager_graph_comparison.json`.

Graph r1 still exited through the parent's default five-second best-effort
process-manager cleanup despite the longer worker setting. The current
harness explicitly invokes
`llm.llm_engine.engine_core.shutdown(timeout=30.0)` after checked release,
then requires all retained EngineCore process objects to be stopped with
exit code zero. This detaches normal finalizers through the existing client
API; it does not patch the production engine. That extra process-exit gate
was added after graph r1 and passed in the image-limit-zero run below.

## Image-limit-zero regression

After the MM eager/graph pair, repeat the graph command with
`--image-limit 0 --checkpoint /tmp/v41-mm-limit0-graph` and a separate output
JSON. The checkpoint deliberately still contains all vision tensors, so the
run exercises the wrapper's loader skip path. Both requests are text-only;
the second contains literal 129264. The harness checks zero MM parameter
elements, no tower allocation, zero encoder calls, no typed image positions,
literal-token text hash routing and successful graph replay.

With every modality limit zero, upstream
`MULTIMODAL_REGISTRY.supports_multimodal_inputs` returns false. This case
must use raw IDs with `inputs_embeds=None`, unlike text-only requests to the
image-enabled wrapper, which retain static embedding buffers. Eager forward
signatures are recorded and checked against the configured image limit.

The real TP8 image-limit-zero graph run passed:
`mm_limit0_graph.json`. All eight ranks report no allocated tower, zero MM
parameter elements and encoder calls, no image prefills, raw-ID-only eager
signatures, six actual graph replays and native capture dispatch. Literal
129264 remains text in all three routers. All Engram shards are released by
the terminal RPC. Explicit client shutdown waits for all eight workers to
exit gracefully (about eight seconds) and EngineCore to exit with code zero
(about eleven seconds total); no forced kill or leaked-shared-memory warning
appears in `/tmp/v41-mm-limit0-graph-r1.log`.
