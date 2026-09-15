# Real full-table Engram acceptance after download

Prepared 2026-09-15 by source audit. The full-table commands below were
**not run** against incomplete shards, and the downloader was untouched.
A subsequently authorized small NPU shutdown test is recorded below. Reuse
existing loaders and diagnostics; do not rerun the synthetic 366 GiB capacity
probe as a substitute for real weights.

## Evidence and remaining gates

| Gate | Current evidence | Still required |
| --- | --- | --- |
| Aggregate registered capacity | 393227699200 payload bytes, 16 owners simultaneously, TP8; passed | Real checkpoint residency/file-cache peak |
| TP head partition | Production range coverage UT; three of 24 heads per rank | Real source/BF16/local-storage boundary samples |
| NUMA and pinned storage | Full synthetic VMAs and sampled pages, pinned DMA passed | Real loaded owner placement and registration |
| Graph offload | Full-shape synthetic staged replay and small TP8 HCCL smoke passed | Full model with real tables and final token history |
| Conversion | Streaming converter and resume/checksum tests | Complete 47/48 input hashes, conversion and BF16 sample oracle |
| Explicit cleanup | Capacity probe: all 16 unregisters and eight child exits passed | Production shutdown after real full-table loading |
| Model quality/performance | Not established by the capacity probe | Full-model correctness, profiling and report |

References: `ENGRAM_FULL_CAPACITY_RESULT.md`,
`engram_host_factory_memory_audit.json`, `ENGRAM_DOWNLOAD_RECOVERY.md`,
and `TYPED_IMAGE_MOE_INTERFACE.md`. Truncated runner fixtures, including
`smoke_runner_tp8.py --layers 40`, replace Engram with **small synthetic
tables**. They must never be reported as full-table weight validation.

## 1. Admit only complete immutable source shards

Run from the repository root after recovery has published both final paths.
Do not rename `.suffix`, `.assembling`, or the original download temporary
prefixes to make an incomplete checkpoint appear complete. Recovery verifies
whole-file SHA256 before publishing and preserves the prefixes.

| File | Exact bytes | Expected SHA256 |
| --- | ---: | --- |
| model-00047-of-00048.safetensors | 101535150936 | 824db4881320407ac340736d14dcee5ecd748c27d0f5836b8127ecc2e3781b0f |
| model-00048-of-00048.safetensors | 101537926640 | 976330f4954338e1ad8b508c32aa912032c7ad908959fd53c8307650fe4520ed |

These hashes identify ModelScope revision
`3bd368ab0f3da472b1adc6e19d37717a6cd0967f`; the audited HF revision
`dba1be0a40aa45a94ad051997016db3960a90277` has the same two hashes. A
download-complete message or safetensors header alone is not payload integrity.

```bash
cd /mnt/models/DeepSeek-V4.1-Flash
sha256sum --check - <<'SHA256'
824db4881320407ac340736d14dcee5ecd748c27d0f5836b8127ecc2e3781b0f  model-00047-of-00048.safetensors
976330f4954338e1ad8b508c32aa912032c7ad908959fd53c8307650fe4520ed  model-00048-of-00048.safetensors
SHA256
```

Run this once centrally, not in each TP worker. Preserve the recovery
manifest/status and source inode/size/mtime/ctime identity. If a complete
authoritative checksum list exists for all 48 shards, verify it once too;
the current converter fingerprints source headers, sizes and mtimes, not
whole-file source payload hashes for all earlier shards.

## 2. Resume conversion and verify the completed publication

Return to the repository root. The existing converter has a lock, bounded
row-block conversion, paired-scale validation and per-output SHA256. Do not
use `--allow-incomplete` for this final gate and do not run a second converter
against the same destination.

```bash
OMP_NUM_THREADS=8 ../.venv/bin/python examples/quantization/convert_deepseek_v41.py \
  --source /mnt/models/DeepSeek-V4.1-Flash \
  --output /mnt/models/DeepSeek-V4.1-Flash-W4A16-G32 \
  --threads 8 --rows-per-block 256

OMP_NUM_THREADS=8 ../.venv/bin/python examples/quantization/convert_deepseek_v41.py \
  --source /mnt/models/DeepSeek-V4.1-Flash \
  --output /mnt/models/DeepSeek-V4.1-Flash-W4A16-G32 --verify-only
```

Record `conversion_manifest.json`: `complete=true`, no missing source shards,
48 verified entries, unchanged paired-scale fingerprints, and published final
config/index. `--verify-only` validates existing outputs and updates the
manifest; it does not convert absent shards or publish a missing config/index.
Exit zero alone is insufficient if its JSON still says incomplete.

CPU/header-only layout inspection, after conversion:

```bash
../.venv/bin/python - <<'PY'
import json
from pathlib import Path
from safetensors import safe_open
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config
from vllm_ascend.ops.engram_hash import HostEngramLayout

root = Path('/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32')
config = DeepseekV41Config(**json.loads((root / 'config.json').read_text()))
layout = HostEngramLayout.from_config(config)
index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
total = 0
for layer, layer_id in enumerate(layout.layer_ids):
    name = f'layers.{layer_id}.engram.embed.weight'
    with safe_open(root / index[name], framework='pt', device='cpu') as reader:
        value = reader.get_slice(name)
        assert value.get_dtype() == 'BF16'
        assert value.get_shape() == [config.engram_num_embeddings[layer], 256]
    cursor = 0
    for rank in range(8):
        heads, ranges = layout.head_shard(layer, rank, 8)
        assert heads == tuple(range(rank * 3, rank * 3 + 3))
        for start, end in ranges:
            assert start == cursor
            cursor = end
            total += (end - start) * 256 * 2
        print(json.dumps(dict(layer=layer_id, rank=rank, heads=heads, ranges=ranges)))
    assert cursor == config.engram_num_embeddings[layer]
assert total == 393227699200
print(json.dumps(dict(payload_bytes=total)))
PY
```

Do not materialize a full table for sampling. For each of the 48 global heads
(24 per layer), read first, middle and final rows using `get_slice`; include
both sides of every TP boundary. For global row `r` and column `c`, the
independent source oracle is:

```text
BF16(FP32(raw_fp8_weight[r,c]) * FP32(raw_scale[r,c // 32]))
```

Engram uses scale blocks **1 row × 32 columns**, unlike dense projections'
32 × 32. Compare all 256 BF16 columns exactly against the converted row.
Resolve weight and scale files independently from the original index; scales
may cross shard boundaries. Save sampled row IDs and value hashes, not whole
tables. This validates conversion for sampled rows only; output SHA256
validates the full converted files against their conversion manifest.

## 3. Preserve single-ownership streaming during actual initialization

The actual factory is
`AscendDeepseekV41ForCausalLM.create_engram_runtime`; the MM wrapper delegates
to it. It checks both table headers before any large allocation, derives
head ranges from the tokenizer-backed layout, and invokes
`EngramTableShard.from_safetensors` once per layer/rank.

The loader obtains a safetensors slice and copies only owned ranges directly
into final NUMA-bound storage, at most 65536 × 256 × 2 = **32 MiB** per chunk.
Registration occurs after those chunks first-touch the destination. It does
not concatenate a full table or allocate a second rank-sized temporary.
All TP ranges together cover each converted table once. Expected registered
payloads are approximately 45.78 GiB/rank, totaling 366.221833 GiB.

For the full run explicitly select `safetensors_load_strategy="lazy"`,
`load_format="safetensors"`, and leave `enable_multithread_load` disabled.
The generic device loader obtains tensors before the model can skip Engram;
lazy `get_tensor` is an mmap view without a full-table copy. The small memory
audit confirms this behavior for the installed stack. Eager `read()+load`
would materialize a whole shard per worker and is rejected by the config
patch. Although `prefetch` is currently allowed, it requests all checkpoint
pages in each worker; exclude it from the no-repeated-full-read acceptance.

Capture per-rank allocation/RSS/Anonymous/PSS and major-fault/read-byte deltas
around factory initialization. Count slice row requests if load-I/O auditing
is enabled: they must partition exactly the owned head ranges with no
duplicates or out-of-range reads. Existing factory UT supplies the narrow
reader wrapper for this check. mmap read-ahead and shared page cache mean
physical disk byte counters alone cannot prove exact logical row ownership.
Large virtual file mappings alone do not establish a replicated private copy.

Use explicit `engram_numa_nodes=[6,7,4,5,0,1,2,3]` on this host, following the
latest per-rank placement validation. The full synthetic capacity run used
`[6,6,4,4,0,0,2,2]`; retain that historical map in its result. Reconfirm
topology if devices are remapped; the list indexes TP rank, not device ID.
Budget real file-cache plus model-loading peaks in addition to the registered
payload. Preserve the capacity probe's 256 GiB global and 32 GiB/local-node
reserves as initial admission floors, not as a guarantee of full-model fit.

## 4. Inspect real runtime owners and TP boundaries

After the real factory returns, run one bounded diagnostic RPC on every
worker, outside graph capture. Reuse the existing `EngramSmokeProbe` pattern
but point at the actual full model; its existing fixture constructors must
not be used. Record the following per layer/rank:

- Owner shape, BF16 dtype, `is_pinned`, NUMA node, tensor address and page-
  rounded registration size. Require 16 independent registered owners.
- `head_indices` and `head_ranges`, compared to the CPU manifest above.
- For each local head, compare local first/middle/final rows with converted
  file slices at the corresponding global row. Local index is the sum of
  preceding local head sizes plus `global_row - head_global_start`.
- Compare the same 144 sampled rows to the original FP8/scale oracle from
  step 2. This checks conversion, partitioning and actual loaded content.
- Use `probe_engram_full_capacity.sample_placement` for bounded page samples
  and `audit_engram_numa.mapping` for the complete `numa_maps` VMA record.
  Do not call the small-buffer `query_or_move` over a full 22.89 GiB owner;
  it builds a Python address for every page. No page migration is needed.
- Query page-rounded anonymous resident count and require all resident pages
  on the intended NUMA node. VmLck/VmPin being zero is not evidence of failed
  CANN MAPPED registration; verify the owner and pinned-DMA behavior.

The necessary diagnostic is a small RPC in the actual model driver, not a
new standalone allocator/framework. Existing tools provide the header, layout,
sampling and placement primitives; a full-model driver must still wire the
RPC and persist per-rank results before this gate can be claimed complete.

## 5. Full-model invocation and shutdown gate

Example first full-model launch, after root has scheduled all eight cards and
the checks above pass. Run directly, not under torchrun. This is a bounded
context configuration, not a throughput setting:

```bash
../.venv/bin/vllm serve /mnt/models/DeepSeek-V4.1-Flash-W4A16-G32 \
  --tensor-parallel-size 8 --dtype bfloat16 --load-format safetensors \
  --safetensors-load-strategy lazy --max-model-len 512 \
  --max-num-batched-tokens 512 --max-num-seqs 1 --block-size 32 \
  --kv-cache-memory-bytes 268435456 --enforce-eager \
  --no-enable-prefix-caching --no-async-scheduling \
  --limit-mm-per-prompt '{"image":0}' \
  --additional-config '{"enable_w4a16_decode":true,"engram_numa_nodes":[6,7,4,5,0,1,2,3]}'
```

This server command starts the actual converted model; it does not install
the diagnostic RPC described above. Prefer the existing LLM driver pattern
with the same arguments for the audited run, preserving its worker reports.
The supported MM admission policy may evolve; keep the first table-load gate
text-only and record the exact registry/config revision used.

Run deterministic short and chunk-boundary prompts; compare eager with a
separately launched `mode=0`, `FULL_DECODE_ONLY` graph run. Compare selected
token IDs/logprobs against the agreed reference and report numerical drift.
Validate actual finalized-token hashes, DEAD boundaries, stable device row
and mask addresses, changed-input replay, and TP gathered head order. Gather
real timings only after instrumentation and synchronous diagnostic reads are
disabled. Conversion time, file-cache state, load time, register time and
decode throughput are separate measurements.

Production shutdown must explicitly release runtime owners. Source audit
found the inherited runner shutdown did not know `engram_runtime`; the runner
now calls the explicit runtime shutdown hook. The new `shutdown()` path
first synchronizes **all streams on the owning device**, then abandons any
pending step and uses checked `close()` fences/unregister. A failed forward
may not have recorded its consumed event, so that event alone is insufficient.
Normal `close()` continues to reject an unconsumed step. Sync/unregister
failures must propagate and retain owner references for retry.

Record all 16 successful unregisters, zero cleanup errors, all worker exit
codes, and post-close anonymous memory. Test normal shutdown and a small
injected pending-step shutdown before a full real-table failure exercise.
An engine process disappearing or kernel reclaiming it is not evidence that
explicit checked unregister succeeded. Do not close mappings still used by
another process or DMA; tensor aliases can retain unregistered CPU storage
until dropped, and RSS reclamation follows that alias lifetime.

For the audited launch, allow sufficient checked-cleanup time using upstream's
existing `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` (default 5 seconds). The
executor sends SIGTERM after that grace period and SIGKILL after another four
seconds. A process-wide 60–120 second grace is a reasonable first full-table
diagnostic setting, not evidence of successful cleanup. The MM eager r3
functional smoke was reported to reach its generation assertions but exceed
the original worker grace; it must not be counted as checked full-model
unregister/normal-exit success.

## Small shutdown regression results

The shutdown fix passed 64 CPU tests in 3.00 seconds after startup across
`test_engram_numa_loader.py`, `test_engram_pinned_host.py`,
`test_engram_offload.py` and `test_engram_runner.py`. Real CPU mmap owners and
mocked CANN calls verify all-device fence order, pending-step cleanup,
sync/copy/unregister failure propagation with retained owners and retry,
normal-close rejection, capture rejection and idempotence. Log:
`/tmp/v41-engram-shutdown-tests.log`.

The bounded NPU 2 test passed both cases in 4.82 seconds; the process exited
zero and released its device. It uses one real registered 16 × 256 BF16 owner
per case (8 KiB), real staging DMA, and a small NPUGraph. One case terminates
after an unconsumed H2D submission; the other after graph replay without
recording final consumption. Normal close rejects both, shutdown completes
checked unregister, `mapping.registered` becomes false, and a graph snapshot
matches exactly. Repeated shutdown is harmless. The tiny transfers may finish
before the fence; this is a pending-protocol test, not a forced DMA race.
It does not establish TP8 worker shutdown/grace behavior or full-table release.

```bash
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest \
  tests/e2e/single_node/ops/test_engram_shutdown.py -q
```

NPU log: `/tmp/v41-engram-shutdown-npu.log`. Ruff, Markdown lint and whitespace
checks passed for the owned shutdown and documentation changes.

## Final report fields

Archive source hashes/revision, conversion manifest, exact code/CANN/torch
versions, full-model config, 16 owner records, 144 boundary sample checks,
logical row-read coverage, real load peaks, NUMA placement, graph correctness,
generation comparison, explicit cleanup and measured performance. Keep any
missing gate marked untested. A capacity PASS or small-table smoke cannot
stand in for any of these real-weight results.
