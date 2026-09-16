# Full-model and real-table acceptance readiness

Current results (2026-09-16): conversion is complete and the user released
device 7. Real-table factory acceptance passed; see
[ENGRAM_REAL_FACTORY_RESULT.md](ENGRAM_REAL_FACTORY_RESULT.md). The full
40-layer eager run also passed; see [FULL_MODEL_RESULT.md](FULL_MODEL_RESULT.md).
The preparation snapshots below preserve the earlier admission state.

CPU preparation snapshot: 2026-09-15 23:53 UTC. **Neither real full-table
loading nor full-model execution has been run by these harnesses.** The r2
reports read all 48 source headers and verified 47 published converted shard
headers. Conversion of shard 48 was still in progress; the complete manifest
and final config/index were absent. Existing conversion/download jobs were
left running without modification or duplicate checksum scans.

The independent table probe is prepared but blocked on final conversion and
a coordinated eight-card window. Full-model execution also fails capacity
admission on device 7: 27.058 GiB free, below even the MoE-only lower bound.
The unrelated device-7 process (PID 3836133 at this snapshot) was untouched.

## Capacity and admission

Header-derived ownership follows the current TP8, EP1, text-only target.
All 40 layers and 384 experts per layer are included. Vision and the three
MTP draft layers are excluded; both real Engram tables stay on the host.

| Per-rank device allocation | GiB |
| --- | ---: |
| Packed MoE INT4 | 31.640625 |
| MoE group32 BF16 scales | 3.955078 |
| TP embedding and head | 0.308228 |
| TP attention | 1.093751 |
| Replicated dense, norm, HC, Engram projections | 1.678920 |
| TP shared experts | 0.329590 |
| Additional retained FP32 router copies | 0.292969 |
| Fused runtime shape metadata | 0.000229 |
| Estimated static device total | **39.299390** |
| Admission: static + 1 GiB loading reserve + 0.25 GiB KV + 8 GiB workspace reserve | **48.549390** |

The MoE-only hard lower bound is **38,220,595,200 bytes / rank = 35.595703 GiB**.
The static estimate is **42,197,398,208 bytes / rank**. The 1 GiB loading reserve
covers the current bounded expert repack (one extra fused w13 buffer and
single-expert temporary tensors). The 8 GiB reserve is an admission policy;
allocator, NZ layouts, graph pools and workspace peaks remain unmeasured.

Real Engram layers 1 and 14 have 384,006,168 and 384,016,682 rows, respectively,
with 256 BF16 columns. Total registered payload is **393,227,699,200 bytes =
366.221833 GiB**, held by 16 owners, approximately 45.78 GiB / rank. Preferred
rank-to-NUMA placement is `[6, 7, 4, 5, 0, 1, 2, 3]`. Historical synthetic
capacity results used `[6, 6, 4, 4, 0, 0, 2, 2]`; this report does not alter them.

Host admission requires payload + 256 GiB globally and each rank's payload +
32 GiB on its assigned node. The r2 snapshot had about 1,945 GiB MemAvailable.
Converted file pages, including dense/MoE weights in full-model mode, create
additional reclaimable cache pressure; private RSS/PSS, peak RSS, faults,
registration time and I/O counters are recorded during actual loading.
The r2 snapshots predate cgroup v1 fallback and show an empty cgroup map.
The final preflight also reads the container's cgroup v1 memory limit, usage
and hierarchical limit when v2 counters are absent. The r3 snapshot records
this host's effectively unlimited v1 limit; host/node admission still applies.

The independent table probe reserves 4 GiB free HBM per device for framework
context and staging, with a measured Torch allocator ceiling of 1 GiB / rank.
Its actual staging payload is only **12,288 bytes / rank** (two tables, four
tokens, three local heads, width 256, BF16). This is why it can pass HBM
admission with the external process present once conversion completes; it
still needs a coordinated launch window and has not been launched.

## Real code paths and acceptance checks

`preflight_full_model.py` reuses converter inventory/output specifications,
reads finalized safetensors headers only, and compares all published tensor
names, shapes, dtypes and shard sizes. Once metadata is published it requires
the complete tensor index, preserved full-model shapes, exact converted
quantization/packing metadata (signed-scale INT4 group32 RTN), and matching
source config/index SHA256. It does not rewrite a checkpoint or hash hundreds
of gigabytes again. Final central payload-checksum evidence remains required.

`validate_real_engram_factory.py` invokes the production
`AscendDeepseekV41ForCausalLM.create_engram_runtime` for each of eight isolated
processes. It supplies TP rank/world-size and NUMA context explicitly, with
a one-element NPU device anchor. This bypasses model construction and HCCL;
it proves real host factory ownership and offload, **not** full-model or
collective execution. Ranks initialize sequentially, keep all 16 owners
resident together, then release in reverse order.

The factory checks both table headers before allocation. Real loading uses
`EngramTableShard.from_safetensors`, copies only the three owned head ranges
in at most 65,536-row / 32 MiB slices, first-touches NUMA-bound anonymous
storage, then registers it through the existing CANN mapped-host API.
`full_engram_audit.py` observes actual slice requests and register/unregister
calls; it rejects duplicated or missing ranges. Placement sampling is bounded
and does not scan page placement for the complete 366 GiB payload.

For each owned head, its first, middle and last row must satisfy exact
`BF16(FP32(source_FP8) * FP32(source_scale)) == converted_BF16 == loaded_row`.
The source layout is checked as one-row by 32-column scale groups. This yields
144 real row oracles across eight ranks and two tables. The factory probe then
executes 20 changed-input graph replays per rank, including DEAD and padded
rows, stable device pointers and exact BF16 output checks. The graph captures
consumption of the stable staging buffers; the production offload manager
performs gather/DMA and its stream handoff outside capture before replay.
This probe does not validate full-model token-history scheduling.

`validate_full_model_tp8.py` loads the actual unmodified complete checkpoint
through the production registry and safetensors loader. The observation-only
worker audits all 40 target layers, both real tables, no vision allocation,
the same source-row oracles, stable staging pointers and graph replay counts.
The initial backend is CANN W4A16; `--native-decode` is an explicit later option.
Prompts have lengths 40, 3 (valid token 0 and literal 129264), 129 and 384,
covering the 128-token chunk boundary. Each generates four tokens twice with
finite selected logprobs and exact repeated token IDs. Logprob differences
are reported; no new arbitrary whole-model precision threshold is imposed.
Every `--run --graph` requires `--reference`; a missing reference is rejected
before CPU preflight. Supplied eager references are read and checked for passed
status, eager mode, identical checkpoint, backend and prompts before any model
launch, then reused for output comparison. CPU-only `--graph` preparation can
still omit a reference. Neither harness claims language-quality validation.

Both harnesses require explicit owner shutdown and all 16 checked unregisters.
Full-model validation also requires a clean EngineCore exit. Independent
factory validation requires every release acknowledgement and owned worker
exit; only its own stuck children can be terminated by its cleanup fallback.

## Prepared commands

Run from the repository root. Output paths must be new; existing evidence is
never overwritten. CPU preparation does not initialize NPU:

```bash
OMP_NUM_THREADS=4 ../.venv/bin/python benchmarks/deepseek_v41/preflight_full_model.py \
  --output /tmp/v41-full-preflight-next.json
OMP_NUM_THREADS=4 ../.venv/bin/python benchmarks/deepseek_v41/validate_real_engram_factory.py \
  --output /tmp/v41-real-factory-next.json
OMP_NUM_THREADS=4 ../.venv/bin/python benchmarks/deepseek_v41/validate_full_model_tp8.py \
  --graph --output /tmp/v41-full-graph-next.json
```

Only after final conversion publication and a coordinated window, the
independent real-table probe is ready to attempt:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 \
  ../.venv/bin/python benchmarks/deepseek_v41/validate_real_engram_factory.py \
  --run --output /tmp/v41-real-factory-run.json
```

Full-model runs additionally need all eight devices above current full-model
HBM admission. Use eager first, then graph against that exact passed result:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HCCL_DETERMINISTIC=strict OMP_NUM_THREADS=4 \
  ../.venv/bin/python benchmarks/deepseek_v41/validate_full_model_tp8.py \
  --run --output /tmp/v41-full-eager-run.json
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HCCL_DETERMINISTIC=strict OMP_NUM_THREADS=4 \
  ../.venv/bin/python benchmarks/deepseek_v41/validate_full_model_tp8.py \
  --run --graph --reference /tmp/v41-full-eager-run.json \
  --output /tmp/v41-full-graph-run.json
```

## CPU evidence and remaining work

Preserved preparation snapshots are `full_model_preflight_r1.json`,
`full_model_graph_prepared_r1.json`, `real_engram_factory_prepared_r1.json`
(46 converted headers), corresponding `r2` files (47 converted headers), and
`full_model_preflight_r3.json` (47 converted headers, cgroup v1 fallback).
All r2 preparations exited successfully, recorded NPU uninitialized, and
correctly remained blocked. The installed `LLM`/`EngineArgs` signatures accept
all 22 launch keyword arguments; no engine configuration or model was created
for that check. Ruff lint/format checks pass for the six added Python files.
CPU tests passed: **15 passed**, with 14 existing Torch JIT deprecation warnings.
They exercise capacity ownership, independent-factory admission, actual
FP8 group32 source/BF16/local row equality, incorrect data/range rejection,
and rejection of incompatible eager references before startup, mandatory
references for actual graph runs, and reference-free CPU preparation.

Pending: actual real-table startup and cleanup, complete-checkpoint eager and
graph generation, actual load/HBM/host peaks, long-context validation, language
quality, native-decode comparison and performance profiling. The 144 sampled
rows do not replace all-row payload checksums. No timing from CPU preparation
is reported as model performance.
