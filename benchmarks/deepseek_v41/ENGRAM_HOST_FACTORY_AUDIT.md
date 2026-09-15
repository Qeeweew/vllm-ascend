# Engram host factory audit

The factory uses the configured converted-model directory and index, loads
three disjoint hash heads per TP rank, and defaults to pinned BF16 storage.
**All 21 dedicated CPU unit tests pass** (0.29 s test time). They use the real upstream `DeepseekV41Config`, tokenizer
normalization, real small safetensors files, and the actual table loader. Only
the pinned allocator and NPU staging manager are replaced, so these tests do
not establish full-table pinned-allocation success on hardware.

The original release nests Engram fields under `text_config`; upstream
`DeepseekV41Config` flattens them onto `hf_config`, as required by the factory.
The release has layers 1 and 14, 24 heads per layer, head dimension 256, and
384006168/384016682 table rows. The generated prime buckets exactly cover both
tables. The converted index preserves the native names
`layers.{1,14}.engram.embed.weight` and points to converted BF16 files.

The dedicated tests cover every TP rank, exact file-backed row contents,
index-based paths, tokenizer path and trust setting, scheduler capacity,
default pin requests, missing tables/files, non-BF16 rejection, pin failure
without fallback, disabled Engram, and slice-only reads of owned heads.

```bash
OMP_NUM_THREADS=8 .venv/bin/python -m pytest \
  vllm-ascend/tests/ut/models/test_deepseek_v41_engram_factory.py -q
```

## Host memory accounting

| Allocation | Across TP8 | Per rank |
| --- | --- | --- |
| Final pinned BF16 tables | 366.221833 GiB | 45.777168–45.778257 GiB |
| File-backed source pages, if fully cached | up to 366.221833 GiB | shared physical page cache |
| Maximum logical source-copy slice | up to 256 MiB across eight workers | 32 MiB |
| Two pinned staging buffers, two layers | 49152 bytes per token capacity | 6144 bytes per token capacity |
| Compressed token maps, history, metadata | additional, workload dependent | rank-local |

The exact final table allocation is **393227699200 bytes**, not eight full
copies. Each rank allocates only its heads and copies in chunks of at most
65536 rows. `get_slice` source tensors may be file-backed views; the 32 MiB
slice size bounds the logical read/copy operation, not necessarily an extra
anonymous allocation. File mapping itself consumes virtual address space,
not physical RAM equal to the entire mapped file. Loaded source pages and
pinned destination pages are distinct physical copies; their combined table
footprint can reach about 732.44 GiB while the source remains cached. Source
page cache is reclaimable, whereas final pinned storage is not. Other model
checkpoint pages, dense loading temporaries, allocator overhead, tokenizer
state, and staging must be added for a process/node peak estimate.

A single-process sparse-file probe retained in
[engram_host_factory_memory_audit.json](engram_host_factory_memory_audit.json)
created a 256 MiB BF16 safetensor and called `get_tensor` without touching its
values. RSS increased by 1508 KiB and anonymous memory by 32 KiB; the tensor
pointer belonged to the file mapping. Copying one 32 MiB slice then increased
RSS by about 81.5 MiB, including file pages, destination and runtime/allocator
overhead. These are synthetic pageable measurements, not measured production
TP8 peaks or a pinned-memory capacity guarantee.

## Issues for integration

The upstream default loader calls `get_tensor` before the model's
`load_weights` skips Engram weights. With `lazy`, this remains a file-backed
view, as verified above. With `eager`, upstream executes `load(f.read())` and
materializes an entire shard in each worker before the model can skip it.
A production Engram shard is about 183.11 GiB: eight simultaneous full-file
reads alone can consume about 1464.9 GiB of private bytes, before decoded
tensors and the rest of model loading. The Ascend config resolver now defaults to `lazy` and rejects `eager`,
`torchao`, and device-direct load formats for this path. Prefetch shares physical page cache but may repeat
full-file reads across ranks and should be evaluated separately.

The model owner fixed both initialization issues found during this audit.
The factory now preflights all names, files, BF16 dtypes, and exact table shapes
before tokenizer construction or any pinned allocation. Previously a wrong
head dimension survived until forward, and a missing second table was detected
only after loading about 22.89 GiB per rank for the first table. Regression
tests cover missing second-table entries and wrong row/head dimensions in
either table, requiring zero pin requests and no tokenizer construction.

## NUMA placement before pinning

There is an unresolved placement risk: `NPUWorker.load_model` creates the
Engram runtime and allocates/copies about 45.78 GiB of pinned tables per rank,
but `worker.py::compile_or_warm_up_model` calls `bind_cpus` only after warmup
and graph capture. The comment at that call explicitly assumes hot allocations
can subsequently be migrated. `CpuAlloc.bind_memory` invokes `migratepages`
after `taskset`, discards its return code, and discards stderr through
`execute_command`. A `[migrate]` log is therefore only an attempted action.
It neither proves that table pages moved nor establishes a memory policy for
future allocations.

DMA-pinned or driver-owned pages cannot be assumed migratable through ordinary
Linux page migration. Pin/reference counts and special mappings can prevent
migration; mlocked pageable memory and driver-pinned DMA memory are different
cases. The installed torch-npu header identifies host storage allocated by
`aclrtMallocHost` or host registration. The upstream
[torch-npu v2.10.0 allocator](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/core/npu/CachingHostAllocator.cpp)
uses `aclrtMallocHostWithCfg`/`aclrtMallocHost` and optional host registration;
its host-allocation config carries a VA flag, not a selected NUMA node. Its
expandable host allocator also uses a physical-memory allocation path.
The small-buffer probes below now establish the installed 2.10.0.post4
allocator's mapping behavior. They operate only on their own process buffers;
no other process or full table was allocated or migrated.

Read-only topology checks on this machine found eight NUMA nodes, permitted
CPUs 0–191 and memory nodes 0–7, default memory policy, and automatic NUMA
balancing disabled (`/proc/sys/kernel/numa_balancing = 0`).

| Physical NPUs | `npu-smi info -t topo` CPU affinity | Nearest host NUMA node |
| --- | --- | --- |
| 0, 1 | 144–167 | 6 |
| 2, 3 | 96–119 | 4 |
| 4, 5 | 0–23 | 0 |
| 6, 7 | 48–71 | 2 |

These are hardware affinity nodes, not necessarily the late binder's target.
The 910B topology binder extends each affinity pool to the next NUMA node,
then partitions it across devices; `bind_memory` selects the node of the
resulting pool's first CPU. For an unrestricted TP8 deployment this can place
one member of each NPU pair's CPU workers on the adjacent node. Engram table
lookup is CPU work, so table locality should follow the planned lookup threads;
only gathered staging rows are transferred to the NPU. Table and DMA-staging
locality should be measured separately rather than assuming both require the
same node. Each pair owns about 91.56 GiB of final tables, before allocator
rounding, staging, and other model data.

## Small-buffer placement and migration evidence

On Linux `5.10.0-216.0.0.115.oe2203sp4.aarch64`, torch 2.10.0+cpu,
torch-npu 2.10.0.post4, and this 910B3/CANN stack, both 4 MiB default-allocator
probes gave the same result in opposite directions:

- A pageable control started entirely on its requested node (0 or 2) and moved
  all 1024 queried pages to the other node. Some individual migration statuses
  were transient `EBUSY`; the subsequent location query is the final result.
- `torch.empty(pin_memory=True)` returned a pinned tensor in the special
  `/dev/davinci_manager` mapping. Query and targeted migration returned `EFAULT`
  for all 1024 pages; `numa_maps` carried the requested VMA policy but no resident
  `N0`/`N2` counts. A displayed `bind:0` therefore does not establish actual
  physical placement, and ordinary late page migration cannot repair this
  mapping through the tested API. The whole-process `migratepages` command was
  not run; only the diagnostic's own buffer pages were targeted.

Raw results: [node 0 to 2](engram_numa_node0_to2.json) and
[node 2 to 0](engram_numa_node2_to0.json).

A second probe started with both CPU affinity and default memory policy on
node 2, initialized NPU 7, then bound only a fresh anonymous 4 MiB mapping to
node 0 with `mbind` before first touch. All 1024 pages landed on node 0 despite
the process defaults. The registration results were:

| Registration | Return | torch sees pinned | Location after registration | Attempted late move to node 2 |
| --- | --- | --- | --- | --- |
| `aclrtHostRegisterV2`, `MAPPED=0x2` | 0 | yes | all 1024 pages on node 0 | blocked; all remain on node 0 |
| `aclrtHostRegister`, legacy `MAPPED=0` | 0 | yes | all 1024 pages on node 0 | blocked; all remain on node 0 |
| `aclrtHostRegisterV2`, `PINNED=0x10000000` | 0 | yes | all 1024 pages on node 0 | all move to node 2 |

`MAPPED` migration produced `EBUSY` and a positive count of pages not processed;
unfilled status-array entries are retained as the diagnostic sentinel `-999`.
The subsequent query proves the final node distribution. In every case,
`aclrtHostUnregister` returned 0, torch then reported unpinned storage, all pages
could move to node 2, and contents remained unchanged. `PINNED` and `MAPPED`
flags therefore have materially different behavior on this machine.

Explicit `ACL_MEM_LOCATION_TYPE_HOST_NUMA` physical allocations also accepted
a 2 MiB handle for nodes 0 and 2 and freed successfully. However, allocation
property readback returned `207000` (`ACL_ERROR_RT_FEATURE_NOT_SUPPORT`).
This is weaker placement evidence than the ordinary-mmap registration path,
which exposes independently queryable resident pages. Full raw data is in
[engram_host_registration_910b.json](engram_host_registration_910b.json).

Reproduce from the repository root; the scripts use small private buffers and
may initialize the specified NPU context, so coordinate that device first:

```bash
OMP_NUM_THREADS=1 numactl --cpunodebind=0 --membind=0 ../.venv/bin/python \
  benchmarks/deepseek_v41/audit_engram_numa.py --device 7 --target-node 2 \
  --output /tmp/engram-numa-node0-to2.json
OMP_NUM_THREADS=1 numactl --cpunodebind=2 --membind=2 ../.venv/bin/python \
  benchmarks/deepseek_v41/audit_engram_numa.py --device 7 --target-node 0 \
  --output /tmp/engram-numa-node2-to0.json
OMP_NUM_THREADS=1 numactl --cpunodebind=2 --membind=2 ../.venv/bin/python \
  benchmarks/deepseek_v41/audit_engram_host_registration.py --device 7 \
  --allocation-node 0 --target-node 2 --output /tmp/engram-registration.json
```

## Opt-in owner and graph validation

`vllm_ascend/ops/engram_pinned_host.py::EngramPinnedHostTensor` implements the
measured path: private mmap, per-range `mbind`, first-touch population, then
`aclrtHostRegisterV2(MAPPED)`. It takes an explicit NUMA node and initialized
NPU device. It changes neither global affinity nor the default torch allocator,
and has not replaced the factory's default allocation path. A streaming loader
can fill the final storage through its `initialize` callback before registration,
avoiding a redundant zero pass over a large table; otherwise storage is zeroed.

`EngramTableShard.from_safetensors` now accepts optional `numa_node` and
`device` together. That explicit path streams only the selected head chunks
directly into the owner's final storage through `initialize`, then registers
it; no full table or concatenated intermediate is materialized. Omitting both
parameters retains the existing torch allocation path. A contradictory
`pin_memory=False` request is rejected. The shard holds the owner and exposes
idempotent `close`. Manager close first synchronizes its copy stream and last
consumption event, then closes its shards; registration and table references
are released only after those fences. A closed shard rejects further lookup.

The owner holds the mapping through the tensor's `frombuffer` storage. Tensor
aliases remain valid CPU memory after explicit close, although they are then
unpinned. Each DMA completion event must be recorded with the owner; close waits
those events before unregistering. The torch-npu source only automatically
records caching-allocator events for pointers it allocated itself, so externally
registered memory requires this explicit lifetime management. Initialization,
registration, page queries, and unregistration stay outside graph capture.
Graphs read the stable device staging destination; no host lookup is captured.

Validation passed:

- **18 CPU unit tests**: ordering, page alignment, first-touch callback,
  allocation/registration/pin-detection failures, DMA-event cleanup and retries,
  surviving tensor aliases, last-storage release, and capture rejection.
- **Two NPU 0 functionality tests**: verified node-0 residency, actual
  `non_blocking=True` H2D from registered `frombuffer` storage, close fencing DMA
  without a prior host wait, and two host slots feeding 20 changed-input graph
  replays through a fixed device buffer. Every output matched exactly. The
  tests used 256 KiB per host slot and ran alongside another functionality test;
  no timing or throughput acceptance is claimed.
- **55 combined CPU tests** passed after optional loader integration: the 18
  owner tests, eight new NUMA loader/cleanup tests, eight existing offload tests,
  and 21 host factory tests. Real tiny safetensors verify the exact selected
  bytes already occupy final storage when registration begins. Partial-copy,
  post-registration validation, and cleanup-fence failures are covered.
- **Five offload NPU tests** passed: ordinary and registered-table loading each
  feed the actual manager's two staging slots in eager and graph modes through
  64 changed-token steps, alternating graph buckets 4/8 and zero padding. The
  lifecycle test verifies close ordering and invalid reuse. Together with the
  owner tests this is seven small NPU functionality tests, with no performance
  measurement or full-table allocation.

```bash
OMP_NUM_THREADS=1 ../.venv/bin/python -m pytest \
  tests/ut/ops/test_engram_pinned_host.py \
  tests/ut/ops/test_engram_numa_loader.py tests/ut/ops/test_engram_offload.py \
  tests/ut/models/test_deepseek_v41_engram_factory.py -q
OMP_NUM_THREADS=1 ../.venv/bin/python -m pytest \
  --confcutdir=tests/e2e/single_node/ops \
  tests/e2e/single_node/ops/test_engram_pinned_host.py \
  tests/e2e/single_node/ops/test_engram_offload.py -q
```

Before default integration, choose table and staging nodes from the actual
worker CPU plan and NPU topology, respecting cgroup memory-node allowances.
The existing topology parsers are reusable, but `DeviceInfo` itself requires
already-running NPU processes and should not be used unchanged for an early
pure planner. The factory/runner must invoke existing manager close during shutdown and
close previously loaded shards if a later initialization stage fails. Validate larger registration limits, node-local capacity,
registration/page-table overhead, gather/H2D performance, and full TP8 peak
memory before enabling full tables. Small-buffer success does not establish
45.78 GiB per-rank registration capacity or a full-model performance gain.
