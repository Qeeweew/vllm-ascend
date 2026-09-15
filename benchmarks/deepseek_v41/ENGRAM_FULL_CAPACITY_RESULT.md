# Full Engram host capacity on 8 × 910B

## Result

**PASS: 366.221833 GiB of production-shaped BF16 host tables were registered
and resident simultaneously across eight ranks and eight NPU contexts.**
All correctness checks and explicit cleanup passed. The controller exited
zero and every one of its eight child processes exited zero. The existing
external process on NPU7 was not modified or signaled.

This closes the aggregate host-registration capacity question for the tested
explicit NUMA/MAPPED allocator. It does **not** establish full-weight loading,
model accuracy, token throughput, or the unfinished checkpoint download.

Run: 2026-09-15, approximately 20:54–21:03 UTC. Raw result:
`benchmarks/deepseek_v41/engram_full_capacity_910b.json`. Process log:
`/tmp/v41-engram-full-capacity.log`. The final JSON event is
`finished`, `complete=true`.

## What was exercised

The probe used released config and the real `HostEngramLayout.head_shard`
to derive each rank's three local heads for both Engram layers. Sixteen
independent anonymous BF16 mappings followed the actual owner lifecycle:
per-VMA mbind → first touch → `aclrtHostRegisterV2(MAPPED=0x2)` → explicit
event fencing/unregister. No checkpoint tensors were loaded.

- Table payload: **393227699200 bytes**.
- Page-aligned registered/resident storage: **393227730944 bytes**.
- Approximately **45.78 GiB/rank**, two approximately 22.89 GiB tables each.
- NUMA placement: ranks 0/1→node 6, 2/3→node 4, 4/5→node 0, 6/7→node 2.
- 400 sampled pages all returned the intended node through `move_pages`.
- All 16 complete VMA `numa_maps` records contained only the intended node;
  each resident-page count equaled that table's page-rounded allocation.
- 144 head-first/head-middle/head-last rows were copied directly from their
  registered host addresses to NPU and matched their exact BF16 markers.
- The actual `EngramOffloadManager` gathered changing head hashes, staged
  pinned transfers and fed fixed device buffers to **160 graph replays**
  across eight ranks. Both layer outputs, alternating DEAD rows and graph
  padding matched exactly (`rtol=0, atol=0`).

CPU gather stayed outside graph execution. This is graph-compatible staged
offload, not a graph kernel directly reading the full host table. All ranks
remained alive and retained both owners until the controller recorded
`all_ranks_resident`, then cleanup proceeded in reverse rank order.

Torch device reservations were **6 MiB/rank**, with 32 KiB live tensor
allocations at each ready event. These counters describe torch allocations;
they are not a measurement of all driver/context overhead. The test used
basic CANN copy/add operations and did not validate newly compiled custom
OPP artifacts.

## Measured costs

The values below are observed wall times for this diagnostic run, not an
isolated serving-performance benchmark. Background compilation and a separate
small NPU1 functionality task could coexist; no complete model was loaded.

| Operation | Count | Minimum | Median | Maximum | Sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-table first touch | 16 | 3.958 s | 6.234 s | 7.597 s | 98.076 s |
| MAPPED registration call | 16 | 8.572 s | 9.688 s | 10.400 s | 153.950 s |
| Host unregister call | 16 | 0.673 s | 0.943 s | 1.201 s | 14.085 s |

The interval from rank 0's initialized baseline through rank 0's final
release was **524.90 seconds**. It includes sequential process startup for
later ranks, allocation, checks, simultaneous residency and reverse cleanup;
the initial controller/import time and final process shutdown are excluded.

At simultaneous residency, global MemAvailable remained
**1674112352256 bytes**. After all children exited it was
**2080820965376 bytes**. File cache was naturally reclaimed during first
touch; no `drop_caches` operation was used.

## Cleanup and limitations

All 16 explicit unregister calls succeeded, all eight cleanup error lists
were empty and all child exit codes were zero. RSS fell from roughly
48 GiB per held worker to **1.97–2.20 GiB** after close, then the isolated
processes exited. Remaining pre-exit RSS was Python/torch/driver context
overhead, not retained table aliases. No forced process termination was
needed, and no system memlock limit or NUMA policy was changed.

The probe intentionally excludes checkpoint file-cache/private-copy peaks,
loading/conversion time, full prompt hashing, multi-rank collectives, actual
Engram GEMM/gate work and model generation. Production initialization reads
real table values and may retain additional file cache. The table owner and
DMA/graph lifecycle were validated at aggregate production shape; the full
model acceptance gates remain separate.

The diagnostic script subsequently tightened two reporting guards: the JSON
top-level `complete` flag is now set only at final cleanup completion, and
the controller explicitly checks that every ready process is still alive
before recording aggregate residency. The successful raw run independently
records all eight normal release acknowledgments and zero exits; no capacity
or data path changed with those reporting guards.
