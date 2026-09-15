# Engram full host-registration capacity probe

## Scope and admission

This is a synthetic capacity and correctness probe, independent of checkpoint
download/conversion. It does not establish full-model throughput or validate
the unfinished Engram weight shards. The controller computes exact production
TP8 head-shard shapes with `HostEngramLayout.head_shard` from released config.

| Rank / NPU | NUMA node | Layer 1 bytes | Layer 14 bytes | Rank bytes |
| --- | --- | ---: | ---: | ---: |
| 0 | 6 | 24576111104 | 24576749056 | 49152860160 |
| 1 | 6 | 24576177664 | 24576814592 | 49152992256 |
| 2 | 4 | 24576258560 | 24576974336 | 49153232896 |
| 3 | 4 | 24576343552 | 24577054208 | 49153397760 |
| 4 | 0 | 24576446976 | 24577125888 | 49153572864 |
| 5 | 0 | 24576532992 | 24577202688 | 49153735680 |
| 6 | 2 | 24576611840 | 24577266176 | 49153878016 |
| 7 | 2 | 24576675328 | 24577354240 | 49154029568 |

Payload total: **393227699200 bytes / 366.221833 GiB**. Registering whole
pages adds only alignment padding. Each rank keeps two independent original
shape owners, approximately 22.89 GiB per owner and 45.78 GiB per rank.
Each NUMA node receives two ranks, approximately 91.56 GiB.

At 20:45 UTC the host reported approximately 1.9 TiB MemAvailable. The
container memory cgroup limit is effectively unlimited. Target nodes had
only 5.5–13.5 GiB MemFree but approximately 226–232 GiB FilePages each.
First touch therefore reclaims file cache even though overall capacity is
ample. Do not overlap with model loading or performance measurement. The
background suffix downloader may continue; it uses little bandwidth.

Before each rank, require global MemAvailable >= all remaining payload plus
256 GiB reserve, and local estimated available memory
`MemFree + FilePages - Shmem + SReclaimable` >= remaining payload for that
node plus 32 GiB. Before each layer, repeat the reserve checks. This local
sum is a conservative-operational admission estimate, not a kernel promise
that every file page is reclaimable; allocation can still fail cleanly.

## Preliminary driver evidence

`RLIMIT_MEMLOCK` soft/hard are both 64 MiB and CAP_IPC_LOCK is present.
Neither limits nor capabilities are changed by this work. A 256 MiB
anonymous mmap → node-6 mbind → first touch → MAPPED registration pilot on
NPU0 succeeded at 20:48 UTC:

- First touch: about 31 ms.
- End of touch through registration/pin confirmation: about 83 ms.
- First/middle/last-row asynchronous H2D: exact BF16 equality.
- Explicit event fence and unregister succeeded; process RSS fell from
  2465144 KiB to 2213624 KiB (baseline 2174396 KiB).
- VmLck and VmPin remained zero throughout; those counters do not establish
  whether this driver registration pinned physical pages.

Pilot log: `/tmp/v41-engram-capacity-pilot.log`. No full-table capacity claim
follows from this small pilot. Per-registration and aggregate driver limits,
large-map pin cost, and release cost require the full probe.

## Staged protocol

Script: `benchmarks/deepseek_v41/probe_engram_full_capacity.py`.

1. Spawn one isolated process for rank 0. Set its NPU and four CPU torch
   threads. Apply per-VMA NUMA binding before first touch; do not alter
   process-global NUMA policy or system settings.
2. Allocate and zero the first full BF16 shape, write distinct values to
   each local head's first/middle/last rows, then MAPPED-register it. Time
   first touch and actual `aclrtHostRegisterV2` separately. Repeat for its
   second layer, keeping the first registered.
3. Check sampled resident pages with read-only `move_pages`; record the
   entire mapping's `/proc/self/numa_maps` row. DMA all sampled rows directly
   from their registered addresses and verify exact expected values.
4. Construct the real `EngramOffloadManager` over these two tables. Run
   20 changing first/middle/last hash selections, alternating DEAD rows and
   padded buckets through CPU gather → pinned staging → H2D → graph replay.
   Host gather remains outside graph capture/replay. Check exact outputs
   and padded values, then hold both owners resident.
5. Only after this process reports ready, start the next rank. Repeat until
   all eight ranks and all 16 owners remain resident simultaneously. Record
   `all_ranks_resident`; testing ranks serially while freeing each would not
   establish aggregate capacity.
6. Release all processes in reverse rank order, timing every unregister and
   recording post-close RSS, cleanup errors and child exit codes.

NPU tensor reservations must stay below 1 GiB per rank; actual tensors are
only a few small staging/output buffers. Driver/context overhead is recorded
separately through device free-memory observations where available and is
not the same as torch tensor reservations. NPU7's existing external process
is left untouched; the root task authorized only this additional small
context. Another agent's small NPU1 functionality test is separate from this
capacity result. No HCCL or model weights are needed.

## Failure and cleanup

Any error stops new allocations. The controller requests release from only
its own children. Each worker synchronizes DMA/device work, closes the
offload manager, removes table aliases, closes owners in reverse order,
unregisters and drops storage, then runs garbage collection and reports RSS.
Cleanup errors are retained in the result; they cannot become a passing
capacity result.

Each rank has a 600-second stage deadline. Cleanup also gets a bounded grace
period. If explicit release cannot complete, terminate then, only as a last
resort, kill that controller-owned child so its isolated driver context and
address space can be reclaimed. Do not signal any other process. Record this
as failure, never as successful explicit cleanup. The probe never calls
`drop_caches`, changes memlock limits, modifies checkpoint files, or changes
the user's existing NPU7 workload.

Reserve an initial **10–30 minute diagnostic window**, with stage watchdogs
enforcing individual bounds. This is a planning allowance, not a measured
full-scale prediction; large driver registration and file-cache reclaim may
dominate. Replace it with actual per-stage timings in the result report.
