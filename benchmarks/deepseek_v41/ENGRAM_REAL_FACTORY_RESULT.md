# Real full-table Engram factory acceptance: PASS

Executed 2026-09-16, production factory start 00:25:24 UTC through final owner
release 00:35:43 UTC. Controller exited **0** after its final `finished/passed`
event. All eight devices were returned with no running NPU processes and
3,457–3,474 MiB base HBM usage before the next full-model run started.

## Accepted real payload

- **393,227,699,200 bytes / 366.221833 GiB** of actual converted BF16 Engram data,
  16 owners simultaneously resident. Page-rounded registered mappings totaled
  393,227,730,944 bytes (31,744 bytes of padding).
- Both complete checkpoint tables: layers 1 and 14, respectively 384,006,168
  and 384,016,682 rows, width 256. No reduced table or synthetic values.
- **144 exact row oracles**: first/middle/last of each rank's three owned heads
  in each table, checking source FP8 with group32 scale → BF16 → converted file
  → actual loaded local row.
- **160 changed-input graph replays**, 20 per rank: exact BF16 results, stable
  staging pointers, DEAD and padded rows included.
- All 16 owners used their requested NUMA nodes. Each mapping sampled 25 pages
  through `move_pages` queries, 400 page samples total, all on the expected
  node. Recorded `/proc` mapping information is included in raw evidence.
- Each rank copied 1,470 contiguous slices, each at most **32 MiB**. All reads
  exactly partitioned its owned ranges; no full-table `get_tensor` occurred.
- **16 checked unregisters**, eight successful release acknowledgements,
  eight worker exits **0**, no cleanup errors or forced terminations.

TP rank/world-size and NUMA context were explicitly supplied to eight isolated
factory processes. The actual production factory, real tokenizer, source
config, converted table loader, host registration and offload manager were
used. This run did **not** instantiate the 40-layer target or HCCL; complete
model generation and token-history scheduling are separate acceptance gates.
The graph captured consumption of stable staging buffers, with real host
gather/DMA and stream handoff performed before each replay.

## Memory and initialization measurements

| Rank | NUMA node | Factory load s | Two registrations s | Peak RSS GiB | Resident PSS GiB | Released PSS GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 6 | 31.29 | 15.13 | 70.567 | 47.489 | 1.715 |
| 1 | 7 | 42.27 | 18.36 | 70.951 | 47.692 | 1.919 |
| 2 | 4 | 37.41 | 18.52 | 70.946 | 47.612 | 1.835 |
| 3 | 5 | 43.99 | 19.19 | 70.581 | 47.220 | 1.445 |
| 4 | 0 | 42.82 | 19.31 | 70.722 | 47.335 | 1.561 |
| 5 | 1 | 50.37 | 19.51 | 70.753 | 47.348 | 1.573 |
| 6 | 2 | 48.22 | 19.14 | 70.952 | 47.514 | 1.740 |
| 7 | 3 | 44.51 | 19.45 | 70.649 | 47.201 | 1.423 |

Factory scope includes tokenizer/hash construction, actual table copies,
registration and staging setup. It excludes interpreter/model import startup
and post-load oracle/replay work. Ranks initialized sequentially while earlier
owners remained live. Total factory scope across ranks was **340.89 s**;
individual loads took **31.29–50.37 s**. These are initialization observations,
not inference throughput or steady-state offload benchmarks.

The allocator held **42,496 bytes allocated / 6 MiB reserved per rank**, below
the 1 GiB ceiling. The row staging payload itself is 12,288 bytes / rank;
framework/device context overhead is outside Torch allocator reporting and
was allowed by the 4 GiB admission reserve.

Observed per-process peak RSS was **70.57–70.95 GiB**, versus resident PSS
**47.20–47.69 GiB**, demonstrating the additional file-mapping/loading cost
beyond the roughly 45.78 GiB pinned payload per rank. Individual peak RSS values
were not simultaneous; their sum is not a measured aggregate peak. Sum of
per-rank ready PSS samples is 379.41 GiB, also sampled at different times.
After explicit owner release, PSS fell to **1.42–1.92 GiB** before each worker
exited. Physical file reads varied by page-cache state; these times do not
constitute a controlled cold-cache load benchmark.

## Checkpoint publication and evidence

At 00:23 UTC the existing converter/watcher published all 48 shards,
`complete=true`, no missing source shards, and final config/index. No second
converter or repeated full payload checksum job was started. CPU admission
verified all 48 header maps and exact quantization/packing metadata; both
factory and full-model capacity admission had no blockers after the user
released the former device-7 workload.

Manifest SHA256:
`4d48bba36c91edca7cae890cd6186a7728f67603c8efc4ae358d1447ddab2bed`.
`full_conversion_publication_r1.json` records metadata identities and the
converter's final tail-shard checksums. The source-row samples are bounded
oracles, not a replacement for all-row payload integrity evidence.

Preserved files:

- `full_conversion_publication_r1.json`: complete publication identities.
- `real_engram_factory_prepared_complete_r1.json`: CPU admission, 48 headers.
- `real_engram_factory_run_r1.json`: full actual slice, row, owner and cleanup
  evidence. SHA256 `9698df783e50ec5be32932e3a9ad142a1cf4099524a718f0c271b6c82d3c6928`.
- `real_engram_factory_run_r1.log.txt`: complete captured execution log.
- `real_engram_factory_summary_r1.json`: compact derived measurements.
- `real_engram_factory_post_npu_smi_r1.txt`: empty-device handoff evidence.

## Result status correction after the run

The original controller set top-level `status=passed` after all owners became
resident, before cleanup finished. The preserved r1 evidence is unchanged;
its acceptance here requires the final `finished/passed` event, all eight
successful releases with 16 unregisters, eight zero exits and controller zero
exit. Intermediate snapshots alone were not treated as final success.

The driver now publishes `resident_checks_passed_cleanup_pending` during
cleanup. It can publish final `passed` only after all eight distinct ranks
acknowledge both unregisters and all eight distinct workers exit zero. The
CPU regression covers incomplete cleanup, a failed exit, a missing unregister
and preservation of prior failure. **16 CPU tests passed**, with 14 existing
Torch JIT warnings; Ruff lint/format pass. No 366 GiB rerun was needed for this
reporting correction. Full-model eager/graph validation proceeds separately.
