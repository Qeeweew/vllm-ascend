# V4.1 DSpark AscendC metadata

Status: isolated 910B operator acceptance passed (127 NPU tests, including 576
changed-input graph replays across K=1..8). Full TP8 DSpark proposer/serving acceptance remains
separate. See [RESULTS.md](RESULTS.md) for measurements and artifact provenance.

Interface: `v41_dspark_metadata(cu_q, lengths, topk_lengths, schedule) -> None`.
All tensors are contiguous INT32 on one NPU. Shapes are `[B+1]`, `[B]`,
`[T,1]`, and caller-owned `[1024]`; schedule must not alias any input.
Initial limits: B <= 4096, T <= 32768, 910B with 20 AIC. B/T zero remain valid
and clear the full output. No host reads of device values or AICPU launch.

The required SMLA invocation has Hq=8, Hkv=1, D=512, CR=0, TND query,
PA_BBND BF16 cache, explicit original sparse indices, topk capacity 256,
and original mask mode 0. Visibility contains 128 prefix keys plus K draft
query keys, subject to the fixed 256-key capacity. The tested small draft
lengths are K=1..8; this is not an arbitrary-K model support claim.
K=5 has a maximum candidate span of **133**, not 132; 133 was the initial
acceptance workload, never a kernel limit. The native API's `ori_win_left`
is a distance, not the candidate count. This schedule supports all candidate
spans 0..256 and query counts up to the total T limit, with no K-specific host
or device branch. Visibility remains defined by the indices/length generator
and consumer.

The arch22 SWA consumer uses mBaseSize=gSize=8: one query is one M tile.
Its s2BaseSize=512 covers every candidate in one tile. Therefore no S2 split
or FD reduction is needed. This kernel partitions the range `[0,cu_q[-1])`
nearly equally over min(20, total queries) AIC cores. It does not compact,
sort, or otherwise change query visibility, and does not add dependencies
on runtime KV lengths when allocating work.

The first 324 words contain 36 FA records of nine words each. FD records
begin at word 324 (72 records of eight words). All unused entries, all FD
records, and the remaining tail up to word 1024 are zero. Enabled FA rows
contain start/end `(batch, local query, S2=0)` and no FD workspace index.
The first core's start is explicitly zero because the consumer ignores it.
An end exactly at a batch boundary is normalized to `(batch+1,0,0)` using
the batch owning the final included query. This avoids assigning trailing
empty requests to the last core. Later starts skip empty request slots.

The whole schedule is overwritten on each launch. Zero candidate rows remain
inside a core's contiguous range and are handled by the existing sparse SMLA
path plus the wrapper's required zero-output mask. Zero total queries disable
all cores. Invalid negative/decreasing query offsets or total > T disable all
cores without accessing out-of-range input offsets; callers remain responsible
for producing valid DSpark metadata.

Validation must compare final SMLA outputs with native metadata and an
independent attention oracle, including B=1/2/4/8/16/32, ragged/empty requests,
padding, rejection transitions, spans 0/133/256, and changed-content NPU graph
replays. Schedules need not be byte-identical to CANN's cost-based planner.
