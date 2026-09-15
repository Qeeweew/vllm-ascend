# V4.1 CR2 prefix-cache ring audit

## Finding

The observed 32-token prefix hit does **not** require restoring the CR2
compressor ring. A fresh ring may contain arbitrary data: resuming at an
even raw-token position never reads that data before replacing the required
row. The official V4.1 implementation uses the same rule. This audit does
not identify the cause of the reported eager/graph logprob difference of
approximately 0.27, and does not justify disabling prefix caching.

Audit date: 2026-09-15. Upstream vLLM checkout:
`836bb3839ffefcda8283ea7d41671a89e1a613df`. Ascend working tree is based on
`b49962987e89b850586f1819ce8f85daa85a0f81`, including uncommitted V4.1 work.
Only CPU tests and this report were added; no product implementation changed.

## Official and Ascend behavior

- `CircularBufferSpec.prefix_cacheable` is false. Its manager allocates one
  private block per request and neither caches nor restores that block.
  Hybrid coordinators omit this scratch group from prefix-hit reconciliation.
- Official `models/deepseek_v41/compressor.py` and Ascend
  `models/deepseek_v4/compressor.py` map each token to
  `block_table[request, 0] * capacity + position % capacity`.
- Official `common/ops/fused_compress_quant_cache.py` reads the predecessor
  from the ring only when the chunk's first position is odd. Other completed
  pairs use two projected rows from the current chunk. The request program
  then saves the final rows to its ring.
- Ascend `csrc/attention/compressor_v41/op_kernel/compressor_v41.cpp`,
  `Boundary`, has the same explicit `(position & 1) != 0` guard before the
  only predecessor `DataCopy` from ring state.
- Current smoke configuration uses raw-token cache blocks of 32. Neither
  coordinator enables partial hash hits for the V4.1 cache group set, which
  contains no Mamba group. The hit alignment is 32. Ring capacity 8 is
  excluded from scheduling/hash alignment by the Ascend resolver.
- A CR2 paged cache spec independently requires its raw-token block size to
  be divisible by 2. The current planner limits pages to 32 KiB; this audit
  does not claim it universally forces every cache spec to block size 32.
- Ordinary scheduler preemption frees blocks and resets
  `request.num_computed_tokens = 0`; rescheduling recomputes or uses the
  reconciled prefix hit. An odd preemption position is not itself an odd
  fresh-ring resume position.

For example, a hit of 32 resumes at position 32. Position 32 stores its raw
projection without producing a compressed row. Position 33 pairs with
position 32, using either the same chunk's input or the newly stored ring
row. A 33-token prompt therefore safely recomputes only its final token at
position 32, and its first decode token at position 33 has valid history.

The runner token totals `[32, 9, 2, 2, 1]` are batch totals, not a proof of
individual request positions or cache ownership. A CPU same-request replay
with those chunk sizes also passes, including its odd-start continuations.

## Reproducible CPU evidence

Test: `tests/ut/models/test_compressor_v41_prefix_audit.py`.

```bash
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest \
  tests/ut/models/test_compressor_v41_prefix_audit.py -q
```

Result: **15 passed**, 0.34 seconds pytest test time. Process startup/import
time is excluded. Full output: `/tmp/compressor-v41-prefix-audit.log`.

Ten cases instantiate actual upstream and Ascend hybrid coordinators with
all six relevant cache roles: main CR1/CR2, index CR1/CR2, SWA, circular state.
They allocate/cache a writer's request, look up a matching reader, and
allocate its private ring through the real managers, without mocks.

| Prompt tokens | Prefix hit | First resumed position | Recomputed suffix |
| --- | --- | --- | --- |
| 32 | 0 | 0 | 32 |
| 33 | 32 | 32 | 1 |
| 41 | 32 | 32 | 9 |
| 65 | 64 | 64 | 1 |
| 97 | 96 | 96 | 1 |

In every case, ring hit blocks are empty and the reader's private ring is
different from the writer's live ring. Filling every fresh ring row with
NaN and running the CPU compressor oracle from the returned hit produces
finite suffix/decode outputs exactly equal to full-prefill outputs, with
`rtol=0, atol=0`. Subsequent odd-position decode starts are covered.

Four negative controls start a fresh request at odd positions 1, 31, 33,
and 65 without predecessor history. All read NaN into the first latent.
Zero-initializing the ring instead produces finite but incorrect latents;
recomputing the preceding even-position token restores exact output. These
are kernel-contract counterexamples, **not reachable prefix-hit examples
under the tested scheduler configuration**.

One final test compares full prefill against persistent-ring chunks
`[32, 9, 2, 2, 1]` and obtains exact equality.

## Limits and next diagnostic

No NPU kernels, complete scheduler loop, graph replay, or model logits were
executed here. The real coordinator evidence and CPU numerical evidence
rule out a missing ring restore at the tested even prefix boundaries; they
do not verify the runner's actual block tables, positions, stream ordering,
or cache contents during the failing smoke.

Record request IDs, per-request positions, prefix-hit lengths and CR2 ring
block IDs at the first divergent model step. In particular, check whether
an odd first position belongs to an existing request with its previous
even projection in the same ring block. Compare compressed cache contents
and first differing layer outputs before attributing a final logprob
difference to prefix caching.

If future partial-hit or external-cache paths admit odd fresh-ring starts,
they must resume at a compression-group boundary, replay the predecessor
projection, or restore equivalent valid state. Zero-fill alone is not a
correct fix. Those unsupported paths were not exercised by this audit.
