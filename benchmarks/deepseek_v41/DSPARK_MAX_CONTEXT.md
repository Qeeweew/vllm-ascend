# V4.1 DSpark maximum-context component semantics

The final CPU suite passed 106 tests and the NPU0 suite passed 12 tests on
2026-09-16. Both processes exited 0. Production DSpark admission remains
disabled (the guard stays in place): real `_propose`, scheduler acceptance/rejection and full serving
admission are still unverified by this component test.

## Request-end behavior

The real proposer retains five query slots. After subtracting rejected tokens,
it sets the virtual sequence length to `prefix + 5`, which can exceed the
configured maximum. Those normal request tails must retain their valid query
rows instead of failing the entire request or moving its prefix backwards.

The dedicated draft metadata path now applies these rules:

- Derive the prefix from the original virtual sequence length before clamping.
- A query is valid only when its logical position is below the model maximum.
  Its visible keys are `[max(prefix - 128, 0), min(prefix + 5, maximum))`.
- Invalid queries and graph bucket padding receive `-1` indices/slots and zero
  attention spans. The physical padding at the end of a non-full final page
  never extends the logical model limit.
- Clamp native sequence lengths to the model maximum. Requests with zero
  query rows receive native length zero, preserving the next active request's
  boundary when query offsets repeat. The original common lengths remain intact.
- Only the draft model's RoPE lookup receives position zero for invalid query
  rows. The logical positions used for cache metadata remain unchanged.
- Explicitly zero native draft attention output and optional LSE for zero-span
  rows. Native attention can leave those rows uninitialized; allowing allocator
  contents or NaNs into HC/MoE is incorrect padding behavior.

The ordinary target metadata/attention branch retains its previous behavior.
No initial-input Triton kernel or other model implementation was changed.
All changing lengths, masks and positions stay on device, with fixed metadata
buffer addresses across replay. The low-level `build_metadata` API retains its
caller-owned sequence-length alias; clamping and empty-request normalization
belong to the dedicated draft cache builder. This work makes no latency or
throughput claim.

## Evidence and failures retained

| Attempt | Result | Finding |
| --- | --- | --- |
| CPU before correction | 12 failed | Maxima 128/129 and rejected counts 0–5 exposed page checks and invalid tail visibility/slots |
| NPU initial correction | 8 passed, 2 failed | Zero-span native rows contained allocator data and NaNs |
| NPU output-mask correction | 8 passed, 2 failed | An empty middle request with nonzero native sequence length corrupted the following boundary query |
| NPU empty-request correction | 2 boundary tests passed | All rejection counts and empty-middle graph replays passed |
| Final CPU suite | 106 passed | Helper, actual metadata builder, RoPE padding, target contracts and existing draft loader/model tests |
| Final NPU0 suite | 12 passed | Existing noncausal attention plus all new maximum-context graph cases |

The final device tests cover both an aligned maximum of 128 and a non-aligned
maximum of 129, all rejected counts from zero through five, mixed request
lengths, empty middle requests, reversed/changed physical page tables, gapped
cache storage, exact cache writes, bucket padding and batches whose entire
five-query block is beyond the maximum. The graph reuses stable metadata
addresses while lengths, page mappings, keys and query contents change.

An independent CPU enumeration gathers keys through the logical page table
and computes dense attention with a denominator-only sink. The original gates
remain unchanged: RMS error at most `0.006 * reference RMS + 1e-6`, elementwise
`rtol=0.025, atol=0.012`; LSE uses `rtol=0.003, atol=0.015`. Cache contents and
invalid output/LSE rows are checked exactly. The empty-middle diagnosis showed
the same failing row in eager and graph execution while cache contents matched,
isolating metadata semantics rather than cache writes or graph pointer reuse.

The [evidence manifest](dspark_max_context/manifest.json) records all retained
failure/final XML files, raw log locations and SHA256 values, and the seven
source/test fingerprints. Raw diagnostic logs remain under `/tmp`; small XML
artifacts are tracked in `dspark_max_context/`.
Failure XML retains the original traceback whitespace and checksums; source
and documentation whitespace checks exclude those two raw failure artifacts.
The manifest records the test-time fingerprint separately where the pinned
Ruff hook subsequently added a blank line between imports.

## Reproduction

Run from the repository root. The NPU command selects device 0 through the
test fixture and requires the coordinated device-0 execution window.

```bash
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest \
  tests/ut/ops/test_dsa_v41.py \
  tests/ut/attention/test_dsa_v41_metadata.py \
  tests/ut/models/test_deepseek_v41_dspark.py -q

OMP_NUM_THREADS=8 ../.venv/bin/python -m pytest \
  --confcutdir=tests/e2e/single_node/ops \
  tests/e2e/single_node/ops/test_dspark_v41_attention.py -q
```
