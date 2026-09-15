# V4.1 DSpark CR0 attention component

The fixed-K5 noncausal attention component and its dedicated metadata builder
pass 53 CPU regression tests and 8 real NPU tests on Ascend910B3. This is
component evidence; DSpark proposal/verification, acceptance, full-model
quality and serving performance remain separate integration gates. No Csrc,
model, proposer, runner or registry changes belong to this component patch.

## Visibility contract and native ABI

The official V4.1 attention uses the shared SWA backend. In
`sources/vllm/vllm/v1/attention/backends/mla/sparse_swa.py`,
`ComputeDSparkNoncausalSWAIndicesKernel` derives
`prefix_len = seq_len - query_len` and
`start_pos = max(prefix_len - window_size, 0)`. For V4.1 window 128 and K5,
every query in one draft request therefore sees the same last **128 prefix
tokens plus all five query tokens**, including future rows inside the block.
The maximum list length is 133, padded to an index width of 256. Target
causal attention retains left127/right0 and never inherits this draft mode.

The older Ascend V4 helper enumerates physical slots. That representation
cannot be passed directly to the existing arch22 sparse-attention kernel:
`GetOriSparseKeyGmOffset` in
`csrc/attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_swa_block_vector.h`
treats `ori_sparse_indices` as logical token IDs and applies the request's
block table itself. Its masking also compares those IDs with sequence length.
The new helper emits logical IDs, checks the corresponding physical page
against actual cache capacity on device, and masks invalid IDs to -1.

Missing or invalid pages leave holes. `ori_topk_length` spans all visible
columns rather than counting only valid pages, so an interior hole cannot
hide later valid columns. Padding rows have all -1 IDs and zero lengths.
Active requests must contain five query rows; empty request slots are allowed.
Offsets must be nondecreasing and active rows must fit the static capacity.
Other query lengths fail closed in the device visibility helper and are not
an admitted draft mode.

## Interfaces

`build_dspark_v41_swa_indices` in `ops/dsa_v41.py` takes the device block table,
query offsets and sequence lengths, static page size/cache block count, and
caller-owned INT32 outputs `[T,1,256]` and `[T,1]`. It uses device operations
without `.item()`, `.tolist()` or `.cpu()` and writes stable output addresses.
Intermediate allocations are ordinary graph-captured device operations.

`AscendDSAV41Metadata` adds optional `draft_swa_indices` and
`draft_swa_lengths`. Both are required together and only supported with CR0.
Explicit draft metadata selects native `ori_mask_mode=0`, `ori_topk=256`,
`ori_topk_length` and `ori_sparse_indices`; the existing causal call arguments
are preserved when the fields are absent. BF16 cache rows already contain
RoPE-transformed K=V; output inverse RoPE remains the caller's responsibility.
Sinks retain their single denominator-only contribution per head.

`AscendV41CacheMetadataBuilder.enable_dspark_device_metadata(max_query_tokens)`
opts a dedicated SWA builder into this mode before graph capture. The capacity
must fit its existing metadata allocation. Repeating the same preparation is
allowed; changing capacity or first preparing during capture is rejected.
The builder requires `common.causal=False` only after explicit enablement.
Target builders remain causal even if their common object says otherwise.

Each draft build obtains the actual SWA cache block count from its group's
first registered cache layer in `static_forward_context`. The cache must have
been bound. The builder refreshes visible indices, lengths and the native
schedule once per group, and also masks draft write slots outside physical
cache capacity. `make_v41_attention_metadata` passes the stable draft buffers
through and rejects combining them with compressed main-cache metadata.

## Validation

CPU command:

```bash
OMP_NUM_THREADS=8 ../.venv/bin/python -m pytest --confcutdir=tests/ut \
  tests/ut/ops/test_dsa_v41.py tests/ut/attention/test_dsa_v41_metadata.py -q
```

Result: **53 passed**. Coverage includes the unchanged causal native contract,
prefix lengths 0/1/127/128/129/1024, permuted pages, invalid physical pages,
holes, empty request slots, graph padding, stable buffer addresses, explicit
draft-only dispatch, cache capacity checks and compressed-path rejection.
The helper test forbids value-dependent host tensor reads while it executes.

NPU command, reserved device 0 only:

```bash
OMP_NUM_THREADS=8 ../.venv/bin/python -m pytest \
  --confcutdir=tests/e2e/single_node/ops \
  tests/e2e/single_node/ops/test_dspark_v41_attention.py -q -x \
  --junitxml=benchmarks/deepseek_v41/dspark_attention_component.xml
```

Result: **8 passed in 23.03 seconds**; raw result is
[dspark_attention_component.xml](dspark_attention_component.xml). The tests
cover short/window-edge contexts, disjoint requests, permuted page tables,
gapped BF16 cache strides, a dominant sink, and two graph scenarios:

- Captured index construction plus attention, replayed with changed query
  vectors, KV values, lengths, page tables, empty requests and padding.
- Captured attention consuming the original metadata object after the real
  cache builder refreshes all fixed-address buffers for changing requests.

The independent CPU oracle enumerates the official logical visibility
predicate, directly gathers through the CPU page table and performs one
FP32 softmax with a zero-value sink. It does not derive visibility from the
new helper's compact layout. Existing V4.1 attention tolerances are unchanged:
output RMS error at most `0.006 * reference_RMS + 1e-6`, elementwise
`atol=0.012, rtol=0.025`, and LSE `atol=0.015, rtol=0.003`.

No performance threshold or kernel dispatch was widened. This patch does
not establish draft model loading, context KV projection/RoPE, target
verification/rollback, sampling correctness, end-to-end graph support, or
a DSpark throughput improvement. Those remain the integration work described
in [DSPARK_INTEGRATION_AUDIT.md](DSPARK_INTEGRATION_AUDIT.md).
