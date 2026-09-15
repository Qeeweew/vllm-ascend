# V4.1 paged cache writes and cache specs

Implemented in `ops/cache_v41.py`, new classes in `core/kv_cache_interface.py`, and an early V4.1 planner branch in `patch/platform/patch_kv_cache_utils.py`. The parent task owns model and runner integration. No C++ code was changed by this cache task. Validation used the parent's fully installed r7 binary. Performance acceptance remains open: no latency/profiling run was started after device 0 was reserved for the eight-card Engram smoke.

## Cache store contract

`write_main_cache_v41(cache, values, slot_mapping, positions=None, compress_ratio=1)` stores precomputed BF16 `[T,512]` or `[T,1,512]` rows into `[pages,physical_rows,512]` or `[pages,physical_rows,1,512]`.

`write_index_cache_v41(key_cache, scale_cache, keys, scales, slot_mapping, positions=None, compress_ratio=1)` stores INT8 K128 and FP16 scale1 using the same fixed-shape indices. Keys are `[T,128]` or `[T,1,128]`; scales can be `[T]`, `[T,1]` or `[T,1,1]`. Scale cache is `[pages,physical_rows,1]` or `[pages,physical_rows,1,1]`.

Slots are already **physical cache slots in compressed-position units**; they are not divided by CR again. Negative/out-of-capacity slots are ignored. CR2 requires real original-token positions and writes only positions with `(position+1)%2==0`. Negative positions never write. Incomplete groups and padding map to `(-1,0)` page/offset indices, which the existing AscendC scatter skips. All input shapes remain fixed; there is no dynamic `nonzero`, host `.item()`, host branch on device contents or dummy write into a real cache row. Active slots must be unique within an invocation; repeated writes across invocations support rollback naturally.

Only axis 0 may have extra page stride; inner axes must be contiguous and pages cannot overlap. The operator receives actual tensor strides. All returned caches alias their inputs. Store code performs no GEMM, normalization, RoPE or quantization. Main/index CR2 values must have RoPE applied at **group-first position `p+1-ratio`**, while publication occurs at the group-last row.

The reused operator is `npu_scatter_nd_update_sk`. Its arch22 implementation computes linear indices, rejects negative/out-of-range rows, and addresses gapped pages using explicit `var.strides()`. BF16/INT8/FP16 all passed real-device tests.

## New specs

| Class | `cache_layout` marker | Content per physical row | Scheduler manager |
|---|---|---:|---|
| `AscendV41MainCacheSpec` | `v41_bf16_latent` | 512 BF16 = 1024 bytes | FullAttentionManager |
| `AscendV41IndexerCacheSpec` | `v41_int8_index_scale` | 128 INT8 + 1 FP16 = 130 bytes | FullAttentionManager |
| `AscendV41SWACacheSpec` | `v41_bf16_swa` | 512 BF16 = 1024 bytes | SlidingWindowManager |

Each class is registered with its own uniform base class in `register_ascend_kv_cache_specs`. Group compatibility cannot merge V4 specs, main/index roles, differing compression ratios, or differing SWA retention rules. Strict merges retain all fields, padding, layout and retention metadata.

Constructor geometry on the installed vLLM main lane:

```python
main = AscendV41MainCacheSpec(
    block_size=64, tokens_per_state=2,
    num_kv_heads=1, head_size=512, dtype=torch.bfloat16,
)
index = AscendV41IndexerCacheSpec(
    block_size=64, tokens_per_state=2,
    num_kv_heads=1, head_size=128, dtype=torch.int8,
)
swa = AscendV41SWACacheSpec(
    block_size=32, tokens_per_state=1, sliding_window=128,
    num_kv_heads=1, head_size=512, dtype=torch.bfloat16,
)
```

`block_size` is in **original-token scheduler units**. `physical_block_size` and `get_storage_block_size(spec)` return `block_size/tokens_per_state`; both examples with CR2 have 32 physical rows. Main and index `num_states`, `state_content_size_bytes`, actual bytes and memory budgeting agree. The new implementation was tested on the installed main API; it is not an independent validation of historical v0.28 APIs.

Every index page stores its contiguous K array first, followed by its contiguous FP16 scales. `scale_offset_bytes = physical_block_size*128`; the actual content is `physical_block_size*130` bytes. Page padding follows the scales. This is a two-view page, not an interleaved 130-byte tensor row or FP8 packed record. `page_size_padded` must be large enough and even. Alignment can increase the physical page but never silently shrink a larger hybrid common page supplied by the planner. Dataclass replacement with a new block size recalculates physical rows and preserves valid common-page padding.

SWA is a paged sliding window, **not a fixed 128-token ring**. The inherited manager retains the current in-flight chunk plus 127 previous tokens and any `extra_retained_tokens`. At block size 32, chunk 4096, long context and no extra retention, the admission bound is `ceil((4096+127)/32)+1 = 133` pages. The extra page covers an unaligned window start. Block-table address space still covers the request's logical sequence; freed blocks are managed by SlidingWindowManager.

## Runner integration contract

The parent task has implemented an explicit V4.1 dispatch before legacy MLA/SFA allocation and reshape branches. The following defines the required contract; runner test evidence is tracked by that task.

1. In allocation, honor standardized descriptor backing size, layer offset/stride and block stride. Do not alias every `shared_layers` entry solely because `self.use_compress` is true. The existing `use_compressed_cache` allocation branch does that, and the current `requires_padded_page_layout`/GLM branch has its own alias semantics. New V4.1 owners require explicit descriptor interpretation; model consumers share their owner's cache without requesting another allocation.
2. Main/SWA reshape produces one BF16 latent tensor, never separate nope/RoPE K/V tensors. The current CR1 `AscendMLAAttentionSpec` page-strided branch always calls `_get_attention_kv_cache_dims` and expects `MLAAttention`; it is inappropriate for a V4.1 cache-only layer with a packed 512-wide latent. Do not fabricate a V4 or sparse-C8 marker to bypass it.
3. Index reshape produces separate K and scale views into each page using **physical rows**. Existing `AscendSFAIndexerCacheSpec` allocation and reshape use uncompressed `block_size` and cannot be reused for CR2 without changes. The new index spec uses the main packed-page accounting and its explicit marker instead.
4. `get_kv_cache_spec` must retain the new subclass rather than reconstructing a plain `AscendMLAAttentionSpec`. Keep compression in original scheduler token units. Attention/indexer metadata must use each cache's correct page table; the spec has no KV-source mapping, which belongs to the model/cache owner.
5. NPU kernels need ND tensor views with actual axis-0 strides, not NZ conversion. Shared buffers and indices must retain addresses during graph replay.

For a per-layer raw view whose physical page stride equals `spec.page_size_bytes`, existing `_adjust_kv_layout` is reusable after the marker dispatch:

```python
p = spec.physical_block_size
if spec.cache_layout == "v41_int8_index_scale":
    views = self._adjust_kv_layout(
        raw, [(num_blocks, p, 1, 128), (num_blocks, p, 1, 1)],
        [torch.int8, torch.float16], spec.page_size_bytes,
    )
else:
    views = self._adjust_kv_layout(
        raw, [(num_blocks, p, 1, 512)], [torch.bfloat16], spec.page_size_bytes,
    )
```

When standardized descriptors expose a different actual `block_stride`, use that stride and `descriptor.offset + layer_index*descriptor.layer_stride` in the `as_strided` views instead. Require the physical stride to cover the layer's page content and be divisible by each view's element size. Do not allocate `descriptor.size` independently for every layer, or reinterpret the entire shared backing as one contiguous layer.

## Validation evidence

- `tests/e2e/single_node/ops/test_cache_v41.py`: **57 passed** on device 0. CR1/2, T0–4096, 3D/4D layouts, true gapped storage, unchanged gap bytes, invalid slots/positions, single-token first/last slots, twenty dynamic graph replays with rollback and all-padding steps. BF16 exhaustively covers all 65536 bit patterns, including NaNs, signed zero and subnormals; FP16 scale tests compare raw bits as well.
- `tests/ut/ops/test_cache_v41.py`: **6 passed**, fixed-shape group-end masking, shared K/scale index tensor, validation before either store, empty inputs and rejected legacy ratios/invalid layouts.
- `tests/ut/core/test_cache_v41_spec.py` plus existing `test_kv_cache_interface.py`: **30 passed**, including two existing core regressions. Physical row/page sizes, scale offsets, padded common pages, alignment, dataclass resizing, strict merges, registry isolation, full-cache budgets and SWA in-flight/extra retention are checked.
- `tests/ut/patch/platform/test_v41_cache_planner.py`: **27 passed**. Combined with existing `test_prefix_cache_cp_patches.py`: **67 passed, 1 skipped**, proving role/layer preservation, both CR page geometries, circular and SWA retention semantics, early dispatch, rejected configurations, default layout, real standard descriptor bounds and unchanged V4/prefix paths. The skip is an existing version-specific regression. Results: `planner_results.xml`.
- Ruff and diff whitespace checks pass. XML results are in this directory.

No device profiling claims are made for these stores. The next performance measurement should separate metadata index construction from native scatter and include one-row decode, CR2 incomplete groups and large prefill buckets; correctness alone does not close that gate.

## V4.1 planner and common page

The public and packed grouping hooks detect actual V4.1 spec classes before legacy V4 C4/C128 grouping. The V4 unifier declines V4.1 input. The new branch pads only stride-aware V4.1 main/index/SWA specs to a common 32768-byte page and calls the captured upstream general equal-page grouper. Every input layer must occur exactly once in the result; role classes, compression ratios and circular state remain distinct. Original frozen specs remain unchanged. Standard main allocation then provides a single backing with group overlays and per-layer descriptors; there is no V4 tuple allocator in this path.

At scheduler block 32 the unpadded page contents are:

| Role | CR1 bytes | CR2 bytes | Allocated page bytes |
|---|---:|---:|---:|
| Main BF16 latent | 32768 | 16384 | 32768 |
| Index INT8 K + FP16 scale | 4160 | 2080 | 32768 |
| SWA BF16 latent | 32768 | unsupported | 32768 |
| Compressor FP32 capacity 8 | 32768 | 32768 | 32768 |

Compression keeps original-token block size 32 and physical rows 32/16. Padding changes only the page stride; it never divides slot mappings again. Ring `CircularBufferSpec` keeps capacity 8, one block per request, no slot mapping, no prefix caching and its exact unpadded 32768-byte page. The inherited SWA manager still admits the current chunk plus 127 previous tokens; it is never converted to a fixed 128-token ring.

Initial planner rejects attention pages larger than 32 KiB, any other mixed cache class, disabled hybrid manager and the historical v0.28 descriptor API. Circular state must be FP32 `[8,1024]` with no V or padding; capacity 16, changed geometry and padded ring pages are rejected. With the current model capacity formula, speculative token counts 0–6 use capacity 8; 7–14 use capacity 16 and are rejected by this initial planner. Raw block 64 with CR1 main/SWA likewise exceeds the supported page size. This restriction avoids silently changing model ring capacity during planning.

Current runner and compressor require both layer-compact and block-compact allocation. Upstream's default `LBNHC` satisfies this; `LBHNC` also works because all V4.1 cache roles have one head. Explicit block-outer layouts and `LHBNC` fail in the V4.1 planner before allocation. Layout resolution precedes profiling and is sent to workers by RPC, so the planner does not rewrite an already resolved layout.

### Future ring page-stride support

Do not pass a padded `as_strided` ring to the current contiguous-only kernel. A minimal future AscendC extension can keep logical capacity and inner row stride unchanged:

1. Pass `state_stride0_elements` from the binding through a 64-bit tiling field; validate it is at least `capacity*1024` and that inner axes are contiguous.
2. For previous-ring reads and tail writes, replace `slot*1024` by `(slot//capacity)*state_stride0_elements + (slot%capacity)*1024`. These are the only two state addressing sites; GEMM, normalization and synchronization are independent.
3. Add real gapped-page guard-byte, odd-start previous-read, rollback and changing-block-ID graph tests, then repeat latency acceptance after a complete build.

This is an audit, not an implemented kernel change. An alternative for larger power-of-two pages is selecting a larger capacity before model and spec construction; that also requires every model/metadata/kernel consumer to agree and cannot be achieved by planner-only dataclass replacement.

### Separate compressor numerical risk

The gate investigation measured AscendC basic `Rsqrt(1.0)` as `0.998046875`. Compressor normalization currently also uses this primitive, so its earlier BF16 tolerance does not establish the tighter normalization contract. A subsequent source window replaced compressor Rsqrt with Sqrt+Div, and the parent completed the r10 full build/install. The compressor suite now passes 54 tests, including eight strict NPU normalization cases; its updated graph performance acceptance passes 38 main cases and 12 closing-group cases, while eager noise acceptance remains open. Details are in `../compressor_v41/report.md`. Cache-store bit-exact tests above do not resolve this arithmetic issue.
