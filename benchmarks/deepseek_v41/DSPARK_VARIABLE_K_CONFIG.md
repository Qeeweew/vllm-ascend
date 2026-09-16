# V4.1 DSpark configurable proposal count

The runtime proposal count is configured by `num_speculative_tokens`, with the
current supported range K1 through K8. The checkpoint's `dspark_block_size=5`
remains training metadata, and its three draft layers remain unchanged.

## Configuration normalization

Upstream sets the draft `n_predict` from the training block size, then applies
an MTP module-reuse divisibility rule. That incorrectly rejects K6, K7 and K8.
The Ascend `update_arch_` wrapper preserves the original architecture update,
then sets `n_predict` to the requested runtime K only for `DSparkV41DraftModel`.
The parallel DSpark computation does not reuse MTP modules K/5 times. Neither
the target configuration nor the checkpoint file is changed. Other model
architectures, methods and the omitted-K default retain upstream behavior.

The real `EngineArgs.create_engine_config()` CPU cases cover every K1..8.
They assert the exact runtime K, draft architecture, retained training block
size, three MTP stages, unchanged complete target HF configuration, unchanged
configuration file and no initialized NPU. Restoring the original upstream
architecture updater rejects K6/7/8 with its divisibility error in the same
EngineArgs path. Existing Qwen3/K3/legacy-V4 configuration tests still pass.

## Physical cache pages

The target compressor already chooses a power-of-two ring that retains the
verification rows and preceding pair state. The planner now uses that actual
ring capacity when choosing its common physical page:

| Runtime K | Ring rows | Ring shape per block | Common page |
|---|---:|---|---:|
| 1..6 | 8 | FP32 `[1,8,1024]` | 32 KiB |
| 7..8 | 16 | FP32 `[1,16,1024]` | 64 KiB |

The ring is contiguous and has no page padding. An old eight-row ring, even
with a forged 64 KiB padded size, is rejected for K7/8. Attention and index
cache logical block sizes remain unchanged; their existing page-stride
arguments address the larger common page. Packed INT8 keys and FP16 scales
retain the same within-page offsets. No runner or native kernel change is
needed for these views: the runner's actual allocate/reshape paths use the
specification's page bytes and ring capacity.

K7/8 doubles the common page bytes. A fixed KV budget therefore allocates
approximately half as many blocks as K1..6; model admission and performance
comparisons must account for this capacity cost.

## Validation and limits

The complete configuration/planner/retention/compressor metadata CPU set passes
**92 tests**. Besides EngineArgs checks, tests call the real standard planner,
runner allocation/reshape and compressor binding; they inspect shared backing,
contiguous ring stride, final-row writes, packed scale offsets and padding
sentinels. Every acceptance length for every K1..8 is checked against CR2
recomputation, and the real sliding-window manager is exercised across all
64 page phases with the configured extra-K retention. Existing unsupported
geometry checks and non-V4.1 dispatch isolation remain in place.

Restoring the original planner in a separate process yields **2 failed / 6
passed** for the K1..8 view matrix: precisely K7/8 fail its capacity-eight
restriction. Evidence, source hashes and complete logs are retained under
`artifacts/dspark-variable-k/` at the workspace root. No NPU is used in this
validation. Native variable-K metadata, actual draft execution, complete target
verification and end-to-end performance remain separate acceptance layers.
