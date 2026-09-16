# Native operator scratch lifetime acceptance

Status: the revised complete isolated Torch extension passes CPU lifetime and
native numerical/memory checks in all three queue modes. Production r12 remains
installed. The first isolated extension passed numerical checks but retained
completed scratch through queue callback copies; those failed memory results
are preserved below.
These checks do not replace the actual DSpark proposer numerical gate or
enable its production guard.

## Revised ownership acceptance

Extension SHA256: `5717bbd0f872b51c9a928ec6a3e9de3feac4a45875ffe0bb60d62cc9fee88fcd`.
All three fresh processes completed the unchanged 96-iteration numerical gate
and the new completed-scratch memory gate. Allocated memory after each of the
eight synchronized bursts was exactly **1,968,128 bytes** in every mode, with
no accumulation. Queue mode 2 still permits more simultaneously pending native
allocations than modes 0/1; peak memory is reported separately.

| Queue mode | Peak allocated bytes | Peak reserved bytes |
| --- | --- | --- |
| 0 | 65,661,952 | 88,080,384 |
| 1 | 157,947,904 | 180,355,072 |
| 2 | 1,005,739,008 | 1,140,850,688 |

Every mode passed 288 exact cache checks and 288 analytical attention checks,
with maximum attention absolute error `5.696527659893036e-05`. Raw first and
revised reports/logs and SHA256 values are in `workspace_lifetime/manifest.json`.
The CPU fixture also rejects the first ownership fix when a completed queue
slot retains a handler copy. This is correctness and allocation validation,
not a latency measurement or DSpark integration acceptance.

## First native run and memory failure

Extension SHA256: `22d1d373655ecb9d9c7e0b40b58b03b47b8471c7fd57fd84777c1f1665cf841f`.
The extension was loaded with `run_v41_small_ops.py`, using the unchanged
production r12 OPP. Every fresh process completed 96 iterations, 288 exact
cache comparisons and 288 analytical attention comparisons. Maximum attention
absolute error was `5.696527659893036e-05` in each mode.

| Queue mode | Peak allocated bytes | Peak reserved bytes |
| --- | --- | --- |
| 0 | 65,661,952 | 88,080,384 |
| 1 | 8,580,786,176 | 9,168,748,544 |
| 2 | 8,580,786,176 | 9,168,748,544 |

Capturing an owning tensor by value fixes premature release, but lets retained
handler copies keep completed scratch allocations alive. The revised callback
shares one tensor owner across its copies and resets that owner immediately
after the native submission. The CPU regression explicitly keeps a completed
queue-slot copy alive: the first fix fails this check, and the revised fix
passes all four immediate/deferred and zero/nonzero-workspace cases.

The native script now records allocated bytes after each synchronized burst
and requires less than 256 MiB; numerical success alone cannot pass acceptance.
The first raw reports retain their original numerical `status=passed`, which
does not certify memory correctness or final acceptance.

## Why three fresh processes

The installed `torch_npu` is `2.10.0.post4`, commit
`5dd8ef3f9b375b5ae4a83538d5785754148c3302`. Its actual source defines:

- [OptionsManager.cpp](https://github.com/Ascend/pytorch/blob/5dd8ef3f9b375b5ae4a83538d5785754148c3302/torch_npu/csrc/core/npu/register/OptionsManager.cpp):
  `GetTaskQueueEnable()` reads `TASK_QUEUE_ENABLE` once and accepts 0, 1, or 2.
  `ASCEND_LAUNCH_BLOCKING=1` overrides it to 0.
- [OpCommand.cpp](https://github.com/Ascend/pytorch/blob/5dd8ef3f9b375b5ae4a83538d5785754148c3302/torch_npu/csrc/framework/OpCommand.cpp):
  `OpCommand::Run()` queues a custom handler when the stream is not a
  `SyncLaunchStream`, task queue mode is nonzero, and the command is not marked
  synchronous. Mode 0 submits directly. This is the API used by our shared macro.

The script reads the installed library's exported `OpApiGetTaskQueueEnable()`
and requires it to equal the requested mode. It uses the ordinary default
stream. Each mode must start in a separate Python process because the option is
cached. No new production environment variable is added.

## Commands

From the repository root, with physical card 1 explicitly released for testing:

```bash
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_LAUNCH_BLOCKING=0 TASK_QUEUE_ENABLE=0 \
  ../.venv/bin/python benchmarks/deepseek_v41/check_op_api_workspace_npu.py \
  --queue-mode 0 --output /tmp/v41-workspace-r13-queue0.json
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_LAUNCH_BLOCKING=0 TASK_QUEUE_ENABLE=1 \
  ../.venv/bin/python benchmarks/deepseek_v41/check_op_api_workspace_npu.py \
  --queue-mode 1 --output /tmp/v41-workspace-r13-queue1.json
ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_LAUNCH_BLOCKING=0 TASK_QUEUE_ENABLE=2 \
  ../.venv/bin/python benchmarks/deepseek_v41/check_op_api_workspace_npu.py \
  --queue-mode 2 --output /tmp/v41-workspace-r13-queue2.json
```

Do not run the three processes concurrently. Existing result files are never
overwritten. Use new filenames for reruns. Defaults are 96 iterations, with a
single synchronization after each burst of 12 iterations.

## Required evidence

1. Native `aclnnScatterNdUpdateSkGetWorkspaceSize` reports **nonzero** scratch for
   context lengths 9, 33, 129 and the five query rows. The setup probe launches
   the native executor once with explicitly retained scratch and synchronizes
   before releasing descriptors. Actual stress iterations use the production
   Torch binding and its shared macro.
2. Each iteration writes three BF16 caches through context and query scatter
   calls. Immediately following each call, an allocation of exactly the reported
   scratch size is filled and released to exercise allocator reuse. It then
   builds the real fixed-K5 metadata and runs three native attention calls.
3. Every iteration retains each cache snapshot and each attention output until
   the burst check. Cache contents must match an independent CPU indexed store
   **bit for bit**; a later iteration cannot hide an earlier corrupted store.
   Metadata spans must be 14, 38, or 133 for the three prefixes.
4. Q and sinks are zero. The independent FP32 attention reference is the sum of
   visible BF16 values divided by `(visible_keys + 1)`, including the sink in the
   denominator. Require `atol=0.001, rtol=0.01`; report the maximum observed
   absolute error. These are synthetic scratch validation inputs, not a relaxed
   threshold for the existing DSpark model gate.
5. Each JSON must end in `status=passed`, `checked_iterations=96`, and 288 cache
   bitwise checks plus 288 attention checks. Record extension and opapi hashes,
   actual queue mode, peak allocated/reserved memory, and errors. A failure
   preserves its traceback and the last completed burst count.
6. After every burst, allocated memory must remain below 256 MiB, including
   retained outputs. Record the full series to detect accumulation across
   completed queue slots, together with peak allocated/reserved HBM.

No per-operator diagnostic synchronization or host copy occurs inside a stress
burst. Initial native workspace measurement does synchronize; it is separate
from the stress loop. This test is neither graph acceptance nor profiling.

The CPU regression executes the production macro with a fake runtime and proves
ownership through immediate and deferred handler submission. The NPU test adds
real allocator, queue, native scratch, and downstream consumer coverage. A pass
cannot by itself prove that scratch lifetime caused the DSpark metadata failure:
that metadata invocation has zero scratch, and its original failure remains a
separate investigation until the actual proposer passes without observers.
