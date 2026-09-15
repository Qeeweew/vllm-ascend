# Development commit checkpoint

This checkpoint organizes the existing adaptation on personal fork
`Qeeweew/vllm-ascend`, branch `deepseek-v41-910b-w4a16-engram`, starting at
`b49962987e89b850586f1819ce8f85daa85a0f81`. It does not mark the full-model
adaptation complete. Runtime reference: vLLM
`836bb3839ffefcda8283ea7d41671a89e1a613df`.

## Commit boundaries

1. Align FastAPI requirements with the installed upstream serving API.
2. Add FP8/BF16 and MXFP4/INT4 group32 conversion and safe tail continuation.
3. Add AscendC kernels, bindings and component operations. Compressor GEMM
   remains separate; W4A16 decode originates from the personal V4 branch.
4. Integrate V4.1 target, multimodal routing, cache planning, pinned Engram
   staging/lifecycle and opt-in decode optimizations, with regression tests.
5. Add the guarded DSpark draft/loader and proposer plumbing. Noncausal K5
   attention and target auxiliary-state infrastructure are included in the
   preceding component/runtime commits. Speculative admission stays disabled.
6. Archive reproducible harnesses, compact text evidence, plans and reports.

These commits form an ordered development series. NPU measurements below
refer to their recorded build/source manifests; CPU checks cover the final
combined tree. Intermediate commits were not individually run on NPU.

## Validation at this checkpoint

- All 38 new/modified unit-test files plus the existing base-proposer suite:
  **864 passed**, 14 upstream TorchScript deprecation warnings, 53.03 seconds
  excluding process bootstrap. Raw result: [commit_checkpoint_cpu.xml](commit_checkpoint_cpu.xml).
- DSpark proposer and base-proposer rerun: **75 passed**. Test doubles now
  provide metadata builders and correctly compare a block-table tensor view.
- Conversion/recovery/watcher rerun after adopting the repository-required
  `regex` import: **29 passed**. Import ordering and test formatting were also
  normalized by the pinned Ruff hook.
- Applicable manual hooks passed: Ruff check/format, codespell, typos,
  markdownlint, Gitleaks staged scan, ShellCheck, filename/package checks,
  forbidden-import/logger/context-manager/long-function and symbolic-meta checks.
  The actionlint environment download failed; no workflow files changed, so
  this unrelated hook was not required in the targeted rerun. Repository
  clang-format excludes `csrc/`; no other C++ files changed.
- Current AscendC installation remains the successful r12 editable build;
  no kernel logic changed during commit organization (only spelling-check
  annotations for the inherited `PrefetchS` identifier).
- Prior NPU component, TP8 and HTTP profiling evidence and its limitations:
  [stage report](../../docs/performance/deepseek_v41_910b.md).

Raw tensor captures (`*.pt`, `*.pth`, `*.safetensors`) remain local and are
ignored by Git. JSON/CSV/XML evidence is preserved, including failed and
provisional runs. CSV line endings are normalized to LF without changing
measurement fields. Full profiler traces remain under the workspace artifact
directory referenced in the profiling report.

## Remaining acceptance work

Source shards 47/48 and complete converted-checkpoint publication are still
pending. The user resumed downloading; the independent recovery process
stopped after detecting a changed source prefix. The conversion watcher keeps
waiting for final published files without modifying download temporaries.

Real 40-layer inference, real full Engram tables, long-context quality,
DSpark acceptance/rollback and final whole-model profiling remain unverified.
The latest pre-embedding Engram staging change has CPU and single-NPU coverage;
production TP8 text/image regression is still required after that change.
