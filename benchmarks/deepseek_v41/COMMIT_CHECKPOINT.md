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
7. Register the V4.1 draft architecture and test actual proposer weight sharing.
8. Restore DSpark cache lookback and cover K5 Engram history acceptance cases.
9. Record Engram preprocessing TP8 controls, including failed repeatability gates.
10. Verify real target auxiliary outputs in eager execution and graph replay.
11. Add the real-weight DSpark component harness and independent CPU oracle.
12. Update this checkpoint and document the unresolved maximum-context issue.

Every development commit carries a `Signed-off-by` trailer. GitHub CLI confirms
the authenticated account and repository owner are `Qeeweew`, with origin
`https://github.com/Qeeweew/vllm-ascend.git`. This checkpoint is committed locally;
it has not been pushed and no pull request has been opened.

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
  annotations for an inherited prefetch identifier).
- Prior NPU component, TP8 and HTTP profiling evidence and its limitations:
  [stage report](../../docs/performance/deepseek_v41_910b.md).

Subsequent focused validation:

- Draft registry/loading/sharing and admission: **57 passed**.
- Cache planner/spec/metadata with the retention fix: **100 passed**;
  expanded retention suite: **11 passed**. Engram history/runner: **61 passed**.
- Production Engram preprocessing controls completed on eight ranks. Strict
  CANN text repetition passed exactly. Native text/image repetition failed
  the new `1e-4` selected-logprob repeatability gate, while structural and
  cleanup checks passed. This does not replace the existing kernel numerical
  accuracy criteria. See [TP8 controls](ENGRAM_PREPROCESS_TP8_REGRESSION.md).
- Three real target layers exported exactly correct auxiliary HC means on
  all eight ranks: two eager calls and six graph replays per rank after
  resetting warmup counters. Repeated outputs matched exactly and cleanup
  passed. The initial startup failure and forced cleanup are retained in the
  [auxiliary-state report](TARGET_AUX_GRAPH_PROBE.md).

Raw tensor captures (`*.pt`, `*.pth`, `*.safetensors`) remain local and are
ignored by Git. JSON/CSV/XML evidence is preserved, including failed and
provisional runs. CSV line endings are normalized to LF without changing
measurement fields. Full profiler traces remain under the workspace artifact
directory referenced in the profiling report.

## Remaining acceptance work

As of 2026-09-15 23:27 UTC, source shards 47 and 48 are both published at their
expected sizes, 101535150936 and 101537926640 bytes. The conversion watcher
has started shard 47; the manifest still lists 46 shards and `complete=false`.
Full converted-checkpoint publication remains pending. The independent recovery
process remains stopped and download temporaries are untouched.

Real 40-layer inference, real full Engram tables, long-context quality,
DSpark acceptance/rollback and final whole-model profiling remain unverified.
The latest pre-embedding Engram staging change has CPU, single-NPU and bounded
TP8 coverage as detailed above; the native repeatability diagnostic remains
unresolved. The initial DSpark input kernel also needs a maximum-context
boundary correction and integration tests, documented in the
[maximum-context audit](DSPARK_MAX_CONTEXT_AUDIT.md). Production speculative
admission remains disabled.
