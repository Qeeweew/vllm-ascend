# Bounded TP8 HTTP timeline result

Eight real Torch-NPU timelines were collected and exported on 2026-09-15.
This validates collection and analysis for the production server. It is a
three-real-language-layer/E384 fixture with small synthetic Engram tables,
not final 40-layer performance or model-quality acceptance.

## Collection, export and shutdown

`http_serving_graph_profile_r2.json` retains its original failed status:
HTTP requests and profiler start/stop all returned 200, but automatic export
found zero timelines. Torch-NPU refuses parsing inside vLLM daemon workers.
Raw data was preserved, then the official
`torch_npu.profiler.profiler.analyse` offline interface produced all eight
nonempty timelines in 13.197 seconds without initializing NPU. Evidence:
`http_profile_r2_offline_export.json` and
`/tmp/v41-http-profile-r2-offline.log`. No NPU recollection was needed.

The server used explicit `--shutdown-timeout 30`, exited with code 0, left
no owned descendants, and logged all eight workers exiting gracefully.
There were no internal force-kill or resource-tracker leak warnings.
An AsyncLLM output-handler `EngineDeadError` was logged after MPClient
teardown completed; this shutdown ordering diagnostic remains recorded in
`HTTP_SERVING_SMOKE.md`. The earlier timeout-zero r1 cleanup failure is not
rewritten as a pass.

## Coverage and initial observations

Each rank contains 5,838 kernel rows. Every timeline includes:

| Observed record | Count per rank |
| --- | ---: |
| Graph `MODEL_EXECUTE` and corresponding execute API | 6 each |
| Native W4A16 decode kernel | 18 |
| Engram gate kernel | 8 |
| Compressor kernel | 8 |
| Host-side device-to-host copy calls | 8 |
| Host-side host-to-device copy calls | 118 |

The request pair contains one text completion and one image chat completion,
both with four generated tokens. The repeated image used a warm cache:
there is no vision fused-infer attention kernel in this capture. Thus it
does not measure the 32-layer encoder. The three-layer fixture also omits
the later CR1 candidate consumers and second Engram layer. Copy counts
include all metadata transfers; they do not isolate Engram transfer cost or
provide a standalone CPU hash/gather measurement.

Rank 0's largest summed kernel categories are shown below. Summed durations
can overlap across streams, include collective waiting, and are **not**
request latency or wall-time percentages.

| Kernel category | Calls | Sum of observed duration, ms |
| --- | ---: | ---: |
| HCCL all-reduce | 56 | 77.225 |
| Grouped matmul | 12 | 10.958 |
| MatMulV2 | 184 | 6.742 |
| HCCL all-gather | 16 | 2.877 |
| SearchSorted | 48 | 2.142 |
| StridedSlice | 264 | 2.076 |
| Native W4A16 | 18 | 1.668 |

Communication synchronization and rank-to-rank readiness merit investigation
in the full model. These observations do not establish that network transfer
is the bottleneck, and do not justify changing HCCL numerical settings.
The report records separate initial, warm-unprofiled and profiled request
wall times. Single observations with cache and cold-start differences cannot
serve as a profiler-overhead estimate or a TTFT/TPOT regression gate.

## Durable artifacts and reproduction

The exported data and raw collection are copied to the workspace artifact
directory, outside the source repository:

`/home/xwj/workspace/deepseek-v41-flash/artifacts/deepseek_v41/http_profile_r2`

There are 900 files totaling 285,832,345 bytes. Per-file SHA256, all-rank
coverage and per-type duration sums are in `http_profile_r2_summary.json`.
The original `/tmp/v41-http-profile-r2` collection is also retained.
Recompute the summary from the workspace root:

```bash
.venv/bin/python vllm-ascend/benchmarks/deepseek_v41/summarize_http_profile.py \
  --profile-dir artifacts/deepseek_v41/http_profile_r2 \
  --output vllm-ascend/benchmarks/deepseek_v41/http_profile_r2_summary.json
```

Two CPU tests check that overlapping durations remain sums rather than wall
time, flow events are excluded from execution counts, and missing ranks
cannot be reported as TP8 coverage. Full-model traces must still cover cold
vision encoding, real Engram lookup/DMA, long context, concurrency and
agreed unprofiled performance baselines.
