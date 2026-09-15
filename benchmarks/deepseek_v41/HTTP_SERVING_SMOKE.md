# Localhost HTTP serving smoke

`smoke_http_serving.py` launches standard `python -m
vllm.entrypoints.cli.main serve` with the production model registry and
default Ascend worker. It does not use the diagnostic MM worker or register
a test model. Without `--run`, it only checks the fixture and parses the
actual installed server CLI; it does not open a socket or initialize NPU.

## Scope and checks

The fixed fixture `/tmp/v41-mm-production-numa-graph-r1` has three real
device-weight language layers, 384 experts and one **synthetic 4096-row
Engram table**. Image-enabled runs load the real vision parameters. This is
an HTTP execution check, not full-model quality, full-table loading or a
performance measurement.

The script uses a unique served model name and requests only
`http://127.0.0.1`; inherited external proxies are ignored. An occupied port
is rejected, so an existing server is never reused. Checks include:

- HTTP 200 `/health` before and after requests.
- `/v1/models` contains this run's unique served model name.
- `/v1/completions` accepts raw prompt IDs `[100,129264,101]`, preserving a
  literal image-token ID in a text request; four generated IDs, finite
  selected-token logprobs, usage and `finish_reason=length` are required.
- SSE completion requires valid data frames, returned token IDs, a final
  usage event, finish reason and `[DONE]`. Its deterministic generated IDs
  and text must match the non-streaming request.
- With `--image`, `/v1/chat/completions` receives a local photo converted to
  a <=512-pixel JPEG data URL, and returns four generated tokens. The image
  is sent over localhost only; no remote image download is requested.

Graph mode requests `mode=0` and `FULL_DECODE_ONLY`; encoder graphs remain
disabled. A graph setting in the command is not by itself proof of replay;
server logs and existing model/runner graph acceptance provide that evidence.
This HTTP script does not inspect internal Engram hashes or every expert ID.

## Process bounds and cleanup

Defaults: startup 600 seconds, each request 120 seconds, controller shutdown
90 seconds and actual server `--shutdown-timeout 30`. Responses/SSE are capped
at 4 MiB and generations at four tokens.
The server has a fresh process session/group. Cleanup first sends SIGINT to
its API-server parent and allows the engine to stop its workers. If needed,
SIGTERM and then SIGKILL target only that new owned session/group. An unrelated
process is never selected by executable name, port or global PID pattern.

The JSON records signals, return code, surviving members and escalation.
`passed` requires HTTP checks plus parent exit zero, no live group members
and no forced escalation, internal engine/worker force kill or resource-tracker
leak warning in the server log. HTTP success followed by forced cleanup is
`failed_cleanup`. An artifact always records runtime failures after launch.
This process-level observation is not a per-owner unregister counter; explicit
runtime shutdown is separately tested and server cleanup logs must be retained.

The child uses existing upstream settings `VLLM_WORKER_MULTIPROC_METHOD=spawn`
and `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=60`. No new production environment
variable or global system setting is introduced.

The current default NUMA map is `[6,7,4,5,0,1,2,3]`, following the latest
per-rank placement validation. The first HTTP test used
`[6,6,4,4,0,0,2,2]`; it is retained as a functional test and is not a NUMA
performance comparison. Every report preserves its exact command/map.

## Commands

CPU-only argument preparation:

```bash
OMP_NUM_THREADS=4 ../.venv/bin/python benchmarks/deepseek_v41/smoke_http_serving.py \
  --graph --output /tmp/v41-http-prepared.json
```

Run only during the allocated TP8 device window:

```bash
OMP_NUM_THREADS=4 ../.venv/bin/python -u benchmarks/deepseek_v41/smoke_http_serving.py \
  --run --graph \
  --image ../sources/vllm/tests/v1/ec_connector/integration/hato.jpg \
  --profile-dir /tmp/v41-http-profile-r2 \
  --output benchmarks/deepseek_v41/http_serving_graph_profile_r2.json
```

The server log is the output filename with `.server.log` suffix. An existing
log is not overwritten. Omit `--image` for text-only admission and omit
`--graph` for eager execution. The fixture is reused without rewriting weights
or config; a full production-sized checkpoint is rejected by this bounded
script's shape checks.

The optional `--profile-dir` must be new or empty. It enables the standard
Torch-NPU profiler with stack capture disabled and frontend profiling ignored.
After initial HTTP checks, one additional unprofiled text/image round records
warm observations; POST `/start_profile` and `/stop_profile` then enclose one
text/image round. Stop/export has a separate 300-second deadline. The collection
gate requires eight nonempty worker timelines. These are three-layer functional
timelines, not final performance acceptance. Initial, warm and profiled wall
observations are recorded separately and do not estimate profiler overhead.
Repeated images may use processor/encoder caches; actual vision execution
coverage must be determined from the trace. Omit `--profile-dir` for HTTP only.

## CPU preparation result

Actual installed CLI parsing passed, including the production fixture, TP8,
lazy safetensors, graph configuration and NUMA map. NPU remained uninitialized.
Preparation with profiling and shutdown arguments also passed. Artifact:
`/tmp/v41-http-profile-prepared.json`; log:
`/tmp/v41-http-profile-cli-validation.log`.

Five CPU tests passed in 0.18 seconds after startup:
`tests/ut/models/test_http_serving_smoke.py`. They cover SSE multiline/comment
frames, truncated/oversized responses, finite token evidence, and actual
owned-process cleanup while an unrelated session remains alive, and rejection
of internal force-kill/resource-leak logs despite parent exit zero. Log:
`/tmp/v41-http-script-tests.log`. Ruff passed for the script and tests.

## HTTP run status

First run: `http_serving_graph_r1.json`; controller log:
`/tmp/v41-http-serving-r1.log`. All HTTP checks passed: health/models/text/SSE/
image returned 200, text and SSE generated identical IDs
`[91488,55063,44099,34954]`, image returned four tokens from 224 prompt tokens,
and the server logged actual aclgraph replay. Initial text, SSE and image wall
observations were 0.7839, 0.1097 and 0.4307 seconds, respectively; these are not
benchmark timings.

The overall first-run result is **failed_cleanup**. Although its API parent
exited zero and no owned processes survived, the existing upstream default
`shutdown_timeout=0` selected `mode=abort timeout=0s`, force-killed EngineCore
and emitted eight leaked-semaphore and ten shared-memory warnings. The report
preserves its original observations and records this post-run audit. The script
now passes the existing server option `--shutdown-timeout 30`; no production
shutdown implementation change was needed for this HTTP-specific default.

Second run: `http_serving_graph_profile_r2.json`, with updated NUMA placement,
30-second engine grace and optional profiling. All HTTP checks passed, including
the separate warm round and profiled text/image requests. `/start_profile` and
`/stop_profile` returned 200. Warm and profiled request-round wall observations
were 0.3040 and 0.3660 seconds; start/stop wall observations were 0.1394 and
1.2023 seconds. No performance or collection-overhead claim follows from these
single samples.

The r2 controller returned 1 and its original **failed** state is retained:
Torch-NPU refuses automatic analysis in daemon workers, leaving zero exported
timelines immediately after `/stop_profile`. Raw profiling data exists for all
eight workers under `/tmp/v41-http-profile-r2`. The eight "stop while RECORD"
warnings are retained; trace content must be audited for actual coverage.
The official offline `torch_npu.profiler.profiler.analyse` interface completed
with four analysis processes in 13.197 seconds, exporting exactly eight nonempty
rank timelines (17.33–17.95 MB each). NPU remained uninitialized before and after
offline analysis. Its independent result is recorded in
`http_profile_r2_offline_export.json`; offline success does not reclassify daemon
automatic export as successful. Controller and export logs:
`/tmp/v41-http-serving-profile-r2.log` and
`/tmp/v41-http-profile-r2-offline.log`.

The CPU-only offline export invocation was:

```python
from torch_npu.profiler.profiler import analyse

analyse("/tmp/v41-http-profile-r2", max_process_number=4)
```

Timeline paths are listed in the export JSON and use
`*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json`. Successful export verifies
file production; operator and graph coverage, missing data and final performance
still require separate trace analysis.

Process cleanup passed: the engine entered `mode=drain timeout=30s` at 22:27:11,
all eight workers exited gracefully at 22:27:20, and the MPClient completed at
22:27:24. The API parent exited zero after only parent SIGINT, no owned group
members survived, and no internal force kill or resource-tracker leak warning
appeared. An `AsyncLLM output_handler` `EngineDeadError` was logged **after**
MPClient completion and before the HTTP application's successful shutdown.
This appears associated with shutdown ordering; the test does not establish
its cause or claim an error-free shutdown log. The production worker has no
per-owner unregister counter, so this is process cleanup evidence together with
the separately validated checked Engram shutdown path.
