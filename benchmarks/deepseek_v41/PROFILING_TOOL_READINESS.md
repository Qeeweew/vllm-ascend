# Profiling tool readiness

CPU-only inspection on 2026-09-15. No profiler context was started, no trace
was collected or parsed, no model/NPU was initialized, and no package was
changed. This complements `SERVING_DEPENDENCY_AUDIT.md`; successful event
timing benchmarks are not a substitute for the final timeline profile.

## Subsequent runtime validation

The later bounded TP8 HTTP run collected real raw data. Automatic parsing in
daemon workers failed; the official offline analyse interface exported eight
nonempty timelines successfully. Coverage, durable artifacts and limitations
are in [HTTP_PROFILE_RESULT.md](HTTP_PROFILE_RESULT.md). The import-only
observations below are retained as the earlier readiness audit.

## What works now

| Check | Result | Evidence |
| --- | --- | --- |
| CANN `msprof --help` | Exit 0 | `/tmp/v41-msprof-help.log` |
| CANN Python analysis CLI `--help` | Exit 0; import/export/query/analyze subcommands | `/tmp/v41-msprof-python-help.log` |
| CANN Python `export --help` | Exit 0 | `/tmp/v41-msprof-export-help.log` |
| CANN Python `analyze --help` | Exit 0 | `/tmp/v41-msprof-analyze-help.log` |
| `torch_npu.profiler` import | PASS | `/tmp/v41-torch-profiler-import.log` |
| Torch-NPU trace/kernel/memory view parsers | All three imports PASS | Same log |
| ProfilerActivity CPU and NPU, profile/schedule/trace handler symbols | Available; not invoked | Same log |
| `torch.npu.is_initialized()` after imports | False | Same log |
| `mskpp` package import | PASS | `/tmp/v41-kpp-import.log` |

The binary resolves to `/usr/local/Ascend/cann-9.1.0/bin/msprof`. This binary
does not implement `--version`: it returns 255 and says “unrecognized option”.
That result is a CLI limitation, not a broken import or a profiling failure.
The installation is CANN 9.1.0 and torch-npu is 2.10.0.post4. Toolkit package
info reports platform-profiler 1.0.0; service-profiler package-info says 26.1.0
while installed Python distribution metadata says 26.0.0. Record both if
investigating parser compatibility; do not silently substitute one version.

Current core route for the eventual report is Torch-NPU CPU+NPU capture,
then its timeline/kernel/memory export and/or CANN offline export. These
entry points are usable at import/help level without repairing the optional
service-profiler environment. Actual collection, device permissions, trace
completeness, parser execution on real data and cross-rank time alignment
remain untested by this audit.

## Real blocker versus metadata warnings

`../.venv/bin/python -m ms_service_profiler --help` exits 1 before argparse
can print help. Its plugin loader imports `ms_service_profiler.analyze`,
which immediately imports `msguard`; that package is missing. The same
missing module blocks direct imports of `exporter_summary` and `split`.
This is a **reproduced blocker for the service-profiler CLI**, not merely
an unsatisfied metadata line. Log: `/tmp/v41-msserviceprofiler-help.log`.

| Dependency finding | Concrete evidence and scope |
| --- | --- |
| msguard missing | Stops service-profiler CLI, analyze/summary/split imports |
| matplotlib missing | Direct import fails; MoE, EP-balance and EPLB chart functions contain deferred pyplot imports |
| plotly missing | Direct import fails; KPP visualization imports it lazily; base mskpp import still succeeds |
| openpyxl missing | Direct import fails; Excel/report paths not executed in this audit |
| tzdata missing | Package import fails; system timezone behavior not tested or assumed broken |
| pandas 3.0.5 versus profiler ~=2.2 | Version mismatch; compare/parse modules imported, actual data processing untested |
| OTLP grpc/http 1.44.0 versus profiler ==1.33.1 | Version mismatch; no telemetry export was attempted |
| affinity-sched argparse distribution | Metadata issue; Python's standard argparse works, including CANN export/analyze help |

Optional-module results are recorded in
`/tmp/v41-profiler-optional-imports.log`. Importing compare/parse individually
does not make the service-profiler CLI usable: its global entry-point loading
still encounters the missing analyze dependency first. Do not bypass that
failure with fake modules or modified system packages.

## Minimal workspace-only repair if service analysis is needed

Do not change the serving `.venv` just to enable optional offline charts.
Create a separate `../.venv-profiler-analysis` overlay with system-site access
to the vendor-installed CANN modules. This avoids copying a vendor wheel that
is no longer present at the temporary installation path in `direct_url.json`.
Keep the environment dedicated to offline trace analysis, not model serving.

The smallest first repair is installing a vendor-compatible **msguard** in
that overlay, then repeating the CLI help/import checks; no version has been
selected or installed by this audit. For the full optional reporting stack,
also supply matplotlib, openpyxl, tzdata, Plotly >=5.11, pandas 2.2.x, and a
coherent OpenTelemetry 1.33.1 family matching service-profiler requirements.
Pin its SDK/API/exporter/proto packages together; installing only grpc/http
exporters 1.33.1 conflicts with the base `opentelemetry-exporter-otlp==1.44.0`
meta-package. Resolve this in the overlay and inspect the resulting metadata
before running analysis.

Preserve NumPy 1.26.4 for the current Ascend toolchain unless a separate
compatibility decision is made. Triton Ascend 3.2.2 pins it exactly. A blanket
NumPy upgrade to address unrelated OpenCV metadata is outside this repair.
An overlay `pip check` will still see unrelated inherited packages; classify
those rather than claiming a globally clean environment.

Preparation command for a later authorized repair (not run here):

```bash
../.venv/bin/python -m venv --system-site-packages ../.venv-profiler-analysis
```

Package installation requires choosing the compatible vendor msguard source
and producing the narrow offline constraints above. It is not a prerequisite
for trying the already-importable core Torch-NPU/CANN timeline path.

## Reproduction and next acceptance gate

Safe help-only commands used in this audit:

```bash
msprof --help
../.venv/bin/python \
  /usr/local/Ascend/cann-9.1.0/tools/profiler/profiler_tool/analysis/msprof/msprof.py \
  export --help
../.venv/bin/python \
  /usr/local/Ascend/cann-9.1.0/tools/profiler/profiler_tool/analysis/msprof/msprof.py \
  analyze --help
../.venv/bin/python -m ms_service_profiler --help
```

The final command is expected to fail until msguard is supplied. Core import
check without instantiating a profiler:

```bash
../.venv/bin/python - <<'PY'
import torch
import torch_npu
from torch_npu.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from torch_npu.profiler.profiler import analyse
from torch_npu.profiler.analysis.prof_view import (
    _trace_view_parser, _kernel_view_parser, _memory_view_parser,
)
assert not torch.npu.is_initialized()
print('Core profiler and offline parser imports passed; no collection started')
PY
```

After functional model admission, schedule a separate bounded NPU profiling
window. First produce and successfully export one small real trace, then
profile the agreed prefill/decode shapes. Retain CPU hash/gather, pinned H2D,
stream waits, compressor projection/vector work, W4A16 expert kernels, HCCL
and graph replay timelines. Check trace data is nonempty and covers intended
ranks/steps before drawing conclusions. Compare unprofiled latency with the
profiled run to quantify collection overhead. The final report needs actual
trace artifacts and observed bottlenecks; tool import readiness alone cannot
support performance claims.
