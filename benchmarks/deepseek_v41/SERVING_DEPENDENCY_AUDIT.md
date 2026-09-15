# HTTP serving dependency audit

2026-09-15, current workspace `.venv`, read-only CPU validation. No packages
were installed/removed, no system dependencies changed, no model loaded and
no socket opened. `torch.npu.is_initialized()` remained false after building
the real OpenAI app. This establishes import/app construction compatibility;
it does not claim HTTP inference or full-model startup has passed.

## Result

**No remaining dependency blocker was reproduced for the HTTP server entry.**
The previous FastAPI/Starlette conflict has already been corrected in this
environment. No additional dependency change is necessary before the scheduled
serving smoke.

The audit successfully imported `vllm.engine.arg_utils.EngineArgs` and
`vllm.entrypoints.openai.api_server`, constructed arguments for the real
converted model path with TP8, constructed the actual FastAPI application,
and generated its OpenAPI schema: **26 routes, 20 schema paths**, including
`/v1/chat/completions` and `/v1/completions`. This took 24.831 seconds including
imports and plugin discovery. Log: `/tmp/v41-serving-app-audit.log`.

The first import probe also tried the former
`vllm.entrypoints.openai.cli_args`, which no longer exists in the installed
upstream checkout. This is an API relocation, not a missing dependency. The
successful probe uses `vllm.entrypoints.launchers.cli_args.make_arg_parser`.
The `openai.api_server` compatibility module itself still imports successfully
but is deprecated. Initial probe log: `/tmp/v41-serving-import-audit.log`.

## Effective versions

| Package | Version | Relevance |
| --- | --- | --- |
| vllm | 0.1.dev1+g836bb3839.empty | Editable local upstream source |
| vllm-ascend | 0.1.dev5255+gb49962987.d20260915 | Editable personal workspace |
| torch / torch-npu | 2.10.0+cpu / 2.10.0.post4 | Existing Ascend runtime pair |
| fastapi | 0.136.3 | Meets upstream >=0.133.0,<0.137.0 |
| starlette | 1.6.0 | Meets upstream >=1.0.1 |
| pydantic | 2.13.4 | Meets upstream >=2.12.0 |
| uvicorn / uvloop / httptools | 0.52.1 / 0.22.1 / 0.8.0 | HTTP server stack present |
| httpx / openai | 0.28.1 / 2.54.0 | Installed client constraints satisfied |
| transformers / tokenizers | 5.14.1 / 0.22.2 | Current plugin pin / upstream requirement |
| prometheus-fastapi-instrumentator | 8.1.0 | Meets upstream >=8.0.0 |
| model-hosting-container-standards | 0.1.16 | Actual app bootstrap succeeded |
| safetensors / setuptools | 0.8.0 / 80.10.2 | Current upstream constraints satisfied |
| numpy / opencv-python-headless | 1.26.4 / 5.0.0.93 | Declared metadata conflict; see below |
| triton-ascend | 3.2.2 | Pins numpy==1.26.4 |

FastAPI and Starlette resolve from `.venv/lib/python3.12/site-packages`.
`pyvenv.cfg` sets `include-system-site-packages=true`; CANN profiler packages
resolve from `/usr/local/Ascend/cann-9.1.0/python/site-packages`, and NumPy,
OpenCV and Triton Ascend resolve from the base Python site-packages. Multiple
historical vLLM/plugin metadata records are visible in the base environment,
but effective imports resolve to the current editable workspace. Do not
infer the active version from an arbitrary duplicate metadata entry.

## pip check findings

`../.venv/bin/python -m pip check` exits 1 with these ten findings. Log:
`/tmp/v41-serving-pip-check.log`.

| Finding | Origin and impact |
| --- | --- |
| mindstudio-kpp: missing plotly | System CANN visualization tooling |
| ms-service-profiler: missing matplotlib | System profiler chart tooling |
| ms-service-profiler: missing msguard | System profiler optional analysis dependency |
| ms-service-profiler: missing openpyxl | System profiler export tooling |
| ms-service-profiler: missing tzdata | System profiler declared dependency |
| affinity-sched: missing argparse distribution | CANN metadata asks for a separately installed distribution; Python 3.12 already has stdlib argparse |
| ms-service-profiler: OTLP grpc 1.33.1 required, 1.44.0 present | Inherited profiler/telemetry version conflict |
| ms-service-profiler: OTLP http 1.33.1 required, 1.44.0 present | Same coherent telemetry-family conflict |
| ms-service-profiler: pandas~=2.2 required, 3.0.5 present | Inherited profiler data-processing version conflict |
| opencv-python-headless: numpy>=2 required, 1.26.4 present | Image/video dependency metadata conflict |

The first nine findings are inherited CANN/tooling issues, not newly missing
vLLM HTTP dependencies. ServiceProfiler's import hook ran during this app
probe without preventing construction, but its later profiling/export paths
were not exercised. Those optional workflows can still fail and require
their own environment validation when selected for the profiling report.
Installing an obsolete PyPI argparse backport merely to silence metadata
is not an appropriate Python 3.12 serving fix.

The OpenCV issue is relevant to the wider image/video stack. Its actual
extension imported as `cv2 5.0.0` under NumPy 1.26.4, and an 8×8 uint8 RGB
array resized to 4×4 correctly. That limited ABI check passed; it does not
erase the declared version mismatch or validate all video codecs.

## Minimal remediation guidance

1. **Serving now:** retain the current FastAPI/Starlette/Pydantic stack and
   proceed to the real HTTP smoke. Do not downgrade FastAPI or force a full
   dependency reinstall to make unrelated profiler metadata quiet.
2. **NumPy/OpenCV metadata:** do not upgrade NumPy alone. Installed
   `triton_ascend==3.2.2` declares `numpy==1.26.4`, whereas OpenCV 5 declares
   NumPy >=2. No NumPy version satisfies both. A future clean environment
   needs an Ascend Triton build supporting NumPy 2, or an independently
   validated OpenCV build with compatible metadata and the upstream-required
   API. Blindly pinning an older OpenCV can violate upstream's >=4.13.0
   requirement. The present successful import is a recorded compatibility
   exception, not a resolver-clean solution.
3. **Profiler tooling:** if this specific service-profiler stack is required,
   use a dedicated environment or a coordinated `.venv`-only dependency set.
   Supply its missing analysis packages and pandas 2.2.x; align the whole
   OpenTelemetry family to the profiler-supported version. Downgrading only
   its grpc/http exporters would conflict with the installed
   `opentelemetry-exporter-otlp==1.44.0`, which pins both to 1.44.0. None of
   these changes was made or is needed for the demonstrated HTTP app build.

The upstream build-system torch 2.13 requirement is distinct from the active
installed empty-platform wheel and Ascend torch 2.10 runtime. Do not rebuild
upstream with default build isolation as an incidental HTTP dependency fix.

## Reproduction without a model or listening socket

From the repository root:

```bash
../.venv/bin/python -m pip check
../.venv/bin/python - <<'PY'
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.api_server import build_app
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser
import torch

parser = make_arg_parser(FlexibleArgumentParser())
args = parser.parse_args([
    '--model', '/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32',
    '--tensor-parallel-size', '8',
])
engine_args = EngineArgs.from_cli_args(args)
app = build_app(args, supported_tasks=('generate',))
schema = app.openapi()
assert '/v1/chat/completions' in schema['paths']
assert '/v1/completions' in schema['paths']
assert engine_args.tensor_parallel_size == 8
assert not torch.npu.is_initialized()
print(len(app.routes), len(schema['paths']))
PY
```

The app has no initialized engine state. Calling model-dependent endpoints,
starting lifespan/server loops, actual network serving, streaming responses
and request-time model execution remain separate acceptance steps.
