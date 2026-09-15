# SPDX-License-Identifier: Apache-2.0
"""Bounded localhost OpenAI smoke through standard vllm serve and production registry.

Default mode validates CLI arguments and prepares a report without starting
a server or initializing NPU. --run requires the separately scheduled TP8
window. The supplied fixture has three real device-weight layers and small
synthetic Engram tables; outputs are execution checks, not quality evidence.
"""

import argparse
import base64
import io
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import psutil

MAX_RESPONSE_BYTES = 4 * 1024**2
MAX_NEW_TOKENS = 4
DEFAULT_CHECKPOINT = Path("/tmp/v41-mm-production-numa-graph-r1")


def fixture_info(checkpoint):
    config = json.loads((checkpoint / "config.json").read_text())
    text = config.get("text_config", config)
    if config.get("architectures") != ["DeepseekV41ForCausalLM"]:
        raise ValueError("The fixture must use the production DeepseekV41ForCausalLM registry entry")
    if text.get("num_hidden_layers") != 3 or text.get("n_routed_experts") != 384:
        raise ValueError("This smoke is bounded to the existing three-layer, 384-expert fixture")
    if text.get("engram_layer_ids") != [1] or text.get("engram_num_embeddings") != [4096]:
        raise ValueError("This smoke requires the explicitly synthetic small Engram fixture")
    if not (checkpoint / "model.safetensors.index.json").is_file():
        raise ValueError("The fixture index is missing")
    return {
        "checkpoint": str(checkpoint),
        "language_layers": 3,
        "experts": 384,
        "engram_layer_ids": [1],
        "engram_rows": [4096],
        "synthetic_engram": True,
        "quality_validation": False,
        "performance_measurement": False,
        "production_registry": True,
        "test_worker_or_registration": False,
    }


def serve_args(args, model_name):
    values = [
        str(args.checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        model_name,
        "--tokenizer",
        str(args.tokenizer),
        "--tensor-parallel-size",
        "8",
        "--dtype",
        "bfloat16",
        "--load-format",
        "safetensors",
        "--safetensors-load-strategy",
        "lazy",
        "--shutdown-timeout",
        str(args.engine_shutdown_timeout),
        "--max-model-len",
        "512",
        "--max-num-batched-tokens",
        "1024",
        "--max-num-seqs",
        "1",
        "--block-size",
        "32",
        "--gpu-memory-utilization",
        "0.2",
        "--kv-cache-memory-bytes",
        str(256 * 1024**2),
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--disable-chunked-mm-input",
        "--limit-mm-per-prompt",
        json.dumps({"image": int(args.image is not None)}),
        "--skip-mm-profiling",
        "--mm-encoder-tp-mode",
        "weights",
        "--additional-config",
        json.dumps({"enable_w4a16_decode": True, "engram_numa_nodes": args.engram_numa_nodes}),
        "--compilation-config",
        json.dumps(
            {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY" if args.graph else "NONE",
                "cudagraph_mm_encoder": False,
                "compile_mm_encoder": False,
            }
        ),
    ]
    if not args.graph:
        values.append("--enforce-eager")
    if args.profile_dir is not None:
        values.extend(
            [
                "--profiler-config",
                json.dumps(
                    {
                        "profiler": "torch",
                        "torch_profiler_dir": str(args.profile_dir.resolve()),
                        "torch_profiler_with_stack": False,
                        "ignore_frontend": True,
                    }
                ),
            ]
        )
    return values


def validate_server_args(values):
    # Parse the actual installed CLI only. Never call create_engine_config,
    # run_server, get_model or any registration/testing hook here.
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parsed = make_arg_parser(FlexibleArgumentParser()).parse_args(values)
    parsed.model = parsed.model_tag
    validate_parsed_serve_args(parsed)
    engine = EngineArgs.from_cli_args(parsed)
    if torch.npu.is_initialized():
        raise RuntimeError("CLI preparation unexpectedly initialized NPU")
    return {
        "model": engine.model,
        "tensor_parallel_size": engine.tensor_parallel_size,
        "host": parsed.host,
        "port": parsed.port,
        "npu_initialized": False,
    }


def group_members(group_id):
    """Return only the fresh session/group created by our Popen call."""
    result = []
    for process in psutil.process_iter(["pid", "status", "create_time"]):
        try:
            pid = process.info["pid"]
            if (
                process.info["status"] != psutil.STATUS_ZOMBIE
                and os.getpgid(pid) == group_id
                and os.getsid(pid) == group_id
            ):
                result.append({"pid": pid, "created": process.info["create_time"]})
        except (ProcessLookupError, PermissionError, psutil.NoSuchProcess):
            continue
    return result


def cleanup_server(process, grace_seconds):
    """Graceful parent shutdown first; escalation targets only our new group."""
    record = {"pid": process.pid, "signals": [], "forced": False}

    def wait_empty(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            process.poll()
            if not group_members(process.pid):
                return True
            time.sleep(0.2)
        return False

    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        record["signals"].append("parent_SIGINT")
    if not wait_empty(grace_seconds):
        for sig, seconds in ((signal.SIGTERM, 15), (signal.SIGKILL, 10)):
            if not group_members(process.pid):
                break
            record["forced"] = True
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            record["signals"].append(f"owned_group_{sig.name}")
            if wait_empty(seconds):
                break
    record["survivors"] = group_members(process.pid)
    try:
        record["returncode"] = process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        record["returncode"] = None
    record["clean"] = not record["survivors"] and not record["forced"] and record["returncode"] == 0
    return record


def audit_cleanup_log(log_path):
    """Parent exit zero cannot hide an internal engine/worker force kill."""
    lines = log_path.read_text(errors="replace").splitlines()
    forced = [
        line
        for line in lines
        if any(
            marker in line
            for marker in (
                "force killing remaining",
                "sending SIGTERM count=",
                "sending SIGKILL count=",
            )
        )
    ]
    leaks = [line for line in lines if "resource_tracker:" in line and "leaked" in line]
    return {"internal_forced_lines": forced, "resource_leak_warnings": leaks, "clean": not forced and not leaks}


def json_request(opener, base, path, timeout, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with opener.open(request, timeout=timeout) as response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError("HTTP response exceeded the bounded smoke size")
        value = json.loads(body) if body else None
        return {"status": response.status, "seconds": time.monotonic() - start, "body": value}


def sse_events(lines):
    """Parse UTF-8 SSE data frames, including comments and multiline data."""
    data = []
    size = 0
    for raw in lines:
        size += len(raw)
        if size > MAX_RESPONSE_BYTES:
            raise ValueError("SSE response exceeded the bounded smoke size")
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        raise ValueError("SSE ended in an incomplete event")


def stream_completion(opener, base, timeout, payload):
    payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    start = time.monotonic()
    chunks, token_ids, text_parts, finish_reasons, usage = [], [], [], [], None
    done = False
    with opener.open(request, timeout=timeout) as response:
        if response.status != 200 or "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise ValueError("Streaming endpoint did not return HTTP 200 SSE")
        for event in sse_events(response):
            if time.monotonic() - start > timeout:
                raise TimeoutError("SSE smoke exceeded its request deadline")
            if event == "[DONE]":
                done = True
                break
            chunk = json.loads(event)
            if "error" in chunk:
                raise ValueError(f"SSE server error: {chunk['error']}")
            chunks.append(chunk)
            for choice in chunk.get("choices", []):
                token_ids.extend(choice.get("token_ids") or [])
                text_parts.append(choice.get("text", ""))
                if choice.get("finish_reason") is not None:
                    finish_reasons.append(choice["finish_reason"])
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
    if not done or not chunks or not finish_reasons or usage is None:
        raise ValueError("SSE must include data, finish_reason, final usage, and [DONE]")
    if usage.get("completion_tokens") != MAX_NEW_TOKENS or len(token_ids) != MAX_NEW_TOKENS:
        raise ValueError("SSE did not return exactly four generated tokens")
    return {
        "status": 200,
        "seconds": time.monotonic() - start,
        "done": done,
        "chunks": chunks,
        "token_ids": token_ids,
        "text": "".join(text_parts),
        "finish_reasons": finish_reasons,
        "usage": usage,
    }


def validate_completion(result):
    if result["status"] != 200:
        raise ValueError("Completion did not return HTTP 200")
    body = result["body"]
    if body.get("usage", {}).get("completion_tokens") != MAX_NEW_TOKENS or len(body.get("choices", [])) != 1:
        raise ValueError("Completion must contain one choice and four generated tokens")
    choice = body["choices"][0]
    if len(choice.get("token_ids") or []) != MAX_NEW_TOKENS or choice.get("finish_reason") != "length":
        raise ValueError("Completion token IDs or finish reason are missing")
    logprobs = choice.get("logprobs", {}).get("token_logprobs", [])
    if len(logprobs) != MAX_NEW_TOKENS or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in logprobs):
        raise ValueError("Completion selected-token logprobs must be finite")
    return choice


def image_request(path, model_name):
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
    data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    return {
        "model": model_name,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0,
        "ignore_eos": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }


def run_requests(args, process, model_name, report):
    base = f"http://127.0.0.1:{args.port}"
    # Ignore inherited external HTTP proxies for every localhost request.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + args.startup_timeout
    next_notice = 0.0
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited before health readiness: {process.returncode}")
        try:
            health = json_request(opener, base, "/health", 2)
            if health["status"] == 200:
                report["health"] = health
                break
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = str(error)
        if time.monotonic() >= next_notice:
            print(json.dumps({"event": "waiting_for_http_health", "pid": process.pid}), flush=True)
            next_notice = time.monotonic() + 30
        time.sleep(0.5)
    else:
        raise TimeoutError(f"HTTP startup exceeded {args.startup_timeout}s: {last_error}")
    models = json_request(opener, base, "/v1/models", args.request_timeout)
    if models["status"] != 200 or model_name not in [item["id"] for item in models["body"].get("data", [])]:
        raise ValueError("/v1/models does not contain this invocation's unique served model name")
    report["models"] = models
    payload = {
        "model": model_name,
        "prompt": [100, 129264, 101],
        "temperature": 0,
        "max_tokens": MAX_NEW_TOKENS,
        "ignore_eos": True,
        "logprobs": 1,
        "return_token_ids": True,
    }
    report["text_completion"] = json_request(opener, base, "/v1/completions", args.request_timeout, payload)
    choice = validate_completion(report["text_completion"])
    report["sse_completion"] = stream_completion(opener, base, args.request_timeout, payload)
    if (
        report["sse_completion"]["token_ids"] != choice["token_ids"]
        or report["sse_completion"]["text"] != choice["text"]
    ):
        raise ValueError("Deterministic SSE and non-streaming completions differ")
    if args.image is not None:
        report["image_completion"] = json_request(
            opener, base, "/v1/chat/completions", args.request_timeout, image_request(args.image, model_name)
        )
        body = report["image_completion"]["body"]
        if body.get("usage", {}).get("completion_tokens") != MAX_NEW_TOKENS or len(body.get("choices", [])) != 1:
            raise ValueError("Image chat completion must return one choice and four generated tokens")
        if body["choices"][0].get("finish_reason") != "length":
            raise ValueError("Image completion ended unexpectedly")
    report["health_after_requests"] = json_request(opener, base, "/health", args.request_timeout)
    report["http_checks_passed"] = True
    if args.profile_dir is not None:
        run_profile_round(args, opener, base, model_name, payload, report)


def run_profile_round(args, opener, base, model_name, text_payload, report):
    """One bounded profiled text/image round after the unprofiled HTTP checks."""
    profile = {
        "scope": "three_layer_functional_timeline_not_final_performance",
        "directory": str(args.profile_dir.resolve()),
        "initial_unprofiled_text_seconds": report["text_completion"]["seconds"],
        "initial_unprofiled_image_seconds": report.get("image_completion", {}).get("seconds"),
        "timing_caveat": "Single-request wall observations are not a performance gate or profiler overhead estimate",
        "cache_caveat": "Repeated image may use warm processor/encoder caches; inspect actual trace coverage",
    }
    report["profile"] = profile
    warm = {}
    profile["warm_unprofiled"] = warm
    warm_start = time.monotonic()
    warm["text_completion"] = json_request(opener, base, "/v1/completions", args.request_timeout, text_payload)
    warm_choice = validate_completion(warm["text_completion"])
    if warm_choice["token_ids"] != report["text_completion"]["body"]["choices"][0]["token_ids"]:
        raise ValueError("Warm and initial deterministic text token IDs differ")
    if args.image is not None:
        warm["image_completion"] = json_request(
            opener, base, "/v1/chat/completions", args.request_timeout, image_request(args.image, model_name)
        )
        body = warm["image_completion"]["body"]
        if (
            body.get("usage", {}).get("completion_tokens") != MAX_NEW_TOKENS
            or len(body.get("choices", [])) != 1
            or body["choices"][0].get("finish_reason") != "length"
        ):
            raise ValueError("Warm image completion must return four generated tokens and length finish")
    warm["request_round_wall_seconds"] = time.monotonic() - warm_start
    profile["start"] = json_request(opener, base, "/start_profile", args.request_timeout, {})
    start = time.monotonic()
    request_error = None
    try:
        profile["text_completion"] = json_request(opener, base, "/v1/completions", args.request_timeout, text_payload)
        choice = validate_completion(profile["text_completion"])
        if choice["token_ids"] != report["text_completion"]["body"]["choices"][0]["token_ids"]:
            raise ValueError("Profiled and unprofiled deterministic text token IDs differ")
        if args.image is not None:
            profile["image_completion"] = json_request(
                opener, base, "/v1/chat/completions", args.request_timeout, image_request(args.image, model_name)
            )
            body = profile["image_completion"]["body"]
            if body.get("usage", {}).get("completion_tokens") != MAX_NEW_TOKENS or len(body.get("choices", [])) != 1:
                raise ValueError("Profiled image completion must return four generated tokens")
            if body["choices"][0].get("finish_reason") != "length":
                raise ValueError("Profiled image completion ended unexpectedly")
        profile["request_round_wall_seconds"] = time.monotonic() - start
    except Exception as error:
        request_error = error
        profile["request_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # Stop/export even when the profiled request fails; preserve errors.
        try:
            profile["stop"] = json_request(opener, base, "/stop_profile", args.profile_stop_timeout, {})
        except Exception as error:
            profile["stop_error"] = f"{type(error).__name__}: {error}"
            if request_error is None:
                raise
    artifacts = [
        {"path": str(path.relative_to(args.profile_dir)), "bytes": path.stat().st_size}
        for path in sorted(args.profile_dir.rglob("*"))
        if path.is_file()
    ]
    profile["artifacts"] = artifacts
    timelines = [
        item
        for item in artifacts
        if item["bytes"] > 0
        and (
            Path(item["path"]).name == "trace_view.json"
            or item["path"].endswith((".pt.trace.json", ".pt.trace.json.gz"))
        )
    ]
    profile["nonempty_timeline_count"] = len(timelines)
    if len(timelines) < 8:
        raise ValueError(f"TP8 profiling requires eight nonempty timelines; found {len(timelines)}")
    profile["collection_passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--image", type=Path, help="Optional local photo for the chat endpoint")
    parser.add_argument("--port", type=int, default=18141)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--run", action="store_true", help="Launch TP8 only in the allocated device window")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--shutdown-timeout", type=float, default=90)
    parser.add_argument("--engine-shutdown-timeout", type=int, default=30)
    parser.add_argument("--profile-dir", type=Path, help="Optional new directory for one real Torch-NPU timeline round")
    parser.add_argument("--profile-stop-timeout", type=float, default=300)
    parser.add_argument("--engram-numa-nodes", type=int, nargs=8, default=[6, 7, 4, 5, 0, 1, 2, 3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or min(args.startup_timeout, args.request_timeout, args.shutdown_timeout) <= 0:
        parser.error("Use an unprivileged TCP port and positive timeouts")
    if args.engine_shutdown_timeout <= 0 or args.profile_stop_timeout <= 0:
        parser.error("Engine shutdown and profile export timeouts must be positive")
    if args.engine_shutdown_timeout >= args.shutdown_timeout:
        parser.error("Controller shutdown timeout must exceed the engine shutdown timeout")
    args.checkpoint, args.tokenizer = args.checkpoint.resolve(), args.tokenizer.resolve()
    if args.image is not None and not args.image.is_file():
        parser.error("The optional image must be an existing local file")
    if args.profile_dir is not None:
        args.profile_dir = args.profile_dir.resolve()
        if args.profile_dir.exists() and any(args.profile_dir.iterdir()):
            parser.error("Use a new or empty profile directory so prior traces cannot satisfy the gate")
    model_name = "v41-http-smoke-" + uuid.uuid4().hex[:12]
    values = serve_args(args, model_name)
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", *values]
    report = {
        "status": "prepared_only",
        "fixture": fixture_info(args.checkpoint),
        "command": command,
        "graph_requested": args.graph,
        "image_requested": args.image is not None,
        "cli_validation": validate_server_args(values),
        "model_name": model_name,
        "timeouts": {
            "startup": args.startup_timeout,
            "request": args.request_timeout,
            "shutdown": args.shutdown_timeout,
            "engine_shutdown": args.engine_shutdown_timeout,
            "profile_stop": args.profile_stop_timeout,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process = None
    if args.run:
        report["status"] = "running"
        log_path = args.output.with_suffix(".server.log")
        report["server_log"] = str(log_path.resolve())
        try:
            with socket.socket() as probe:
                probe.settimeout(1)
                if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                    raise RuntimeError(
                        "Requested localhost port is already occupied; no existing server will be reused"
                    )
            child_env = os.environ.copy()
            child_env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            child_env["VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"] = "60"
            with log_path.open("x") as log:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=child_env, start_new_session=True
                )
                report["server_pid"] = process.pid
                run_requests(args, process, model_name, report)
                report["all_requested_checks_passed"] = True
        except (Exception, KeyboardInterrupt) as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
        finally:
            if process is not None:
                try:
                    report["cleanup"] = cleanup_server(process, args.shutdown_timeout)
                    report["cleanup"]["log_audit"] = audit_cleanup_log(log_path)
                    report["cleanup"]["clean"] &= report["cleanup"]["log_audit"]["clean"]
                except Exception as error:
                    report["cleanup"] = {"clean": False, "error": f"{type(error).__name__}: {error}"}
            if report.get("all_requested_checks_passed"):
                report["status"] = "passed" if report.get("cleanup", {}).get("clean") else "failed_cleanup"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}), flush=True)
    return 0 if report["status"] in {"passed", "prepared_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
