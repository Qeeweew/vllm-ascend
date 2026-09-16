# SPDX-License-Identifier: Apache-2.0
"""Real 40-layer TP8 DSpark serving benchmark; CPU preparation unless --run.

Example (use a fresh output directory):
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HCCL_DETERMINISTIC=strict \
    OMP_NUM_THREADS=4 ../.venv/bin/python benchmarks/deepseek_v41/bench_full_dspark.py \
    --run --output /tmp/v41-full-dspark-bench-r1

Uses standard production workers and installed native operators. No fixture,
worker instrumentation, or autoregressive fallback. Random-token throughput
is not natural-language quality or representative-text acceptance evidence.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from preflight_full_model import NUMA_NODES, build_preflight
from prometheus_client.parser import text_string_to_metric_families
from smoke_http_serving import audit_cleanup_log, cleanup_server, json_request, validate_server_args

GIB = 1024**3
SUCCESS = "vllm:request_success_total"
DRAFTS = "vllm:spec_decode_num_drafts_total"
DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
PER_POSITION = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


def serve_values(args, name):
    query_tokens = 1 if args.disable_dspark else args.dspark_tokens + 1
    additional = {
        "enable_w4a16_decode": args.native_decode,
        "enable_v41_rope": args.fused_rope,
        "enable_v41_cache_store": args.fused_cache_store,
        "enable_v41_router": args.fused_router,
        "engram_numa_nodes": list(NUMA_NODES),
    }
    compilation = {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [query_tokens * batch for batch in range(1, 9)],
        "cudagraph_mm_encoder": False,
        "compile_mm_encoder": False,
    }
    values = [
        str(args.checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        name,
        "--tokenizer",
        str(args.checkpoint),
        "--tensor-parallel-size",
        "8",
        "--dtype",
        "bfloat16",
        "--load-format",
        "safetensors",
        "--safetensors-load-strategy",
        "lazy",
        "--additional-config",
        json.dumps(additional),
        "--compilation-config",
        json.dumps(compilation),
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        str(args.chunk_size),
        "--max-num-seqs",
        "8",
        "--block-size",
        "32",
        "--kv-cache-memory-bytes",
        str(int(args.kv_gib * GIB)),
        "--gpu-memory-utilization",
        "0.9",
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--disable-chunked-mm-input",
        "--limit-mm-per-prompt",
        '{"image": 0}',
        "--skip-mm-profiling",
        "--mm-encoder-tp-mode",
        "weights",
        "--shutdown-timeout",
        "180",
    ]
    if not args.disable_dspark:
        values += [
            "--speculative-config",
            json.dumps({"method": "dspark", "num_speculative_tokens": args.dspark_tokens}),
        ]
    if args.profile_after_bench or args.profile_warmup:
        values += [
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str(args.output / "traces"),
                    "torch_profiler_with_stack": False,
                    "ignore_frontend": True,
                    "max_iterations": 24,
                    "delay_iterations": 0,
                }
            ),
        ]
    return values


def bench_command(args, name, input_len, concurrency, count, filename):
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        name,
        "--tokenizer",
        str(args.checkpoint),
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        "128",
        "--random-range-ratio",
        "0",
        "--random-prefix-len",
        "0",
        "--num-prompts",
        str(count),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--seed",
        "41",
        "--num-warmups",
        "0",
        "--ready-check-timeout-sec",
        "0",
        "--extra-body",
        '{"temperature": 0}',
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(args.output),
        "--result-filename",
        filename,
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,95,99",
    ]


def parse_metrics(raw, name):
    result = {}
    for family in text_string_to_metric_families(raw):
        for sample in family.samples:
            if sample.labels.get("model_name") != name:
                continue
            if sample.name not in {SUCCESS, DRAFTS, DRAFT_TOKENS, ACCEPTED, PER_POSITION}:
                continue
            key = sample.name
            if sample.name == PER_POSITION:
                key += ":" + sample.labels["position"]
            result[key] = result.get(key, 0.0) + sample.value
    return result


def snapshot(opener, base, name):
    with opener.open(base + "/metrics", timeout=15) as response:
        raw = response.read(32 * 1024**2).decode()
    return raw, parse_metrics(raw, name)


def settled_metrics(opener, base, name, expected_success, process):
    # Log-stat propagation may lag HTTP completion. Wait for request accounting,
    # then a further full logging interval before accepting the snapshot.
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited while waiting for metrics")
        raw, values = snapshot(opener, base, name)
        if values.get(SUCCESS, 0) >= expected_success:
            time.sleep(12)
            return snapshot(opener, base, name)
        time.sleep(1)
    raise TimeoutError(f"Metrics did not account for {expected_success} completed requests")


def metric_delta(before, after, count, *, dspark_enabled=True):
    delta = {key: after.get(key, 0) - before.get(key, 0) for key in before.keys() | after.keys()}
    if any(value < 0 for value in delta.values()):
        raise ValueError("Metrics reset during benchmark")
    if delta.get(SUCCESS) != count:
        raise ValueError(f"Expected exactly {count} measured requests, got {delta.get(SUCCESS)}")
    drafts, tokens, accepted = (delta.get(key, 0) for key in (DRAFTS, DRAFT_TOKENS, ACCEPTED))
    if not dspark_enabled:
        if any((drafts, tokens, accepted)):
            raise ValueError("Speculative decoding ran despite explicit autoregressive benchmark mode")
        return {"counters": delta, "speculative_decoding": False}
    if drafts <= 0 or tokens <= 0 or not 0 <= accepted <= tokens:
        raise ValueError("Missing or invalid actual DSpark execution counters; no fallback permitted")
    return {
        "counters": delta,
        "draft_token_acceptance_rate": accepted / tokens,
        "accepted_tokens_per_draft": accepted / drafts,
        "mean_acceptance_length_including_bonus": 1 + accepted / drafts,
        "acceptance_by_position": {
            key.removeprefix(PER_POSITION + ":"): value / drafts
            for key, value in delta.items()
            if key.startswith(PER_POSITION + ":")
        },
    }


def wait_ready(opener, base, args, process, name):
    deadline = time.monotonic() + args.startup_timeout
    notice = 0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited during startup: {process.returncode}")
        try:
            json_request(opener, base, "/health", 2)
            models = json_request(opener, base, "/v1/models", 15)["body"]
            if name not in [item["id"] for item in models["data"]]:
                raise ValueError("Unexpected server model identity")
            return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        if time.monotonic() >= notice:
            print(json.dumps({"event": "waiting_for_server", "pid": process.pid}), flush=True)
            notice = time.monotonic() + 30
        time.sleep(1)
    raise TimeoutError("Server startup deadline exceeded")


def run_bench(command, output, env, count):
    with output.open("x") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
    result = json.loads(
        Path(command[command.index("--result-dir") + 1], command[command.index("--result-filename") + 1]).read_text()
    )
    if result.get("completed") != count or result.get("total_output_tokens") != count * 128:
        raise ValueError("vllm bench did not complete every request with 128 output tokens")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--port", type=int, default=18141)
    parser.add_argument("--dspark-tokens", type=int, choices=range(1, 9), default=5)
    parser.add_argument("--disable-dspark", action="store_true", help="Explicit autoregressive comparison mode")
    parser.add_argument("--profile-after-bench", action="store_true", help="Collect separate bounded torch NPU traces")
    parser.add_argument("--profile-warmup", action="store_true", help="Profile warmup before measurement")
    parser.add_argument("--input-lengths", nargs="+", type=int, choices=(128, 1024), default=[128, 1024])
    parser.add_argument("--concurrencies", nargs="+", type=int, choices=(1, 4, 8), default=[1, 4, 8])
    parser.add_argument("--num-prompts", type=int, choices=(16, 32), default=16)
    parser.add_argument("--kv-gib", type=float, default=2)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument(
        "--native-decode", action="store_true", help="Enable optimized W4 decode; default uses CANN W4A16"
    )
    parser.add_argument("--fused-rope", action="store_true")
    parser.add_argument("--fused-cache-store", action="store_true")
    parser.add_argument("--fused-router", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    if not 1024 <= args.port <= 65535 or min(args.kv_gib, args.chunk_size, args.startup_timeout) <= 0:
        parser.error("Invalid port, KV budget, chunk size or timeout")
    args.output = args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output.mkdir(parents=True)
    name = ("v41-full-ar-" if args.disable_dspark else "v41-full-dspark-") + uuid.uuid4().hex[:12]
    values = serve_values(args, name)
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", *values]
    report = {
        "status": "prepared_only",
        "model_name": name,
        "command": command,
        "cli_validation": validate_server_args(values),
        "preflight": build_preflight(
            args.source,
            args.checkpoint,
            query_hbm=args.run,
            reserve_gib=11 + args.kv_gib,
            include_dspark=not args.disable_dspark,
        ),
        "num_speculative_tokens": 0 if args.disable_dspark else args.dspark_tokens,
        "measurement": "Real full-model random-token HTTP throughput; not natural-text quality or speedup",
        "graph_validation": "Graph configuration required; actual replay validation is a separate full-model audit",
        "cases": [],
    }
    for input_len in args.input_lengths:
        for concurrency in args.concurrencies:
            stem = f"input{input_len}-output128-c{concurrency}"
            report["cases"].append(
                {
                    "name": stem,
                    "input_tokens": input_len,
                    "output_tokens": 128,
                    "concurrency": concurrency,
                    "warmup_command": bench_command(
                        args, name, input_len, concurrency, concurrency, stem + "-warmup.json"
                    ),
                    "command": bench_command(args, name, input_len, concurrency, args.num_prompts, stem + ".json"),
                }
            )
    path = args.output / "report.json"
    if args.profile_warmup:
        for case in report["cases"]:
            case["warmup_command"].append("--profile")
            case["warmup_profiled"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")
    if not args.run:
        print(json.dumps({"status": report["status"], "report": str(path)}), flush=True)
        return 0
    process = None
    log_path = args.output / "server.log"
    try:
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3,4,5,6,7":
            raise ValueError("Require physical devices 0..7 in order for real Engram NUMA mapping")
        if os.environ.get("HCCL_DETERMINISTIC") != "strict":
            raise ValueError("Require HCCL_DETERMINISTIC=strict")
        if report["preflight"]["status"] != "ready_for_scheduled_launch":
            raise ValueError("Full-model capacity/conversion preflight is not ready")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                raise ValueError("Port already occupied; refusing to reuse another server")
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"] = "180"
        env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        base = f"http://127.0.0.1:{args.port}"
        with log_path.open("x") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        report["server_pid"] = process.pid
        wait_ready(opener, base, args, process, name)
        successes = 0
        for case in report["cases"]:
            stem = case["name"]
            run_bench(case["warmup_command"], args.output / (stem + "-warmup.log"), env, case["concurrency"])
            successes += case["concurrency"]
            raw, before = settled_metrics(opener, base, name, successes, process)
            (args.output / (stem + "-metrics-before.txt")).write_text(raw)
            case["benchmark"] = run_bench(case["command"], args.output / (stem + ".log"), env, args.num_prompts)
            successes += args.num_prompts
            raw, after = settled_metrics(opener, base, name, successes, process)
            (args.output / (stem + "-metrics-after.txt")).write_text(raw)
            case["speculative_metrics"] = metric_delta(
                before, after, args.num_prompts, dspark_enabled=not args.disable_dspark
            )
            case["status"] = "passed"
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"event": "case_complete", "name": stem, **case["speculative_metrics"]}), flush=True)
        if args.profile_after_bench:
            report["profiles"] = []
            for input_len, concurrency in ((128, 1), (128, 8), (1024, 1), (1024, 8)):
                stem = f"profile-input{input_len}-c{concurrency}"
                profile_command = bench_command(args, name, input_len, concurrency, concurrency, stem + ".json")
                profile_command.append("--profile")
                result = run_bench(profile_command, args.output / (stem + ".log"), env, concurrency)
                report["profiles"].append({"name": stem, "command": profile_command, "benchmark": result})
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"event": "profile_complete", "name": stem}), flush=True)
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if process is not None:
            report["cleanup"] = cleanup_server(process, 240)
            report["cleanup"]["log_audit"] = audit_cleanup_log(log_path)
            if not report["cleanup"]["clean"] or not report["cleanup"]["log_audit"]["clean"]:
                report["status"] = "failed_cleanup"
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(path)}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
