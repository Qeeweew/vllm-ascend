# SPDX-License-Identifier: Apache-2.0
"""Small real-checkpoint chat/long-context/image smoke, not a quality benchmark.

Default mode tokenizes with the official V4.1 chat encoder on CPU. Explicit
--run requires a free TP8 window. Save outputs even if an answer check fails.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

from preflight_full_model import NUMA_NODES, build_preflight


def cases():
    padding = "背景记录：今天气温适宜，仓库正常开放，设备运行平稳。\n"
    retrieval = (
        "阅读以下记录，找出唯一的核验码。最终只输出核验码，不要解释。\n"
        + padding * 140
        + "唯一核验码是 NPU-7319。\n"
        + padding * 140
        + "记录结束。核验码是什么？"
    )
    return [
        {"name": "arithmetic", "prompt": "17 + 25 等于多少？只输出十进制整数。", "expected": "42"},
        {"name": "chinese", "prompt": "中国的首都是哪座城市？只输出城市名。", "expected": "北京"},
        {
            "name": "english_extraction",
            "prompt": "The booking code is Q7M2. Return only the booking code, without punctuation.",
            "expected": "Q7M2",
        },
        {
            "name": "json_sort",
            "prompt": "Sort the numbers 9, 1, 4 in ascending order. Output only a JSON array.",
            "expected_json": [1, 4, 9],
        },
        {"name": "long_retrieval", "prompt": retrieval, "expected": "NPU-7319"},
    ]


def answer_matches(case, text):
    if "expected_json" in case:
        try:
            return json.loads(text.strip()) == case["expected_json"]
        except ValueError:
            return False
    return text.strip() == case["expected"]


def check_native_dispatch(final, *, enabled, name, graph):
    """Require replay of the same concrete graph that captured the native op."""
    counts = final["operator_dispatch"]
    calls = counts.get(f"{name}_eager", 0) + counts.get(f"{name}_capture", 0)
    if not enabled:
        assert calls == 0, name
        return
    assert calls > 0, name
    if graph:
        assert any(
            entry["request_replays"] > 0 and entry["captured_native_ops"].get(name, 0) > 0
            for entry in final["graph_dispatch"]
        ), f"{name}: no request replay of its captured graph entry"


def execute(args, report):
    """Run the production model, record every answer and check resource cleanup."""
    from vllm import LLM, SamplingParams

    llm = None
    with_vision = args.image is not None
    photo = None
    if with_vision:
        from smoke_mm_runner_tp8 import load_photo

        photo, _ = load_photo(args.image)
    try:
        llm = LLM(
            model=str(args.checkpoint),
            tokenizer=str(args.checkpoint),
            tensor_parallel_size=8,
            dtype="bfloat16",
            worker_cls="full_model_worker.V41FullModelWorker",
            load_format="safetensors",
            safetensors_load_strategy="lazy",
            additional_config={
                "enable_w4a16_decode": args.native_decode,
                "enable_v41_rope": args.fused_rope,
                "enable_v41_cache_store": args.fused_cache_store,
                "enable_v41_router": args.fused_router,
                "engram_numa_nodes": list(NUMA_NODES),
            },
            enforce_eager=not args.graph,
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY" if args.graph else "NONE",
                "cudagraph_mm_encoder": False,
                "compile_mm_encoder": False,
            },
            max_model_len=8192,
            max_num_batched_tokens=1024 if with_vision else 128,
            max_num_seqs=1,
            block_size=32,
            kv_cache_memory_bytes=512 * 1024**2,
            gpu_memory_utilization=0.9,
            async_scheduling=False,
            enable_prefix_caching=False,
            disable_chunked_mm_input=True,
            limit_mm_per_prompt={"image": int(with_vision)},
            skip_mm_profiling=True,
            mm_encoder_tp_mode="weights",
            shutdown_timeout=180,
        )
        report["workers_loaded"] = llm.collective_rpc("start_full_model_audit", args=(str(args.source), with_vision))
        params = SamplingParams(temperature=0, max_tokens=32, logprobs=1)
        report["answers"] = []
        for case in report["cases"]:
            started = time.perf_counter()
            prompt = {"prompt_token_ids": case["prompt_token_ids"]}
            if case.get("with_image"):
                prompt["multi_modal_data"] = {"image": photo}
            output = llm.generate([prompt], params, use_tqdm=False)[0]
            answer = output.outputs[0]
            selected = [row[token].logprob for token, row in zip(answer.token_ids, answer.logprobs, strict=True)]
            report["answers"].append(
                {
                    "name": case["name"],
                    "text": answer.text,
                    "token_ids": list(answer.token_ids),
                    "selected_logprobs": selected,
                    "finite": bool(selected) and all(math.isfinite(value) for value in selected),
                    "answer_matches": answer_matches(case, answer.text),
                    "finish_reason": answer.finish_reason,
                    "elapsed_seconds_diagnostic_only": time.perf_counter() - started,
                }
            )
            print(json.dumps({"event": "text_case_complete", **report["answers"][-1]}, ensure_ascii=False), flush=True)
        report["workers_final"] = llm.collective_rpc("inspect_full_model_audit")
        for loaded, final in zip(report["workers_loaded"], report["workers_final"], strict=True):
            initial = loaded["state"]
            assert initial["rows"] == final["rows"] and initial["mask"] == final["mask"]
            assert not final["prepared"] and final["offload_steps"] > initial["offload_steps"]
            assert initial["image_mask"] == final["image_mask"]
            assert final["encoder_spans"] == ([189] if with_vision else [])
            if args.graph:
                assert final["graph_replays"] > 0
            for enabled, name in (
                (args.native_decode, "w4_native"),
                (args.fused_rope, "v41_rope"),
                (args.fused_cache_store, "v41_main_cache_store"),
                (args.fused_cache_store, "v41_index_cache_store"),
                (args.fused_router, "v41_moe_router"),
            ):
                check_native_dispatch(final, enabled=enabled, name=name, graph=args.graph)
        report["checks_passed"] = all(answer["finite"] and answer["answer_matches"] for answer in report["answers"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if llm is not None:
            try:
                report["workers_released"] = llm.collective_rpc("finish_full_model_audit")
                assert sum(worker["checked_unregisters"] for worker in report["workers_released"]) == 16
            except Exception as error:
                report["cleanup_error"] = f"{type(error).__name__}: {error}"
            try:
                client = llm.llm_engine.engine_core
                processes = list(client.resources.engine_manager.processes)
                client.shutdown(timeout=240.0)
                report["engine_shutdown"] = [
                    {"pid": process.pid, "exitcode": process.exitcode, "alive": process.is_alive()}
                    for process in processes
                ]
                assert all(not item["alive"] and item["exitcode"] == 0 for item in report["engine_shutdown"])
            except Exception as error:
                report["cleanup_error"] = f"{type(error).__name__}: {error}"
    report["status"] = "passed" if report.get("checks_passed") and "cleanup_error" not in report else "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--native-decode", action="store_true")
    parser.add_argument("--fused-rope", action="store_true")
    parser.add_argument("--fused-cache-store", action="store_true")
    parser.add_argument("--fused-router", action="store_true")
    parser.add_argument("--image", type=Path, help="Optional fixed hato.jpg fixture; runs real vision before text")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new result path")
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(str(args.checkpoint), tokenizer_mode="deepseek_v41")
    prepared = cases()
    photo_info = None
    if args.image is not None:
        from smoke_mm_runner_tp8 import load_photo

        _, photo_info = load_photo(args.image)
        if photo_info["source_sha256"] != "8f7e776cf614298af55cb64b7116a513c37f8710959fb90a5e2babedece489b4":
            parser.error("The fixed image oracle requires the original hato.jpg fixture")
        prepared.insert(
            0,
            {
                "name": "image_animal",
                "prompt": [
                    {"type": "image_url", "image_url": {"url": "fixture:hato.jpg"}},
                    {
                        "type": "text",
                        "text": "Which animal is closest to the camera? Answer with one lowercase English word.",
                    },
                ],
                "expected": "pigeon",
                "with_image": True,
            },
        )
    for case in prepared:
        case["prompt_token_ids"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": case["prompt"]}], thinking=False, tokenize=True
        )
        case["prompt_tokens"] = len(case["prompt_token_ids"])
        assert case["prompt_tokens"] + 32 <= 8192
    assert prepared[-1]["prompt_tokens"] > 4096
    report = {
        "status": "prepared_only",
        "cases": prepared,
        # Add 1 GiB over the short-run admission floor for the larger KV pool
        # and long-context workspace. Actual peaks still require observation.
        "preflight": build_preflight(args.source, args.checkpoint, reserve_gib=11 if args.image else 9),
        "photo": photo_info,
        "graph": args.graph,
        "native_decode": args.native_decode,
        "fused_rope": args.fused_rope,
        "fused_cache_store": args.fused_cache_store,
        "fused_router": args.fused_router,
        "thinking": False,
        "quality_benchmark": False,
        "performance_benchmark": False,
    }
    if args.run:
        if os.environ.get("HCCL_DETERMINISTIC") != "strict":
            parser.error("Require HCCL_DETERMINISTIC=strict")
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3,4,5,6,7":
            parser.error("Require physical devices 0–7 in order for the NUMA mapping")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "180")
        if report["preflight"]["status"] != "ready_for_scheduled_launch":
            report["status"] = "blocked_before_launch"
        else:
            execute(args, report)
    else:
        import torch

        assert not torch.npu.is_initialized()
        report["npu_initialized"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)
    return 0 if report["status"] in {"prepared_only", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
