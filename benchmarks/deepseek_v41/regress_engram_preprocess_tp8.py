# SPDX-License-Identifier: Apache-2.0
"""Repeat production-registry MM/text graphs using an existing three-layer fixture."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from smoke_http_serving import fixture_info
from smoke_mm_runner_tp8 import EXPECTED_IMAGE_TOKENS, IMAGE_TOKEN_ID, load_photo, selected_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("/tmp/v41-mm-production-numa-graph-r1"))
    parser.add_argument("--source", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-limit", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--cann-decode", action="store_true", help="Use existing CANN fallback for decode diagnosis")
    parser.add_argument("--rows-diagnostic", action="store_true", help="Extra synchronous D2H row-content diagnostics")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path; existing evidence must not be overwritten")
    fixture = fixture_info(args.checkpoint)
    fixture.pop("test_worker_or_registration")  # This harness uses an observation-only worker.
    photo, photo_info = load_photo(args.image)
    root = Path(__file__).resolve().parents[2]
    result = {
        "status": "prepared_only",
        "fixture": fixture,
        "photo": photo_info,
        "image_limit": args.image_limit,
        "engram_numa_nodes": [6, 7, 4, 5, 0, 1, 2, 3],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "runner_sha256": hashlib.sha256((root / "vllm_ascend/worker/model_runner_v1.py").read_bytes()).hexdigest(),
        "production_registry": True,
        "diagnostic_worker": (
            "smoke_mm_rows_worker.V41MMRowsSmokeWorker"
            if args.rows_diagnostic
            else "smoke_mm_runner_worker.V41MMSmokeWorker"
        ),
        "graph_mode": "FULL_DECODE_ONLY",
        "performance_measurement": False,
        "repeat_logprob_tolerance": 1e-4,
        "repeatability_gate_kind": "new_diagnostic_threshold_not_native_numerical_accuracy_acceptance",
        "native_decode": not args.cann_decode,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "HCCL_DETERMINISTIC",
                "ASCEND_LAUNCH_BLOCKING",
                "OMP_NUM_THREADS",
                "VLLM_WORKER_MULTIPROC_METHOD",
                "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
            )
        },
        "engram_row_content_observed": args.rows_diagnostic,
    }
    if args.run:
        from vllm import LLM, SamplingParams

        llm = None
        try:
            llm = LLM(
                model=str(args.checkpoint.resolve()),
                tokenizer=str(args.source.resolve()),
                dtype="bfloat16",
                tensor_parallel_size=8,
                worker_cls=result["diagnostic_worker"],
                additional_config={
                    "enable_w4a16_decode": not args.cann_decode,
                    "engram_numa_nodes": result["engram_numa_nodes"],
                },
                load_format="safetensors",
                safetensors_load_strategy="lazy",
                enforce_eager=False,
                compilation_config={
                    "mode": 0,
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_mm_encoder": False,
                    "compile_mm_encoder": False,
                },
                max_model_len=512,
                max_num_batched_tokens=1024,
                max_num_seqs=1,
                block_size=32,
                gpu_memory_utilization=0.2,
                kv_cache_memory_bytes=256 * 1024**2,
                async_scheduling=False,
                enable_prefix_caching=False,
                disable_chunked_mm_input=True,
                limit_mm_per_prompt={"image": args.image_limit},
                skip_mm_profiling=True,
                mm_encoder_tp_mode="weights",
            )
            result["workers_before"] = llm.collective_rpc("start_mm_smoke")
            sampling = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=1)
            prompts = {
                "literal_image_id_text": {"prompt_token_ids": [100, IMAGE_TOKEN_ID, 101]},
                "valid_zero_text": {"prompt_token_ids": [0, IMAGE_TOKEN_ID, 101]},
            }
            if args.image_limit:
                prompts["typed_image"] = {
                    "prompt_token_ids": [100, IMAGE_TOKEN_ID, 101],
                    "multi_modal_data": {"image": photo},
                }
            outputs = {name: [] for name in prompts}
            result["outputs"] = outputs
            for repeat in range(2):
                for name, prompt in prompts.items():
                    outputs[name].append(selected_outputs(llm.generate([prompt], sampling, use_tqdm=False))[0])
                    print(json.dumps({"event": "request_complete", "request": name, "repeat": repeat}), flush=True)
            final = llm.collective_rpc("inspect_mm_smoke")
            result["workers_final"] = final
            comparisons = {}
            result["repeat_comparisons"] = comparisons
            for name, (first, second) in outputs.items():
                delta = max(
                    abs(a - b) for a, b in zip(first["selected_logprobs"], second["selected_logprobs"], strict=True)
                )
                comparisons[name] = {
                    "token_ids_exact": first["token_ids"] == second["token_ids"],
                    "max_selected_logprob_abs_difference": delta,
                }
            for rank, (before, after) in enumerate(zip(result["workers_before"], final, strict=True)):
                assert before["rows"] == after["rows"] and before["mask"] == after["mask"]
                assert before["image_mask_ptr"] == after["image_mask_ptr"]
                assert not after["prepared"] and after["image_mask_sum"] == 0
                assert after["captured_graphs"] > 0 and after["graph_replays"] >= 3 * 2 * len(prompts)
                assert after["tower_allocated"] == after["supports_mm_inputs"] == bool(args.image_limit)
                assert all(
                    item["raw_ids"] and item["embeddings"] == bool(args.image_limit)
                    for item in after["forward_signatures"]
                )
                if args.cann_decode:
                    assert after["w4_dispatch"]["native_capture"] == 0
                    assert after["w4_dispatch"]["fallback_capture"] > 0
                else:
                    assert after["w4_dispatch"]["native_capture"] > 0
                    assert after["w4_dispatch"]["fallback_capture"] == 0
                assert after["w4_dispatch"]["fallback_eager"] > before["w4_dispatch"]["fallback_eager"]
                if args.image_limit:
                    assert after["encoder_calls"] >= 1
                    assert all(span == EXPECTED_IMAGE_TOKENS for span in after["encoder_span_lengths"])
                    assert len(after["image_prefills"]) == 2
                    assert all(
                        item == {"image_tokens": EXPECTED_IMAGE_TOKENS, "text_tokens": 2}
                        for item in after["image_prefills"]
                    )
                else:
                    assert after["encoder_calls"] == after["mm_parameter_elements"] == 0
                    assert not after["image_prefills"]
                for layer in range(3):
                    observed = [item for item in after["router_observations"] if item["layer"] == layer]
                    assert sum(item["zero_text_tokens"] for item in observed) >= 2
                    assert sum(item["literal_image_id_text_tokens"] for item in observed) >= 4
                    assert any(item["image_tokens"] == EXPECTED_IMAGE_TOKENS for item in observed) == bool(
                        args.image_limit
                    )
                for table in after["host_tables"]:
                    node = result["engram_numa_nodes"][rank]
                    assert table["pinned"] and table["numa_node"] == node
                    assert table["placement"]["return"] == 0 and table["placement"]["pages"] > 0
                    assert set(table["placement"]["status_counts"]) == {str(node)}
            result["structural_checks_passed"] = True
            if args.rows_diagnostic:
                per_round = 4 * len(prompts)
                result["engram_content_comparisons"] = []
                for rank, worker in enumerate(final):
                    steps = worker["engram_content_steps"]
                    assert len(steps) == 2 * per_round, (rank, len(steps), per_round)
                    comparable = [{key: value for key, value in step.items() if key != "request_ids"} for step in steps]
                    exact = comparable[:per_round] == comparable[per_round:]
                    result["engram_content_comparisons"].append(
                        {"rank": rank, "steps_per_round": per_round, "exact": exact}
                    )
                    assert exact, (rank, "Repeated Engram snapshot/row/mask contents differ")
            for name, comparison in comparisons.items():
                assert comparison["token_ids_exact"] and comparison["max_selected_logprob_abs_difference"] <= 1e-4, (
                    name,
                    comparison,
                )
            result["checks_passed"] = True
        except Exception as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            if llm is not None:
                try:
                    released = llm.collective_rpc("finish_mm_smoke")
                    result["workers_released"] = released
                    assert len(released) == 8 and all(
                        worker["closed"] and all(worker["shard_weights_released"]) for worker in released
                    )
                except Exception as error:
                    result["release_error"] = f"{type(error).__name__}: {error}"
                try:
                    client = llm.llm_engine.engine_core
                    processes = list(client.resources.engine_manager.processes)
                    client.shutdown(timeout=30.0)
                    exited = [
                        {"pid": process.pid, "exitcode": process.exitcode, "alive": process.is_alive()}
                        for process in processes
                    ]
                    result["engine_shutdown"] = {"timeout_seconds": 30, "processes": exited}
                    assert all(not process["alive"] and process["exitcode"] == 0 for process in exited)
                except Exception as error:
                    result["shutdown_error"] = f"{type(error).__name__}: {error}"
            if result.get("checks_passed"):
                result["status"] = (
                    "passed"
                    if not any(key in result for key in ("release_error", "shutdown_error"))
                    else "failed_cleanup"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)
    return 0 if result["status"] in {"prepared_only", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
