# SPDX-License-Identifier: Apache-2.0
"""Admission-first eager/graph validation of all real target weights and Engram tables.

Default mode prepares a CPU report. --run requires a separately allocated TP8
window and passing current conversion/host/HBM admission. No fixture is created.
"""

import argparse
import json
import os
from pathlib import Path

from preflight_full_model import NUMA_NODES, build_preflight


def main():
    """Prepare CPU admission; validate references before launch and require checked owner/engine cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--converted", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--native-decode", action="store_true", help="Opt into native W4A16 after the CANN baseline")
    parser.add_argument("--reference", type=Path, help="Prior full-model eager result for graph comparison")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new result path")
    if args.run and args.graph and args.reference is None:
        parser.error("--run --graph requires --reference to a passed eager full-model result")
    prompts = [list(range(100, 140)), [0, 129264, 101], list(range(100, 229)), list(range(100, 484))]
    report = {
        "status": "prepared_only",
        "preflight": build_preflight(args.source, args.converted),
        "graph": args.graph,
        "native_decode": args.native_decode,
        "synthetic_weights": False,
        "speculative_decoding": False,
        "quality_validation": False,
        "performance_measurement": False,
    }
    previous = None
    if args.run and args.reference is not None:
        try:
            previous = json.loads(args.reference.read_text())
            validate_reference(previous, report, prompts)
        except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
            parser.error(f"Invalid eager reference: {error}")
        report["reference"] = str(args.reference)
    if args.run and report["preflight"]["status"] != "ready_for_scheduled_launch":
        report["status"] = "blocked_before_launch"
    elif args.run:
        if os.environ.get("HCCL_DETERMINISTIC") != "strict":
            parser.error("Use HCCL_DETERMINISTIC=strict for the initial correctness baseline")
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7") != "0,1,2,3,4,5,6,7":
            parser.error("This host-specific NUMA/HBM admission requires physical devices 0–7 in order")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "180")
        from vllm import LLM, SamplingParams

        llm = None
        try:
            llm = LLM(
                model=str(args.converted),
                tokenizer=str(args.converted),
                dtype="bfloat16",
                tensor_parallel_size=8,
                worker_cls="full_model_worker.V41FullModelWorker",
                load_format="safetensors",
                safetensors_load_strategy="lazy",
                additional_config={"enable_w4a16_decode": args.native_decode, "engram_numa_nodes": list(NUMA_NODES)},
                enforce_eager=not args.graph,
                compilation_config={
                    "mode": 0,
                    "cudagraph_mode": "FULL_DECODE_ONLY" if args.graph else "NONE",
                    "cudagraph_mm_encoder": False,
                    "compile_mm_encoder": False,
                },
                max_model_len=512,
                max_num_batched_tokens=128,
                max_num_seqs=1,
                block_size=32,
                kv_cache_memory_bytes=256 * 1024**2,
                gpu_memory_utilization=0.9,
                async_scheduling=False,
                enable_prefix_caching=False,
                disable_chunked_mm_input=True,
                limit_mm_per_prompt={"image": 0},
                skip_mm_profiling=True,
                shutdown_timeout=180,
            )
            report["workers_loaded"] = llm.collective_rpc("start_full_model_audit", args=(str(args.source),))
            from smoke_mm_runner_tp8 import selected_outputs

            params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=1)
            outputs = []
            for repeat in range(2):
                round_outputs = []
                for prompt in prompts:
                    round_outputs.extend(
                        selected_outputs(llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False))
                    )
                outputs.append(round_outputs)
                print(json.dumps({"event": "full_generation_round_complete", "repeat": repeat}), flush=True)
            report["prompts"] = prompts
            report["outputs"] = outputs
            report["workers_final"] = llm.collective_rpc("inspect_full_model_audit")
            for loaded, final in zip(report["workers_loaded"], report["workers_final"], strict=True):
                initial = loaded["state"]
                assert initial["rows"] == final["rows"] and initial["mask"] == final["mask"]
                assert not final["prepared"] and final["offload_steps"] > initial["offload_steps"]
                assert final["graph_replays"] >= (24 if args.graph else 0)
            report["repeat_comparisons"] = compare_outputs(outputs[0], outputs[1])
            assert all(item["token_ids_equal"] for item in report["repeat_comparisons"])
            if previous is not None:
                report["reference_comparisons"] = compare_outputs(previous["outputs"][0], outputs[0])
                assert all(item["token_ids_equal"] for item in report["reference_comparisons"])
            report["checks_passed"] = True
        except Exception as error:
            report["status"] = "failed"
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
            if report.get("checks_passed"):
                report["status"] = "failed_cleanup" if "cleanup_error" in report else "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)
    return 0 if report["status"] in {"prepared_only", "passed"} else 1


def validate_reference(previous, current, prompts):
    if (
        previous.get("status") != "passed"
        or previous.get("graph") is not False
        or previous.get("synthetic_weights") is not False
        or previous.get("prompts") != prompts
        or previous.get("native_decode") != current["native_decode"]
        or previous.get("preflight", {}).get("source_fingerprints") != current["preflight"]["source_fingerprints"]
        or previous.get("preflight", {}).get("converted") != current["preflight"]["converted"]
    ):
        raise ValueError(
            "Reference must be a passed eager full-model run with identical checkpoint, backend and prompts"
        )


def compare_outputs(first, second):
    return [
        {
            "token_ids_equal": a["token_ids"] == b["token_ids"],
            "max_selected_logprob_abs_difference": max(
                abs(x - y) for x, y in zip(a["selected_logprobs"], b["selected_logprobs"], strict=True)
            ),
        }
        for a, b in zip(first, second, strict=True)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
