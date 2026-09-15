# SPDX-License-Identifier: Apache-2.0
"""Check real three-layer target HC auxiliary outputs in eager and decode graph.

This diagnostic enables the existing auxiliary-output runner protocol without
a drafter. It uses real target weights and small synthetic Engram tables from
the existing bounded fixture; it does not validate 40-layer DSpark integration.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path; never overwrite prior evidence")
    from smoke_mm_runner_tp8 import selected_outputs
    from vllm import LLM, SamplingParams

    result = {
        "status": "running",
        "real_target_layers": 3,
        "synthetic_small_engram": True,
        "performance_measurement": False,
    }
    llm = None
    try:
        llm = LLM(
            model=str(args.checkpoint.resolve()),
            tokenizer=str(args.source.resolve()),
            dtype="bfloat16",
            tensor_parallel_size=8,
            worker_cls="probe_target_aux_worker.V41AuxProbeWorker",
            additional_config={"enable_w4a16_decode": False, "engram_numa_nodes": [6, 7, 4, 5, 0, 1, 2, 3]},
            load_format="safetensors",
            enforce_eager=False,
            compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
            max_model_len=512,
            max_num_batched_tokens=1024,
            max_num_seqs=1,
            block_size=32,
            gpu_memory_utilization=0.2,
            kv_cache_memory_bytes=256 * 1024**2,
            async_scheduling=False,
            enable_prefix_caching=False,
            disable_chunked_mm_input=True,
            limit_mm_per_prompt={"image": 0},
            skip_mm_profiling=True,
            mm_encoder_tp_mode="weights",
        )
        result["workers_started"] = llm.collective_rpc("reset_aux_probe")
        sampling = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=1)
        runs = []
        for _ in range(2):
            runs.append(
                selected_outputs(llm.generate([{"prompt_token_ids": [0, 100, 102, 101]}], sampling, use_tqdm=False))
            )
        result["outputs"] = runs
        result["workers"] = llm.collective_rpc("inspect_aux_probe")
        for worker in result["workers"]:
            assert worker["max_abs_error"] == {"eager": [0.0] * 3, "graph": [0.0] * 3}, worker
            assert worker["calls"]["eager"] >= 2 and worker["calls"]["graph"] >= 6, worker
        assert runs[0] == runs[1], "Strict CANN repeated token/logprob output mismatch"
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if llm is not None:
            try:
                result["workers_released"] = llm.collective_rpc("finish_mm_smoke")
            except BaseException as error:
                result.update(status="failed_cleanup", release_error=f"{type(error).__name__}: {error}")
            try:
                client = llm.llm_engine.engine_core
                processes = list(client.resources.engine_manager.processes)
                client.shutdown(timeout=30.0)
                result["engine_shutdown"] = [
                    {"exitcode": process.exitcode, "alive": process.is_alive()} for process in processes
                ]
                if any(process.is_alive() or process.exitcode != 0 for process in processes):
                    result["status"] = "failed_cleanup"
            except BaseException as error:
                result.update(status="failed_cleanup", shutdown_error=f"{type(error).__name__}: {error}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "passed":
        raise RuntimeError(result["status"])


if __name__ == "__main__":
    main()
