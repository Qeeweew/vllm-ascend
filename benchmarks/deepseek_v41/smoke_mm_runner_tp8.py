# SPDX-License-Identifier: Apache-2.0
"""Prepare or run a real-photo V4.1 MM prefill -> text decode smoke.

Preparation is CPU-only and is the default. --run launches all eight NPUs;
run directly, never under torchrun. Device weights are real, but the language
model has only three layers and synthetic small Engram tables. Results are
execution checks, not a quality or performance benchmark.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image
from safetensors import safe_open
from smoke_runner_tp8 import make_checkpoint

MM_DELIMITERS = {"image_start", "image_end", "image_newline"}
MM_PARAMETER_COUNT = 266
EXPECTED_IMAGE_TOKENS = 189
IMAGE_TOKEN_ID = 129264


def make_mm_checkpoint(source: Path, converted: Path, destination: Path) -> dict:
    """Reuse the bounded LM fixture, then include all real vision parameters."""
    destination.mkdir(parents=True, exist_ok=False)
    make_checkpoint(source, destination, layers=3, experts=384, converted=converted)
    index_path = destination / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    mm_names = set()
    for shard in sorted(converted.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as reader:
            names = set(reader.keys())
            mm = {name for name in names if name.startswith(("vision.", "aligner.")) or name in MM_DELIMITERS}
            if not mm:
                continue
            if mm_names & mm:
                raise ValueError(f"Duplicate vision weights in {shard.name}")
            for name in mm:
                if reader.get_slice(name).get_dtype() != "BF16":
                    raise ValueError(f"Vision weight {name} is not BF16")
        target = destination / shard.name
        if not target.exists():
            if names != mm:
                raise ValueError(f"Unselected vision shard contains unrelated weights: {shard.name}")
            target.symlink_to(shard.resolve())
        weight_map.update({name: shard.name for name in mm})
        mm_names.update(mm)
    if len(mm_names) != MM_PARAMETER_COUNT or not MM_DELIMITERS.issubset(mm_names):
        raise ValueError(f"Incomplete vision checkpoint: {len(mm_names)} / {MM_PARAMETER_COUNT} parameters")
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return {"vision_parameter_count": len(mm_names), "language_layers": 3, "experts": 384}


def load_photo(path: Path):
    with Image.open(path) as original:
        image = original.convert("RGB")
    original_size = list(image.size)
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    return image, {
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "original_size": original_size,
        "thumbnail_size": list(image.size),
        "thumbnail_pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def selected_outputs(outputs):
    result = []
    for request in outputs:
        completion = request.outputs[0]
        probabilities = [
            distribution[token].logprob
            for token, distribution in zip(completion.token_ids, completion.logprobs, strict=True)
        ]
        assert len(completion.token_ids) == 4 and all(map(math.isfinite, probabilities))
        result.append({"token_ids": completion.token_ids, "selected_logprobs": probabilities})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--converted", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="New directory for the bounded fixture")
    parser.add_argument(
        "--image", type=Path, required=True, help="Use the reported hato.jpg fixture for the fixed span gate"
    )
    parser.add_argument("--run", action="store_true", help="Launch TP8; otherwise only prepare CPU artifacts")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--image-limit", type=int, choices=(0, 1), default=1)
    parser.add_argument("--engram-numa-nodes", type=int, nargs=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = make_mm_checkpoint(args.source.resolve(), args.converted.resolve(), args.checkpoint.resolve())
    photo, photo_info = load_photo(args.image)
    result = {
        "status": "prepared_only",
        "fixture": fixture,
        "photo": photo_info,
        "graph_requested": args.graph,
        "image_limit": args.image_limit,
        "synthetic_small_engram": True,
        "performance_measurement": False,
        "checkpoint": str(args.checkpoint.resolve()),
    }
    # No imports that initialize workers until the explicitly requested run.
    if args.run:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(args.checkpoint.resolve()),
            tokenizer=str(args.source.resolve()),
            dtype="bfloat16",
            tensor_parallel_size=8,
            worker_cls="smoke_mm_runner_worker.V41MMSmokeWorker",
            additional_config={"enable_w4a16_decode": True, "engram_numa_nodes": args.engram_numa_nodes},
            load_format="safetensors",
            enforce_eager=not args.graph,
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY" if args.graph else "NONE",
                "cudagraph_mm_encoder": False,
                "compile_mm_encoder": False,
            },
            max_model_len=512,
            # Admission checks the processor's maximum span budget (1024),
            # even though this fixed photo's actual prompt has 191 tokens.
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
        before = llm.collective_rpc("start_mm_smoke")
        sampling = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=1)
        first_prompt = (
            {"prompt_token_ids": [100, IMAGE_TOKEN_ID, 101], "multi_modal_data": {"image": photo}}
            if args.image_limit
            else {"prompt_token_ids": [100, 102, 101]}
        )
        image_outputs = llm.generate([first_prompt], sampling, use_tqdm=False)
        after_image = llm.collective_rpc("inspect_mm_smoke")
        text_outputs = llm.generate([{"prompt_token_ids": [100, IMAGE_TOKEN_ID, 101]}], sampling, use_tqdm=False)
        after_text = llm.collective_rpc("inspect_mm_smoke")
        for initial, encoded, final in zip(before, after_image, after_text, strict=True):
            assert initial["rows"] == encoded["rows"] == final["rows"]
            assert initial["mask"] == encoded["mask"] == final["mask"]
            assert initial["image_mask_ptr"] == encoded["image_mask_ptr"] == final["image_mask_ptr"]
            assert encoded["encoder_calls"] == final["encoder_calls"] == args.image_limit
            assert encoded["encoder_span_lengths"] == ([EXPECTED_IMAGE_TOKENS] if args.image_limit else [])
            expected_prefills = [{"image_tokens": EXPECTED_IMAGE_TOKENS, "text_tokens": 2}] if args.image_limit else []
            assert encoded["image_prefills"] == expected_prefills
            assert final["supports_mm_inputs"] == bool(args.image_limit)
            assert final["tower_allocated"] == bool(args.image_limit)
            if not args.image_limit:
                assert final["mm_parameter_elements"] == 0
            assert all(signature["embeddings"] == bool(args.image_limit) for signature in final["forward_signatures"])
            assert not final["prepared"] and final["image_mask_sum"] == 0
            assert all(table["pinned"] for table in final["host_tables"])
            for layer in range(3):
                observations = [item for item in final["router_observations"] if item["layer"] == layer]
                if args.image_limit:
                    assert any(item["image_tokens"] == EXPECTED_IMAGE_TOKENS for item in observations)
                else:
                    assert all(item["image_tokens"] == 0 for item in observations)
                assert any(item["literal_image_id_text_tokens"] == 1 for item in observations)
            if args.graph:
                assert final["captured_graphs"] > 0
                assert final["graph_replays"] > encoded["graph_replays"] > 0
            native_phase = "native_capture" if args.graph else "native_eager"
            assert final["w4_dispatch"][native_phase] > 0
            assert final["w4_dispatch"]["fallback_eager"] > initial["w4_dispatch"]["fallback_eager"]
        image_completions = selected_outputs(image_outputs)
        text_completions = selected_outputs(text_outputs)
        # This is terminal: all assertions/inspection precede checked host
        # unregister. There must be no later generate or inspect RPC.
        released = llm.collective_rpc("finish_mm_smoke")
        assert all(worker["closed"] and all(worker["shard_weights_released"]) for worker in released)
        client = llm.llm_engine.engine_core
        processes = list(client.resources.engine_manager.processes)
        client.shutdown(timeout=30.0)
        exited = [
            {"pid": process.pid, "exitcode": process.exitcode, "alive": process.is_alive()} for process in processes
        ]
        assert all(not process["alive"] and process["exitcode"] == 0 for process in exited), exited
        result.update(
            status="passed",
            image_outputs=image_completions,
            text_outputs=text_completions,
            workers_before=before,
            workers_after_image=after_image,
            workers_final=after_text,
            workers_released=released,
            engine_shutdown={"timeout_seconds": 30, "processes": exited},
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
