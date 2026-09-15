# SPDX-License-Identifier: Apache-2.0
"""Synthetic TP8 runner smoke, including actual host Engram.

This checks execution plumbing with dummy device weights and a small real
host table. It provides no evidence of full-model quality or throughput.
Run this directly, not under torchrun; LLM launches the eight workers.
"""

import argparse
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import torch
from audit_engram_numa import query_or_move
from safetensors import safe_open
from safetensors.torch import save_file


def make_checkpoint(
    source: Path, destination: Path, layers: int = 3, experts: int = 8, converted: Path | None = None
) -> None:
    converter_path = Path(__file__).resolve().parents[2] / "examples/quantization/convert_deepseek_v41.py"
    spec = importlib.util.spec_from_file_location("v41_smoke_converter", converter_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = module.converted_config(json.loads((source / "config.json").read_text()))
    text = config["text_config"]
    text.update(
        num_hidden_layers=layers,
        n_routed_experts=experts,
        engram_vocab_size=17,
        num_nextn_predict_layers=0,
    )
    text["compress_ratios"] = text["compress_ratios"][:layers]
    for key in ("kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids"):
        text[key] = [layer for layer in text[key] if layer < layers]
    table_rows = 4096 if layers == 3 else 8192
    text["engram_num_embeddings"] = [table_rows] * len(text["engram_layer_ids"])
    if converted is None:
        config["vision_config"] = {}
    (destination / "config.json").write_text(json.dumps(config))
    # AutoTokenizer in the host factory resolves the converted model folder.
    for path in source.iterdir():
        if path.is_file() and (
            path.name.startswith("tokenizer") or path.name in {"special_tokens_map.json", "added_tokens.json"}
        ):
            (destination / path.name).symlink_to(path)
    tables = {
        f"layers.{layer}.engram.embed.weight": torch.arange(table_rows * 256)
        .reshape(table_rows, 256)
        .remainder(127)
        .float()
        .div_(127)
        .bfloat16()
        for layer in text["engram_layer_ids"]
    }
    save_file(tables, str(destination / "engram.safetensors"))
    weight_map = {name: "engram.safetensors" for name in tables}
    if converted is not None:
        # Use real device weights without publishing an incomplete full model.
        # The release shards align to layers; reject a different layout rather
        # than silently loading unwanted layers from a mixed shard.
        prefixes = tuple(f"layers.{layer}." for layer in range(layers))
        global_weights = {"embed.weight", "head.weight", "norm.weight"}
        for path in sorted(converted.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as reader:
                names = reader.keys()
            wanted = [name for name in names if name.startswith(prefixes) or name in global_weights]
            if not wanted:
                continue
            extra = set(names) - set(wanted) - {"image_start", "image_end", "image_newline"}
            if extra:
                raise ValueError(f"Selected checkpoint shard mixes unsupported weights: {path.name}")
            (destination / path.name).symlink_to(path.resolve())
            weight_map.update({name: path.name for name in names})
        if not global_weights.issubset(weight_map):
            raise ValueError("Converted checkpoint is missing embedding/head/norm weights")
    (destination / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


class EngramSmokeProbe:
    @torch.inference_mode()
    def prepare_engram_smoke(self):
        # Upstream dummy loading skips integer tensors. Replace undefined
        # packed storage with valid deterministic two's-complement INT4 ones,
        # allowing separately launched eager/graph runs to compare weights.
        if self.vllm_config.load_config.load_format == "dummy":
            for name, parameter in self.model_runner.get_model().named_parameters():
                if name.endswith("weight_packed"):
                    assert parameter.dtype == torch.int32
                    parameter.fill_(0x11111111)
        return self.inspect_engram_smoke()

    def inspect_engram_smoke(self):
        runner = self.model_runner
        runtime = runner.engram_runtime
        host_tables = []
        for shard in runtime.offload.shards:
            owner = shard._pinned_owner
            host_tables.append(
                {
                    "pinned": shard.weight.is_pinned(),
                    "numa_node": None if owner is None else owner.numa_node,
                    "placement": None
                    if owner is None
                    else query_or_move(shard.weight.data_ptr(), shard.weight.numel() * shard.weight.element_size()),
                }
            )
        return {
            "rows": [rows.data_ptr() for rows in runtime.offload.device_rows],
            "mask": runtime.token_mask.data_ptr(),
            "prepared": runtime._prepared,
            "steps": runtime.offload._step,
            "graph_mode": str(runner.compilation_config.cudagraph_mode),
            "model_wrapper": type(runner.model).__name__,
            "captured_graphs": len(getattr(runner.model, "concrete_aclgraph_entries", {})),
            "w4_dispatch": dict(getattr(self, "w4_dispatch", {})),
            "candidate_dispatch": dict(getattr(self, "candidate_dispatch", {})),
            "engram_tokens": list(getattr(self, "engram_tokens", [])),
            "host_tables": host_tables,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--layers", type=int, choices=(3, 40), default=3)
    parser.add_argument("--experts", type=int, choices=(8, 384), default=8)
    parser.add_argument("--native-decode", action="store_true")
    parser.add_argument("--candidate-decode", action="store_true")
    parser.add_argument("--prompt-length", type=int, choices=(32, 40, 1152), default=40)
    parser.add_argument("--converted-weights", type=Path)
    parser.add_argument("--engram-numa-nodes", type=int, nargs=8)
    parser.add_argument("--capture-moe-inputs", type=Path)
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeat-batch", action="store_true")
    parser.add_argument("--trace-layers", type=Path)
    parser.add_argument("--trace-all-ranks", action="store_true", help="Trace only the first layer on all eight ranks")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.native_decode and args.experts != 384:
        parser.error("--native-decode requires --experts 384 to exercise production dispatch")
    if args.candidate_decode and args.layers != 40:
        parser.error("--candidate-decode requires --layers 40 to include candidate consumers")
    if args.converted_weights is not None and args.experts != 384:
        parser.error("--converted-weights requires --experts 384")
    if args.graph and args.trace_layers is not None:
        parser.error("--trace-layers requires eager execution")
    if args.trace_all_ranks and args.trace_layers is None:
        parser.error("--trace-all-ranks requires --trace-layers")
    from vllm import LLM, SamplingParams

    with tempfile.TemporaryDirectory(prefix="v41-runner-smoke-") as temporary:
        checkpoint = Path(temporary)
        make_checkpoint(args.source, checkpoint, args.layers, args.experts, args.converted_weights)
        llm = LLM(
            limit_mm_per_prompt={"image": 0},
            model=str(checkpoint),
            tokenizer=str(args.source),
            tensor_parallel_size=8,
            worker_cls="smoke_runner_worker.EngramSmokeWorker",
            additional_config={
                "enable_w4a16_decode": args.native_decode,
                "enable_indexer_candidate_decode": args.candidate_decode,
                "engram_numa_nodes": args.engram_numa_nodes,
            },
            load_format="dummy" if args.converted_weights is None else "safetensors",
            enforce_eager=not args.graph,
            compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY" if args.graph else "NONE"},
            max_model_len=max(128, args.prompt_length + 32),
            max_num_batched_tokens=32,
            max_num_seqs=2,
            block_size=32,
            gpu_memory_utilization=0.1,
            kv_cache_memory_bytes=(64 if args.prompt_length <= 40 else 256) * 1024**2,
            async_scheduling=False,
            enable_prefix_caching=args.prefix_cache,
        )
        sampling = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=1)
        initial = llm.collective_rpc("prepare_engram_smoke")
        if args.capture_moe_inputs is not None:
            llm.collective_rpc("capture_moe_inputs", args=(str(args.capture_moe_inputs.resolve()),))
        if args.trace_layers is not None:
            llm.collective_rpc(
                "capture_layer_trace",
                args=(str(args.trace_layers.resolve()), args.trace_all_ranks, 1 if args.trace_all_ranks else 3),
            )
        prompts = [list(range(100, 100 + args.prompt_length + extra)) for extra in (0, 8)]
        first = llm.generate([{"prompt_token_ids": ids} for ids in prompts], sampling, use_tqdm=False)
        after_first = llm.collective_rpc("inspect_engram_smoke")
        # Reuse the long prompt prefix in a second execute sequence.
        second = llm.generate([{"prompt_token_ids": prompts[0] + [170]}], sampling, use_tqdm=False)
        final = llm.collective_rpc("inspect_engram_smoke")
        for rank, (before, intermediate, after) in enumerate(zip(initial, after_first, final, strict=True)):
            assert before["rows"] == intermediate["rows"] == after["rows"]
            assert before["mask"] == intermediate["mask"] == after["mask"]
            assert not after["prepared"]
            assert after["steps"] > intermediate["steps"] > before["steps"]
            if args.candidate_decode:
                assert after["candidate_dispatch"]["capture" if args.graph else "eager"] > 0
            if args.engram_numa_nodes is not None:
                for snapshot in (before, intermediate, after):
                    for table in snapshot["host_tables"]:
                        assert table["pinned"] and table["numa_node"] == args.engram_numa_nodes[rank]
                        placement = table["placement"]
                        assert placement["return"] == 0 and placement["pages"] > 0, placement
                        assert set(placement["status_counts"]) == {str(args.engram_numa_nodes[rank])}, placement
            if args.graph:
                assert after["model_wrapper"] == "ACLGraphWrapper"
                assert after["captured_graphs"] > 0
            if args.native_decode:
                dispatch_key = "native_capture" if args.graph else "native_eager"
                assert after["w4_dispatch"][dispatch_key] > 0, after["w4_dispatch"]
                assert after["w4_dispatch"]["fallback_eager"] > 0, after["w4_dispatch"]
                if args.graph and args.prompt_length == 32 and args.prefix_cache:
                    # A fully cached 32-token block leaves exactly one prompt
                    # token, followed by three graph decode steps. The prompt
                    # must still run each layer's CANN prefill path.
                    start = len(intermediate["engram_tokens"])
                    assert after["engram_tokens"][start:] == [1, 1, 1, 1]
                    fallbacks = after["w4_dispatch"]["fallback_eager"] - intermediate["w4_dispatch"]["fallback_eager"]
                    assert fallbacks == args.layers, fallbacks
        outputs = [result.outputs[0].token_ids for result in [*first, *second]]
        assert all(len(ids) == 4 for ids in outputs), outputs
        selected_logprobs = []
        for request in [*first, *second]:
            selected = []
            for token_id, probabilities in zip(request.outputs[0].token_ids, request.outputs[0].logprobs, strict=True):
                assert math.isfinite(probabilities[token_id].logprob)
                selected.append(probabilities[token_id].logprob)
            selected_logprobs.append(selected)
        result = {
            "synthetic": True,
            "device_weights": "dummy" if args.converted_weights is None else str(args.converted_weights),
            "engram_weights": "synthetic_small_tables",
            "engram_numa_nodes": args.engram_numa_nodes,
            "layers": args.layers,
            "experts": args.experts,
            "native_decode_requested": args.native_decode,
            "candidate_decode_requested": args.candidate_decode,
            "tensor_parallel_size": 8,
            "graph_requested": args.graph,
            "host_engram": True,
            "chunked_prefill": True,
            "prefix_reuse_requested": args.prefix_cache,
            "prompt_lengths": [len(prompt) for prompt in prompts],
            "outputs": outputs,
            "selected_logprobs_finite": True,
            "selected_logprobs": selected_logprobs,
            "packed_weight_initialization": (
                "all_nibbles_signed_one_after_dummy_load"
                if args.converted_weights is None
                else "converted_checkpoint_unmodified"
            ),
            "workers_after_capture": initial,
            "workers_after_first": after_first,
            "workers_final": final,
        }
        if args.repeat_batch:
            repeated = llm.generate([{"prompt_token_ids": ids} for ids in prompts], sampling, use_tqdm=False)
            result["repeat_outputs"] = [request.outputs[0].token_ids for request in repeated]
            result["repeat_selected_logprobs"] = [
                [
                    probabilities[token_id].logprob
                    for token_id, probabilities in zip(
                        request.outputs[0].token_ids, request.outputs[0].logprobs, strict=True
                    )
                ]
                for request in repeated
            ]
            result["workers_after_repeat"] = llm.collective_rpc("inspect_engram_smoke")
        result["workers_released"] = llm.collective_rpc("finish_engram_smoke")
        client = llm.llm_engine.engine_core
        processes = list(client.resources.engine_manager.processes)
        client.shutdown(timeout=30.0)
        result["engine_shutdown"] = [
            {"pid": process.pid, "exitcode": process.exitcode, "alive": process.is_alive()} for process in processes
        ]
        assert all(not process["alive"] and process["exitcode"] == 0 for process in result["engine_shutdown"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
