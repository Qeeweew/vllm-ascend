# SPDX-License-Identifier: Apache-2.0
"""Bounded real-weight DSpark component capture, with CPU-only default.

--prepare reads headers/config only. --run requires torchrun with eight ranks
and an allocated NPU window. --compare performs CPU stage-oracle checks after
all ranks exit. This is not a serving, acceptance-rate or performance test.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import torch
from dspark_v41_reference import ConvertedWeights, compare

PEAK_BUDGET_BYTES = 8 * 1024**3


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, choices=(9, 33, 129), default=9)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--compare", action="store_true")
    return parser.parse_args()


def prepare(args):
    path = Path(__file__).resolve().parents[2] / "examples/quantization/convert_deepseek_v41.py"
    spec = importlib.util.spec_from_file_location("v41_dspark_conversion", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    source_config = json.loads((args.source / "config.json").read_text())
    config = module.converted_config(source_config)
    # This fixture only builds a component engine config. It never constructs
    # a target, scheduler or host Engram owner, and never alters source files.
    config["vision_config"]["num_hidden_layers"] = 0
    config["text_config"]["engram_layer_ids"] = []
    directory = args.output / "config"
    directory.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(config, indent=2) + "\n"
    config_path = directory / "config.json"
    if config_path.exists() and config_path.read_text() != serialized:
        raise ValueError("Existing fixture config differs; choose a fresh output directory")
    config_path.write_text(serialized)
    weights = ConvertedWeights(args.checkpoint)
    names = sorted(name for name in weights.index if name.startswith("mtp."))
    required = {
        "embed.weight",
        "head.weight",
        "mtp.0.main_proj.weight",
        "mtp.2.norm.weight",
        "mtp.2.markov_head.embed.weight",
        "mtp.2.markov_head.head.weight",
        "mtp.2.confidence_head.proj.weight",
    }
    if not required.issubset(weights.index):
        raise ValueError(f"Missing draft/shared-vocabulary weights: {required - weights.index.keys()}")
    counts = [sum(name.startswith(f"mtp.{stage}.") for name in names) for stage in range(3)]
    if counts != [1176, 1174, 1178]:
        raise ValueError(f"Unexpected released draft inventory: {counts}")
    for stage in range(3):
        assert weights.headers[f"mtp.{stage}.ffn.gate.weight"]["shape"] == [128, 5120]
    result = dict(
        status="prepared_only",
        checkpoint=str(args.checkpoint),
        source=str(args.source),
        context_tokens=args.context_tokens,
        query_tokens=5,
        experts=128,
        top_k=3,
        layers=3,
        actual_weights=True,
        target_aux_input="deterministic synthetic BF16; no target layers executed",
        stage_weight_counts=counts,
        peak_budget_per_rank_bytes=PEAK_BUDGET_BYTES,
        manifest_sha256=hashlib.sha256((args.checkpoint / "conversion_manifest.json").read_bytes()).hexdigest(),
        npu_initialized=bool(hasattr(torch, "npu") and torch.npu.is_initialized()),
    )
    engine = component_config(args)
    result["draft_architecture"] = engine.speculative_config.draft_model_config.architectures
    result["engine_config_validated"] = True
    result["npu_initialized"] = bool(torch.npu.is_initialized())
    assert not result["npu_initialized"]
    (args.output / "prepared.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def component_config(args):
    from vllm.engine.arg_utils import EngineArgs

    return EngineArgs(
        model=str(args.output / "config"),
        tokenizer=str(args.source),
        tensor_parallel_size=8,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=1,
        dtype="bfloat16",
        enforce_eager=True,
        block_size=32,
        async_scheduling=False,
        enable_prefix_caching=False,
        distributed_executor_backend="external_launcher",
        limit_mm_per_prompt={"image": 0},
        additional_config={"enable_w4a16_decode": False},
        speculative_config={"method": "dspark", "num_speculative_tokens": 5},
    ).create_engine_config()


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        return [snapshot(item) for item in value]
    return value


class Captures:
    """Process-local correctness hooks; every hook is removed after the call."""

    def __init__(self, model, record):
        from vllm_ascend.models.deepseek_v4 import model as model_module

        self.restore = []
        self.handles = []
        self.active = None
        self.record = record
        self.module = model_module
        self.hook(model.model.main_proj, lambda args, kwargs, output: record.update(main_projection=snapshot(output)))
        self.hook(model.model.norm, lambda args, kwargs, output: record.update(normalized=snapshot(output)))
        for index, layer in enumerate(model.model.layers):
            data = record["layers"][index]
            self.handles.append(
                layer.register_forward_pre_hook(lambda owner, args, index=index: setattr(self, "active", index))
            )
            self.hook(layer, lambda args, kwargs, output, data=data: data.update(block_output=snapshot(output)))
            self.hook(
                layer.self_attn,
                lambda args, kwargs, output, data=data: data.update(
                    attention_input=snapshot(kwargs["hidden_states"]), attention_output=snapshot(output)
                ),
            )
            self.wrap(
                layer.self_attn,
                "project_inputs",
                lambda args, kwargs, output, data=data: data.update(
                    query=snapshot(output[1]), query_kv=snapshot(output[2])
                ),
            )
            self.wrap(
                layer.self_attn.sparse,
                "forward",
                lambda args, kwargs, output, data=data: data.update(sparse_output=snapshot(output[0])),
            )
            self.hook(
                layer.self_attn.wo_b,
                lambda args, kwargs, output, data=data: data.update(output_b_input=snapshot(args[0])),
            )
            self.wrap(
                layer.self_attn.wo_b.quant_method,
                "apply",
                lambda args, kwargs, output, data=data: data.update(output_b_local=snapshot(output)),
            )
            self.hook(
                layer.mlp.shared_experts,
                lambda args, kwargs, output, data=data: data.update(
                    shared=dict(input=snapshot(args[0] if args else kwargs["hidden_states"]), output=snapshot(output))
                ),
            )
            self.wrap(
                layer.mlp.experts.routed_experts.quant_method.quant_method,
                "apply",
                lambda args, kwargs, output, data=data: data.update(
                    moe=dict(
                        x=snapshot(kwargs["x"]),
                        ids=snapshot(kwargs["topk_ids"]),
                        weights=snapshot(kwargs["topk_weights"]),
                        output=snapshot(output.routed_out),
                    )
                ),
            )
        self.wrap(model_module, "mhc_pre_delayed", self.capture_pre)
        self.wrap(model_module, "mhc_post", self.capture_post)

    def hook(self, module, callback):
        self.handles.append(
            module.register_forward_hook(
                lambda owner, args, kwargs, output: callback(args, kwargs, output), with_kwargs=True
            )
        )

    def wrap(self, owner, name, callback):
        original = getattr(owner, name)

        def instrumented(*args, **kwargs):
            # Clone input tensors before the operation can mutate them.
            saved_args = snapshot(args) if owner is self.module else args
            result = original(*args, **kwargs)
            callback(saved_args, kwargs, result)
            return result

        setattr(owner, name, instrumented)
        self.restore.append((owner, name, original))

    def capture_pre(self, args, kwargs, output):
        self.record["layers"][self.active]["hc_pre"].append(
            dict(hidden=args[0], incoming=args[4], outputs=snapshot(output))
        )

    def capture_post(self, args, kwargs, output):
        self.record["layers"][self.active]["hc_post"].append(dict(inputs=args, output=snapshot(output)))

    def close(self):
        for handle in self.handles:
            handle.remove()
        for owner, name, original in reversed(self.restore):
            setattr(owner, name, original)


def load_vocabulary(model, config, weights, rank, engine_config):
    from torch import nn
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

    target = nn.Module()
    target.model = nn.Module()
    target.model.embed_tokens = VocabParallelEmbedding(
        config.vocab_size, config.hidden_size, params_dtype=torch.bfloat16
    )
    target.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, params_dtype=torch.bfloat16)
    for module, name in ((target.model.embed_tokens, "embed.weight"), (target.lm_head, "head.weight")):
        width = config.vocab_size // 8
        module.weight.copy_(weights.read(name, rows=slice(rank * width, (rank + 1) * width)))
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.method, proposer.model = "dspark", model
    proposer.vllm_config, proposer.use_cuda_graph = engine_config, False
    proposer._maybe_share_embeddings(target)
    proposer._maybe_share_lm_head(target)
    assert model.model.embed_tokens is target.model.embed_tokens
    assert model.lm_head is target.lm_head
    return target


def bind_caches(model, config, context_tokens, device):
    from vllm_ascend.attention.dsa_v41 import AscendV41CacheMetadata
    from vllm_ascend.ops.dsa_v41 import build_dspark_v41_swa_indices

    page_size = 32
    pages = (context_tokens + 5 + page_size - 1) // page_size
    table = torch.arange(pages - 1, -1, -1, device=device, dtype=torch.int32)[None]
    query_offsets = torch.tensor([0, 5], dtype=torch.int32, device=device)
    lengths = torch.tensor([context_tokens + 5], dtype=torch.int32, device=device)
    positions = torch.arange(context_tokens, context_tokens + 5, device=device)
    context_positions = torch.arange(context_tokens, device=device)
    all_positions = torch.arange(context_tokens + 5, device=device)
    slots = (table[0, all_positions // page_size] * page_size + all_positions % page_size).long()
    indices = torch.empty((5, 1, 256), dtype=torch.int32, device=device)
    visible_lengths = torch.empty((5, 1), dtype=torch.int32, device=device)
    build_dspark_v41_swa_indices(
        table,
        query_offsets,
        lengths,
        page_size=32,
        num_cache_blocks=pages,
        indices_output=indices,
        lengths_output=visible_lengths,
    )
    metadata, context_slots = {}, []
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.swa_cache_layer.bind_kv_cache(torch.zeros((pages, 32, 1, 512), dtype=torch.bfloat16, device=device))
        sparse_meta = attn.sparse.build_metadata(
            query_offsets,
            lengths,
            table,
            max_seqlen_q=5,
            max_seqlen_kv=pages * 32,
            draft_swa_indices=indices,
            draft_swa_lengths=visible_lengths,
        )
        metadata[attn.swa_cache_layer.prefix] = AscendV41CacheMetadata(
            role="swa",
            compress_ratio=0,
            physical_block_size=32,
            positions=positions,
            cu_seqlens_q=query_offsets,
            seqused_kv=lengths,
            block_table=table,
            slot_mapping=slots[context_tokens:],
            token_to_req_indices=torch.zeros(5, dtype=torch.int32, device=device),
            schedule=sparse_meta.schedule,
            num_prefills=1,
            num_decode_tokens=0,
            draft_swa_indices=indices,
            draft_swa_lengths=visible_lengths,
        )
        context_slots.append(slots[:context_tokens])
    return metadata, positions, context_positions, context_slots


@torch.inference_mode()
def run(args):
    """Load the real TP8 draft, capture component outputs, and record cleanup.

    Custom-op registration precedes cache metadata construction. Captures use
    synthetic target auxiliary inputs and shared real vocabulary weights;
    numerical acceptance happens later in the independent CPU comparison.
    Every rank records its failure and teardown status before exiting.
    """
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        raise ValueError("--run requires torchrun --nproc-per-node=8 and an allocated eight-device window")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if (args.output / f"rank{rank}.pt").exists():
        raise ValueError("Capture exists; choose a fresh prepared output directory")
    result = dict(rank=rank, status="failed", peak_budget_bytes=PEAK_BUDGET_BYTES)
    initialized = False
    try:
        import torch_npu
        from vllm import ModelRegistry
        from vllm.config import set_current_vllm_config
        from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
        from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading

        from vllm_ascend import ops
        from vllm_ascend.ascend_config import init_ascend_config
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel
        from vllm_ascend.models import register_model
        from vllm_ascend.utils import adapt_patch, enable_custom_op, register_ascend_customop

        torch.set_num_threads(4)
        torch.npu.set_device(local_rank)
        device = torch.device("npu", local_rank)
        config = component_config(args)
        adapt_patch()
        register_model()
        ops.register_dummy_fusion_op()
        register_ascend_customop(config)
        init_ascend_config(config)
        if not enable_custom_op():
            raise RuntimeError("The real draft capture requires the compiled Ascend custom operators")
        with set_current_vllm_config(config):
            init_distributed_environment(8, rank, "env://", local_rank, "hccl")
            initialized = True
            ensure_model_parallel_initialized(8, 1, 1, 1)
            init_ascend_model_parallel(config.parallel_config)
            result["phase"] = "constructing_draft"
            print(f"rank {rank}: TP8 ready; constructing three real E128 draft blocks", flush=True)
            torch.set_default_dtype(torch.bfloat16)
            model_class, _ = ModelRegistry.resolve_model_cls(["DSparkV41DraftModel"], config.model_config)
            if model_class.__module__ != "vllm_ascend.models.deepseek_v4.dspark":
                raise ValueError("Production registry did not select the Ascend V4.1 drafter")
            weights = ConvertedWeights(args.checkpoint)
            with torch.device(device):
                model = initialize_model(config, model_class=model_class, prefix="draft")
                if torch.npu.max_memory_allocated() >= PEAK_BUDGET_BYTES:
                    raise RuntimeError("Draft construction exceeded the fixed component memory budget")
                loaded = model.load_weights(
                    (name, weights.read(name)) for name in sorted(weights.index) if name.startswith("mtp.")
                )
                target_vocab = load_vocabulary(model, model.config, weights, rank, config)
                process_weights_after_loading(model, config.model_config, device)
                model.eval()
            result["phase"] = "weights_loaded"
            print(f"rank {rank}: strict real weight load and shared vocabulary complete", flush=True)
            assert len(model.model.layers) == 3 and model.num_routed_experts == 128
            for layer in model.model.layers:
                assert layer.mlp.n_routed_experts == 128 and layer.mlp.gate.weight.shape == (128, 5120)
                assert layer.mlp.gate.bias_vl is None and layer.engram is None
            metadata, positions, context_positions, context_slots = bind_caches(
                model, config, args.context_tokens, device
            )
            generator = torch.Generator(device="cpu").manual_seed(41061)
            aux = (
                torch.randn((args.context_tokens, 15360), generator=generator, dtype=torch.float32)
                .mul_(0.125)
                .bfloat16()
            )
            ids = torch.tensor([100, 128799, 101, 129264, 102], dtype=torch.int64)
            record = dict(
                rank=rank,
                aux=aux,
                positions=positions.cpu(),
                context_positions=context_positions.cpu(),
                ids=ids,
                layers=[dict(hc_pre=[], hc_post=[]) for _ in range(3)],
            )
            hooks = Captures(model, record)
            result["phase"] = "capturing_forward"
            try:
                with set_ascend_forward_context(
                    metadata,
                    config,
                    num_tokens=5,
                    num_actual_tokens=5,
                    model_instance=model,
                    is_draft_model=True,
                    has_sinks=True,
                ):
                    context = model.combine_hidden_states(aux.to(device))
                    record["context"] = snapshot(context)
                    model.precompute_and_store_context_kv(context, context_positions, context_slots)
                    for index, layer in enumerate(model.model.layers):
                        cache = layer.self_attn.swa_cache_layer.kv_cache.reshape(-1, 512)
                        record["layers"][index]["context_kv"] = snapshot(cache[context_slots[index]])
                    head_hidden = model(ids.to(device), positions)
                    record["head_hidden"] = snapshot(head_hidden)
                    record["logits"] = snapshot(model.compute_logits(head_hidden))
                    markov_ids = torch.tensor([100, 101, 102, 103, 104], device=device)
                    markov = model.markov_embed(markov_ids)
                    record["markov_ids"], record["markov_embed"] = snapshot(markov_ids), snapshot(markov)
                    record["markov_bias"] = snapshot(model.markov_bias(markov))
                    record["confidence"] = snapshot(model.compute_confidence(head_hidden, markov))
            finally:
                hooks.close()
            torch.npu.synchronize()
            result.update(
                status="captured_pending_cpu_oracle",
                loaded_parameter_names=len(loaded),
                layers=3,
                experts=128,
                top_k=3,
                context_tokens=args.context_tokens,
                allocated_bytes=torch.npu.memory_allocated(),
                peak_allocated_bytes=torch.npu.max_memory_allocated(),
                reserved_bytes=torch.npu.memory_reserved(),
                torch=str(torch.__version__),
                torch_npu=str(torch_npu.__version__),
                shared_embed_identity=model.model.embed_tokens is target_vocab.model.embed_tokens,
                shared_head_identity=model.lm_head is target_vocab.lm_head,
            )
            if result["peak_allocated_bytes"] >= PEAK_BUDGET_BYTES:
                raise RuntimeError("Fixed 8 GiB component peak memory gate failed")
            torch.save(record, args.output / f"rank{rank}.pt")
            result["phase"] = "capture_saved"
            print(f"rank {rank}: capture saved; entering checked distributed cleanup", flush=True)
            torch.distributed.barrier()
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        if initialized:
            from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

            try:
                torch.npu.synchronize()
                destroy_model_parallel()
                destroy_distributed_environment()
                result["distributed_cleanup"] = True
            except Exception:
                result["distributed_cleanup"] = False
                result["cleanup_error"] = traceback.format_exc()
                result["status"] = "failed_cleanup"
        (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")


def compare_all(args):
    torch.set_num_threads(8)
    config = json.loads((args.source / "config.json").read_text())["text_config"]
    weights = ConvertedWeights(args.checkpoint)
    results, captures = [], []
    for rank in range(8):
        status = json.loads((args.output / f"rank{rank}.json").read_text())
        if status["status"] != "captured_pending_cpu_oracle" or not status.get("distributed_cleanup"):
            raise ValueError(f"Rank {rank} did not finish capture and cleanup")
        capture = torch.load(args.output / f"rank{rank}.pt", map_location="cpu", weights_only=True)
        results.append(compare(capture, config, weights))
        captures.append(capture)
        print(json.dumps(dict(rank=rank, passed=results[-1]["passed"])), flush=True)
    # Report the BF16 TP reduction separately from per-rank GEMM. Strict HCCL
    # defines reproducibility, not equivalence to one final FP32-sum rounding.
    from dspark_v41_reference import metrics

    reductions = []
    for stage in range(3):
        local = (
            torch.stack([capture["layers"][stage]["output_b_local"].float() for capture in captures]).sum(0).bfloat16()
        )
        actual = captures[0]["layers"][stage]["attention_output"]
        reductions.append(
            dict(
                stage=stage,
                rank_outputs_equal=all(
                    torch.equal(actual, capture["layers"][stage]["attention_output"]) for capture in captures
                ),
                **metrics(actual, local),
            )
        )
    passed = all(item["passed"] for item in results) and all(item["rank_outputs_equal"] for item in reductions)
    result = dict(
        status="passed" if passed else "failed_numerics",
        stage_oracle_only=True,
        full_model_independent_oracle=False,
        performance_measurement=False,
        ranks=results,
        tp_attention_reduction=reductions,
    )
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    args = arguments()
    if args.run:
        run(args)
    elif args.compare:
        compare_all(args)
    else:
        print(json.dumps(prepare(args), indent=2))
