# SPDX-License-Identifier: Apache-2.0
"""Construct the registered V4.1 model on eight real NPU ranks.

Prepare a synthetic 3-layer/E8 config, then run with torchrun --nproc-per-node=8.
Production hidden/projection/Engram dimensions are retained; no checkpoint is
loaded and no model forward or Engram gate is called. Each rank validates the
actual Ascend MoE factory/quantization choice and TP parameter geometry.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    return parser.parse_args()


def prepare(args):
    conversion_path = Path(__file__).resolve().parents[2] / "examples/quantization/convert_deepseek_v41.py"
    spec = importlib.util.spec_from_file_location("v41_smoke_conversion", conversion_path)
    conversion = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = conversion
    spec.loader.exec_module(conversion)
    config = conversion.converted_config(json.loads((args.source / "config.json").read_text()))
    text = config["text_config"]
    text.update(
        num_hidden_layers=3,
        n_routed_experts=8,
        compress_ratios=[0, 0, 2],
        kv_source_layer_ids=[2],
        index_source_layer_ids=[2],
        candidate_source_layer_id=-1,
        engram_layer_ids=[1],
        engram_num_embeddings=text["engram_num_embeddings"][:1],
        num_nextn_predict_layers=0,
        dspark_target_layer_ids=[],
    )
    directory = args.output / "config"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Prepared synthetic model config: {directory}", flush=True)


def validate(model, config, rank):
    import torch
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV41ForCausalLM
    from vllm_ascend.quantization.methods.wna16.w4a16 import AscendW4A16FusedMoEMethod

    assert isinstance(model, AscendDeepseekV41ForCausalLM), type(model)
    assert config.cache_config.block_size == 32
    assert len(model.model.layers) == model.num_moe_layers == 3
    assert model.num_routed_experts == model.num_logical_experts == 8
    assert model.num_local_physical_experts == 8, (
        "TP-only MoE must retain all experts on each rank",
        model.num_local_physical_experts,
    )
    assert model.model.engram_layer_ids == (1,)
    assert model.model.topk_indices.shape == (128, 1, 512)
    assert model.model.candidate_blocks.shape == (128, 1, 2048)
    layers = []
    for layer_id, layer in enumerate(model.model.layers):
        attention, moe = layer.self_attn, layer.mlp
        routed = moe.experts.routed_experts
        scheme = routed.quant_method.quant_method
        assert isinstance(scheme, AscendW4A16FusedMoEMethod), type(scheme)
        assert scheme.group_size == 32 and scheme.num_bits == 4
        assert not scheme.enable_native_decode  # E8 is outside native decode's E384 guard.
        assert routed.moe_config.ep_size == 1 and routed.moe_config.tp_size == 8
        assert moe.n_local_physical_experts == 8 and moe.ep_size == 1
        assert not moe.hash and moe.gate.tid2eid is None
        expected = {
            "w13_weight_packed": ((8, 576, 640), torch.int32),
            "w2_weight_packed": ((8, 5120, 36), torch.int32),
            "w13_weight_scale": ((8, 576, 160), torch.bfloat16),
            "w2_weight_scale": ((8, 5120, 9), torch.bfloat16),
            "w13_weight_shape": ((8, 2), torch.int32),
            "w2_weight_shape": ((8, 2), torch.int32),
        }
        for name, (shape, dtype) in expected.items():
            parameter = getattr(routed, name)
            assert tuple(parameter.shape) == shape, (layer_id, name, parameter.shape, shape)
            assert parameter.dtype == dtype
        assert attention.fused_wqa_wkv.weight.shape == (1792, 5120)
        assert attention.wq_b.weight.shape == (4096, 1280)
        assert attention.wo_a.weight.shape == (1024, 4096)
        assert attention.wo_b.weight.shape == (5120, 1024)
        assert attention.attn_sink.shape == (8,)
        assert attention.fused_wqa_wkv.tp_rank == 0
        assert attention.wq_b.tp_rank == rank
        assert moe.shared_experts.gate_up_proj.weight.dtype == torch.bfloat16
        assert moe.shared_experts.gate_up_proj.weight.shape == (576, 5120)
        assert moe.shared_experts.down_proj.weight.shape == (5120, 288)
        assert not hasattr(moe.shared_experts.gate_up_proj, "weight_packed")
        assert layer.hc_attn_fn.dtype == layer.hc_ffn_fn.dtype == torch.float32
        layers.append(
            {
                "layer": layer_id,
                "runner": type(moe.experts).__name__,
                "routed_experts": type(routed).__name__,
                "quant_adapter": type(routed.quant_method).__name__,
                "quant_scheme": type(scheme).__name__,
                "tp_size": routed.moe_config.tp_size,
                "ep_size": routed.moe_config.ep_size,
                "outer_ep_size": moe.ep_size,
                "parameters": {
                    name: {"shape": shape, "dtype": str(dtype)} for name, (shape, dtype) in expected.items()
                },
            }
        )
    engram = model.model.layers[1].engram
    assert engram.wkv.weight.shape == (25600, 6144)
    assert engram.q_weight.shape == engram.k_weight.shape == (4, 5120)
    assert engram.local_heads == 3
    assert not any("engram.embed" in name for name, _ in model.named_parameters())
    assert model.model.embed_tokens.weight.shape == model.lm_head.weight.shape == (16160, 5120)
    assert model.model.norm.weight.shape == (5120,)
    context = {
        name: layer
        for name, layer in config.compilation_config.static_forward_context.items()
        if isinstance(layer, AttentionLayerBase)
    }
    assert len(context) == 6, list(context)
    caches = {
        name: {
            "spec": type(layer.get_kv_cache_spec(config)).__name__,
            "backend": layer.get_attn_backend().get_name(),
        }
        for name, layer in context.items()
    }
    parameters = list(model.parameters())
    assert all(parameter.device.type == "npu" for parameter in parameters)
    return {
        "model_class": type(model).__name__,
        "num_local_physical_experts": model.num_local_physical_experts,
        "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters),
        "layers": layers,
        "caches": caches,
        "engram_host_tables_allocated": False,
        "forward_executed": False,
        "checkpoint_loaded": False,
    }


def run(args):
    import torch
    import torch_npu  # noqa: F401
    from vllm.config import set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import initialize_model

    from vllm_ascend import ops
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel
    from vllm_ascend.utils import adapt_patch, enable_custom_op, register_ascend_customop

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 8
    result_path = args.output / f"rank{rank}.json"
    started = time.monotonic()
    result = {"rank": rank, "local_rank": local_rank, "success": False}
    try:
        torch.set_num_threads(4)
        torch.npu.set_device(local_rank)
        config = EngineArgs(
            model=str(args.output / "config"),
            tokenizer=str(args.source),
            tensor_parallel_size=8,
            max_model_len=2048,
            max_num_batched_tokens=128,
            max_num_seqs=4,
            enforce_eager=True,
            block_size=32,
            async_scheduling=False,
            distributed_executor_backend="external_launcher",
        ).create_engine_config()
        adapt_patch()
        ops.register_dummy_fusion_op()
        register_ascend_customop(config)
        init_ascend_config(config)
        assert enable_custom_op()
        with set_current_vllm_config(config):
            init_distributed_environment(8, rank, "env://", local_rank, "hccl")
            ensure_model_parallel_initialized(8, 1, 1, 1)
            init_ascend_model_parallel(config.parallel_config)
            print(f"rank {rank}: TP8 initialized; constructing synthetic V4.1", flush=True)
            torch.set_default_dtype(torch.bfloat16)
            with torch.device(f"npu:{local_rank}"):
                model = initialize_model(config)
            torch.npu.synchronize()
            result["constructed_model_class"] = type(model).__name__
            result["constructed_moe"] = [
                {
                    "runner": type(layer.mlp.experts).__name__,
                    "routed_experts": type(layer.mlp.experts.routed_experts).__name__,
                    "quant_adapter": type(layer.mlp.experts.routed_experts.quant_method).__name__,
                    "quant_scheme": type(
                        getattr(layer.mlp.experts.routed_experts.quant_method, "quant_method", None)
                    ).__name__,
                    "outer_ep_size": layer.mlp.ep_size,
                    "outer_num_local_experts": layer.mlp.n_local_physical_experts,
                    "parameters": {
                        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                        for name, value in layer.mlp.experts.routed_experts.named_parameters(recurse=False)
                    },
                }
                for layer in model.model.layers
            ]
            result.update(validate(model, config, rank))
            result.update(
                success=True,
                allocated_bytes=torch.npu.memory_allocated(),
                peak_allocated_bytes=torch.npu.max_memory_allocated(),
                elapsed_seconds=time.monotonic() - started,
            )
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            torch.distributed.barrier()
            if rank == 0:
                all_ranks = [json.loads((args.output / f"rank{i}.json").read_text()) for i in range(8)]
                assert all(item["success"] for item in all_ranks)
                (args.output / "result.json").write_text(
                    json.dumps({"success": True, "world_size": 8, "ranks": all_ranks}, indent=2) + "\n"
                )
                print("V4.1 TP8 constructor smoke: PASS", flush=True)
    except Exception:
        result["error"] = traceback.format_exc()
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = arguments()
    if args.prepare:
        prepare(args)
    else:
        run(args)
