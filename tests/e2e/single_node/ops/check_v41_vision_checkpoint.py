# SPDX-License-Identifier: Apache-2.0
"""Full real-weight V4.1 ViT/aligner numerical check, not a benchmark.

Run only on an explicitly allocated device. The CPU reference runs first;
there is only one tower+aligner parameter copy on the NPU. No latency samples
are collected. Encoder output is not a multimodal language-model result.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from PIL import Image
from safetensors import safe_open
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.models.deepseek_v4.model import (
    AscendV41VisionAligner,
    AscendV41VisionTower,
    _v41_vision_cos_sin,
    _v41_vision_rotary,
)
from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention


class FP16FIAProbe(torch.nn.Module):
    """Diagnostic only: distinguish BF16 intermediate attention rounding."""

    def __init__(self, heads, dim):
        super().__init__()
        self.attention = AscendMMEncoderAttention(heads, dim)

    def forward(self, q, k, v):
        return self.attention(q.half(), k.half(), v.half()).to(q.dtype)


def load_image_input(args, cfg):
    path = args.reference.parent / "image_processor.py"
    spec = importlib.util.spec_from_file_location("released_v41_checkpoint_image_processor", path)
    processor = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = processor
    spec.loader.exec_module(processor)
    source_bytes = args.image.read_bytes()
    with Image.open(io.BytesIO(source_bytes)) as source:
        pixels = source.convert("RGB")
    metadata = {
        "source": str(args.image),
        "source_bytes_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_rgb_pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "source_size": list(pixels.size),
        "processor_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if args.image_thumbnail_max_edge:
        pixels.thumbnail((args.image_thumbnail_max_edge, args.image_thumbnail_max_edge), Image.Resampling.LANCZOS)
    metadata.update(
        thumbnail_max_edge=args.image_thumbnail_max_edge,
        input_rgb_pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
        input_size=list(pixels.size),
    )
    encoded = io.BytesIO()
    pixels.save(encoded, format="PNG")
    patches, height, width, llm_height, llm_width = processor.load_image({"data": encoded.getvalue()}, cfg)
    metadata.update(
        llm_grid=[llm_height, llm_width],
        image_span_tokens=processor.num_image_tokens(llm_height, llm_width),
        patches_sha256=hashlib.sha256(patches.view(torch.uint8).numpy().tobytes()).hexdigest(),
    )
    return patches, height, width, metadata


def errors(actual, expected):
    actual, expected = actual.cpu().double(), expected.double()
    difference = actual - expected
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "nrmse": (difference.square().sum() / expected.square().sum().clamp_min(1e-30)).sqrt().item(),
        "cosine_error": 1 - torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
        "max_abs": difference.abs().max().item(),
        "reference_rms": expected.square().mean().sqrt().item(),
    }


def cpu_batch_attention_probe(reference, official, patches, height, width, expected_features, expected_rows):
    """Same CPU weights/math, alternate SDPA dispatch shape; diagnostic only."""
    for index, block in enumerate(reference.vision.blocks):
        attention = block.attn

        def batched_attention(x, cos, sin, layer=attention):
            tokens = x.shape[0]
            q, k, v = (value.view(tokens, layer.n_heads, layer.head_dim) for value in layer.wqkv(x).chunk(3, dim=-1))
            q, k = official.apply_rotary(q, cos, sin), official.apply_rotary(k, cos, sin)
            output = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0)
            )
            return layer.wo(output[0].transpose(0, 1).reshape(tokens, -1))

        attention.forward = batched_attention
        block.register_forward_hook(
            lambda _, inputs, output, layer=index: print(f"CPU 4D SDPA block {layer}", flush=True)
        )
    features = reference.vision(patches, height, width)
    rows = reference.aligner(features, height, width)
    return {"tower": errors(features, expected_features), "aligner": errors(rows, expected_rows)}


def capture_reference(module, tensors):
    handles = []
    for name, child in module.named_modules():
        if (
            name == "vision.patch_embed"
            or name == "vision.norm"
            or (
                name.startswith("vision.blocks.")
                and (name.count(".") == 2 or name.endswith(("norm1", "norm2", "wqkv", "attn.wo", "mlp")))
            )
        ):

            def capture(_, inputs, output, label=name):
                tensors[label] = {"input": inputs[0].detach().clone(), "output": output.detach().clone()}
                if label.startswith("vision.blocks.") and label.count(".") == 2:
                    print(f"CPU reference completed {label}", flush=True)

            handles.append(child.register_forward_hook(capture))
    return handles


def diagnose(native, reference_tensors, cfg, official, height, width, device):
    """Replay modules with CPU oracle inputs to separate local from accumulated error."""
    results = {}
    for name, values in reference_tensors.items():
        if name.startswith("vision.blocks.") and name.count(".") == 2:
            continue  # Whole-block forward also requires position embeddings.
        module = native.get_submodule(name)
        inputs = values["input"].to(device)
        results[name] = errors(module(inputs), values["output"])
        if name.endswith(("norm1", "norm2")) or name == "vision.norm":
            x = inputs.float()
            variance = x.square().mean(-1, keepdim=True) + 1e-6
            alternative = (module.weight.float() * (x / variance.sqrt())).to(inputs.dtype)
            results[name + ".sqrt_div"] = errors(alternative, values["output"])
    cos, sin = official.get_vision_cos_sin(
        height, width, cfg.vision_dim // cfg.vision_n_heads // 2, cfg.vision_rope_theta
    )
    native_cos, native_sin = _v41_vision_cos_sin(
        height, width, cfg.vision_dim // cfg.vision_n_heads, cfg.vision_rope_theta, device
    )
    for index, block in enumerate(native.vision.blocks):
        block_name = f"vision.blocks.{index}"
        values = reference_tensors[block_name]
        results[block_name + ".isolated"] = errors(
            block(values["input"].to(device), native_cos, native_sin), values["output"]
        )
        residual = values["input"].to(device) + reference_tensors[block_name + ".attn.wo"]["output"].to(device)
        results[block_name + ".residual1"] = errors(residual, reference_tensors[block_name + ".norm2"]["input"])
        residual = reference_tensors[block_name + ".norm2"]["input"].to(device)
        residual = residual + reference_tensors[block_name + ".mlp"]["output"].to(device)
        results[block_name + ".residual2"] = errors(residual, values["output"])
        prefix = f"vision.blocks.{index}.attn"
        q, k, v = (
            tensor.view(height * width, cfg.vision_n_heads, -1)
            for tensor in reference_tensors[prefix + ".wqkv"]["output"].chunk(3, dim=-1)
        )
        for label, tensor in (("q", q), ("k", k)):
            results[prefix + f".rope_{label}"] = errors(
                _v41_vision_rotary(tensor.to(device), native_cos, native_sin), official.apply_rotary(tensor, cos, sin)
            )
        q, k = official.apply_rotary(q, cos, sin), official.apply_rotary(k, cos, sin)
        q, k, v = (tensor.unsqueeze(0).to(device) for tensor in (q, k, v))
        actual = block.attn.attention(q, k, v).reshape(height * width, cfg.vision_dim)
        results[prefix + ".fia"] = errors(actual, reference_tensors[prefix + ".wo"]["input"])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--reference", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash/inference/vision.py"))
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--grid-height", type=int, default=32)
    parser.add_argument("--grid-width", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--attention", choices=("fia", "fia_fp16", "sdpa"), default="fia")
    parser.add_argument("--cpu-batch-attention-probe", action="store_true")
    parser.add_argument("--cpu-fp32-probe", action="store_true")
    parser.add_argument("--output-tensors", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--image-thumbnail-max-edge", type=int, default=0)
    args = parser.parse_args()
    if min(args.grid_height, args.grid_width, args.cpu_threads) <= 0:
        parser.error("grid and CPU thread counts must be positive")
    if args.image_thumbnail_max_edge < 0:
        parser.error("thumbnail max edge cannot be negative")
    if args.cpu_batch_attention_probe and args.cpu_fp32_probe:
        parser.error("select only one CPU precision probe")
    torch.set_num_threads(args.cpu_threads)
    if not (args.cpu_batch_attention_probe or args.cpu_fp32_probe):
        torch.npu.set_device(args.device)
    torch.manual_seed(4132)
    document = json.loads((args.model / "config.json").read_text())
    vision = document["vision_config"]
    cfg = SimpleNamespace(
        vision_dim=vision["hidden_size"],
        vision_n_heads=vision["num_attention_heads"],
        vision_n_layers=vision["num_hidden_layers"],
        vision_inter_dim=vision["intermediate_size"],
        vision_patch_size=vision["patch_size"],
        vision_rope_theta=vision["rope_theta"],
        vision_downsample_ratio=vision["downsample_ratio"],
        vision_min_pixels=vision["min_pixels"],
        vision_max_n_token=vision["max_image_tokens"],
        vision_max_wh_ratio=vision["max_wh_ratio"],
        hidden_size=document["text_config"]["hidden_size"],
        dim=document["text_config"]["hidden_size"],
    )
    spec = importlib.util.spec_from_file_location("released_v41_full_vision", args.reference)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        reference = torch.nn.Module()
        reference.vision, reference.aligner = official.ViT(cfg), official.Aligner(cfg)
        parameters = dict(reference.named_parameters())
        index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
        checkpoint_names = {name for name in index if name.startswith(("vision.", "aligner."))}
        if checkpoint_names != set(parameters):
            raise ValueError("Full vision checkpoint keys do not match the reference parameters")
        checkpoint_digest = hashlib.sha256()
        with torch.no_grad():
            for shard in sorted({index[name] for name in checkpoint_names}):
                with safe_open(args.model / shard, framework="pt", device="cpu") as reader:
                    for name in sorted(name for name in checkpoint_names if index[name] == shard):
                        value = reader.get_tensor(name)
                        if value.dtype != torch.bfloat16 or value.shape != parameters[name].shape:
                            raise ValueError(f"Invalid real vision tensor {name}")
                        checkpoint_digest.update(name.encode())
                        checkpoint_digest.update(value.view(torch.uint8).numpy().tobytes())
                        parameters[name].copy_(value)
        print(f"Loaded all {len(parameters)} real BF16 vision/aligner tensors", flush=True)
        # Values mimic preprocessed normalized RGB patches; this isolates the
        # real-weight encoder from image resize and language embedding merge.
        if args.image:
            patches, args.grid_height, args.grid_width, input_description = load_image_input(args, cfg)
            print(
                f"Released image processor: grid {args.grid_height}x{args.grid_width}, {input_description}", flush=True
            )
        else:
            patches = torch.linspace(-1, 1, args.grid_height * args.grid_width * 3 * cfg.vision_patch_size**2)
            patches = patches.reshape(
                args.grid_height * args.grid_width, 3, cfg.vision_patch_size, cfg.vision_patch_size
            )
            patches = patches.bfloat16()
            input_description = "BF16 linspace(-1,1) RGB patches, not an image_processor output"
        with torch.inference_mode():
            cache_key = {
                "checkpoint": checkpoint_digest.hexdigest(),
                "reference": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                "grid": [args.grid_height, args.grid_width],
                "torch": str(torch.__version__),
                "input": input_description,
            }
            reference_tensors = {}
            if args.reference_cache and args.reference_cache.is_file():
                cached = torch.load(args.reference_cache, map_location="cpu", weights_only=True)
                if cached["key"] != cache_key or (args.diagnose and not cached["intermediates"]):
                    raise ValueError("CPU reference cache does not match this run")
                expected_features, expected_rows = cached["features"], cached["rows"]
                reference_tensors = cached["intermediates"]
                print("Loaded verified CPU reference activation cache", flush=True)
            else:
                handles = capture_reference(reference, reference_tensors) if args.diagnose else []
                print("CPU released-reference ViT forward started", flush=True)
                expected_features = reference.vision(patches, args.grid_height, args.grid_width)
                expected_rows = reference.aligner(expected_features, args.grid_height, args.grid_width)
                for handle in handles:
                    handle.remove()
                if args.reference_cache:
                    args.reference_cache.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "key": cache_key,
                            "features": expected_features,
                            "rows": expected_rows,
                            "intermediates": reference_tensors,
                        },
                        args.reference_cache,
                    )
            print("CPU released-reference outputs complete", flush=True)
            if args.cpu_fp32_probe:
                reference.float()
                print("CPU FP32 reference forward started", flush=True)
                features = reference.vision(patches.float(), args.grid_height, args.grid_width)
                rows = reference.aligner(features, args.grid_height, args.grid_width)
                result = dict(
                    cache_key,
                    scope="CPU full FP32 arithmetic with original BF16 weight values; diagnostic only",
                    cpu_bf16_tower_vs_fp32=errors(expected_features, features),
                    cpu_bf16_aligner_vs_fp32=errors(expected_rows, rows),
                )
                if args.output_tensors:
                    torch.save({"key": cache_key, "tower": features, "aligner": rows}, args.output_tensors)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result, indent=2), flush=True)
                return
            if args.cpu_batch_attention_probe:
                result = cpu_batch_attention_probe(
                    reference, official, patches, args.grid_height, args.grid_width, expected_features, expected_rows
                )
                result.update(
                    cache_key,
                    scope="CPU 4D versus released 3D SDPA dispatch sensitivity; diagnostic only, no acceptance claim",
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result, indent=2), flush=True)
                return
            with set_current_vllm_config(VllmConfig()):
                # Meta construction avoids a second large CPU weight allocation
                # and initializes exactly one NPU parameter set from reference.
                with torch.device("meta"):
                    native = torch.nn.Module()
                    factory = {
                        "fia": lambda heads, dim: AscendMMEncoderAttention(heads, dim),
                        "fia_fp16": FP16FIAProbe,
                        "sdpa": None,
                    }[args.attention]
                    native.vision = AscendV41VisionTower(cfg, attention_factory=factory)
                    native.aligner = AscendV41VisionAligner(cfg)
                device = torch.device("npu", args.device)
                native.to_empty(device=device)
                native.load_state_dict(reference.state_dict(), strict=True)
                allocated = torch.npu.memory_allocated(device)
                if allocated >= 2 * 1024**3:
                    raise RuntimeError("Vision parameters exceed the 2 GiB NPU test budget")
                print("NPU full real-weight ViT forward started", flush=True)
                accumulated = {}
                handles = []
                if args.diagnose:
                    for index, block in enumerate(native.vision.blocks):
                        label = f"vision.blocks.{index}"

                        def capture_block(_, inputs, output, name=label):
                            accumulated[name] = errors(output, reference_tensors[name]["output"])

                        handles.append(block.register_forward_hook(capture_block))
                actual_features = native.vision(patches.to(device), args.grid_height, args.grid_width)
                actual_rows = native.aligner(actual_features, args.grid_height, args.grid_width)
                for handle in handles:
                    handle.remove()
                if args.output_tensors:
                    torch.save(
                        {"key": cache_key, "tower": actual_features.cpu(), "aligner": actual_rows.cpu()},
                        args.output_tensors,
                    )
                result = {
                    "model": str(args.model),
                    "device": str(device),
                    "grid": [args.grid_height, args.grid_width],
                    "layers": cfg.vision_n_layers,
                    "loaded_tensors": len(parameters),
                    "checkpoint_vision_tensor_sha256": checkpoint_digest.hexdigest(),
                    "reference_source_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                    "torch": torch.__version__,
                    "scope": "real-weight replicated ViT and aligner, no LLM or performance claim",
                    "input": cache_key["input"],
                    "attention": args.attention,
                    "tower": errors(actual_features, expected_features),
                    "aligner": errors(actual_rows, expected_rows),
                    "peak_allocated_bytes": torch.npu.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.npu.max_memory_reserved(device),
                    "tower_shape": list(actual_features.shape),
                    "aligner_shape": list(actual_rows.shape),
                }
                if args.diagnose:
                    result["accumulated_blocks"] = accumulated
                    result["isolated_modules"] = diagnose(
                        native, reference_tensors, cfg, official, args.grid_height, args.grid_width, device
                    )
                    result["peak_allocated_bytes"] = torch.npu.max_memory_allocated(device)
                    result["peak_reserved_bytes"] = torch.npu.max_memory_reserved(device)
                # Full-depth accumulation is reported separately from the
                # shallow component suite. Thresholds are fixed before running.
                result["gate"] = "finite, NRMSE<0.03, cosine_error<0.0005, peak_allocated<2GiB"
                result["passed"] = result["peak_allocated_bytes"] < 2 * 1024**3 and all(
                    result[name]["finite"] and result[name]["nrmse"] < 0.03 and result[name]["cosine_error"] < 5e-4
                    for name in ("tower", "aligner")
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result, indent=2), flush=True)
                if not result["passed"]:
                    raise AssertionError("Full real-weight vision component acceptance failed")
    finally:
        torch.set_default_dtype(previous_dtype)


if __name__ == "__main__":
    main()
