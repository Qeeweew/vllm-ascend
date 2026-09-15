# SPDX-License-Identifier: Apache-2.0
"""Worst declared V4.1 image shape: finite/shape/memory and eager latency only.

Default is CPU preparation and safetensor-header validation. --run explicitly
uses the selected NPU. No CPU attention oracle, NRMSE or model-quality claim.
The fixed 4 GiB encoder component budget is not full-model HBM admission.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.processing import InputProcessingContext
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

from vllm_ascend.models.deepseek_v4.model import (
    AscendDeepseekV41ForConditionalGeneration,
    AscendV41VisionAligner,
    AscendV41VisionTower,
)
from vllm_ascend.patch.worker.patch_deepseek_v41_mm import (
    DeepseekV41VLDummyInputsBuilder,
    DeepseekV41VLProcessingInfo,
    DeepseekV41VLProcessor,
)

ENCODER_PEAK_BUDGET_BYTES = 4 * 1024**3
MM_TENSOR_COUNT = 266


class CapacityEncoder(AscendDeepseekV41ForConditionalGeneration):
    """Only encoder parameters; reuse the actual production encoding method."""

    def __init__(self, config, attention_factory=None):
        torch.nn.Module.__init__(self)
        self.config = config
        self.image_limit = 1
        self.vision = AscendV41VisionTower(config, attention_factory=attention_factory)
        self.aligner = AscendV41VisionAligner(config)
        for name in ("image_start", "image_end", "image_newline"):
            self.register_parameter(name, torch.nn.Parameter(torch.empty(config.hidden_size, dtype=torch.bfloat16)))


def prepare_input(model: Path, config):
    mm = MultiModalConfig(limit_per_prompt={"image": 1})
    model_config = SimpleNamespace(
        model=str(model),
        hf_config=config,
        dtype=torch.bfloat16,
        max_model_len=config.vision_max_n_token,
        encoder_config=None,
        multimodal_config=mm,
        get_multimodal_config=lambda: mm,
    )
    info = DeepseekV41VLProcessingInfo(InputProcessingContext(model_config, tokenizer=None))
    size = info.get_image_size_with_most_features()
    images = DeepseekV41VLDummyInputsBuilder(info).get_dummy_mm_data(config.vision_max_n_token, {"image": 1}, {})[
        "image"
    ]
    values = DeepseekV41VLProcessor(config)(images=images)
    expected_patch_grid, expected_llm_grid = [3, 3063], [1, 1021]
    if config.vision_max_wh_ratio is not None or config.vision_max_n_token != 1024:
        raise ValueError("This fixed acceptance case requires the release's uncapped 1024-token image configuration")
    assert values["vit_grid"].tolist() == [expected_patch_grid]
    assert values["llm_grid"].tolist() == [expected_llm_grid]
    assert values["patches"].shape == (9189, 3, 14, 14)
    assert values["types"].numel() == 1024
    return dict(values), {
        "dummy_pixels": [size.width, size.height],
        "patch_grid": expected_patch_grid,
        "llm_grid": expected_llm_grid,
        "patch_count": 9189,
        "image_span_tokens": 1024,
        "patches_sha256": hashlib.sha256(values["patches"].view(torch.uint8).numpy().tobytes()).hexdigest(),
        "source": "unmodified local processor's actual maximum-size dummy image",
    }


def validate_headers(model, native):
    weight_map = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    parameters = dict(native.named_parameters())
    selected = {
        name
        for name in weight_map
        if name.startswith(("vision.", "aligner.")) or name in {"image_start", "image_end", "image_newline"}
    }
    if selected != set(parameters) or len(selected) != MM_TENSOR_COUNT:
        raise ValueError("Checkpoint must contain exactly the complete 266 encoder tensors")
    for shard in sorted({weight_map[name] for name in selected}):
        with safe_open(model / shard, framework="pt", device="cpu") as reader:
            for name in sorted(name for name in selected if weight_map[name] == shard):
                value = reader.get_slice(name)
                if value.get_dtype() != "BF16" or tuple(value.get_shape()) != tuple(parameters[name].shape):
                    raise ValueError(f"Invalid real checkpoint header for {name}")
    return weight_map


@torch.inference_mode()
def run_npu(args, config, values, result):
    # No device selection, allocation, or quadratic CPU reference in prepare.
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention

    torch.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    with set_current_vllm_config(VllmConfig()):
        with torch.device("meta"):
            native = CapacityEncoder(config, lambda heads, dim: AscendMMEncoderAttention(heads, dim))
        weight_map = validate_headers(args.model, native)
        native.to_empty(device=device)
        native.eval()
        parameters = dict(native.named_parameters())
        for shard in sorted({weight_map[name] for name in parameters}):
            with safe_open(args.model / shard, framework="pt", device="cpu") as reader:
                for name in sorted(name for name in parameters if weight_map[name] == shard):
                    parameters[name].copy_(reader.get_tensor(name))
        values["patches"] = values["patches"].to(device)
        observed = {}
        handles = [
            native.vision.register_forward_hook(lambda module, inputs, output: observed.update(tower=output)),
            native.aligner.register_forward_hook(lambda module, inputs, output: observed.update(aligner=output)),
        ]
        torch.npu.synchronize(device)
        baseline = {
            "allocated_bytes": torch.npu.memory_allocated(device),
            "reserved_bytes": torch.npu.memory_reserved(device),
            "free_total_bytes": list(torch.npu.mem_get_info(device)),
        }
        records = []
        for iteration in range(2 + args.repeats):
            observed.clear()
            torch.npu.synchronize(device)
            torch.npu.reset_peak_memory_stats(device)
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            started = time.perf_counter()
            begin.record()
            spans = native.embed_multimodal(**values)
            end.record()
            torch.npu.synchronize(device)
            wall_ms = (time.perf_counter() - started) * 1000
            shapes = {
                "tower": list(observed["tower"].shape),
                "aligner": list(observed["aligner"].shape),
                "span": list(spans[0].shape),
            }
            finite = {name: bool(torch.isfinite(value).all()) for name, value in observed.items()}
            finite["span"] = bool(torch.isfinite(spans[0]).all())
            expected = {
                "tower": [9189, config.vision_dim],
                "aligner": [1021, config.hidden_size],
                "span": [1024, config.hidden_size],
            }
            records.append(
                {
                    "phase": "cold" if iteration == 0 else "warmup" if iteration == 1 else "steady",
                    "wall_ms": wall_ms,
                    "npu_event_ms": begin.elapsed_time(end),
                    "shapes": shapes,
                    "finite": finite,
                    "shape_passed": shapes == expected and len(spans) == 1,
                    "peak_allocated_bytes": torch.npu.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.npu.max_memory_reserved(device),
                    "free_total_bytes": list(torch.npu.mem_get_info(device)),
                }
            )
            print(json.dumps(records[-1]), flush=True)
            del spans
        for handle in handles:
            handle.remove()
        result.update(
            status="measured",
            device=str(device),
            device_name=torch.npu.get_device_name(device),
            torch=str(torch.__version__),
            torch_npu=str(torch_npu.__version__),
            baseline=baseline,
            iterations=records,
            peak_allocated_bytes=max(record["peak_allocated_bytes"] for record in records),
            peak_reserved_bytes=max(record["peak_reserved_bytes"] for record in records),
            steady_wall_median_ms=statistics.median(
                record["wall_ms"] for record in records if record["phase"] == "steady"
            ),
            steady_event_median_ms=statistics.median(
                record["npu_event_ms"] for record in records if record["phase"] == "steady"
            ),
        )
        result["passed"] = result["peak_allocated_bytes"] < ENCODER_PEAK_BUDGET_BYTES and all(
            record["shape_passed"] and all(record["finite"].values()) for record in records
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats <= 0 or args.cpu_threads <= 0:
        parser.error("repeat and CPU thread counts must be positive")
    torch.set_num_threads(args.cpu_threads)
    config = DeepseekV41Config(**json.loads((args.model / "config.json").read_text()))
    if config.vision_n_layers != 32:
        parser.error("The capacity gate requires all 32 real vision layers")
    values, input_info = prepare_input(args.model, config)
    with torch.device("meta"):
        meta = CapacityEncoder(config)
    validate_headers(args.model, meta)
    result = {
        "status": "prepared_only",
        "model": str(args.model),
        "input": input_info,
        "layers": config.vision_n_layers,
        "checkpoint_tensors": MM_TENSOR_COUNT,
        "gate": {"finite": True, "exact_shapes": True, "peak_allocated_bytes_lt": ENCODER_PEAK_BUDGET_BYTES},
        "scope": "encoder component only; no CPU accuracy oracle, model quality, graph, TP8, or full-model HBM claim",
        "timing_scope": (
            "preprocessed device patches to complete span; excludes CPU resize, H2D, loading and finite checks"
        ),
    }
    if args.run:
        run_npu(args, config, values, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "passed": result.get("passed"), "output": str(args.output)}))
    if args.run and not result["passed"]:
        raise SystemExit("Fixed encoder shape/finite/memory gate failed; details retained in JSON")


if __name__ == "__main__":
    main()
