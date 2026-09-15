# SPDX-License-Identifier: Apache-2.0
"""Evaluate existing 910B matmul APIs; no custom GEMM or compressor fusion."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu
from benchmark_compressor_v41 import measure, summarize
from safetensors import safe_open


def errors(actual, reference):
    diff = actual.double() - reference.double()
    return {
        "max_abs": diff.abs().max().item(),
        "nrmse": (diff.square().mean() / reference.double().square().mean()).sqrt().item(),
        "bf16_grid_fraction": (actual == actual.bfloat16().float()).float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 4096])
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.set_num_threads(16)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    keys = [f"layers.2.attn.compressor.{name}.weight" for name in ("wkv", "wgate")]
    weights = []
    for key in keys:
        with safe_open(args.model / index[key], framework="pt", device="cpu") as checkpoint:
            weights.append(checkpoint.get_tensor(key))
    weight_cpu = torch.cat(weights, dim=0).contiguous()
    assert weight_cpu.dtype == torch.bfloat16 and weight_cpu.shape == (1024, 5120)
    weight_bf16 = weight_cpu.npu()
    weight_fp32 = weight_bf16.float()
    # FP64 samples separate accumulation error from CPU FP32 GEMM roundoff.
    columns = torch.arange(0, 1024, 16)
    rng = torch.Generator().manual_seed(41022)
    report = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "allow_hf32": torch.npu.matmul.allow_hf32,
        "weight_keys": keys,
        "weight_sha256": hashlib.sha256(weight_cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "mm_dtype_schema": str(torch.ops.aten.mm.dtype._schema),
        "graph_unroll": 16,
        "rounds": 5,
        "iterations_per_round": 20,
        "scope": "Independent projection only; no compression, normalization, RoPE, or cache store",
        "cases": [],
    }
    for tokens in args.tokens:
        hidden_cpu = torch.randn((tokens, 5120), generator=rng).bfloat16()
        hidden_bf16 = hidden_cpu.npu()
        hidden_fp32 = hidden_bf16.float()
        reference = hidden_cpu.float() @ weight_cpu.float().T
        fp64_samples = hidden_cpu.double() @ weight_cpu[columns].double().T
        fns = {
            "fp32_mm": lambda x=hidden_fp32: torch.mm(x, weight_fp32.T),
            "fp32_mm_with_activation_cast": lambda x=hidden_bf16: torch.mm(x.float(), weight_fp32.T),
            "bf16_inputs_fp32_output": lambda x=hidden_bf16: torch.mm(x, weight_bf16.T, out_dtype=torch.float32),
        }
        result = {"M": tokens, "K": 5120, "N": 1024, "paths": {}}
        snapshots = {}
        graphs = {}
        for name, fn in fns.items():
            output = fn()
            assert output.dtype == torch.float32
            snapshot = output.cpu()
            snapshots[name] = snapshot
            result["paths"][name] = {
                "vs_cpu_fp32": errors(snapshot, reference),
                "vs_cpu_fp64_columns_0_16_32_etc": errors(snapshot[:, columns], fp64_samples),
            }
            for _ in range(10):
                output = fn()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(report["graph_unroll"]):
                    output = fn()
            # Keep all inputs and the graph output alive until after timing.
            graph.replay()
            torch.testing.assert_close(output.cpu(), snapshot, rtol=0, atol=0)
            graphs[name] = (graph, output)
        result["bf16_inputs_fp32_output_vs_npu_fp32"] = errors(
            snapshots["bf16_inputs_fp32_output"], snapshots["fp32_mm"]
        )
        invalid_control = torch.mm(hidden_bf16, weight_bf16.T).float().cpu()
        result["invalid_bf16_output_then_float_control"] = errors(invalid_control, reference)
        samples = {name: [] for name in fns}
        for round_id in range(report["rounds"]):
            names = list(fns) if round_id % 2 == 0 else list(reversed(fns))
            for name in names:
                graph = graphs[name][0]
                samples[name].append(measure(graph.replay, report["iterations_per_round"], report["graph_unroll"]))
        for name, values in samples.items():
            result["paths"][name]["timing"] = summarize(values)
        result["bf16_to_fp32_speedup"] = (
            result["paths"]["fp32_mm"]["timing"]["median_us"]
            / result["paths"]["bf16_inputs_fp32_output"]["timing"]["median_us"]
        )
        report["cases"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"T={tokens}: FP32 {result['paths']['fp32_mm']['timing']['median_us']:.3f} us, "
            f"BF16->FP32 {result['paths']['bf16_inputs_fp32_output']['timing']['median_us']:.3f} us, "
            f"speedup {result['bf16_to_fp32_speedup']:.3f}x, "
            f"NRMSE vs NPU FP32 {result['bf16_inputs_fp32_output_vs_npu_fp32']['nrmse']:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
