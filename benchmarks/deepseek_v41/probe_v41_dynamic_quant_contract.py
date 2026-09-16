# SPDX-License-Identifier: Apache-2.0
"""Baseline-only CANN dynamic quant contract; no custom operator is loaded."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu


def inputs():
    rows = [torch.zeros(128), torch.full((128,), -0.0)]
    names = ["zero", "negative_zero"]
    for maximum in (127.0, 63.5, 1.0, 0.125, 256.0, 1e-6):
        row = torch.linspace(-maximum, maximum, 128)
        row[0], row[1] = maximum, -maximum
        rows.append(row)
        names.append(f"range_{maximum}")
    for sign in (1.0, -1.0):
        row = torch.arange(128).float().sub_(63).add_(0.5).mul_(sign)
        row[0] = 127
        rows.append(row)
        names.append(f"halfway_sign_{sign}")
    generator = torch.Generator().manual_seed(412)
    for index in range(6):
        rows.append(torch.randn(128, generator=generator) * (2.0 ** (index - 3)))
        names.append(f"random_{index}")
    return names, torch.stack(rows).bfloat16()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path")
    names, values = inputs()
    torch.npu.set_device(0)
    report = {"status": "running", "torch_npu": torch_npu.__version__, "custom_operators_loaded": False, "cases": []}
    for tokens, offset in [(t, o) for t in (1, 2, 4, 16, 64, 128, 1024) for o in range(0, len(names), t)]:
        selected = values[(torch.arange(tokens) + offset) % len(names)]
        quantized, scale = torch_npu.npu_dynamic_quant(selected.npu(), dst_type=torch.int8)
        actual, actual_scale = quantized.cpu(), scale.cpu()
        fp16_scale = scale.half().cpu()
        maximum = selected.float().abs().amax(-1)
        reference_scale = maximum * torch.tensor(1.0 / 127.0, dtype=torch.float32)
        scaled = selected.float() * (127.0 / maximum[:, None])
        expected = scaled.round().clamp(-128, 127).to(torch.int8)
        nonzero = maximum != 0
        record = {
            "tokens": tokens,
            "pattern_offset": offset,
            "scale_dtype": str(actual_scale.dtype),
            "nonzero_quant_exact": torch.equal(actual[nonzero], expected[nonzero]),
            "fp32_scale_bits_exact": torch.equal(actual_scale.view(torch.int32), reference_scale.view(torch.int32)),
            "fp16_scale_bits_exact": torch.equal(
                fp16_scale.view(torch.int16), reference_scale.half().view(torch.int16)
            ),
            "mismatched_quant_count": int((actual[nonzero] != expected[nonzero]).sum()),
            "rows": [],
        }
        for index in range(min(tokens, len(names))):
            record["rows"].append(
                {
                    "name": names[(index + offset) % len(names)],
                    "input_bf16_bits": selected[index].view(torch.int16).tolist(),
                    "quantized": actual[index].tolist(),
                    "scale_fp32": actual_scale[index].item(),
                    "scale_fp32_bits": actual_scale[index].view(torch.int32).item(),
                    "scale_fp16_bits": fp16_scale[index].view(torch.int16).item(),
                }
            )
        report["cases"].append(record)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    source = Path("/usr/local/Ascend/cann").resolve() / (
        "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/dynamic_quant"
    )
    report["installed_source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source.glob("dynamic_quant*.h"))
    }
    report["status"] = "observed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": [{k: v for k, v in case.items() if k != "rows"} for case in report["cases"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
