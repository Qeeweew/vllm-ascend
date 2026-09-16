# SPDX-License-Identifier: Apache-2.0
"""Record CANN quantization and generic division at observed INT8 boundaries."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    torch.npu.set_device(0)
    rows = []
    for value, maximum in ((9.8125, 19.625), (-11.875, 23.75), (10.6875, 21.375), (9.375, 18.75)):
        for tokens in (1, 64):
            key = torch.zeros((tokens, 128), dtype=torch.bfloat16)
            key[:, 0], key[:, 1] = maximum, value
            maxima = key.float().abs().amax(-1, keepdim=True)
            numerator = torch.full_like(maxima, 127)
            cpu_multiplier = 127.0 / maxima
            device_maxima, device_numerator = maxima.npu(), numerator.npu()
            generic_device_div = (device_numerator / device_maxima).cpu()
            generic_device_reciprocal = device_maxima.reciprocal().cpu()
            q, scale = torch_npu.npu_dynamic_quant(key.npu(), dst_type=torch.int8)
            rows.append(
                {
                    "tokens": tokens,
                    "value": value,
                    "maximum": maximum,
                    "numerator": 127.0,
                    "cpu_reciprocal_times_127": float(cpu_multiplier[0, 0]),
                    "cpu_direct_tensor_division": float((numerator / maxima)[0, 0]),
                    "cpu_product": float((key.float() * cpu_multiplier)[0, 1]),
                    "cpu_quantized": int((key.float() * cpu_multiplier).round()[0, 1]),
                    "generic_npu_division": float(generic_device_div[0, 0]),
                    "generic_npu_reciprocal": float(generic_device_reciprocal[0, 0]),
                    "generic_npu_division_bits": int(generic_device_div.view(torch.int32)[0, 0]),
                    "cann_dynamic_quantized": int(q.cpu()[0, 1]),
                    "cann_scale_fp32": float(scale.cpu()[0]),
                    "cann_scale_fp16_bits": int(scale.half().cpu().view(torch.int16)[0]),
                }
            )
    source_root = (
        Path("/usr/local/Ascend/cann").resolve() / "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/dynamic_quant"
    )
    sources = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source_root / "dynamic_quant_single_row.h", source_root / "dynamic_quant_multi_row.h")
    }
    report = {
        "status": "recorded",
        "scope": "CANN baseline only; generic NPU division is observed separately, not read from quantizer internal UB",
        "source_sha256": sources,
        "cases": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
