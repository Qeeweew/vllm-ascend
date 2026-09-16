# SPDX-License-Identifier: Apache-2.0
"""Isolate BF16 context projection sensitivity to row padding on one NPU."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401 -- register NPU device
from dspark_v41_reference import ConvertedWeights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    weights = ConvertedWeights(args.checkpoint)
    weight = weights.read("mtp.0.main_proj.weight")
    device_weight = weight.to("npu")
    results = []
    with torch.inference_mode():
        for rows, bucket, seed in [(9, 16, 0), (33, 64, 1), (171, 256, 3)]:
            inputs = (
                torch.randn((rows, 15360), generator=torch.Generator().manual_seed(91203 + seed)).mul_(0.125).bfloat16()
            )
            device_inputs = inputs.to("npu")
            padded = torch.zeros((bucket, 15360), dtype=torch.bfloat16, device="npu")
            padded[:rows].copy_(device_inputs)
            unpadded_output = torch.nn.functional.linear(device_inputs, device_weight).cpu()
            padded_output = torch.nn.functional.linear(padded, device_weight)[:rows].cpu()
            reference = torch.nn.functional.linear(inputs[:8].float(), weight.float())
            difference = (unpadded_output.float() - padded_output.float()).abs()
            results.append(
                dict(
                    rows=rows,
                    bucket=bucket,
                    max_abs=float(difference.max()),
                    different_values=int((unpadded_output != padded_output).sum()),
                    total_values=unpadded_output.numel(),
                    nrmse=float((difference.square().mean() / unpadded_output.float().square().mean()).sqrt()),
                    unpadded_cpu_fp32_nrmse=float(
                        ((unpadded_output[:8].float() - reference).square().mean() / reference.square().mean()).sqrt()
                    ),
                    padded_cpu_fp32_nrmse=float(
                        ((padded_output[:8].float() - reference).square().mean() / reference.square().mean()).sqrt()
                    ),
                )
            )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
