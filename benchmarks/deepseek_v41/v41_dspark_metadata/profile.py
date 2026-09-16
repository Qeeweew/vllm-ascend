# SPDX-License-Identifier: Apache-2.0
"""Small msprof op workload; select the V41DsparkMetadata kernel."""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    torch.npu.set_device(0)
    tokens = 5 * args.batch
    cu = torch.arange(0, tokens + 1, 5, dtype=torch.int32, device="npu")
    lengths = torch.full((args.batch,), 4096, dtype=torch.int32, device="npu")
    spans = torch.full((tokens, 1), 133, dtype=torch.int32, device="npu")
    schedule = torch.empty(1024, dtype=torch.int32, device="npu")
    for _ in range(4):
        torch.ops._C_ascend.v41_dspark_metadata(cu, lengths, spans, schedule)
        torch.npu.synchronize()
    assert schedule[0].cpu().item() == 1


if __name__ == "__main__":
    main()
