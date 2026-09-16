# SPDX-License-Identifier: Apache-2.0
"""Small fixed-input workload for isolated msprof op collection."""

import argparse

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.v41_rope_cache import v41_index_cache_store, v41_main_cache_store, v41_rope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("rope", "main", "index"), required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, choices=(1, 8, 32), default=1)
    parser.add_argument("--width", type=int, choices=(128, 512), default=512)
    parser.add_argument("--ratio", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.tokens < 1 or (args.kind != "rope" and args.heads != 1):
        parser.error("positive tokens and single-head cache rows required")
    torch.npu.set_device(0)
    width = (128 if args.kind == "index" else 512) if args.kind != "rope" else args.width
    shape = (args.tokens, args.heads, width) if args.heads != 1 else (args.tokens, width)
    generator = torch.Generator().manual_seed(4141)
    x = torch.randn(shape, generator=generator).bfloat16().npu()
    positions = (torch.arange(args.tokens, dtype=torch.int64) * args.ratio + args.ratio - 1).npu()
    slots = torch.arange(args.tokens, dtype=torch.int64).npu()
    angles = torch.randn((args.tokens * args.ratio, 32), generator=generator)
    cos, sin = angles.cos().npu(), angles.sin().npu()
    if args.kind == "rope":
        output = torch.empty_like(x)

        def call():
            v41_rope(x, positions, cos, sin, output)
    else:
        blocks = (args.tokens + 31) // 32
        if args.kind == "index":
            stride = 32 * (width + 2)
            raw = torch.zeros(blocks * stride, dtype=torch.uint8, device="npu")
            cache = raw.view(torch.int8).as_strided((blocks, 32, 1, width), (stride, width, width, 1))
            scales = raw.view(torch.float16).as_strided((blocks, 32, 1), (stride // 2, 1, 1), 32 * width // 2)
        else:
            cache = torch.zeros((blocks, 32, 1, width), dtype=torch.bfloat16, device="npu")
            scales = None

        def call():
            if args.kind == "index":
                v41_index_cache_store(x, positions, slots, cos, sin, cache, scales, compress_ratio=args.ratio)
            else:
                v41_main_cache_store(x, positions, slots, cos, sin, cache, compress_ratio=args.ratio)

    for _ in range(6):
        call()
        torch.npu.synchronize()


if __name__ == "__main__":
    main()
