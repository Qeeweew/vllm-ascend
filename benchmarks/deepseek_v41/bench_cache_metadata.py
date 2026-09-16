# SPDX-License-Identifier: Apache-2.0
"""Compare target cache preparation with the equivalent decomposed NPU graph."""

import argparse
import importlib
import json

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--ratio", type=int, choices=(1, 2), default=2)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--fused-only", action="store_true")
    args = parser.parse_args()
    if not hasattr(torch.ops._C_ascend, "v41_cache_metadata"):
        bootstrap_custom_op_env(include_vendor_lib=True)
        importlib.import_module("vllm_ascend.vllm_ascend_C")
    torch.npu.set_device(args.device)
    batch, tokens, ratio = args.batch, args.tokens, args.ratio
    columns = (args.context + 16 * ratio - 1) // (16 * ratio)
    positions = torch.arange(tokens, device="npu", dtype=torch.int64) + args.context // 2
    cu = torch.linspace(0, tokens, batch + 1, dtype=torch.float32).to(device="npu", dtype=torch.int32)
    lengths = torch.full((batch,), args.context, device="npu", dtype=torch.int32)
    table = torch.arange(batch * columns, device="npu", dtype=torch.int32).view(batch, columns)
    po, co, lo, to = [torch.empty_like(x) for x in (positions, cu, lengths, table)]
    requests = torch.empty(tokens, device="npu", dtype=torch.int32)
    slots = torch.empty_like(positions)
    cmp, residual = torch.empty_like(lengths), torch.empty_like(lengths)
    indices = torch.arange(tokens, device="npu", dtype=torch.int32)

    def fused():
        torch.ops._C_ascend.v41_cache_metadata(
            positions, cu, lengths, table, po, co, lo, to, requests, slots, cmp, residual, 16 * ratio, 16, ratio, True
        )

    def split():
        po.copy_(positions)
        co.copy_(cu)
        lo.copy_(lengths)
        to.fill_(-1)
        to.copy_(table)
        torch.searchsorted(co[1:], indices, right=True, out_int32=True, out=requests)
        valid_token = (indices < co[-1]) & (po >= 0)
        valid = valid_token & ((po + 1) % ratio == 0)
        page = po // (16 * ratio)
        valid &= page < to.shape[1]
        physical = to.flatten().index_select(
            0, requests.clamp(0, batch - 1).long() * columns + page.clamp(0, columns - 1)
        )
        slots.copy_(physical.long() * 16 + (po % (16 * ratio)) // ratio)
        slots.masked_fill_(~valid | (physical < 0), -1)
        requests.masked_fill_(~valid_token, -1)
        torch.div(lo, ratio, rounding_mode="floor", out=cmp)
        torch.remainder(lo, ratio, out=residual)

    result = vars(args).copy()
    for name, fn in [("fused", fused)] + ([] if args.fused_only else [("split", split)]):
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(10):
                fn()
        for _ in range(3):
            graph.replay()
        torch.npu.synchronize()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            graph.replay()
        end.record()
        end.synchronize()
        result[name + "_us"] = start.elapsed_time(end) * 1000 / (args.iterations * 10)
    if "split_us" in result:
        result["speedup"] = result["split_us"] / result["fused_us"]
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
