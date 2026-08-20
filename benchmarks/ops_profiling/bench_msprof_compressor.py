"""msprof app 模式采集用：fused vs split2 vs split1 全场景 launch（支持 C4/C128）。

每个 (场景, 实现) 连续 3 次 launch（msprof op_summary 每 launch 一行），
解析时取中位数。权重 concat 在计时外。复用 bench_compressor_graph 的 build/run。
"""
import argparse
import sys

sys.path.insert(0, ".")
import torch
import torch_npu  # noqa: F401

import bench_compressor_graph as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=int, default=4, choices=[4, 128])
    ap.add_argument("--decode", default="1,8,32,128", help="decode B 列表")
    ap.add_argument("--prefill", default="1024,4096,8192", help="prefill M 列表")
    args = ap.parse_args()
    G.set_ratio(args.ratio)

    cases = []
    for b in (int(x) for x in args.decode.split(",")):
        cases.append((f"decode_{b}", G.build(b)))
    for m in (int(x) for x in args.prefill.split(",")):
        cases.append((f"prefill_{m}", G.build(m)))
    impls = (("fused", G.run_fused), ("split2", G.run_split_nopack),
             ("split1", G.run_split_pack))
    for tag, t in cases:
        for name, fn in impls:
            for _ in range(3):
                fn(t)
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
