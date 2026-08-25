"""从 compressor_graph_results.csv 计算全矩阵对比数值（报告数据来源）。

用法: python benchmarks/ops_profiling/analyze_compressor_graph.py [csv路径]
"""
import csv
import sys
from pathlib import Path

# 硬件峰值（AGENTS.md）：BF16 cube 245.76 TFLOPS
P_CUBE_TFLOPS = 245.76
H, OUT_DIM = 7168, 1024


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else (
        "benchmarks/ops_profiling/msprof_out/compressor_graph_results.csv")
    rows = list(csv.DictReader(open(csv_path)))
    d = {(r["tag"], r["mode"]): float(r["time_us"]) for r in rows}

    print("=== decode（B 请求 × 1 token，kv_len=4096）===")
    for B in [1, 2, 4, 8, 16, 32, 64, 128]:
        tag = f"decode B={B}"
        f, p, n = d[(tag, "fused")], d[(tag, "split_pack")], d[(tag, "split_nopack")]
        print(f"B={B:4d} fused={f:7.1f} pack={p:7.1f} nopack={n:7.1f} | "
              f"pack_speedup={f/p:.3f}x nopack_speedup={f/n:.3f}x pack_vs_nopack={p/n:.3f}")

    print("\n=== prefill（单请求连续 M 行，kv_len=M）===")
    for M in [256, 512, 1024, 2048, 4096, 8192]:
        tag = f"prefill M={M}"
        f, p, n = d[(tag, "fused")], d[(tag, "split_pack")], d[(tag, "split_nopack")]
        flops = 4 * M * H * OUT_DIM  # 2 GEMM × 2 (MAC→FLOP)
        tflops = flops / (p * 1e-6) / 1e12
        sol = tflops / P_CUBE_TFLOPS * 100
        print(f"M={M:5d} fused={f:7.1f} pack={p:7.1f} nopack={n:7.1f} | "
              f"pack_speedup={f/p:.3f}x nopack_speedup={f/n:.3f}x pack_vs_nopack={p/n:.3f} | "
              f"pack_TFLOPS={tflops:5.1f} ({sol:.1f}% SOL)")


if __name__ == "__main__":
    main()
