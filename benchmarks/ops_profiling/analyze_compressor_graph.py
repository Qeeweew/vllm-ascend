"""从 compressor_graph_results.csv 计算全矩阵对比数值（报告数据来源）。

用法: python benchmarks/ops_profiling/analyze_compressor_graph.py [csv路径]
"""
import csv
import sys


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else (
        "benchmarks/ops_profiling/msprof_out/compressor_graph_results.csv")
    with open(csv_path) as csv_file:
        rows = list(csv.DictReader(csv_file))
    selected = [r for r in rows if r["scenario"] in ("expected", "normal")]
    d = {(r["tag"], r["mode"]): float(r["time_us"]) for r in selected}
    tags = list(dict.fromkeys(r["tag"] for r in selected))
    for tag in tags:
        fused = d[(tag, "fused")]
        packed = d[(tag, "split_pack")]
        print(f"{tag:24s} fused={fused:8.2f} us  pack={packed:8.2f} us  "
              f"speedup={fused / packed:.3f}x")


if __name__ == "__main__":
    main()
