"""DSA compressor 性能对比：fused vs packed linear + compress_norm_rope。

计时方式：NPU graph 捕获计算图 + replay 多次取平均（与 vllm-ascend 解码 graph 场景一致）。
decode 显式覆盖每请求 1/2/4 个 token；prefill 使用单请求连续 M 行。

两种实现：
- fused  : 单个 compressor 算子（内部 cube mm + 压缩计算）
- split_pack   : 权重 concat 成一个 (2*OUT_DIM, H) 矩阵，单次 linear + chunk + compress_norm_rope

请求位置相位均匀时，decode 每步期望有 B*q_len/cmp_ratio 个请求跨越
压缩边界。benchmark 捕获相邻的整数边界数，并线性插值得到期望耗时。
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

assert enable_custom_op(), "custom op 加载失败"

DEVICE = "npu"
DTYPE = torch.bfloat16
H, HEAD_DIM, ROPE_DIM = 7168, 512, 64
# ratio -> (coff, state_block)：C4A 双组权重 overlap，C128A 单组整行
RATIO_CFG = {4: (2, 8), 128: (1, 32)}
CMP = 4
COFF = 2
OUT_DIM = COFF * HEAD_DIM
STATE_DIM = 2 * OUT_DIM
DSA_STATE_BLOCK = 8
DSA_SEQ_LEN = 4096


def set_ratio(ratio):
    """按 ratio 设置模块全局（C4A: coff=2/state_block=8；C128A: coff=1/state_block=32）。"""
    global CMP, COFF, OUT_DIM, STATE_DIM, DSA_STATE_BLOCK
    CMP = ratio
    COFF, DSA_STATE_BLOCK = RATIO_CFG[ratio]
    OUT_DIM = COFF * HEAD_DIM
    STATE_DIM = 2 * OUT_DIM


def build(B, q_len=None, boundary_count=None):
    """构造 decode（B 个请求）或 prefill（单请求 B 行）输入。"""
    if B <= 128:
        q_len = 1 if q_len is None else q_len
        rows, batch, kv_len = B * q_len, B, DSA_SEQ_LEN
    else:
        rows, batch, q_len, kv_len = B, 1, B, B
    max_blocks = (kv_len + DSA_STATE_BLOCK - 1) // DSA_STATE_BLOCK
    n_state_blocks = max_blocks + 1
    x = torch.randn(rows, H, dtype=DTYPE, device=DEVICE)
    wkv_w = torch.randn(OUT_DIM, H, dtype=DTYPE, device=DEVICE)
    wgate_w = torch.randn(OUT_DIM, H, dtype=DTYPE, device=DEVICE)
    state_cache = torch.zeros(n_state_blocks, DSA_STATE_BLOCK, STATE_DIM,
                              dtype=torch.float32, device=DEVICE)
    ape = torch.randn(CMP, OUT_DIM, dtype=torch.float32, device=DEVICE)
    norm_w = torch.randn(HEAD_DIM, dtype=DTYPE, device=DEVICE)
    rope_tokens = min(rows, rows // CMP + batch)
    sin = torch.randn(rope_tokens, ROPE_DIM, dtype=torch.float32, device=DEVICE)
    cos = torch.randn(rope_tokens, ROPE_DIM, dtype=torch.float32, device=DEVICE)
    sbt = (1 + torch.arange(max_blocks, dtype=torch.int32, device=DEVICE)
           % (n_state_blocks - 1)).unsqueeze(0).expand(batch, -1).contiguous()
    cu_seqlens = torch.arange(0, batch + 1, dtype=torch.int32, device=DEVICE) * q_len
    if B <= 128 and boundary_count is not None:
        if q_len > CMP:
            raise ValueError("decode q_len must not exceed cmp_ratio")
        phase_base = DSA_SEQ_LEN - CMP
        start_pos_cpu = torch.full((batch,), phase_base, dtype=torch.int32)
        if boundary_count:
            start_pos_cpu[:boundary_count] = phase_base + CMP - q_len
        start_pos = start_pos_cpu.to(DEVICE)
    else:
        start_pos = torch.full((batch,), max(kv_len - q_len, 0), dtype=torch.int32, device=DEVICE)
    w_cat = torch.cat([wkv_w, wgate_w], dim=0).contiguous()
    return dict(x=x, wkv_w=wkv_w, wgate_w=wgate_w, w_cat=w_cat, state_cache=state_cache,
                ape=ape, norm_w=norm_w, sin=sin, cos=cos, sbt=sbt, cu_seqlens=cu_seqlens,
                start_pos=start_pos, B=B, q_len=q_len)


def run_fused(t):
    torch.ops._C_ascend.compressor(
        t["x"], t["wkv_w"], t["wgate_w"], t["state_cache"], t["ape"], t["norm_w"],
        t["sin"].view(-1, ROPE_DIM), t["cos"].view(-1, ROPE_DIM),
        state_block_table=t["sbt"], cu_seqlens=t["cu_seqlens"], seqused=None,
        start_pos=t["start_pos"], rope_head_dim=ROPE_DIM, cmp_ratio=CMP, coff=COFF,
        norm_eps=1e-6, rotary_mode=2, cache_mode=1)


def run_split_pack(t):
    mm = torch.nn.functional.linear(t["x"], t["w_cat"])
    mm_kv, mm_score = mm.chunk(2, dim=-1)
    torch.ops._C_ascend.compress_norm_rope(
        mm_kv, mm_score, t["state_cache"], t["ape"], t["norm_w"],
        t["sin"].view(-1, ROPE_DIM), t["cos"].view(-1, ROPE_DIM),
        state_block_table=t["sbt"], cu_seqlens=t["cu_seqlens"], seqused=None,
        start_pos=t["start_pos"], rope_head_dim=ROPE_DIM, cmp_ratio=CMP, coff=COFF,
        norm_eps=1e-6, rotary_mode=2, cache_mode=1)


MODES = {"fused": run_fused, "split_pack": run_split_pack}


class GraphBench:
    """NPU graph 捕获 + replay 计时（vllm-ascend 风格：graph 缓存不释放）。"""

    def __init__(self, warmup=3, iters=500, rounds=5):
        self.warmup = warmup
        self.iters = iters
        self.rounds = rounds
        self.graphs = {}  # (mode, B) -> (graph, t)

    def capture(self, fn, t):
        """warmup + 捕获计算图并缓存。"""
        for _ in range(self.warmup):
            fn(t)
        torch.npu.synchronize()
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            fn(t)
        torch.npu.synchronize()
        return g

    def measure(self, key):
        """对已捕获的 graph 做 replay 计时。"""
        g, _ = self.graphs[key]
        for _ in range(self.warmup):
            g.replay()
        torch.npu.synchronize()
        samples = []
        for _ in range(self.rounds):
            t0 = time.perf_counter()
            for _ in range(self.iters):
                g.replay()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - t0) / self.iters * 1e6)
        return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=int, default=4, choices=[4, 128],
                    help="cmp_ratio: 4 (C4A) or 128 (C128A)")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--decode-tokens", type=int, nargs="+", default=[1, 2, 4],
                    help="每个 decode 请求本步处理的 token 数")
    ap.add_argument("--out", default="msprof_out/compressor_graph_results.csv")
    args = ap.parse_args()
    set_ratio(args.ratio)

    # batch: 1,2,4,...,8192（B<=128 为 decode，B>128 为 prefill M）
    sizes = args.sizes

    bench = GraphBench(warmup=args.warmup, iters=args.iters, rounds=args.rounds)

    def scenarios(B, q_len):
        if B > 128:
            return (("normal", None),)
        if q_len > CMP:
            raise ValueError("--decode-tokens values must not exceed --ratio")
        expected = B * q_len / CMP
        lower = int(expected)
        upper = lower if expected == lower else lower + 1
        counts = {lower, upper}
        if q_len < CMP:
            counts.add(0)
        return tuple((f"boundaries={count}", count) for count in sorted(counts))

    cases = []
    for B in sizes:
        if B <= 128:
            cases.extend((B, q_len) for q_len in args.decode_tokens)
        else:
            cases.append((B, B))

    # 阶段1：全部 capture（graph 缓存，不释放）
    print("== capture all ==", flush=True)
    for B, q_len in cases:
        for scenario, boundary_count in scenarios(B, q_len):
            t = build(B, q_len, boundary_count)
            for name, fn in MODES.items():
                g = bench.capture(fn, t)
                bench.graphs[(name, B, q_len, scenario)] = (g, t)
                print(f"  captured {name:12s} B={B} q={q_len} {scenario}", flush=True)

    # 阶段2：统一 replay 计时
    print("== measure all ==", flush=True)
    rows = []
    for B, q_len in cases:
        tag = f"decode B={B} q={q_len}" if B <= 128 else f"prefill M={B}"
        measured = {}
        for scenario, boundary_count in scenarios(B, q_len):
            for name in MODES:
                us = bench.measure((name, B, q_len, scenario))
                measured[(scenario, name)] = us
                rows.append({"batch": B, "tokens_per_request": q_len, "tag": tag, "scenario": scenario,
                             "boundary_count": boundary_count, "mode": name, "time_us": us})
                print(f"{tag:24s} {scenario:14s} {name:12s} {us:9.2f} us", flush=True)
        if B <= 128:
            boundary_expectation = B * q_len / CMP
            lower = int(boundary_expectation)
            upper = lower if boundary_expectation == lower else lower + 1
            fraction = boundary_expectation - lower
            for name in MODES:
                lower_us = measured[(f"boundaries={lower}", name)]
                upper_us = measured[(f"boundaries={upper}", name)]
                expected_us = lower_us + fraction * (upper_us - lower_us)
                rows.append({"batch": B, "tokens_per_request": q_len, "tag": tag,
                             "scenario": "expected",
                             "boundary_count": boundary_expectation, "mode": name,
                             "time_us": expected_us})
                print(f"{tag:24s} {'expected':14s} {name:12s} {expected_us:9.2f} us", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["batch", "tokens_per_request", "tag", "scenario",
                                         "boundary_count", "mode", "time_us"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
