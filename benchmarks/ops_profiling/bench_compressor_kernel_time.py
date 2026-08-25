"""DSA compressor 设备侧 kernel 时间对比（fused vs split2G vs split1G 合并）。

- 权重 concat 在计时外（一次性预处理，不计入）
- 只统计 device kernel 时间（torch profiler self_device_time）
- 覆盖 prefill M 矩阵与 decode B 矩阵
"""
import os
import sys
import torch
import torch_npu
from torch import profiler

from vllm_ascend.utils import enable_custom_op
assert enable_custom_op(), "custom op 加载失败"

DEVICE = "npu"
DTYPE = torch.bfloat16
H, HEAD_DIM, ROPE_DIM, CMP = 7168, 512, 64, 4
COFF = 2
OUT_DIM = COFF * HEAD_DIM
STATE_DIM = 2 * OUT_DIM
DSA_STATE_BLOCK = 8
DSA_SEQ_LEN = 4096


def build(M, B, kv_len):
    q_len = 1 if B > 1 else M
    max_blocks = (kv_len + DSA_STATE_BLOCK - 1) // DSA_STATE_BLOCK
    n_state_blocks = max_blocks + 1
    x = torch.randn(M, H, dtype=DTYPE, device=DEVICE)
    wkv_w = torch.randn(OUT_DIM, H, dtype=DTYPE, device=DEVICE)
    wgate_w = torch.randn(OUT_DIM, H, dtype=DTYPE, device=DEVICE)
    state_cache = torch.zeros(n_state_blocks, DSA_STATE_BLOCK, STATE_DIM,
                              dtype=torch.float32, device=DEVICE)
    ape = torch.randn(CMP, OUT_DIM, dtype=torch.float32, device=DEVICE)
    norm_w = torch.randn(HEAD_DIM, dtype=DTYPE, device=DEVICE)
    rope_tokens = min(M, M // CMP + B)
    sin = torch.randn(rope_tokens, ROPE_DIM, dtype=torch.float32, device=DEVICE)
    cos = torch.randn(rope_tokens, ROPE_DIM, dtype=torch.float32, device=DEVICE)
    sbt = (1 + torch.arange(max_blocks, dtype=torch.int32, device=DEVICE)
           % (n_state_blocks - 1)).unsqueeze(0).expand(B, -1).contiguous()
    cu_seqlens = torch.arange(0, B + 1, dtype=torch.int32, device=DEVICE) * q_len
    start_pos = torch.full((B,), max(kv_len - q_len, 0), dtype=torch.int32, device=DEVICE)
    # 预合并权重（不计时）
    w_cat = torch.cat([wkv_w, wgate_w], dim=0).contiguous()
    return dict(x=x, wkv_w=wkv_w, wgate_w=wgate_w, w_cat=w_cat, state_cache=state_cache,
                ape=ape, norm_w=norm_w, sin=sin, cos=cos, sbt=sbt, cu_seqlens=cu_seqlens,
                start_pos=start_pos, B=B, M=M)


def run_fused(t):
    torch.ops._C_ascend.compressor(
        t["x"], t["wkv_w"], t["wgate_w"], t["state_cache"], t["ape"], t["norm_w"],
        t["sin"].view(-1, ROPE_DIM), t["cos"].view(-1, ROPE_DIM),
        state_block_table=t["sbt"], cu_seqlens=t["cu_seqlens"], seqused=None,
        start_pos=t["start_pos"], rope_head_dim=ROPE_DIM, cmp_ratio=CMP, coff=COFF,
        norm_eps=1e-6, rotary_mode=2, cache_mode=1)


def run_split2(t):
    mm_kv = torch.nn.functional.linear(t["x"], t["wkv_w"])
    mm_score = torch.nn.functional.linear(t["x"], t["wgate_w"])
    torch.ops._C_ascend.compressor_epilogue(
        mm_kv, mm_score, t["state_cache"], t["ape"], t["norm_w"],
        t["sin"].view(-1, ROPE_DIM), t["cos"].view(-1, ROPE_DIM),
        state_block_table=t["sbt"], cu_seqlens=t["cu_seqlens"], seqused=None,
        start_pos=t["start_pos"], rope_head_dim=ROPE_DIM, cmp_ratio=CMP, coff=COFF,
        norm_eps=1e-6, rotary_mode=2, cache_mode=1)


def run_split1(t):
    mm = torch.nn.functional.linear(t["x"], t["w_cat"])
    mm_kv, mm_score = mm.chunk(2, dim=-1)
    torch.ops._C_ascend.compressor_epilogue(
        mm_kv, mm_score, t["state_cache"], t["ape"], t["norm_w"],
        t["sin"].view(-1, ROPE_DIM), t["cos"].view(-1, ROPE_DIM),
        state_block_table=t["sbt"], cu_seqlens=t["cu_seqlens"], seqused=None,
        start_pos=t["start_pos"], rope_head_dim=ROPE_DIM, cmp_ratio=CMP, coff=COFF,
        norm_eps=1e-6, rotary_mode=2, cache_mode=1)


def measure(fn, t, iters=5, warmup=2):
    for _ in range(warmup):
        fn(t)
    torch.npu.synchronize()
    with profiler.profile(activities=[profiler.ProfilerActivity.CPU,
                                      profiler.ProfilerActivity.PrivateUse1]) as p:
        for _ in range(iters):
            fn(t)
        torch.npu.synchronize()
    # 汇总 device kernel 时间
    total = 0.0
    names = {}
    for e in p.key_averages():
        if e.self_device_time_total > 0:
            total += e.self_device_time_total
            names[e.key] = e.self_device_time_total
    return total / iters / 1000.0, names


def main():
    # 单场景模式：python bench_compressor_kernel_time.py <fused|split2|split1> <M>
    # M > 100 视为 prefill M，否则视为 decode batch B
    mode = sys.argv[1]
    size = int(sys.argv[2])
    if size > 100:
        t = build(size, 1, max(size, DSA_SEQ_LEN))
        tag = f"prefill M={size}"
    else:
        t = build(size, size, DSA_SEQ_LEN)
        tag = f"decode  B={size}"
    fn = {"fused": run_fused, "split2": run_split2, "split1": run_split1}[mode]
    for _ in range(2):
        fn(t)
    torch.npu.synchronize()
    kt, names = measure(fn, t)
    print(f"{tag} {mode:7s} kernel_time={kt:8.1f}us  {names}")


if __name__ == "__main__":
    main()
