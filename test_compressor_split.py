# Numerical validation: fused compressor vs split (MatMulV3 x2 + compressor_epilogue)
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks/ops_profiling"))
import bench_deepseek_v4 as B

from vllm_ascend.utils import enable_custom_op

enable_custom_op()
torch.npu.set_device("npu:0")

H = 7168
HEAD_DIM = 512
ROPE_DIM = 64
OUT_DIM = 2 * HEAD_DIM  # coff * head_dim
CMP = B.DSA_CMP_RATIO  # 4
EPS = 1e-6


def build_case(Bq, q_len, seq_len, start_pos_val, seed=0):
    torch.manual_seed(seed)
    M = Bq * q_len
    x = torch.randn(M, H, dtype=torch.bfloat16, device="npu")
    wkv_w = torch.randn(OUT_DIM, H, dtype=torch.bfloat16, device="npu") * 0.02
    wgate_w = torch.randn(OUT_DIM, H, dtype=torch.bfloat16, device="npu") * 0.02
    max_state_blocks = (seq_len + B.DSA_STATE_BLOCK - 1) // B.DSA_STATE_BLOCK
    n_state_blocks = max_state_blocks + 1
    state_cache = torch.zeros(Bq * n_state_blocks, B.DSA_STATE_BLOCK, 2 * OUT_DIM,
                              dtype=torch.float32, device="npu")
    ape = torch.randn(CMP, OUT_DIM, dtype=torch.float32, device="npu") * 0.1
    norm_w = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="npu")
    rope_tokens = min(M, M // CMP + Bq)
    sin = torch.randn(rope_tokens, ROPE_DIM, dtype=torch.float32, device="npu")
    cos = torch.randn(rope_tokens, ROPE_DIM, dtype=torch.float32, device="npu")
    sbt = (1 + torch.arange(max_state_blocks, dtype=torch.int32, device="npu")
           % n_state_blocks).unsqueeze(0).expand(Bq, -1).contiguous()
    # block table entry 0 is null; remap to per-request disjoint blocks
    sbt = (1 + torch.arange(Bq * max_state_blocks, dtype=torch.int32, device="npu")).view(Bq, max_state_blocks)
    cu = torch.arange(0, Bq + 1, dtype=torch.int32, device="npu") * q_len
    start_pos = torch.full((Bq,), start_pos_val, dtype=torch.int32, device="npu")
    return dict(x=x, wkv_w=wkv_w, wgate_w=wgate_w, state_cache=state_cache, ape=ape, norm_w=norm_w,
                sin=sin, cos=cos, sbt=sbt, cu=cu, start_pos=start_pos, M=M, Bq=Bq)


def run_fused(t):
    out = torch.ops._C_ascend.compressor(
        t["x"], t["wkv_w"], t["wgate_w"], t["state_cache"], t["ape"], t["norm_w"],
        t["sin"], t["cos"], state_block_table=t["sbt"], cu_seqlens=t["cu"], seqused=None,
        start_pos=t["start_pos"], rope_head_dim=ROPE_DIM, cmp_ratio=CMP, coff=2, norm_eps=EPS,
        rotary_mode=2, cache_mode=1)
    return out


def run_split(t):
    mm_kv = torch.nn.functional.linear(t["x"], t["wkv_w"])
    mm_score = torch.nn.functional.linear(t["x"], t["wgate_w"])
    out = torch.ops._C_ascend.compressor_epilogue(
        mm_kv, mm_score, t["state_cache"], t["ape"], t["norm_w"], t["sin"], t["cos"],
        state_block_table=t["sbt"], cu_seqlens=t["cu"], seqused=None, start_pos=t["start_pos"],
        rope_head_dim=ROPE_DIM, cmp_ratio=CMP, coff=2, norm_eps=EPS, rotary_mode=2, cache_mode=1)
    return out


def compare(tag, ref_out, ref_state, new_out, new_state, written_rows=None):
    assert ref_out.shape == new_out.shape, f"{tag}: shape {ref_out.shape} vs {new_out.shape}"
    # 输出行数 = min(T, T/r+B)，比实际写入行数多（未写行是未初始化内存，不可比）；只比写入行
    if written_rows is None:
        written_rows = ref_out.shape[0]
    # state: only compare written (non-zero) rows
    smask = ref_state.abs().sum(-1) > 0
    nmask = new_state.abs().sum(-1) > 0
    sdiff = (ref_state - new_state).abs()
    if written_rows == 0:
        # 该用例设计上不产生压缩行（如 decode 组未完成），仅比较 state 连续性
        print(f"[{tag}] cmp_kv skipped (0 written rows by design) | "
              f"state rows ref={smask.sum().item()} new={nmask.sum().item()} "
              f"state_max_abs={sdiff.max().item():.4e}")
        return
    a, b = ref_out[:written_rows].float(), new_out[:written_rows].float()
    diff = (a - b).abs()
    rel = (diff / (a.abs() + 1e-3))
    print(f"[{tag}] cmp_kv written_rows={written_rows}/{ref_out.shape[0]} max_abs={diff.max().item():.4e} "
          f"mean_abs={diff.mean().item():.4e} rel>1%: {(rel > 0.01).float().mean().item():.4%} | "
          f"state rows ref={smask.sum().item()} new={nmask.sum().item()} "
          f"state_max_abs={sdiff.max().item():.4e}")


def written_rows_of(bq, q_len, start_pos_val):
    # TH 输出按 batch 紧凑排列：每 batch 写入 (startPos+qLen)//r - startPos//r 行
    return sum((start_pos_val + q_len) // CMP - start_pos_val // CMP for _ in range(bq))


def main():
    cases = [
        ("prefill_B1_q8192", 1, 8192, 8192, 0),
        ("prefill_B4_q512", 4, 512, 512, 0),
        ("decode_B8_kv4096", 8, 1, 4096, 4095),
        ("decode_B64_kv4096", 64, 1, 4096, 4095),
        ("prefill_chunk_B2_q300_kv1024", 2, 300, 1024, 724),  # startPos 非 4 对齐，跨 chunk
    ]

    for tag, bq, ql, sl, sp in cases:
        t_ref = build_case(bq, ql, sl, sp, seed=hash(tag) % 10000)
        t_new = build_case(bq, ql, sl, sp, seed=hash(tag) % 10000)
        out_ref = run_fused(t_ref)
        out_new = run_split(t_new)
        torch.npu.synchronize()
        compare(tag, out_ref, t_ref["state_cache"], out_new, t_new["state_cache"],
                written_rows_of(bq, ql, sp))

    # 两步 decode：第二步读第一步写的 state（注意 start_pos 未步进，组始终未完成，
    # 输出一行都不写，仅比较 state 连续性）
    tag = "decode_2step_B8"
    t_ref = build_case(8, 1, 4096, 4094, seed=7)
    t_new = build_case(8, 1, 4096, 4094, seed=7)
    run_fused(t_ref)
    run_split(t_new)
    out_ref = run_fused(t_ref)
    out_new = run_split(t_new)
    torch.npu.synchronize()
    compare(tag, out_ref, t_ref["state_cache"], out_new, t_new["state_cache"], 0)
    print("DONE")


if __name__ == "__main__":
    main()
