"""fused compressor vs split（MatMulV3 + compressor_epilogue）精度与稳定性对比测试。

用法：
    python bench_compressor_accuracy.py               # c4 全量（默认，含确定性/精度/padding 行分析）
    python bench_compressor_accuracy.py --ratio 128   # c128 诊断（崩溃报 KNOWN-BUG，修复后自动转精度对比）

历次排查沉淀的设计要点（勿回退）：
1. state_cache 就地更新：每条路径必须从同一个原始副本 clone，否则结果互相污染。
2. 输入必须用真实模型尺度：x ~ N(0,1)（归一化激活），w ~ N(0, 1/sqrt(hidden))
   （ReplicatedLinear kaiming 初始化，1/sqrt(7168)≈0.012）。纯 randn 时 score
   量级 ~sqrt(H)≈85，softmax 的 exp 放大把 bf16 中间精度差异放大到 ~9%；真实
   尺度 score~1 时实测真实行差异仅 0.3~0.8%。
3. split1 的 mm.chunk(2, -1) 返回非 contiguous 视图（stride=2048），epilogue 对
   非 contiguous 输入处理有缺陷（M=8192 实测 s1vs2 差 13%）；生产路径是独立
   F.linear（天然 contiguous）。脚本显式 .contiguous() 对齐生产语义。
4. cmp_kv 行数是上界 min(tokenSize, tokenSize/ratio+reqs)（dsa_v1.py:607），
   最后 1 行是 padding 预留槽位：compressor_metadata 给无效行 slot_mapping=-1，
   dsa_kv_compress_scatter 跳过（dsa_v1.py:2171），从不被消费。统计必须区分
   "真实行"（前 N-1 行）与 padding 行——padding 行是 at::empty 未初始化内存
   （M=8192 实测 nan/inf/0.266 随机），不计入精度判据。
5. coff：c4=2（overlap），c128=1；state_cache block：c4=8、c128=32
   （DSV4_BLOCK_SIZES[128]=[[128,128,8,32],...]，models/layer/attention/layer.py）。
6. 确定性：同一输入同一路径跑两次，真实行必须逐位一致（diff=0）——M=8192 曾
   出现 padding 行 nan 假象，真实行始终确定。
"""

import argparse
import sys

import torch
import torch_npu  # noqa: F401
from torch import nn

from vllm_ascend.utils import enable_custom_op

assert enable_custom_op(), "custom op 加载失败"

DEVICE = "npu"
DTYPE = torch.bfloat16
H, HEAD_DIM, ROPE_DIM = 7168, 512, 64
DSA_SEQ_LEN = 4096
# 真实模型尺度：x 为归一化激活 ~N(0,1)；wkv/wgate 为 ReplicatedLinear kaiming
# 初始化 ~N(0, 1/sqrt(H))（1/sqrt(7168)≈0.012），score 量级 ~1。
X_STD, W_STD = 1.0, 1.0 / (H**0.5)
# 判据：真实行相对误差阈值（实测 0.3~0.8%）
REL_TOL = 1e-2

# ratio -> (coff, state_block)：c4 overlap=2、block=8；c128 overlap=1、block=32
# （DSV4_BLOCK_SIZES[128] = [[128,128,8,32], ...]）
RATIO_CONFIG = {4: (2, 8), 128: (1, 32)}


def build(M, B, kv_len, ratio, seed, start_pos_override=None, x_std=X_STD, w_std=W_STD):
    coff, state_block = RATIO_CONFIG[ratio]
    out_dim = coff * HEAD_DIM
    state_dim = 2 * out_dim
    torch.manual_seed(seed)
    q_len = 1 if B > 1 else M
    max_blocks = (kv_len + state_block - 1) // state_block
    n_state_blocks = max_blocks + 1
    x = torch.randn(M, H, dtype=DTYPE, device=DEVICE) * x_std
    wkv_w = torch.randn(out_dim, H, dtype=DTYPE, device=DEVICE) * w_std
    wgate_w = torch.randn(out_dim, H, dtype=DTYPE, device=DEVICE) * w_std
    state_cache = torch.zeros(n_state_blocks, state_block, state_dim,
                              dtype=torch.float32, device=DEVICE)
    ape = torch.randn(ratio, out_dim, dtype=torch.float32, device=DEVICE)
    norm_w = torch.randn(HEAD_DIM, dtype=DTYPE, device=DEVICE)
    rope_tokens = min(M, M // ratio + B)
    sin = torch.randn(rope_tokens, ROPE_DIM, dtype=torch.float32, device=DEVICE)
    cos = torch.randn(rope_tokens, ROPE_DIM, dtype=torch.float32, device=DEVICE)
    sbt = (1 + torch.arange(max_blocks, dtype=torch.int32, device=DEVICE)
           % (n_state_blocks - 1)).unsqueeze(0).expand(B, -1).contiguous()
    cu_seqlens = torch.arange(0, B + 1, dtype=torch.int32, device=DEVICE) * q_len
    if start_pos_override is not None:
        start_pos = torch.full((B,), start_pos_override, dtype=torch.int32, device=DEVICE)
    else:
        start_pos = torch.full((B,), max(kv_len - q_len, 0), dtype=torch.int32, device=DEVICE)
    w_cat = torch.cat([wkv_w, wgate_w], dim=0).contiguous()
    return dict(x=x, wkv_w=wkv_w, wgate_w=wgate_w, w_cat=w_cat, state_cache=state_cache,
                ape=ape, norm_w=norm_w, sin=sin, cos=cos, sbt=sbt, cu_seqlens=cu_seqlens,
                start_pos=start_pos, B=B, M=M, ratio=ratio)


def _common_kwargs(t):
    coff = RATIO_CONFIG[t["ratio"]][0]
    return dict(
        state_block_table=t["sbt"], cu_seqlens=t["cu_seqlens"], seqused=None,
        start_pos=t["start_pos"], rope_head_dim=ROPE_DIM, cmp_ratio=t["ratio"],
        coff=coff, norm_eps=1e-6, rotary_mode=2, cache_mode=1,
    )


def run_fused(t):
    out = torch.ops._C_ascend.compressor(
        t["x"], t["wkv_w"], t["wgate_w"], t["state_cache"], t["ape"], t["norm_w"],
        t["sin"].view(-1, ROPE_DIM), t["cos"].view(-1, ROPE_DIM), **_common_kwargs(t))
    return out, t["state_cache"]


def run_split1(t):
    """生产语义对齐：w_cat 拼接 + chunk 后必须 .contiguous()（epilogue 不支持
    非 contiguous 视图，详见文件头第 3 点）。"""
    mm = nn.functional.linear(t["x"], t["w_cat"])
    mm_kv, mm_score = mm.chunk(2, dim=-1)
    mm_kv, mm_score = mm_kv.contiguous(), mm_score.contiguous()
    out = torch.ops._C_ascend.compressor_epilogue(
        mm_kv, mm_score, t["state_cache"], t["ape"], t["norm_w"],
        t["sin"].view(-1, ROPE_DIM), t["cos"].view(-1, ROPE_DIM), **_common_kwargs(t))
    return out, t["state_cache"]


def run_split2(t):
    """生产路径 dsa_v1.py:1600 的写法：两个独立 F.linear（输出天然 contiguous）。"""
    mm_kv = nn.functional.linear(t["x"], t["wkv_w"])
    mm_score = nn.functional.linear(t["x"], t["wgate_w"])
    out = torch.ops._C_ascend.compressor_epilogue(
        mm_kv, mm_score, t["state_cache"], t["ape"], t["norm_w"],
        t["sin"].view(-1, ROPE_DIM), t["cos"].view(-1, ROPE_DIM), **_common_kwargs(t))
    return out, t["state_cache"]


RUNNERS = {"fused": run_fused, "split1": run_split1, "split2": run_split2}


def _stats(tag, ref, got, name=""):
    """分真实行 / padding 行统计。cmp_kv 行数 = min(M, M//ratio+B) 是上界，
    最后一行是预留 padding（slot=-1 从不消费），其未初始化内容不计入判据。"""
    a = ref.detach().float().cpu()
    b = got.detach().float().cpu()
    assert a.shape == b.shape, f"{tag} shape mismatch {a.shape} vs {b.shape}"
    d = (a - b).abs()
    valid = d[:-1]  # 真实压缩行
    denom = a[:-1].abs().max().item() + 1e-12
    rel = valid.max().item() / denom
    pad_abs = d[-1].max().item() if d.shape[0] > 1 else 0.0
    ok = rel <= REL_TOL
    print(f"  {name:14s} real_rows_rel={rel:8.4f}  (pad_row_abs={pad_abs:9.3e})  {'PASS' if ok else 'FAIL'}")
    return ok


def _check_determinism(t, name, reps=2):
    """同一输入同一路径跑两次：真实行必须逐位一致（无竞态）。"""
    base = t["state_cache"].clone()
    outs = []
    for _ in range(reps):
        t["state_cache"] = base.clone()
        out, _ = RUNNERS[name](t)
        outs.append(out)
    torch.npu.synchronize()
    a = outs[0].detach().float().cpu()
    c = outs[1].detach().float().cpu()
    d = (a - c).abs()
    real_max = d[:-1].max().item()
    ok = real_max == 0.0
    print(f"  {'determinism':14s} {name:7s} real_rows_max_diff={real_max:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def run_case(M, B, kv_len, ratio, seed, spo, desc):
    print(f"== {desc} (M={M}, B={B}, kv_len={kv_len}, ratio={ratio}, start_pos={spo}) ==")
    t = build(M, B, kv_len, ratio, seed, spo)
    base = t["state_cache"].clone()
    outs, caches = {}, {}
    for name in ("fused", "split1", "split2"):
        t["state_cache"] = base.clone()
        out, cache = RUNNERS[name](t)
        outs[name], caches[name] = out, cache
    torch.npu.synchronize()

    results = []
    results.append(_check_determinism(t, "fused"))
    results.append(_check_determinism(t, "split2"))
    for name in ("split1", "split2"):
        results.append(_stats("cmp_kv", outs["fused"], outs[name], f"fused vs {name}"))
        results.append(_stats("cache", caches["fused"], caches[name], f"cache {name}"))
    results.append(_stats("s1vs2", outs["split1"], outs["split2"], "split1 vs split2"))
    return all(results)


def run_c128(M, kv_len, seed):
    """c128：fused 应正常。epilogue 有两个待修问题（见 reports/compressor_epilogue_c128_analysis.md）：
    1) UB 越界崩溃（ape = coff*cmpRatio*dSplitSize*4B，c128 dSplitSize=64 时 32KB > apeBuf 16K）
    2) d 分块破坏 rms_norm（epilogue 单核串行、rms_norm col=headDim=512 读 d 片）
    现状 epilogue 崩溃 -> 报 KNOWN-BUG；修复后自动转入精度对比（fused vs epilogue + 确定性）。
    单独跑（--ratio 128 / --c128-diagnose），避免崩溃污染设备状态。"""
    print(f"== c128 诊断 (M={M}, kv_len={kv_len}) ==")
    t = build(M, 1, kv_len, 128, seed)
    base = t["state_cache"].clone()

    t["state_cache"] = base.clone()
    try:
        out_f, cache_f = run_fused(t)
        torch.npu.synchronize()
        print(f"  fused OK, shape={tuple(out_f.shape)}")
    except RuntimeError as e:
        print(f"  fused FAIL: {str(e)[:80]}")
        return False

    t["state_cache"] = base.clone()
    try:
        out_e, cache_e = run_split2(t)
        torch.npu.synchronize()
    except RuntimeError as e:
        print(f"  epilogue KNOWN-BUG (UB 越界): {str(e)[:80]}")
        print("  -> c128 + split（VLLM_ASCEND_DSA_COMPRESSOR_SPLIT=1）需修复 epilogue，"
              "见 reports/compressor_epilogue_c128_analysis.md")
        return False

    # epilogue 不崩 -> 转入精度对比（修复验证）
    print("  epilogue OK，进入精度对比:")
    t["state_cache"] = base.clone()
    ok = _check_determinism(t, "split2")
    ok &= _stats("cmp_kv", out_f, out_e, "fused vs epi")
    ok &= _stats("cache", cache_f, cache_e, "cache epi")
    print("  " + ("PASS" if ok else "FAIL"))
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=int, choices=[4, 128], default=4)
    parser.add_argument("--c128-diagnose", action="store_true",
                        help="只跑 c128 隔离诊断（fused vs epilogue）")
    args = parser.parse_args()

    if args.c128_diagnose or args.ratio == 128:
        ok = run_c128(2048, 4096, 4)
        sys.exit(0 if ok else 1)

    cases = [
        (512, 1, 4096, 4, 0, None, "prefill M=512"),
        (1024, 1, 4096, 4, 1, None, "prefill M=1024"),
        (4096, 1, 4096, 4, 2, None, "prefill M=4096"),
        (8192, 1, 8192, 4, 3, None, "prefill M=8192"),
        (1024, 1, 4096, 4, 5, 3, "start_pos=3 非对齐"),
        (8, 8, 4096, 4, 7, None, "decode B=8"),
    ]
    all_ok = True
    for M, B, kv_len, ratio, seed, spo, desc in cases:
        try:
            all_ok &= run_case(M, B, kv_len, ratio, seed, spo, desc)
        except RuntimeError as e:
            all_ok = False
            print(f"  !! case 执行异常: {str(e)[:100]}")
    print("=" * 60)
    print("ALL PASS" if all_ok else "HAS FAILURES")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
