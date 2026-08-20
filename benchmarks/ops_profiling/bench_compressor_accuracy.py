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


def build(M, B, kv_len, ratio, seed, start_pos_override=None, x_std=X_STD, w_std=W_STD,
          lengths=None, start_pos_arr=None, seqused_arr=None):
    """构造输入。

    扩展参数（覆盖边界压缩场景）：
      lengths:     每 batch token 数（非均匀 cu_seqlens）；None = 均匀（兼容旧调用）
      start_pos_arr: 每 batch 的 start_pos（P，全局位置，决定组对齐/首组 headHolder）；
                     None = 取 start_pos_override，再 None = 默认 kv_len - q_len
      seqused_arr:  每 batch 的 seq_used（S，消费 token 数，< lengths[b] 即尾部空洞，
                     末组不产出）；None = 无空洞（S = lengths[b]）
    """
    coff, state_block = RATIO_CONFIG[ratio]
    out_dim = coff * HEAD_DIM
    state_dim = 2 * out_dim
    torch.manual_seed(seed)
    q_len = 1 if B > 1 else M
    if lengths is None:
        lengths = [M // B] * B
    M = sum(lengths)  # 实际总 token
    max_blocks = (kv_len + state_block - 1) // state_block
    # 每 batch 独立 state block 区间（生产每请求独立 kv cache block 表；
    # 多 batch 写同一 block 会互相覆盖，污染 fused/split 对比）
    n_state_blocks = B * max_blocks + 1
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
    sbt = (torch.arange(B, dtype=torch.int32, device=DEVICE).unsqueeze(1) * max_blocks
           + torch.arange(1, max_blocks + 1, dtype=torch.int32, device=DEVICE))
    cu = [0]
    for l in lengths:
        cu.append(cu[-1] + l)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=DEVICE)
    if start_pos_arr is not None:
        start_pos = torch.tensor(start_pos_arr, dtype=torch.int32, device=DEVICE)
    elif start_pos_override is not None:
        start_pos = torch.full((B,), start_pos_override, dtype=torch.int32, device=DEVICE)
    else:
        start_pos = torch.full((B,), max(kv_len - q_len, 0), dtype=torch.int32, device=DEVICE)
    seqused = None if seqused_arr is None else torch.tensor(
        seqused_arr, dtype=torch.int32, device=DEVICE)
    w_cat = torch.cat([wkv_w, wgate_w], dim=0).contiguous()
    return dict(x=x, wkv_w=wkv_w, wgate_w=wgate_w, w_cat=w_cat, state_cache=state_cache,
                ape=ape, norm_w=norm_w, sin=sin, cos=cos, sbt=sbt, cu_seqlens=cu_seqlens,
                start_pos=start_pos, seqused=seqused, B=B, M=M, ratio=ratio,
                lengths=lengths)


def expected_comp_rows(B, ratio, start_pos_arr, seqused_arr):
    """实际产出的压缩行数 = Σ_b ((P+S)//ratio - P//ratio)：只有完整覆盖的组产出。
    cmp_kv 输出 shape 是上界 min(M, M//ratio+B)，前 expected 行是真实行，
    后续为 padding（未初始化，不计入判据）。"""
    total = 0
    for b in range(B):
        total += (start_pos_arr[b] + seqused_arr[b]) // ratio - start_pos_arr[b] // ratio
    return total


def _common_kwargs(t):
    coff = RATIO_CONFIG[t["ratio"]][0]
    return dict(
        state_block_table=t["sbt"], cu_seqlens=t["cu_seqlens"], seqused=t["seqused"],
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


def _stats(tag, ref, got, name="", n_rows=None):
    """分真实行 / padding 行统计。n_rows = 实际产出压缩行数（边界/空洞场景下
    少于输出上界 min(M, M//ratio+B)），前 n_rows 行为真实行，其余为 padding
    （at::empty 未初始化，不计入判据）。n_rows=None 时用旧语义（末行 padding）。"""
    a = ref.detach().float().cpu()
    b = got.detach().float().cpu()
    assert a.shape == b.shape, f"{tag} shape mismatch {a.shape} vs {b.shape}"
    d = (a - b).abs()
    if n_rows is None:
        valid = d[:-1]  # 真实压缩行
        pad = d[-1:]
    else:
        n_rows = min(n_rows, d.shape[0] - 1)  # 至少留 1 行 padding
        valid = d[:n_rows]
        pad = d[n_rows:]
    if valid.numel() == 0:
        # 无实际产出行（如 M 不满一组）：没有可对比的真实行，视为通过
        print(f"  {name:14s} no real rows (0 产出)  SKIP")
        return True
    denom = a[:valid.shape[0]].abs().max().item() + 1e-12
    rel = valid.max().item() / denom
    pad_abs = pad.max().item() if pad.numel() else 0.0
    ok = rel <= REL_TOL
    print(f"  {name:14s} real_rows_rel={rel:8.4f}  (pad_row_abs={pad_abs:9.3e})  {'PASS' if ok else 'FAIL'}")
    return ok


def _check_determinism(t, name, reps=2, n_rows=None):
    """同一输入同一路径跑两次：真实行必须逐位一致（无竞态）。
    n_rows = 实际产出行数（空洞/边界场景），padding 行未初始化不计入。"""
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
    if n_rows is None:
        real = d[:-1]
    else:
        real = d[:min(n_rows, d.shape[0])]
    real_max = real.max().item() if real.numel() else 0.0
    ok = real_max == 0.0
    print(f"  {'determinism':14s} {name:7s} real_rows_max_diff={real_max:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def run_case(M, B, kv_len, ratio, seed, spo, desc, lengths=None, start_pos_arr=None,
             seqused_arr=None):
    """通用 case：默认 start_pos=spo（None = 默认 kv_len-q_len）。
    传 lengths/start_pos_arr/seqused_arr 可覆盖多 batch 非均匀 / 独立 P / 空洞。"""
    t = build(M, B, kv_len, ratio, seed, spo, lengths=lengths,
              start_pos_arr=start_pos_arr, seqused_arr=seqused_arr)
    B_ = t["B"]
    if start_pos_arr is None:
        sp_arr = [spo if spo is not None else t["start_pos"][0].item()] * B_
    else:
        sp_arr = list(start_pos_arr)
    if seqused_arr is None:
        sq_arr = list(t["lengths"])
    else:
        sq_arr = list(seqused_arr)
    n_rows = expected_comp_rows(B_, ratio, sp_arr, sq_arr)
    print(f"== {desc} (M={t['M']}, B={B_}, kv_len={kv_len}, ratio={ratio}, "
          f"P={sp_arr}, S={sq_arr}) ==")
    base = t["state_cache"].clone()
    outs, caches = {}, {}
    for name in ("fused", "split1", "split2"):
        t["state_cache"] = base.clone()
        out, cache = RUNNERS[name](t)
        outs[name], caches[name] = out, cache
    torch.npu.synchronize()

    results = []
    results.append(_check_determinism(t, "fused", n_rows=n_rows))
    results.append(_check_determinism(t, "split2", n_rows=n_rows))
    for name in ("split1", "split2"):
        results.append(_stats("cmp_kv", outs["fused"], outs[name], f"fused vs {name}", n_rows))
        results.append(_stats("cache", caches["fused"], caches[name], f"cache {name}", n_rows))
    results.append(_stats("s1vs2", outs["split1"], outs["split2"], "split1 vs split2", n_rows))
    return all(results)


def run_decode_chain(B, kv_len, ratio, seed, steps, start_pos0, desc):
    """decode 连续多步：每步 M=B 个 token（每 batch 1 个），P 逐步 +1。
    压缩只在 P 到达 ratio 组边界时发生（decode 步跨边界），state 递归更新长链。
    fused 与 split2 各跑完整链，逐步对比输出 + 链末 state。"""
    print(f"== {desc} (B={B}, kv_len={kv_len}, ratio={ratio}, steps={steps}, "
          f"P0={start_pos0}) ==")
    # 两路独立链（同一权重/ape/norm/rope，各自 state 副本）
    base_t = build(1, B, kv_len, ratio, seed, start_pos_override=start_pos0)
    wkv_w, wgate_w, ape, norm_w = (base_t["wkv_w"], base_t["wgate_w"], base_t["ape"],
                                   base_t["norm_w"])
    # 每 batch 独立 state block 区间（生产 decode 多请求的 sbt 行互不相同，
    # 避免多 batch 写同一 state block 互相覆盖污染对比）
    state_block = RATIO_CONFIG[ratio][1]
    state_dim = 2 * RATIO_CONFIG[ratio][0] * HEAD_DIM
    max_blocks = (kv_len + state_block - 1) // state_block
    n_state_blocks = B * max_blocks + 1
    state_cache = torch.zeros(n_state_blocks, state_block, state_dim,
                              dtype=torch.float32, device=DEVICE)
    sbt = (torch.arange(B, dtype=torch.int32, device=DEVICE).unsqueeze(1) * max_blocks
           + torch.arange(max_blocks, dtype=torch.int32, device=DEVICE) + 1)
    kw = dict(state_block_table=sbt, cu_seqlens=torch.arange(B + 1, dtype=torch.int32,
                                                             device=DEVICE),
              seqused=None, rope_head_dim=ROPE_DIM, cmp_ratio=ratio,
              coff=RATIO_CONFIG[ratio][0], norm_eps=1e-6, rotary_mode=2, cache_mode=1)
    state_f = state_cache.clone()
    state_e = state_cache.clone()
    all_ok = True
    step_rels = []
    for step in range(steps):
        P = start_pos0 + step
        x = torch.randn(B, H, dtype=DTYPE, device=DEVICE) * X_STD
        sp = torch.full((B,), P, dtype=torch.int32, device=DEVICE)
        # 每步 1 个 token/请求；rope 行数必须 = min(M, M/ratio+B)（fused tiling 硬校验）
        produced = (P + 1) // ratio - P // ratio
        rope_tok = min(B, B // ratio + B)
        sin = torch.randn(rope_tok, ROPE_DIM, dtype=torch.float32, device=DEVICE)
        cos = torch.randn(rope_tok, ROPE_DIM, dtype=torch.float32, device=DEVICE)
        o_f = torch.ops._C_ascend.compressor(x, wkv_w, wgate_w, state_f, ape, norm_w,
                                             sin.view(-1, ROPE_DIM), cos.view(-1, ROPE_DIM),
                                             start_pos=sp, **kw)
        o_e = torch.ops._C_ascend.compressor_epilogue(
            nn.functional.linear(x, wkv_w), nn.functional.linear(x, wgate_w),
            state_e, ape, norm_w, sin.view(-1, ROPE_DIM), cos.view(-1, ROPE_DIM),
            start_pos=sp, **kw)
        if produced > 0:
            a = o_f.detach().float().cpu()[:produced * B]
            c = o_e.detach().float().cpu()[:produced * B]
            rel = (a - c).abs().max().item() / (c.abs().max().item() + 1e-12)
            step_rels.append(rel)
    torch.npu.synchronize()
    # 逐步输出对比（仅压缩步）
    if step_rels:
        max_rel = max(step_rels)
        ok_step = max_rel <= REL_TOL
        print(f"  压缩步输出 rel_max={max_rel:8.4f} (steps={len(step_rels)})  "
              f"{'PASS' if ok_step else 'FAIL'}")
        all_ok &= ok_step
    # 链末 state 对比（递归累积精度）
    df = (state_f - state_e).abs().max().item()
    denom = state_e.abs().max().item() + 1e-12
    rel = df / denom
    ok = rel <= REL_TOL
    print(f"  链末 state  rel={rel:8.4f}  (max_abs={df:.3e})  {'PASS' if ok else 'FAIL'}")
    return all_ok and ok


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


def run_c4_full(quick=False):
    """c4 全量：基础 6 case + 边界压缩场景（start_pos 扫描 / seqused 空洞 /
    非均匀多 batch / M 边界 / decode 链）。"""
    cases = [
        # (M, B, kv_len, ratio, seed, spo, desc, lengths, sp_arr, sq_arr)
        (512, 1, 4096, 4, 0, None, "prefill M=512"),
        (1024, 1, 4096, 4, 1, None, "prefill M=1024"),
        (4096, 1, 4096, 4, 2, None, "prefill M=4096"),
        (8192, 1, 8192, 4, 3, None, "prefill M=8192"),
        (1024, 1, 4096, 4, 5, 3, "start_pos=3 非对齐"),
        (8, 8, 4096, 4, 7, None, "decode B=8"),
    ]
    if quick:
        return cases
    # A. start_pos 扫描：P%4 != 0 时首组 headHolder>0（窗口混合 state 历史），
    #    边界压缩位置不同（首个压缩行在 P 对齐后）
    for P in (0, 1, 2, 4, 5, 7, 9, 127, 1021):
        cases.append((1024, 1, 4096, 4, 10 + P, P, f"start_pos={P} 边界扫描"))
    # B. seqused 空洞：S < lengths 时尾部组不产出（produce = gStart+r <= P+S）
    for S in (1024, 1023, 513, 512, 511, 9, 5, 4, 3, 1):
        cases.append((1024, 1, 4096, 4, 20 + S, 0, f"seqused={S} 空洞", None,
                      None, [S]))
    # C. 多 batch 非均匀：cu_seqlens 不等长、每 batch 独立 P/S、混合对齐/空洞
    cases.append((0, 3, 4096, 4, 30, None,
                  "B=3 非均匀 混合对齐", [1024, 2048, 512],
                  [0, 3, 1024], [1024, 1025, 500]))
    cases.append((0, 2, 4096, 4, 31, None,
                  "B=2 非均匀 P 交错", [5, 1000], [1, 4], [5, 999]))
    # D. M 边界：M 略小于/等于/大于 ratio 倍数；单 token 边界
    for M in (3, 4, 5, 7, 8, 9, 15, 16, 17):
        cases.append((M, 1, 4096, 4, 40 + M, 0, f"M={M} 边界"))
    cases.append((1, 1, 4096, 4, 50, 3, "M=1 单token 边界 P=3"))
    return cases


def run_c128_full():
    """c128 全量：基础 prefill + 边界场景（ratio=128：组大，边界更敏感）。"""
    cases = [
        (2048, 1, 4096, 128, 0, None, "c128 prefill M=2048"),
        (4096, 1, 4096, 128, 1, None, "c128 prefill M=4096"),
        (8192, 1, 8192, 128, 2, None, "c128 prefill M=8192"),
        (2048, 1, 4096, 128, 3, 3, "c128 start_pos=3 非对齐"),
        # 边界扫描（大 ratio 下 P 偏移敏感）
        (2048, 1, 4096, 128, 4, 127, "c128 start_pos=127"),
        (2048, 1, 4096, 128, 5, 128, "c128 start_pos=128 对齐"),
        (2048, 1, 4096, 128, 6, 129, "c128 start_pos=129"),
        (2048, 1, 4096, 128, 7, 255, "c128 start_pos=255"),
        # M 边界
        (127, 1, 4096, 128, 8, 0, "c128 M=127 不满一组"),
        (128, 1, 4096, 128, 9, 0, "c128 M=128 恰好一组"),
        (129, 1, 4096, 128, 10, 0, "c128 M=129 跨一组"),
        (256, 1, 4096, 128, 11, 0, "c128 M=256 两组"),
        # seqused 空洞
        (2048, 1, 4096, 128, 12, 0, "c128 seqused=127 空洞", None, None, [127]),
        (2048, 1, 4096, 128, 13, 0, "c128 seqused=128 完整一组", None, None, [128]),
        (2048, 1, 4096, 128, 14, 0, "c128 seqused=129", None, None, [129]),
    ]
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=int, choices=[4, 128], default=4)
    parser.add_argument("--quick", action="store_true",
                        help="只跑基础 case（快速冒烟）")
    parser.add_argument("--c128-diagnose", action="store_true",
                        help="只跑 c128 隔离诊断（fused vs epilogue）")
    args = parser.parse_args()

    if args.c128_diagnose or args.ratio == 128:
        # c128 隔离：先单 case 诊断（避免连续崩溃污染设备），再全量
        ok = run_c128(2048, 4096, 4)
        if not ok:
            sys.exit(1)
        print()
        cases = run_c128_full()
        all_ok = ok
        for c in cases:
            try:
                all_ok &= run_case(*c)
            except RuntimeError as e:
                all_ok = False
                print(f"  !! case 执行异常: {str(e)[:100]}")
        print("=" * 60)
        print("ALL PASS" if all_ok else "HAS FAILURES")
        sys.exit(0 if all_ok else 1)

    cases = run_c4_full(quick=args.quick)
    all_ok = True
    for c in cases:
        try:
            all_ok &= run_case(*c)
        except RuntimeError as e:
            all_ok = False
            print(f"  !! case 执行异常: {str(e)[:100]}")
    # E. decode 连续链（跨组边界 + state 递归累积）
    for B_, kv_len, steps, P0, desc in (
        (1, 4096, 16, 0, "decode 链 B=1 16步 P0=0"),
        (1, 4096, 9, 5, "decode 链 B=1 9步 P0=5（非对齐起）"),
        (4, 4096, 12, 0, "decode 链 B=4 12步"),
        (8, 4096, 6, 3, "decode 链 B=8 6步 P0=3"),
    ):
        try:
            all_ok &= run_decode_chain(B_, kv_len, 4, 30 + B_ + steps, steps, P0, desc)
        except RuntimeError as e:
            all_ok = False
            print(f"  !! decode 链异常: {str(e)[:100]}")
    print("=" * 60)
    print("ALL PASS" if all_ok else "HAS FAILURES")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
