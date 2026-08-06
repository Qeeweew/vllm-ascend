#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark the core vLLM-Ascend operators used by DeepSeek-V4 inference.

This script is a config-driven extension of ``benchmarks/ops_profiling/bench_ops.py``.
It covers the operator categories A-G listed in
``docs/vllm-ascend 推理算子分类与 SOL 评测维度.md`` (H 通信类算子不在本脚本范围内).

For every operator we:

1. Build a synthetic workload that matches the V4-Pro / V4-Flash config.
2. Run warmup + iterations with host-side wall-clock timing (``torch.npu.synchronize``).
3. Estimate the hardware lower-bound (SOL) from the workload bytes/FLOPs.
4. Print a markdown-friendly table and write a JSON result file.

The SOL numbers are *single-card* lower bounds (TP/EP/CP/DP are not applied).
For device-side numbers, wrap the script with ``msprof`` (see ``run_msprof.sh``);
the JSON can be re-loaded with ``--msprof-summary`` to compute real SOL gaps.

Usage (quick host timing, V4-Pro):
    python benchmarks/ops_profiling/bench_deepseek_v4.py --model V4-Pro --iters 100

Usage with msprof app mode:
    bash benchmarks/ops_profiling/run_msprof.sh app all 30
    # then pass the generated op_summary.csv back to the script:
    python benchmarks/ops_profiling/bench_deepseek_v4.py \
        --model V4-Pro --msprof-summary PROF_*/mindstudio_profiler_output/op_summary_*.csv

Usage with msprof op mode for a single case:
    bash benchmarks/ops_profiling/run_msprof.sh op rms_norm_prefill 30
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op

CUSTOM_OP_AVAILABLE = enable_custom_op()

# ---------------------------------------------------------------------------
# Hardware / model constants
# ---------------------------------------------------------------------------
DEVICE = os.environ.get("VLLM_ASCEND_BENCH_DEVICE", "npu:0")
DTYPE = torch.bfloat16

# Ascend 910B4-1 single-die peak parameters (see docs/vllm-ascend 算子性能评测实践指南.md)
P_CUBE_FLOPS = 246e12          # bf16 Cube TFLOPS
BW_HBM_BPS = 1.6e12            # HBM bytes/s
P_VEC_ELEM_S = 4.2e12          # AIV element/s (f16 simple elementwise)

# V4-Pro / V4-Flash configs derived from the classification doc.
# These are fallbacks when the HF config cannot be fetched.
V4_PRO_CONFIG = {
    "model_type": "deepseek_v4",
    "hidden_size": 7168,
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1536,
    "o_lora_rank": 1024,
    "o_groups": 16,
    "sliding_window": 128,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 1024,
    "num_hash_layers": 3,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "scoring_func": "sqrtsoftplus",
    "routed_scaling_factor": 2.5,
    "moe_intermediate_size": 3072,
    "expert_dtype": "fp4",
    "vocab_size": 129280,
    "num_nextn_predict_layers": 1,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 5,
    "rms_norm_eps": 1e-6,
    "hc_eps": 1e-6,
}

V4_FLASH_CONFIG = {
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "num_hidden_layers": 43,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1024,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "sliding_window": 128,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 512,
    "num_hash_layers": 3,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "scoring_func": "sqrtsoftplus",
    "routed_scaling_factor": 2.5,
    "moe_intermediate_size": 2048,
    "expert_dtype": "fp4",
    "vocab_size": 129280,
    "num_nextn_predict_layers": 1,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 5,
    "rms_norm_eps": 1e-6,
    "hc_eps": 1e-6,
}

MODELS = {
    "V4-Pro": V4_PRO_CONFIG,
    "V4-Flash": V4_FLASH_CONFIG,
}

# Default batch/sequence sizes for the two inference stages.
M_PREFILL_DEFAULT = 4096
M_DECODE_DEFAULT = 1
ATTN_SEQ_LEN = 2048


def _round_up(n: int, k: int) -> int:
    return (n + k - 1) // k * k


# ---------------------------------------------------------------------------
# Helpers: timing, SOL estimation, CSV parsing
# ---------------------------------------------------------------------------
def _bench(fn: Callable[[], None], iters: int, warmup: int) -> float:
    """Host-side wall-clock timing with NPU synchronization. Returns avg us."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters * 1e6


def _bench_mstx(name: str, fn: Callable[[], None], iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        rid = torch_npu.npu.mstx.range_start(name)
        fn()
        torch_npu.npu.mstx.range_end(rid)
    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters * 1e6


def _t_cube_us(flops: float) -> float:
    return flops / P_CUBE_FLOPS * 1e6


def _t_gm_us(bytes_moved: float) -> float:
    return bytes_moved / BW_HBM_BPS * 1e6


def _t_vec_us(elements: float) -> float:
    return elements / P_VEC_ELEM_S * 1e6


def _parse_msprof_summary(csv_path: str) -> Dict[str, float]:
    """Return a mapping {op_name_prefix: average Task Duration(us)}.

    Aggregation is by the first 34 characters of Op Name to match the snippet
    in the ops_profiling README.
    """
    rows = list(csv.DictReader(open(csv_path)))
    agg: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        name = r["Op Name"][:34]
        try:
            agg[name].append(float(r["Task Duration(us)"]))
        except (KeyError, ValueError):
            continue
    return {k: sum(v) / len(v) for k, v in agg.items()}


# ---------------------------------------------------------------------------
# Operator cases: builder + SOL estimator
# ---------------------------------------------------------------------------
@dataclass
class Case:
    name: str
    category: str          # A, B, C, D, E, F, G
    stage: str             # prefill / decode / other
    build: Callable[[Dict], Callable[[], None]]
    sol: Callable[[Dict], Dict[str, float]]
    # candidate msprof op names; first substring match is used
    msprof_names: List[str] = field(default_factory=list)


def _cfg_m(cfg: Dict, stage: str) -> int:
    return M_DECODE_DEFAULT if stage == "decode" else M_PREFILL_DEFAULT


# ------------------------- A: RMSNorm -------------------------
def _build_rms_norm(cfg: Dict, stage: str) -> Callable[[], None]:
    M = _cfg_m(cfg, stage)
    H = cfg["hidden_size"]
    x = torch.randn(M, H, dtype=DTYPE, device=DEVICE)
    w = torch.randn(H, dtype=DTYPE, device=DEVICE)
    eps = cfg["rms_norm_eps"]

    def fn() -> None:
        torch_npu.npu_rms_norm(x, w, eps)

    return fn


def _sol_rms_norm(cfg: Dict, stage: str) -> Dict[str, float]:
    M = _cfg_m(cfg, stage)
    H = cfg["hidden_size"]
    bytes_moved = 2 * M * H * 2  # read x + write y; weight cached in UB
    vec_ops = 2 * M * H          # reduce + mul
    return {
        "T_cube_us": _t_cube_us(0.0),
        "T_vec_us": _t_vec_us(vec_ops),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": max(_t_vec_us(vec_ops), _t_gm_us(bytes_moved)),
        "workload": f"M={M}, H={H}",
    }


def _build_add_rms_norm(cfg: Dict, stage: str) -> Callable[[], None]:
    M = _cfg_m(cfg, stage)
    H = cfg["hidden_size"]
    x = torch.randn(M, H, dtype=DTYPE, device=DEVICE)
    r = torch.randn_like(x)
    w = torch.randn(H, dtype=DTYPE, device=DEVICE)
    eps = cfg["rms_norm_eps"]

    def fn() -> None:
        torch_npu.npu_add_rms_norm(x, r, w, eps)

    return fn


def _sol_add_rms_norm(cfg: Dict, stage: str) -> Dict[str, float]:
    M = _cfg_m(cfg, stage)
    H = cfg["hidden_size"]
    bytes_moved = 4 * M * H * 2  # read x, read r, write x, write r
    vec_ops = 2 * M * H
    return {
        "T_cube_us": _t_cube_us(0.0),
        "T_vec_us": _t_vec_us(vec_ops),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": max(_t_vec_us(vec_ops), _t_gm_us(bytes_moved)),
        "workload": f"M={M}, H={H}",
    }


def _build_add_rms_norm_bias(cfg: Dict, stage: str) -> Callable[[], None]:
    if not CUSTOM_OP_AVAILABLE:
        raise RuntimeError("custom op not available")
    M = _cfg_m(cfg, stage)
    H = cfg["hidden_size"]
    x = torch.randn(M, H, dtype=DTYPE, device=DEVICE)
    r = torch.randn_like(x)
    gamma = torch.randn(H, dtype=DTYPE, device=DEVICE)
    bias = torch.randn(H, dtype=DTYPE, device=DEVICE)
    eps = cfg["rms_norm_eps"]

    def fn() -> None:
        torch.ops._C_ascend.npu_add_rms_norm_bias(x, r, gamma, bias, eps)

    return fn


# ------------------------- B: RoPE -------------------------
def _build_rotary_mul(cfg: Dict, stage: str) -> Callable[[], None]:
    M = _cfg_m(cfg, stage)
    n_heads = cfg["num_attention_heads"]
    head_dim = cfg["qk_rope_head_dim"]
    # npu_rotary_mul expects (B, S, N, D) and (B, S, 1, D)
    x = torch.randn(1, M, n_heads, head_dim, dtype=DTYPE, device=DEVICE)
    cos = torch.randn(1, M, 1, head_dim, dtype=DTYPE, device=DEVICE)
    sin = torch.randn(1, M, 1, head_dim, dtype=DTYPE, device=DEVICE)

    def fn() -> None:
        torch_npu.npu_rotary_mul(x, cos, sin)

    return fn


def _sol_rotary_mul(cfg: Dict, stage: str) -> Dict[str, float]:
    M = _cfg_m(cfg, stage)
    n_heads = cfg["num_attention_heads"]
    head_dim = cfg["qk_rope_head_dim"]
    bytes_moved = 2 * M * n_heads * head_dim * 2  # read x + write y
    vec_ops = M * n_heads * head_dim
    return {
        "T_cube_us": _t_cube_us(0.0),
        "T_vec_us": _t_vec_us(vec_ops),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": max(_t_vec_us(vec_ops), _t_gm_us(bytes_moved)),
        "workload": f"M={M}, N={n_heads}, D={head_dim}",
    }


# ------------------------- C: Attention -------------------------
def _build_fused_attention(cfg: Dict, stage: str) -> Callable[[], None]:
    # The fused attention operator on NPU supports head_dim 64/128/192 in TND
    # mode, so we use a representative head_dim=64 for the micro-benchmark while
    # keeping the V4 token/head counts.  The exact V4 MLA path uses mla_preprocess
    # and DSA custom kernels, which are benchmarked separately below.
    # q_len is clamped to the mask/seq size; with large --M-prefill the case
    # measures a 2048-token sub-batch (single request).
    M = min(_cfg_m(cfg, stage), ATTN_SEQ_LEN) if stage != "decode" else _cfg_m(cfg, stage)
    seq_len = ATTN_SEQ_LEN
    n_heads = 64
    n_kv_heads = 1
    head_dim = 64
    block_size = 128
    n_blocks = (seq_len + block_size - 1) // block_size
    query = torch.randn(M, n_heads, head_dim, dtype=DTYPE, device=DEVICE)
    key = torch.randn(n_blocks, n_kv_heads, block_size, head_dim, dtype=DTYPE, device=DEVICE)
    value = key
    block_table = torch.arange(n_blocks, dtype=torch.int32, device=DEVICE).unsqueeze(0)
    attn_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.uint8, device=DEVICE)
    actual_q = torch.tensor([M], dtype=torch.int32, device=DEVICE)
    actual_kv = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
    scale = 1.0 / math.sqrt(head_dim)

    def fn() -> None:
        torch_npu.npu_fused_infer_attention_score_v2(
            query, key, value,
            num_query_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            input_layout="TND",
            pre_tokens=seq_len,
            next_tokens=0,
            atten_mask=attn_mask,
            sparse_mode=3,
            softmax_scale=scale,
            block_table=block_table,
            block_size=block_size,
            actual_seq_qlen=actual_q,
            actual_seq_kvlen=actual_kv,
        )

    return fn


def _sol_fused_attention(cfg: Dict, stage: str) -> Dict[str, float]:
    M = min(_cfg_m(cfg, stage), ATTN_SEQ_LEN) if stage != "decode" else _cfg_m(cfg, stage)
    seq_len = ATTN_SEQ_LEN
    n_heads = 64
    n_kv_heads = 1
    head_dim = 64
    # QK + PV matmuls
    flops = 2 * M * seq_len * head_dim * n_heads * 2
    bytes_moved = (
        M * n_heads * head_dim * 2
        + seq_len * n_kv_heads * head_dim * 2 * 2
        + M * n_heads * head_dim * 2
    )
    return {
        "T_cube_us": _t_cube_us(flops),
        "T_vec_us": _t_vec_us(M * n_heads * head_dim),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": max(_t_cube_us(flops), _t_gm_us(bytes_moved)),
        "workload": f"M={M}, seq={seq_len}, N={n_heads}, D={head_dim}",
    }


def _build_bmm_transpose(cfg: Dict, stage: str) -> Callable[[], None]:
    # DeepSeek V3.2 (SFA) true dims: 128 heads, kv_lora=512, v_head_dim=128.
    # This op is only called by the SFA path (V3.2), V4 DSA does not use it.
    M = min(_cfg_m(cfg, stage), 1024)  # operator limit in SFA
    n_heads = 128
    kv_lora = 512
    v_head = 128
    x = torch.randn(M, n_heads, kv_lora, dtype=DTYPE, device=DEVICE)
    W = torch.randn(n_heads, kv_lora, v_head, dtype=DTYPE, device=DEVICE)
    res = torch.empty(M, n_heads, v_head, dtype=DTYPE, device=DEVICE)

    def fn() -> None:
        torch.ops._C_ascend.batch_matmul_transpose(x, W, res)

    return fn


def _sol_bmm_transpose(cfg: Dict, stage: str) -> Dict[str, float]:
    M = min(_cfg_m(cfg, stage), 1024)
    n_heads = 128
    kv_lora = 512
    v_head = 128
    flops = 2 * M * n_heads * kv_lora * v_head
    bytes_moved = (M * n_heads * kv_lora + n_heads * kv_lora * v_head + M * n_heads * v_head) * 2
    return {
        "T_cube_us": _t_cube_us(flops),
        "T_vec_us": 0.0,
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": max(_t_cube_us(flops), _t_gm_us(bytes_moved)),
        "workload": f"M={M}, N={n_heads}, L={kv_lora}, V={v_head}",
    }


def _build_store_kv_block(cfg: Dict, stage: str) -> Callable[[], None]:
    if not CUSTOM_OP_AVAILABLE:
        raise RuntimeError("custom op not available")
    M = _cfg_m(cfg, stage)
    head_dim = cfg["head_dim"]
    n_blocks = 32
    block_size = 128
    key = torch.randn(M, 1, head_dim, dtype=DTYPE, device=DEVICE)
    kv_cache = torch.randn(n_blocks, block_size, 1, head_dim, dtype=DTYPE, device=DEVICE)
    group_len = torch.full((M,), 1, dtype=torch.int32, device=DEVICE)
    group_key_idx = torch.arange(M, dtype=torch.int32, device=DEVICE)
    group_key_cache_idx = torch.arange(M, dtype=torch.int32, device=DEVICE)

    def fn() -> None:
        torch.ops._C_ascend.store_kv_block(
            key, kv_cache, group_len, group_key_idx, group_key_cache_idx, block_size
        )

    return fn


def _sol_store_kv_block(cfg: Dict, stage: str) -> Dict[str, float]:
    M = _cfg_m(cfg, stage)
    head_dim = cfg["head_dim"]
    bytes_moved = 2 * M * head_dim * 2
    return {
        "T_cube_us": 0.0,
        "T_vec_us": _t_vec_us(M * head_dim),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": _t_gm_us(bytes_moved),
        "workload": f"M={M}, D={head_dim}",
    }


# ------------------------- D: Sparse index / HC -------------------------
def _build_hc_pre(cfg: Dict, stage: str) -> Callable[[], None]:
    if not CUSTOM_OP_AVAILABLE:
        raise RuntimeError("custom op not available")
    M = _cfg_m(cfg, stage)
    H = cfg["hidden_size"]
    hc_mult = cfg["hc_mult"]
    hc_dim = hc_mult * H
    x = torch.randn(M, hc_mult, H, dtype=DTYPE, device=DEVICE)
    hc_fn = torch.randn((2 + hc_mult) * hc_mult, hc_dim, dtype=torch.float32, device=DEVICE)
    hc_scale = torch.randn(3, dtype=torch.float32, device=DEVICE)
    hc_base = torch.randn((2 + hc_mult) * hc_mult, dtype=torch.float32, device=DEVICE)
    eps = cfg["rms_norm_eps"]
    hc_eps = cfg["hc_eps"]
    iters = cfg["hc_sinkhorn_iters"]

    def fn() -> None:
        torch.ops._C_ascend.npu_hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, iters, eps, hc_eps)

    return fn


def _sol_hc_pre(cfg: Dict, stage: str) -> Dict[str, float]:
    M = _cfg_m(cfg, stage)
    H = cfg["hidden_size"]
    hc_mult = cfg["hc_mult"]
    hc_dim = hc_mult * H
    # linear: x_flat [M*hc_mult, H] x hc_fn [(2+hc_mult)*hc_mult, hc_dim]
    flops = 2 * M * hc_mult * H * (2 + hc_mult) * hc_mult
    bytes_moved = (M * hc_mult * H + (2 + hc_mult) * hc_mult * hc_dim + M * hc_mult * H) * 2
    return {
        "T_cube_us": _t_cube_us(flops),
        "T_vec_us": _t_vec_us(M * hc_mult * H * 2),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": max(_t_cube_us(flops), _t_gm_us(bytes_moved)),
        "workload": f"M={M}, hc_mult={hc_mult}, H={H}",
    }


# ------------------------- E: MoE -------------------------
def _build_moe_gating_top_k(cfg: Dict, stage: str) -> Callable[[], None]:
    if not CUSTOM_OP_AVAILABLE:
        raise RuntimeError("custom op not available")
    M = _cfg_m(cfg, stage)
    n_experts = cfg["n_routed_experts"]
    topk = cfg["num_experts_per_tok"]
    router_logits = torch.randn(M, n_experts, dtype=DTYPE, device=DEVICE)
    # V4 path: experts_selector._select_experts_with_fusion_ops, sqrtsoftplus branch.
    # scoring_func=sqrtsoftplus -> moe_gating_top_k_hash(norm_type=2), no groups,
    # renorm done in Python (kernel renorm=0), bias only affects selection.
    bias = torch.randn(n_experts, dtype=DTYPE, device=DEVICE)
    rsf = cfg.get("routed_scaling_factor", 2.5)

    def fn() -> None:
        torch.ops._C_ascend.moe_gating_top_k_hash(
            x=router_logits,
            k=topk,
            bias=bias,
            input_ids=None,
            tid2eid=None,
            k_group=1,
            group_count=1,
            routed_scaling_factor=rsf,
            eps=1e-20,
            group_select_mode=1,
            renorm=0,
            norm_type=2,
            out_flag=False,
        )

    return fn


def _sol_moe_gating_top_k(cfg: Dict, stage: str) -> Dict[str, float]:
    M = _cfg_m(cfg, stage)
    n_experts = cfg["n_routed_experts"]
    bytes_moved = 2 * M * n_experts * 2
    return {
        "T_cube_us": 0.0,
        "T_vec_us": _t_vec_us(M * n_experts),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": _t_gm_us(bytes_moved),
        "workload": f"M={M}, n_experts={n_experts}, k={cfg['num_experts_per_tok']}, sqrtsoftplus",
    }


def _build_moe_init_routing(cfg: Dict, stage: str) -> Callable[[], None]:
    if not CUSTOM_OP_AVAILABLE:
        raise RuntimeError("custom op not available")
    M = _cfg_m(cfg, stage)
    H = cfg["hidden_size"]
    n_experts = cfg["n_routed_experts"]
    topk = cfg["num_experts_per_tok"]
    hidden = torch.randn(M, H, dtype=DTYPE, device=DEVICE)
    topk_ids = torch.randint(0, n_experts, (M, topk), dtype=torch.int32, device=DEVICE)

    def fn() -> None:
        torch.ops._C_ascend.npu_moe_init_routing_custom(
            hidden,
            topk_ids,
            scale=None,
            active_num=M * topk,
            expert_num=n_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, n_experts],
            quant_mode=-1,
        )

    return fn


def _sol_moe_init_routing(cfg: Dict, stage: str) -> Dict[str, float]:
    M = _cfg_m(cfg, stage)
    H = cfg["hidden_size"]
    topk = cfg["num_experts_per_tok"]
    bytes_moved = (M * H + M * topk * H) * 2
    return {
        "T_cube_us": 0.0,
        "T_vec_us": _t_vec_us(M * topk * H),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": _t_gm_us(bytes_moved),
        "workload": f"M={M}, topk={topk}, H={H}",
    }


def _build_grouped_matmul(cfg: Dict, stage: str) -> Callable[[], None]:
    M = _cfg_m(cfg, stage)
    H = cfg["hidden_size"]
    I = cfg["moe_intermediate_size"]
    n_experts = cfg["n_routed_experts"]
    topk = cfg["num_experts_per_tok"]
    n_tokens = M * topk
    # evenly split tokens across experts for the benchmark
    per_expert = n_tokens // n_experts
    hidden = torch.randn(n_tokens, H, dtype=DTYPE, device=DEVICE)
    weight = torch.randn(n_experts, H, I, dtype=DTYPE, device=DEVICE)
    counts = torch.full((n_experts,), per_expert, dtype=torch.int64, device=DEVICE)
    group_list = counts.cumsum(0)

    def fn() -> None:
        torch_npu.npu_grouped_matmul(
            x=[hidden],
            weight=[weight],
            group_list=group_list,
            group_list_type=0,
            group_type=0,
            split_item=2,
        )

    return fn


def _sol_grouped_matmul(cfg: Dict, stage: str) -> Dict[str, float]:
    M = _cfg_m(cfg, stage)
    H = cfg["hidden_size"]
    I = cfg["moe_intermediate_size"]
    topk = cfg["num_experts_per_tok"]
    n_experts = cfg["n_routed_experts"]
    flops = 2 * M * topk * H * I
    bytes_moved = (M * topk * H + n_experts * H * I + M * topk * I) * 2  # simplified
    return {
        "T_cube_us": _t_cube_us(flops),
        "T_vec_us": 0.0,
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": max(_t_cube_us(flops), _t_gm_us(bytes_moved)),
        "workload": f"M={M}, topk={topk}, n_experts={n_experts}, H={H}, I={I}",
    }


def _build_moe_token_unpermute(cfg: Dict, stage: str) -> Callable[[], None]:
    M = _cfg_m(cfg, stage)
    H = cfg["hidden_size"]
    topk = cfg["num_experts_per_tok"]
    permuted = torch.randn(M * topk, H, dtype=DTYPE, device=DEVICE)
    sorted_idx = torch.arange(M * topk, dtype=torch.int32, device=DEVICE)
    probs = torch.randn(M, topk, dtype=DTYPE, device=DEVICE)

    def fn() -> None:
        torch_npu.npu_moe_token_unpermute(permuted, sorted_idx, probs)

    return fn


def _sol_moe_token_unpermute(cfg: Dict, stage: str) -> Dict[str, float]:
    M = _cfg_m(cfg, stage)
    H = cfg["hidden_size"]
    topk = cfg["num_experts_per_tok"]
    bytes_moved = (M * topk * H + M * H) * 2
    return {
        "T_cube_us": 0.0,
        "T_vec_us": _t_vec_us(M * topk * H),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": _t_gm_us(bytes_moved),
        "workload": f"M={M}, topk={topk}, H={H}",
    }


def _build_swiglu(cfg: Dict, stage: str) -> Callable[[], None]:
    M = _cfg_m(cfg, stage)
    I = cfg["moe_intermediate_size"]
    topk = cfg["num_experts_per_tok"]
    x = torch.randn(M * topk, 2 * I, dtype=DTYPE, device=DEVICE)

    def fn() -> None:
        torch_npu.npu_swiglu(x, dim=-1)

    return fn


def _sol_swiglu(cfg: Dict, stage: str) -> Dict[str, float]:
    M = _cfg_m(cfg, stage)
    I = cfg["moe_intermediate_size"]
    topk = cfg["num_experts_per_tok"]
    bytes_moved = 2 * M * topk * 2 * I * 2
    return {
        "T_cube_us": 0.0,
        "T_vec_us": _t_vec_us(M * topk * 2 * I),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": _t_gm_us(bytes_moved),
        "workload": f"M={M}, topk={topk}, I={I}",
    }


# ------------------------- F: Quantization -------------------------
def _build_dynamic_quant(cfg: Dict, stage: str) -> Callable[[], None]:
    M = _cfg_m(cfg, stage)
    H = cfg["hidden_size"]
    x = torch.randn(M, H, dtype=DTYPE, device=DEVICE)

    def fn() -> None:
        torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)

    return fn


def _sol_dynamic_quant(cfg: Dict, stage: str) -> Dict[str, float]:
    M = _cfg_m(cfg, stage)
    H = cfg["hidden_size"]
    bytes_moved = M * H * 2 + M * H * 1 + 4  # bf16 in, int8 out, scale
    return {
        "T_cube_us": 0.0,
        "T_vec_us": _t_vec_us(M * H),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": _t_gm_us(bytes_moved),
        "workload": f"M={M}, H={H}",
    }


# ------------------------- DSA: DeepSeek-V4 default sparse attention -------------------------
# Real call path: vllm_ascend/attention/dsa_v1.py (AscendDSAImpl) +
# vllm_ascend/device/device_op.py (DeviceOperator).  These operators are the
# *default* DeepSeek-V4 attention path (SWA window + compressed KV + lightning
# indexer topk).
DSA_BLOCK_SIZE = 128          # swa / indexer cache block size
DSA_CMP_RATIO = 4             # compress_ratio (c4 path with indexer)
DSA_STATE_BLOCK = 8           # c4 state cache block size
DSA_SEQ_LEN = 4096            # kv history length per request
DSA_CMP_BLOCK_TOKENS = DSA_BLOCK_SIZE // DSA_CMP_RATIO  # 32 compressed tokens per cmp block


def _dsa_stage_shapes(cfg: Dict, stage: str) -> Tuple[int, int, int]:
    """Return (B, q_len_per_req, total_tokens) for the stage."""
    M = _cfg_m(cfg, stage)
    if stage == "decode":
        return M, 1, M
    return 1, M, M


def _dsa_kv_len(cfg: Dict, stage: str) -> int:
    """KV history length per request: prefill attends to its own tokens."""
    _, q_len, _ = _dsa_stage_shapes(cfg, stage)
    return max(q_len, DSA_SEQ_LEN)


def _build_dsa_kv_scatter(cfg: Dict, stage: str) -> Callable[[], None]:
    if not CUSTOM_OP_AVAILABLE:
        raise RuntimeError("custom op not available")
    M = _cfg_m(cfg, stage)
    head_dim = cfg["head_dim"]
    n_blocks = 64
    cache = torch.zeros(n_blocks, DSA_BLOCK_SIZE, 1, head_dim, dtype=DTYPE, device=DEVICE)
    x = torch.randn(M, 1, head_dim, dtype=DTYPE, device=DEVICE)
    # 910B DSA slot_mapping is 2D [block_idx, offset] (device_op.format_dsa_slot_mapping)
    slot_linear = torch.arange(M, dtype=torch.int32, device=DEVICE) % (n_blocks * DSA_BLOCK_SIZE)
    slot_mapping = torch.stack(
        [slot_linear // DSA_BLOCK_SIZE, slot_linear % DSA_BLOCK_SIZE], dim=-1).contiguous()

    def fn() -> None:
        # dsa_kv_compress_scatter on 910B (device_op.py)
        torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, slot_mapping, x)

    return fn


def _sol_dsa_kv_scatter(cfg: Dict, stage: str) -> Dict[str, float]:
    M = _cfg_m(cfg, stage)
    head_dim = cfg["head_dim"]
    bytes_moved = 2 * M * head_dim * 2
    return {
        "T_cube_us": 0.0,
        "T_vec_us": _t_vec_us(M * head_dim),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": _t_gm_us(bytes_moved),
        "workload": f"M={M}, D={head_dim}",
    }


def _build_dsa_compressor(cfg: Dict, stage: str) -> Callable[[], None]:
    if not CUSTOM_OP_AVAILABLE:
        raise RuntimeError("custom op not available")
    B, q_len, M = _dsa_stage_shapes(cfg, stage)
    H = cfg["hidden_size"]
    head_dim = cfg["head_dim"]
    rope_dim = cfg["qk_rope_head_dim"]
    coff = 2  # overlap=True when compress_ratio == 4
    out_dim = coff * head_dim
    state_dim = 2 * out_dim
    # Kernel WriteToCacheState indexes stateBlockTable with curSeqIdx/blockSize
    # where curSeqIdx is in *raw token* units, so the table needs kv_len/8
    # columns.  Entry 0 is treated as null block (write skipped), so use 1..n-1.
    kv_len = max(q_len, DSA_SEQ_LEN)
    max_state_blocks_per_req = (kv_len + DSA_STATE_BLOCK - 1) // DSA_STATE_BLOCK
    n_state_blocks = max_state_blocks_per_req + 1

    x = torch.randn(M, H, dtype=DTYPE, device=DEVICE)
    wkv_w = torch.randn(out_dim, H, dtype=DTYPE, device=DEVICE)
    wgate_w = torch.randn(out_dim, H, dtype=DTYPE, device=DEVICE)
    state_cache = torch.zeros(n_state_blocks, DSA_STATE_BLOCK, state_dim, dtype=torch.float32, device=DEVICE)
    ape = torch.randn(DSA_CMP_RATIO, out_dim, dtype=torch.float32, device=DEVICE)
    norm_w = torch.randn(head_dim, dtype=DTYPE, device=DEVICE)
    # rope sin/cos are indexed by compressed tokens: tiling requires
    # dim0 == min(tokenSize, tokenSize/cmpRatio + batchSize)
    rope_tokens = min(M, M // DSA_CMP_RATIO + B)
    sin = torch.randn(rope_tokens, rope_dim, dtype=torch.float32, device=DEVICE)
    cos = torch.randn(rope_tokens, rope_dim, dtype=torch.float32, device=DEVICE)
    state_block_table = (1 + torch.arange(max_state_blocks_per_req, dtype=torch.int32, device=DEVICE)
                         % (n_state_blocks - 1)).unsqueeze(0).expand(B, -1).contiguous()
    cu_seqlens = torch.arange(0, B + 1, dtype=torch.int32, device=DEVICE) * q_len
    start_pos = torch.full((B,), max(DSA_SEQ_LEN - q_len, 0), dtype=torch.int32, device=DEVICE)

    split = os.getenv("VLLM_ASCEND_DSA_COMPRESSOR_SPLIT", "0") == "1"

    def fn() -> None:
        if split:
            mm_kv = torch.nn.functional.linear(x, wkv_w)
            mm_score = torch.nn.functional.linear(x, wgate_w)
            torch.ops._C_ascend.compressor_epilogue(
                mm_kv,
                mm_score,
                state_cache,
                ape,
                norm_w,
                sin.view(-1, rope_dim),
                cos.view(-1, rope_dim),
                state_block_table=state_block_table,
                cu_seqlens=cu_seqlens,
                seqused=None,
                start_pos=start_pos,
                rope_head_dim=rope_dim,
                cmp_ratio=DSA_CMP_RATIO,
                coff=coff,
                norm_eps=cfg["rms_norm_eps"],
                rotary_mode=2,
                cache_mode=1,
            )
            return
        torch.ops._C_ascend.compressor(
            x,
            wkv_w,
            wgate_w,
            state_cache,
            ape,
            norm_w,
            sin.view(-1, rope_dim),
            cos.view(-1, rope_dim),
            state_block_table=state_block_table,
            cu_seqlens=cu_seqlens,
            seqused=None,
            start_pos=start_pos,
            rope_head_dim=rope_dim,
            cmp_ratio=DSA_CMP_RATIO,
            coff=coff,
            norm_eps=cfg["rms_norm_eps"],
            rotary_mode=2,
            cache_mode=1,
        )

    return fn


def _sol_dsa_compressor(cfg: Dict, stage: str) -> Dict[str, float]:
    B, q_len, M = _dsa_stage_shapes(cfg, stage)
    H = cfg["hidden_size"]
    head_dim = cfg["head_dim"]
    coff = 2
    out_dim = coff * head_dim
    kv_len = max(q_len, DSA_SEQ_LEN)
    # two GEMMs: wkv + wgate
    flops = 2 * M * H * out_dim * 2
    # state cache 只读写新增压缩 token 对应的行（不是全部历史行）
    state_rows = M // DSA_CMP_RATIO + B
    bytes_moved = (
        M * H * 2                          # read x
        + 2 * H * out_dim * 2              # read wkv + wgate weights
        + (M // DSA_CMP_RATIO + 1) * head_dim * 2  # write compressed kv
        + state_rows * 2 * out_dim * 4 * 2  # state cache rw (fp32)
    )
    return {
        "T_cube_us": _t_cube_us(flops),
        "T_vec_us": _t_vec_us(M * out_dim),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": max(_t_cube_us(flops), _t_gm_us(bytes_moved)),
        "workload": f"M={M}, H={H}, out={out_dim}, cmp={DSA_CMP_RATIO}",
    }


def _build_dsa_lightning_indexer(cfg: Dict, stage: str) -> Callable[[], None]:
    if not CUSTOM_OP_AVAILABLE:
        raise RuntimeError("custom op not available")
    B, q_len, M = _dsa_stage_shapes(cfg, stage)
    n_heads = cfg["index_n_heads"]
    head_dim = cfg["index_head_dim"]
    index_topk = cfg["index_topk"]
    kv_len = _dsa_kv_len(cfg, stage)
    n_cmp_tokens = kv_len // DSA_CMP_RATIO
    n_blocks = (n_cmp_tokens + DSA_BLOCK_SIZE - 1) // DSA_BLOCK_SIZE
    max_blocks = n_blocks

    query = torch.randint(-8, 8, (M, n_heads, head_dim), dtype=torch.int8, device=DEVICE)
    key = torch.randint(-8, 8, (n_blocks, DSA_BLOCK_SIZE, 1, head_dim), dtype=torch.int8, device=DEVICE)
    weights = torch.randn(M, n_heads, dtype=torch.float16, device=DEVICE)
    query_scale = (torch.randn(M, n_heads, dtype=torch.float16, device=DEVICE).abs() + 0.5)
    key_scale = (torch.randn(n_blocks, DSA_BLOCK_SIZE, 1, dtype=torch.float16, device=DEVICE).abs() + 0.5)
    qlens = (torch.arange(1, B + 1, dtype=torch.int32, device=DEVICE) * q_len).contiguous()
    kvlens = torch.full((B,), kv_len, dtype=torch.int32, device=DEVICE)
    block_table = torch.arange(n_blocks, dtype=torch.int32, device=DEVICE).unsqueeze(0).expand(B, -1).contiguous()

    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_metadata(
        actual_seq_lengths_query=qlens.clone(),
        actual_seq_lengths_key=kvlens.clone(),
        num_heads_q=n_heads,
        num_heads_k=1,
        head_dim=head_dim,
        query_quant_mode=0,
        key_quant_mode=0,
        batch_size=B,
        max_seqlen_q=q_len,
        max_seqlen_k=kv_len,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=index_topk,
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        cmp_ratio=DSA_CMP_RATIO,
        device=str(DEVICE),
    )

    def fn() -> None:
        torch.ops._C_ascend.npu_quant_lightning_indexer(
            query=query,
            key=key,
            weights=weights,
            query_dequant_scale=query_scale,
            key_dequant_scale=key_scale,
            actual_seq_lengths_query=qlens,
            actual_seq_lengths_key=kvlens,
            block_table=block_table,
            metadata=metadata,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=index_topk,
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=DSA_CMP_RATIO,
            return_value=False,
        )

    return fn


def _sol_dsa_lightning_indexer(cfg: Dict, stage: str) -> Dict[str, float]:
    B, q_len, M = _dsa_stage_shapes(cfg, stage)
    n_heads = cfg["index_n_heads"]
    head_dim = cfg["index_head_dim"]
    index_topk = cfg["index_topk"]
    kv_len = _dsa_kv_len(cfg, stage)
    if stage == "decode":
        # decode query 在序列末尾，causal 下全量扫描有效
        scan_tokens = kv_len // DSA_CMP_RATIO
    else:
        # prefill causal(sparse_mode=3)：query 位置 p 平均有效扫描 p/2，
        # 有效计算量 = 全量的一半（与 kernel 是否跳过无关，SOL 按有用功计）
        scan_tokens = (kv_len // 2) // DSA_CMP_RATIO
    # int8 GEMM: int8 cube rate is 2x bf16 cube rate on 910B
    flops = 2 * M * n_heads * head_dim * scan_tokens
    t_cube_int8 = flops / (2 * P_CUBE_FLOPS) * 1e6
    bytes_moved = (
        B * (kv_len // DSA_CMP_RATIO) * head_dim  # read int8 k cache（每请求一份，请求内 query 共享）
        + M * n_heads * head_dim            # read int8 q
        + M * n_heads * 2                   # weights fp16
        + B * (kv_len // DSA_CMP_RATIO) * 2  # k scale fp16
        + M * index_topk * 4                # write topk indices
    )
    return {
        "T_cube_us": t_cube_int8,
        "T_vec_us": 0.0,
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": max(t_cube_int8, _t_gm_us(bytes_moved)),
        "workload": f"M={M}, heads={n_heads}, D={head_dim}, scan_tokens={scan_tokens}, topk={index_topk}",
    }


def _build_dsa_sparse_attn(cfg: Dict, stage: str) -> Callable[[], None]:
    if not CUSTOM_OP_AVAILABLE:
        raise RuntimeError("custom op not available")
    B, q_len, M = _dsa_stage_shapes(cfg, stage)
    n_heads = cfg["num_attention_heads"]
    head_dim = cfg["head_dim"]
    index_topk = cfg["index_topk"]
    window = cfg["sliding_window"]
    kv_len = _dsa_kv_len(cfg, stage)
    n_ori_blocks = (kv_len + DSA_BLOCK_SIZE - 1) // DSA_BLOCK_SIZE
    n_cmp_tokens = kv_len // DSA_CMP_RATIO
    n_cmp_blocks = (n_cmp_tokens + DSA_CMP_BLOCK_TOKENS - 1) // DSA_CMP_BLOCK_TOKENS

    q = torch.randn(M, n_heads, head_dim, dtype=DTYPE, device=DEVICE)
    ori_kv = torch.randn(n_ori_blocks, DSA_BLOCK_SIZE, 1, head_dim, dtype=DTYPE, device=DEVICE)
    cmp_kv = torch.randn(n_cmp_blocks, DSA_CMP_BLOCK_TOKENS, 1, head_dim, dtype=DTYPE, device=DEVICE)
    cmp_sparse_indices = torch.randint(0, n_cmp_tokens, (M, 1, index_topk), dtype=torch.int32, device=DEVICE)
    if stage != "decode":
        # cmp_mask_mode=3 是 causal：query 位置 p 只能看 <= p/cmp_ratio 的 cmp token。
        # 随机 indices 一半会被 mask（kernel 跳过），为测真实算力按位置 clamp 出全有效 indices。
        pos = torch.arange(M, dtype=torch.int32, device=DEVICE)
        cmp_sparse_indices = torch.minimum(cmp_sparse_indices, (pos // DSA_CMP_RATIO).view(M, 1, 1))
    ori_block_table = torch.arange(n_ori_blocks, dtype=torch.int32, device=DEVICE).unsqueeze(0).expand(
        B, -1).contiguous()
    cmp_block_table = torch.arange(n_cmp_blocks, dtype=torch.int32, device=DEVICE).unsqueeze(0).expand(
        B, -1).contiguous()
    cu_seqlens_q = torch.arange(0, B + 1, dtype=torch.int32, device=DEVICE) * q_len
    seqused_q = torch.full((B,), q_len, dtype=torch.int32, device=DEVICE)
    seqused_kv = torch.full((B,), kv_len, dtype=torch.int32, device=DEVICE)
    sinks = torch.zeros(n_heads, dtype=torch.float32, device=DEVICE)
    cmp_seqlens = torch.full((B,), n_cmp_tokens, dtype=torch.int32, device=DEVICE)
    cu_seqlens_ori_kv = torch.zeros(B + 1, dtype=torch.int32, device=DEVICE)
    cu_seqlens_ori_kv[1:] = torch.cumsum(seqused_kv, dim=0)
    cu_seqlens_cmp_kv = torch.zeros(B + 1, dtype=torch.int32, device=DEVICE)
    cu_seqlens_cmp_kv[1:] = torch.cumsum(cmp_seqlens, dim=0)

    metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=n_heads,
        num_heads_kv=1,
        head_dim=head_dim,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_q if stage != "decode" else cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        max_seqlen_q=seqused_q.max(),
        max_seqlen_kv=seqused_kv.max(),
        batch_size=B,
        cmp_topk=index_topk,
        cmp_ratio=DSA_CMP_RATIO,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=window - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=True,
        device=str(DEVICE),
    )

    softmax_scale = head_dim**-0.5

    def fn() -> None:
        torch.ops._C_ascend.npu_sparse_attn_sharedkv(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_sparse_indices,
            ori_block_table=ori_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=softmax_scale,
            cmp_ratio=DSA_CMP_RATIO,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=window - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
        )

    return fn


def _sol_dsa_sparse_attn(cfg: Dict, stage: str) -> Dict[str, float]:
    B, q_len, M = _dsa_stage_shapes(cfg, stage)
    n_heads = cfg["num_attention_heads"]
    head_dim = cfg["head_dim"]
    index_topk = cfg["index_topk"]
    window = cfg["sliding_window"]
    kv_len = _dsa_kv_len(cfg, stage)
    n_cmp_tokens = kv_len // DSA_CMP_RATIO
    attended_cmp = min(index_topk, n_cmp_tokens)
    attended = window + attended_cmp
    # QK^T + PV for each query token over `attended` kv tokens
    flops = 2 * M * n_heads * head_dim * attended * 2
    bytes_moved = (
        M * n_heads * head_dim * 2 * 2                    # read q + write out
        + M * attended * head_dim * 2                     # per-query KV gather（heads 间共享，query 间不共享）
        + M * index_topk * 4                              # read sparse indices
    )
    return {
        "T_cube_us": _t_cube_us(flops),
        "T_vec_us": _t_vec_us(M * n_heads * attended),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": max(_t_cube_us(flops), _t_gm_us(bytes_moved)),
        "workload": f"M={M}, B={B}, heads={n_heads}, D={head_dim}, win={window}, topk={index_topk}",
    }


# ------------------------- G: Sampling / KV cache / MTP -------------------------
def _build_apply_top_k_top_p(cfg: Dict, stage: str) -> Callable[[], None]:
    M = _cfg_m(cfg, stage)
    vocab = cfg["vocab_size"]
    logits = torch.randn(M, vocab, dtype=torch.float32, device=DEVICE)
    k = torch.full((M,), 50, dtype=torch.int32, device=DEVICE)
    p = torch.full((M,), 0.9, dtype=torch.float32, device=DEVICE)

    def fn() -> None:
        torch.ops._C_ascend.npu_apply_top_k_top_p(logits, k=k, p=p)

    return fn


def _sol_apply_top_k_top_p(cfg: Dict, stage: str) -> Dict[str, float]:
    M = _cfg_m(cfg, stage)
    vocab = cfg["vocab_size"]
    bytes_moved = 2 * M * vocab * 2
    return {
        "T_cube_us": 0.0,
        "T_vec_us": _t_vec_us(M * vocab),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": _t_gm_us(bytes_moved),
        "workload": f"M={M}, vocab={vocab}",
    }


def _build_ngram_spec_decode(cfg: Dict, stage: str) -> Callable[[], None]:
    if not CUSTOM_OP_AVAILABLE:
        raise RuntimeError("custom op not available")
    B = 8
    max_seq = 128
    vocab = cfg["vocab_size"]
    k = 4
    token_ids = torch.randint(0, vocab, (B, max_seq), dtype=torch.int32, device=DEVICE)
    num_tokens = torch.randint(1, max_seq, (B,), dtype=torch.int32, device=DEVICE)
    sampled = torch.randint(0, vocab, (B, k), dtype=torch.int32, device=DEVICE)
    discard = torch.zeros(B, dtype=torch.int32, device=DEVICE)

    def fn() -> None:
        torch.ops._C_ascend.npu_ngram_spec_decode(
            token_ids, num_tokens, sampled, discard, vocab_size=vocab, min_n=1, max_n=5, k=k
        )

    return fn


def _sol_ngram_spec_decode(cfg: Dict, stage: str) -> Dict[str, float]:
    B = 8
    max_seq = 128
    k = 4
    vocab = cfg["vocab_size"]
    bytes_moved = (B * max_seq * 4 + B * k * 4 + B * vocab * 4) * 2
    return {
        "T_cube_us": 0.0,
        "T_vec_us": _t_vec_us(B * max_seq * k),
        "T_gm_us": _t_gm_us(bytes_moved),
        "T_sol_us": _t_gm_us(bytes_moved),
        "workload": f"B={B}, seq={max_seq}, k={k}",
    }


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------
CASES: List[Case] = [
    Case("rms_norm_decode", "A", "decode", _build_rms_norm, _sol_rms_norm, ["(?<!add)RmsNorm"]),
    Case("rms_norm_prefill", "A", "prefill", _build_rms_norm, _sol_rms_norm, ["(?<!add)RmsNorm"]),
    Case("add_rms_norm_prefill", "A", "prefill", _build_add_rms_norm, _sol_add_rms_norm, ["AddRmsNorm(?![a-z0-9])"]),
    Case("add_rms_norm_bias_prefill", "A", "prefill", _build_add_rms_norm_bias, _sol_add_rms_norm, ["AddRmsNormBias"]),
    Case("rotary_mul_decode", "B", "decode", _build_rotary_mul, _sol_rotary_mul, ["RotaryPositionEmbedding", "RotaryMul"]),
    Case("rotary_mul_prefill", "B", "prefill", _build_rotary_mul, _sol_rotary_mul, ["RotaryPositionEmbedding", "RotaryMul"]),
    Case("fused_attention_decode", "C", "decode", _build_fused_attention, _sol_fused_attention, ["FusedInferAttentionScore"]),
    Case("fused_attention_prefill", "C", "prefill", _build_fused_attention, _sol_fused_attention, ["FusedInferAttentionScore"]),
    Case("batch_matmul_transpose", "C", "prefill", _build_bmm_transpose, _sol_bmm_transpose, ["batch_matmul_transpose", "BatchMatmulTranspose"]),
    Case("store_kv_block", "C", "decode", _build_store_kv_block, _sol_store_kv_block, ["StoreKVBlock"]),
    Case("hc_pre_decode", "D", "decode", _build_hc_pre, _sol_hc_pre, ["HcPre"]),
    Case("hc_pre_prefill", "D", "prefill", _build_hc_pre, _sol_hc_pre, ["HcPre"]),
    Case("moe_gating_top_k", "E", "prefill", _build_moe_gating_top_k, _sol_moe_gating_top_k, ["MoeGatingTopK"]),
    Case("moe_gating_top_k_decode", "E", "decode", _build_moe_gating_top_k, _sol_moe_gating_top_k, ["MoeGatingTopK"]),
    Case("moe_init_routing", "E", "prefill", _build_moe_init_routing, _sol_moe_init_routing, ["MoeInitRouting"]),
    Case("grouped_matmul", "E", "prefill", _build_grouped_matmul, _sol_grouped_matmul, ["GroupedMatmul"]),
    Case("moe_token_unpermute", "E", "prefill", _build_moe_token_unpermute, _sol_moe_token_unpermute, ["MoeTokenUnpermute"]),
    Case("swiglu", "E", "prefill", _build_swiglu, _sol_swiglu, ["SwiGlu"]),
    Case("dynamic_quant", "F", "prefill", _build_dynamic_quant, _sol_dynamic_quant, ["DynamicQuant"]),
    # DSA: DeepSeek-V4 default sparse attention path
    Case("dsa_kv_scatter", "C", "decode", _build_dsa_kv_scatter, _sol_dsa_kv_scatter, ["ScatterNdUpdate"]),
    Case("dsa_kv_scatter_prefill", "C", "prefill", _build_dsa_kv_scatter, _sol_dsa_kv_scatter, ["ScatterNdUpdate"]),
    Case("dsa_compressor", "C", "prefill", _build_dsa_compressor, _sol_dsa_compressor, ["Compressor"]),
    Case("dsa_compressor_decode", "C", "decode", _build_dsa_compressor, _sol_dsa_compressor, ["Compressor"]),
    Case("dsa_lightning_indexer", "D", "prefill", _build_dsa_lightning_indexer, _sol_dsa_lightning_indexer, ["QuantLightningIndexer"]),
    Case("dsa_lightning_indexer_decode", "D", "decode", _build_dsa_lightning_indexer, _sol_dsa_lightning_indexer, ["QuantLightningIndexer"]),
    Case("dsa_sparse_attn", "C", "prefill", _build_dsa_sparse_attn, _sol_dsa_sparse_attn, ["SparseAttnSharedkv"]),
    Case("dsa_sparse_attn_decode", "C", "decode", _build_dsa_sparse_attn, _sol_dsa_sparse_attn, ["SparseAttnSharedkv"]),
    Case("apply_top_k_top_p", "G", "decode", _build_apply_top_k_top_p, _sol_apply_top_k_top_p, ["ApplyTopKTopP"]),
    Case("ngram_spec_decode", "G", "decode", _build_ngram_spec_decode, _sol_ngram_spec_decode, ["NgramSpecDecode"]),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_msprof_mapping(summary_csv: str) -> Dict[str, float]:
    agg = _parse_msprof_summary(summary_csv)
    mapping: Dict[str, float] = {}
    regex_metachars = set("()[]{}*+?^$|\\")
    for c in CASES:
        if not c.msprof_names:
            continue
        total = 0.0
        matched = False
        for op_name, duration in agg.items():
            op_name_norm = op_name.lower()
            for cand in c.msprof_names:
                if any(ch in cand for ch in regex_metachars):
                    if re.search(cand, op_name_norm, re.IGNORECASE):
                        total += duration
                        matched = True
                        break
                elif cand.lower() in op_name_norm:
                    total += duration
                    matched = True
                    break
        if matched:
            mapping[c.name] = total
    return mapping


def main():
    global M_PREFILL_DEFAULT, M_DECODE_DEFAULT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="V4-Pro",
                        help="DeepSeek-V4 variant to benchmark")
    parser.add_argument("--case", default="all",
                        help="case name, 'all', or comma-separated list")
    parser.add_argument("--stage", default="all", choices=["all", "prefill", "decode"],
                        help="filter by inference stage")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--mstx", action="store_true",
                        help="wrap iterations with mstx ranges for msprof correlation")
    parser.add_argument("--msprof-summary", type=str, default=None,
                        help="path to msprof op_summary_*.csv to compute real device-time gaps")
    parser.add_argument("--output-dir", type=str, default="benchmarks/ops_profiling/results",
                        help="directory to write JSON results")
    parser.add_argument("--M-prefill", type=int, default=M_PREFILL_DEFAULT,
                        dest="M_prefill")
    parser.add_argument("--M-decode", type=int, default=M_DECODE_DEFAULT,
                        dest="M_decode")
    args = parser.parse_args()

    # override workload sizes
    M_PREFILL_DEFAULT = args.M_prefill
    M_DECODE_DEFAULT = args.M_decode

    torch.npu.set_device(DEVICE)
    init_device_properties_triton()

    cfg = MODELS[args.model]
    if args.case == "all":
        names = [c.name for c in CASES]
    else:
        names = [n.strip() for n in args.case.split(",")]

    selected = [c for c in CASES if c.name in names]
    if args.stage != "all":
        selected = [c for c in selected if c.stage == args.stage]

    device_times: Dict[str, float] = {}
    if args.msprof_summary:
        if not os.path.exists(args.msprof_summary):
            # allow glob pattern
            matches = glob.glob(args.msprof_summary)
            if matches:
                args.msprof_summary = matches[0]
        if args.msprof_summary and os.path.exists(args.msprof_summary):
            device_times = _load_msprof_mapping(args.msprof_summary)
        else:
            print(f"WARNING: msprof summary not found: {args.msprof_summary}")

    results = []
    print(f"Benchmarking {args.model} on {DEVICE} (M_prefill={M_PREFILL_DEFAULT}, M_decode={M_DECODE_DEFAULT})")
    print(f"{'case':<30}{'cat':<4}{'stage':<8}{'avg_host(us)':>12}{'T_sol(us)':>11}{'D_sol(host)':>12}{'T_k(us)':>10}{'D_sol(device)':>14}")
    print("-" * 101)
    for c in selected:
        try:
            fn = c.build(cfg, c.stage)
        except Exception as e:
            print(f"{c.name:<30}{c.category:<4}{c.stage:<8}{'SKIP':>12}  # {e}")
            results.append({"name": c.name, "category": c.category, "stage": c.stage,
                            "status": "skip", "error": str(e)})
            continue

        bench = _bench_mstx if args.mstx else _bench
        avg_host = bench(c.name, fn, args.iters, args.warmup) if args.mstx else bench(fn, args.iters, args.warmup)
        sol = c.sol(cfg, c.stage)
        t_sol = sol["T_sol_us"]
        d_host = avg_host / t_sol if t_sol > 0 else float("inf")
        t_k = device_times.get(c.name)
        d_device = ""
        if t_k is not None and t_sol > 0:
            d_device = f"{t_k / t_sol:.2f}"
        t_k_str = f"{t_k:>10.2f}" if t_k is not None else f"{'':>10}"
        d_device_str = f"{d_device:>14}" if d_device else f"{'':>14}"
        print(f"{c.name:<30}{c.category:<4}{c.stage:<8}{avg_host:>12.2f}{t_sol:>11.2f}{d_host:>12.2f}{t_k_str}{d_device_str}")
        results.append({
            "name": c.name,
            "category": c.category,
            "stage": c.stage,
            "avg_host_us": avg_host,
            "sol": sol,
            "D_sol_host": d_host,
            "T_k_us": t_k,
            "D_sol_device": float(d_device) if d_device else None,
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"bench_deepseek_v4_{args.model.lower()}_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": args.model,
            "device": DEVICE,
            "M_prefill": M_PREFILL_DEFAULT,
            "M_decode": M_DECODE_DEFAULT,
            "iters": args.iters,
            "warmup": args.warmup,
            "results": results,
        }, f, indent=2)
    print(f"\nResults written to {out_file}")


if __name__ == "__main__":
    main()
