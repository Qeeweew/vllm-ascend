#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Ascend W4A16 CPU offload MoE method.

Offloads MoE (W4A16 int4) computation to ARM CPU via nanovllm_ext,
retaining non-MoE layers (attention, shared expert, norms) on NPU.
"""

import os
import threading
from collections.abc import Callable
from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult

_ENV_FLAG = "VLLM_ASCEND_MOE_CPU_OFFLOAD"


def enable_moe_cpu_offload() -> bool:
    return os.environ.get(_ENV_FLAG, "0") == "1" or getattr(
        get_ascend_config(), "enable_moe_cpu_offload", False
    )


def _get_capture_sizes() -> list[int]:
    vllm_config = get_current_vllm_config()
    sizes = getattr(vllm_config.compilation_config, "cudagraph_capture_sizes", None)
    if sizes:
        return sorted(int(x) for x in sizes)
    return [1, 2, 4, 8, 16, 24, 32]


# ---------------------------------------------------------------------------
# NPU callback manager – subscribes a processing thread to the stream so that
# aclrtLaunchCallback inside nanovllm_ext can execute.  Must be created before
# the first moe_forward_npu_stream / moe_forward_npu_graph_out call.
# ---------------------------------------------------------------------------

_CB_LOCK = threading.Lock()
_CB_MANAGERS: dict[tuple[int, int], Any] = {}


def _ensure_callback_manager() -> None:
    """Create (once per stream) a NpuCallbackManager for the current NPU stream."""
    import torch_npu

    stream_ptr = int(torch_npu.npu.current_stream().npu_stream)
    device_id = int(torch_npu.npu.current_device())
    key = (device_id, stream_ptr)
    if key in _CB_MANAGERS:
        return
    with _CB_LOCK:
        if key not in _CB_MANAGERS:
            mgr = torch.classes.nanovllm.NpuCallbackManager(stream_ptr, device_id)
            _CB_MANAGERS[key] = mgr
            logger.info(
                "[MoE CPU Offload] NpuCallbackManager created "
                "(device=%d, stream=0x%x)", device_id, stream_ptr,
            )


# ---------------------------------------------------------------------------
# Dummy weight loader – silently consume non-MoE safetensor expert weights
# ---------------------------------------------------------------------------

def _noop_weight_loader(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str = "",
    shard_id: str = "",
    expert_id: int = -1,
    return_success: bool = False,
) -> bool | None:
    """Always succeeds without copying.  The real W4A16 weights are loaded
    lazily in ``process_weights_after_loading`` from the ``moe-w4a16-*.safetensors``
    shards."""
    return True if return_success else None


# ---------------------------------------------------------------------------
# CPU offload MoE method
# ---------------------------------------------------------------------------

class AscendW4A16CPUOffloadMoEMethod(FusedMoEMethodBase):
    """FusedMoE method that offloads W4A16 MoE compute to ARM CPU.

    Creates minimal CPU placeholder params with a no-op weight_loader so that
    vllm's weight-loading loop finds matching ``params_dict`` entries (no
    KeyError) but copies nothing.  Real W4A16 weights are loaded from the
    ``moe-w4a16-*.safetensors`` shards in ``process_weights_after_loading``.
    """

    quant_type = 4          # QuantType.W4A16
    supports_eplb: bool = False

    def get_fused_moe_quant_config(self, *args, **kwargs):
        return None  # W4A16 has no separate quant config

    def __init__(self, moe_config: FusedMoEConfig | None = None) -> None:
        super().__init__(moe_config)
        self._moe_handle: Any = None
        self._graph_ctx: dict[tuple[int, int, int], Any] = {}
        self._capture_sizes: list[int] = []
        self._top_k: int = 0
        # Mimic AscendFusedMoEMethod wrapper: _get_quant_type() reads
        # self.quant_method.quant_type to determine the MoE comm type.
        self.quant_method = self

    # ---- weight_map cache (key -> shard name) -----------------------------

    _weight_map_cache: dict[str, dict[str, str]] = {}

    @classmethod
    def _load_weight_map(cls, model_dir: str) -> dict[str, str]:
        """Load and cache the safetensors index weight_map for the model."""
        if model_dir in cls._weight_map_cache:
            return cls._weight_map_cache[model_dir]
        import json

        idx_path = os.path.join(model_dir, "quant_model_weights.safetensors.index.json")
        with open(idx_path) as f:
            idx = json.load(f)
        wm = idx.get("weight_map", {})
        cls._weight_map_cache[model_dir] = wm
        logger.info(
            "[MoE CPU Offload] loaded weight_map (%d keys) from %s",
            len(wm), os.path.basename(idx_path),
        )
        return wm

    # ---- create_weights ----------------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        E = int(num_experts)

        for pname in ("w13_weight_packed", "w2_weight_packed"):
            p = torch.nn.Parameter(
                torch.empty(E, 1, 1, dtype=torch.int32, device="cpu"),
                requires_grad=False,
            )
            p.weight_loader = _noop_weight_loader
            layer.register_parameter(pname, p)

        for pname in ("w13_weight_scale", "w2_weight_scale"):
            p = torch.nn.Parameter(
                torch.empty(E, 1, 1, dtype=params_dtype, device="cpu"),
                requires_grad=False,
            )
            p.weight_loader = _noop_weight_loader
            p.quant_method = "group"
            layer.register_parameter(pname, p)

        for pname in ("w13_weight_shape", "w2_weight_shape"):
            p = torch.nn.Parameter(
                torch.empty(E, 2, dtype=torch.int32, device="cpu"),
                requires_grad=False,
            )
            p.weight_loader = _noop_weight_loader
            p.quant_method = "group"
            layer.register_parameter(pname, p)

    # ---- lifecycle ---------------------------------------------------------

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        import nanovllm_ext  # noqa: F401

        experts = int(layer.moe_config.num_experts or layer.local_num_experts)
        hidden = int(layer.hidden_size)
        inter = int(layer.intermediate_size_per_partition)
        layer_id = int(getattr(layer, "layer_id", 0))

        vllm_cfg = get_current_vllm_config()
        model_dir = vllm_cfg.model_config.model

        # Build key -> shard mapping from the safetensors index json.
        # The index covers BOTH non-MoE (quant_model_weights-*) and
        # MoE (moe-w4a16-*) shards; we only read the latter.
        weight_map = AscendW4A16CPUOffloadMoEMethod._load_weight_map(model_dir)

        E, I, H = experts, inter, hidden
        PK, GS = 8, 32

        w13_packed = torch.empty(E, 2 * I, H // PK, dtype=torch.int32, device="cpu")
        w2_packed = torch.empty(E, H, I // PK, dtype=torch.int32, device="cpu")
        w13_scale = torch.empty(E, 2 * I, H // GS, dtype=layer.params_dtype, device="cpu")
        w2_scale = torch.empty(E, H, I // GS, dtype=layer.params_dtype, device="cpu")

        _sf_cache: dict[str, Any] = {}

        def _get_tensor(shard_name: str, key: str) -> torch.Tensor:
            if shard_name not in _sf_cache:
                from safetensors import safe_open
                sf = safe_open(os.path.join(model_dir, shard_name), framework="pt", device="cpu")
                sf.__enter__()
                _sf_cache[shard_name] = sf
            return _sf_cache[shard_name].get_tensor(key)

        def _load_expert(e: int, proj: str, dest_packed, dest_scale, row_start: int):
            key_packed = f"model.layers.{layer_id}.mlp.experts.{e}.{proj}.weight_packed"
            key_scale = f"model.layers.{layer_id}.mlp.experts.{e}.{proj}.weight_scale"
            shard_packed = weight_map.get(key_packed)
            shard_scale = weight_map.get(key_scale)
            if shard_packed is None or shard_scale is None:
                raise KeyError(f"Missing {key_packed} in weight_map")
            p = _get_tensor(shard_packed, key_packed).to(torch.int32)
            K = p.shape[0]
            dest_packed[e, row_start : row_start + K].copy_(p)
            s = _get_tensor(shard_scale, key_scale).to(layer.params_dtype)
            dest_scale[e, row_start : row_start + K].copy_(s)

        logger.info(
            "[MoE CPU Offload] layer %d: loading %d experts from moe-w4a16 shards ...",
            layer_id, experts,
        )
        for e in range(experts):
            _load_expert(e, "gate_proj", w13_packed, w13_scale, row_start=0)
            _load_expert(e, "up_proj", w13_packed, w13_scale, row_start=I)
            _load_expert(e, "down_proj", w2_packed, w2_scale, row_start=0)

        for sf in _sf_cache.values():
            sf.__exit__(None, None, None)

        self._moe_handle = torch.classes.nanovllm.MoEInfer(experts, hidden, inter, 1)
        self._moe_handle.store_quantized_repack(w13_packed, w13_scale, w2_packed, w2_scale)
        del w13_packed, w2_packed, w13_scale, w2_scale

        self._top_k = getattr(layer, "top_k", 8)
        self._capture_sizes = _get_capture_sizes()

        logger.info(
            "[MoE CPU Offload] layer %d: repacked & stored %d experts (Q4_0)",
            layer_id, experts,
        )

    # ---- forward -----------------------------------------------------------

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: Any | None = None,
    ) -> torch.Tensor:
        assert self._moe_handle is not None

        from vllm_ascend.quantization.methods.base import get_moe_num_logical_experts

        n_shared = getattr(layer, "n_shared_experts", 0) or 0
        num_logical = get_moe_num_logical_experts(
            layer, num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=n_shared,
        )

        topk_weights, topk_ids = select_experts(
            hidden_states=x, router_logits=router_logits,
            top_k=top_k, use_grouped_topk=use_grouped_topk,
            renormalize=renormalize, topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_logical, tid2eid=tid2eid,
        )

        num_tokens = int(x.shape[0])
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(torch.float32)
        out = self._run_moe_cpu(x, topk_ids, topk_weights, num_tokens)
        return FusedExpertsResult(routed_out=out)

    def _run_moe_cpu(self, x, topk_ids, topk_weights, num_tokens):
        _ensure_callback_manager()
        # Graph path ONLY during graph capture – the captured op is replayed
        # during decode without calling Python.  Prefill, profile, and eager
        # decode all use the stream path.
        if _EXTRA_CTX.capturing:
            return self._run_graph_out(x, topk_ids, topk_weights, num_tokens)
        return torch.ops.nanovllm.moe_forward_npu_stream(
            x, topk_ids, topk_weights, self._moe_handle,
        )

    def _run_graph_out(self, x, topk_ids, topk_weights, num_tokens):
        dtype_int = 1 if x.dtype == torch.bfloat16 else 0
        top_k = int(topk_ids.shape[1])
        ctx_key = (num_tokens, top_k, dtype_int)
        ctx = self._graph_ctx.get(ctx_key)
        if ctx is None:
            ctx = torch.classes.nanovllm.MoEGraphContext(
                self._moe_handle, num_tokens, top_k, dtype_int,
            )
            self._graph_ctx[ctx_key] = ctx
        out = torch.empty_like(x)
        torch.ops.nanovllm.moe_forward_npu_graph_out(
            x, topk_ids, topk_weights, self._moe_handle, ctx, out,
        )
        return out
