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
"""Ascend W4A16 quantization helpers and fused MoE method."""

from collections.abc import Callable
from typing import Any

import torch
import torch_npu
import triton
import triton.language as tl
from vllm.config import get_current_vllm_config

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input

from .base import AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme


@triton.jit
def _int4_repack_kernel(in_ptr, out_ptr, N, K, stride_in_n, stride_in_k8,
                       stride_out_k, stride_out_n8, NUM_CORES: tl.constexpr,
                       BLOCK_N8: tl.constexpr):
    """Fused unpack + transpose + repack of int4 weights (one NPU vector-core
    kernel). Input ``[K//8, N]`` int32 (8 nibbles packed along K) -> output
    ``[K, N//8]`` int32 (8 nibbles packed along N), two's-complement encoded.
    Bitwise-equivalent to ``unpack_from_int32`` -> ``transpose`` ->
    ``npu_convert_weight_to_int4pack`` but allocates only the output tensor
    (no 8x-int32 intermediate), avoiding OOM in ``process_weights_after_loading``.
    """
    pid = tl.program_id(0)
    total_k8_rows = K // 8
    rows_per_core = (total_k8_rows + NUM_CORES - 1) // NUM_CORES
    start_row = pid * rows_per_core
    if start_row >= total_k8_rows:
        return
    end_row = tl.minimum(start_row + rows_per_core, total_k8_rows)
    for k8_idx in range(start_row, end_row):
        num_n8 = N // 8
        for n8_base in range(0, num_n8, BLOCK_N8):
            n8_idx = n8_base + tl.arange(0, BLOCK_N8)
            mask_out_n = n8_idx < num_n8
            out_0 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_1 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_2 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_3 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_4 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_5 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_6 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_7 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            for i in tl.static_range(8):
                n_idx = n8_idx * 8 + i
                mask_n = n_idx < N
                in_ptrs = in_ptr + n_idx * stride_in_n + k8_idx * stride_in_k8
                packed_in = tl.load(in_ptrs, mask=mask_n, other=0).to(tl.uint32)
                shift_n = i * 4
                out_0 |= ((((packed_in >> 0) & 0xF) - 8) & 0xF) << shift_n
                out_1 |= ((((packed_in >> 4) & 0xF) - 8) & 0xF) << shift_n
                out_2 |= ((((packed_in >> 8) & 0xF) - 8) & 0xF) << shift_n
                out_3 |= ((((packed_in >> 12) & 0xF) - 8) & 0xF) << shift_n
                out_4 |= ((((packed_in >> 16) & 0xF) - 8) & 0xF) << shift_n
                out_5 |= ((((packed_in >> 20) & 0xF) - 8) & 0xF) << shift_n
                out_6 |= ((((packed_in >> 24) & 0xF) - 8) & 0xF) << shift_n
                out_7 |= ((((packed_in >> 28) & 0xF) - 8) & 0xF) << shift_n
            k_idx_base = k8_idx * 8
            tl.store(out_ptr + (k_idx_base + 0) * stride_out_k + n8_idx * stride_out_n8, out_0.to(tl.int32), mask=mask_out_n)
            tl.store(out_ptr + (k_idx_base + 1) * stride_out_k + n8_idx * stride_out_n8, out_1.to(tl.int32), mask=mask_out_n)
            tl.store(out_ptr + (k_idx_base + 2) * stride_out_k + n8_idx * stride_out_n8, out_2.to(tl.int32), mask=mask_out_n)
            tl.store(out_ptr + (k_idx_base + 3) * stride_out_k + n8_idx * stride_out_n8, out_3.to(tl.int32), mask=mask_out_n)
            tl.store(out_ptr + (k_idx_base + 4) * stride_out_k + n8_idx * stride_out_n8, out_4.to(tl.int32), mask=mask_out_n)
            tl.store(out_ptr + (k_idx_base + 5) * stride_out_k + n8_idx * stride_out_n8, out_5.to(tl.int32), mask=mask_out_n)
            tl.store(out_ptr + (k_idx_base + 6) * stride_out_k + n8_idx * stride_out_n8, out_6.to(tl.int32), mask=mask_out_n)
            tl.store(out_ptr + (k_idx_base + 7) * stride_out_k + n8_idx * stride_out_n8, out_7.to(tl.int32), mask=mask_out_n)


def _repack_int4_npu(weight_packed_t: torch.Tensor) -> torch.Tensor:
    """Repack int4 ``[K//8, N]`` int32 -> ``[K, N//8]`` int32 on NPU vector cores."""
    K_8, N = weight_packed_t.shape
    K = K_8 * 8
    out = torch.empty((K, N // 8), device=weight_packed_t.device, dtype=torch.int32)
    num_vectorcore = 48  # Ascend 910B3 AIV cores per die
    _int4_repack_kernel[(num_vectorcore,)](
        weight_packed_t, out, N, K,
        weight_packed_t.stride(1), weight_packed_t.stride(0),
        out.stride(0), out.stride(1),
        NUM_CORES=num_vectorcore, BLOCK_N8=256)
    return out


def _transpose_and_repack_int4(weight_packed: torch.Tensor) -> torch.Tensor:
    """``[E, N, K//8]`` compressed-tensors weight -> ``[E, K, N//8]`` kernel layout.

    Fuses unpack + transpose(1,2) + repack into one triton kernel; only the
    output ``[E, K, N//8]`` int32 tensor is allocated (vs the old path which
    materialised an ``[E, K, N]`` int32 intermediate, 8x larger).
    """
    E, N, K_div_8 = weight_packed.shape
    K = K_div_8 * 8
    weight_t = weight_packed.transpose(1, 2).contiguous()      # [E, K//8, N]
    weight_t_flat = weight_t.view(E * K_div_8, N)              # [E*K//8, N]
    weight_repacked_flat = _repack_int4_npu(weight_t_flat)     # [E*K, N//8]
    return weight_repacked_flat.view(E, K, N // 8)


_ZERO_OFFSET_CACHE: dict = {}


def _get_zero_offset(ref: torch.Tensor) -> torch.Tensor:
    """Zero tensor matching ref's shape/dtype/device, cached per device.

    All MoE layers share the same w13 / w2 offset shape, so this allocates the
    all-zero offset ONCE per (shape, dtype, device) (~75 MB total) instead of a
    5.4 GB per-layer copy, while still not pre-allocating at load time.
    """
    key = (tuple(ref.shape), ref.dtype, ref.device.index)
    z = _ZERO_OFFSET_CACHE.get(key)
    if z is None:
        z = torch.zeros(ref.shape, dtype=ref.dtype, device=ref.device)
        _ZERO_OFFSET_CACHE[key] = z
    return z


def unpack_from_int32(
    weight: torch.Tensor,
    shape: torch.Size,
    num_bits: int,
    packed_dim: int = 1,
) -> torch.Tensor:
    """Unpacks quantized weights from int32 format back to original bits.

    :param weight: The packed int32 tensor containing quantized weights
    :param shape: Original shape to restore, defaults to None
    :param num_bits: The number of bits used for quantization (<= 8)
    :param packed_dim: Dimension along which weights are packed (0 or 1), defaults to 1
    :return: Unpacked tensor with int8 dtype after applying offset correction
    """
    assert weight.dtype == torch.int32, f"Expecting `weight.dtype` is torch.int32 but got {weight.dtype}."
    assert num_bits > 0, f"Expecting `num_bits` should be positive but got {num_bits}."
    assert num_bits <= 8, f"Expecting `num_bits` should not be larger than 8 but got {num_bits}."
    assert 32 % num_bits == 0, f"Expecting `num_bits` {num_bits} to divide 32 exactly."
    assert packed_dim in [0, 1], f"Expecting `packed_dim` is 0 or 1 but got {packed_dim}."

    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
        unpacked_weight = torch.zeros(
            (weight.shape[0], weight.shape[1] * pack_factor),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[:, i::pack_factor] = (weight >> (num_bits * i)) & mask
        original_row_size = int(shape[1])
        unpacked_weight = unpacked_weight[:, :original_row_size]
    else:
        unpacked_weight = torch.zeros(
            (weight.shape[0] * pack_factor, weight.shape[1]),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[i::pack_factor, :] = (weight >> (num_bits * i)) & mask
        original_row_size = int(shape[0])
        unpacked_weight = unpacked_weight[:original_row_size, :]

    offset = pow(2, num_bits) // 2
    unpacked_weight = (unpacked_weight - offset).to(torch.int8)

    return unpacked_weight


def pack_to_int32(weight: torch.Tensor) -> torch.Tensor:
    """Packs quantized weights into int32 format for storage.

    :param weight: The 3D tensor to pack, must be int8 or int32 dtype
    :return: Packed tensor with int32 dtype optimized for storage
    """
    assert weight.dim() == 3, (
        "Expecting `weight.dim()` is 3 ([expert, output_channel, input_channel] or "
        "[expert, input_channel, output_channel]) but got "
        f"{weight.dim()}."
    )
    assert weight.dtype in [torch.int8, torch.int32], (
        f"Expecting `weight.dtype` is torch.int8 or torch.int32 but got {weight.dtype}."
    )

    if weight.dtype == torch.int32:
        assert weight.shape[-1] % 8 == 0, "the last dim of weight needs to be divided by 8."
        packed_weight = torch_npu.npu_convert_weight_to_int4pack(weight.flatten(0, 1))
        packed_weight = packed_weight.view(weight.shape[0], weight.shape[1], -1)
    else:
        assert weight.shape[-1] % 4 == 0, "the last dim of weight needs to be divided by 4."
        packed_weight = weight.view(torch.int32).contiguous()

    return packed_weight


@register_scheme("W4A16", "moe")
class AscendW4A16FusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for Ascend W4A16.

    This method supports only weights generated by LLM-Compressor, for
    example ``moonshotai/Kimi-K2-Thinking``.

    Each original routed MoE expert in the checkpoint stores separate
    LLM-Compressor tensors. The names below use ``L`` for the layer index and
    ``E`` for the expert index. For these 4-bit weights, ``pack_factor`` is
    8, so one int32 element stores eight 4-bit weight values.

    - ``model.layers.L.mlp.experts.E.gate_proj.weight_packed``:
      ``torch.int32``,
      ``[moe_intermediate_size, hidden_sizes // pack_factor]``.
    - ``model.layers.L.mlp.experts.E.gate_proj.weight_scale``:
      ``torch.bfloat16``,
      ``[moe_intermediate_size, hidden_sizes // group_size]``.
    - ``model.layers.L.mlp.experts.E.gate_proj.weight_shape``:
      ``torch.int32``, ``[2]``.
    - ``model.layers.L.mlp.experts.E.up_proj.weight_packed``:
      ``torch.int32``,
      ``[moe_intermediate_size, hidden_sizes // pack_factor]``.
    - ``model.layers.L.mlp.experts.E.up_proj.weight_scale``:
      ``torch.bfloat16``,
      ``[moe_intermediate_size, hidden_sizes // group_size]``.
    - ``model.layers.L.mlp.experts.E.up_proj.weight_shape``:
      ``torch.int32``, ``[2]``.
    - ``model.layers.L.mlp.experts.E.down_proj.weight_packed``:
      ``torch.int32``,
      ``[hidden_sizes, moe_intermediate_size // pack_factor]``.
    - ``model.layers.L.mlp.experts.E.down_proj.weight_scale``:
      ``torch.bfloat16``,
      ``[hidden_sizes, moe_intermediate_size // group_size]``.
    - ``model.layers.L.mlp.experts.E.down_proj.weight_shape``:
      ``torch.int32``, ``[2]``.

    During loading, the gate and up projections are fused into ``w13`` and the
    down projection is loaded as ``w2``. In
    :meth:`process_weights_after_loading`, weight tensors are unpacked,
    transposed into the data layout required by the Ascend fused MoE operator,
    and repacked into the int32 dtype. The offset tensors are not loaded from the
    checkpoint; they are all-zero tensors constructed because the operator
    requires offset inputs.

    After :meth:`process_weights_after_loading`, ``apply`` consumes:

    - ``w13_weight_packed``: ``torch.int32``,
      ``[num_experts, hidden_sizes,
      2 * moe_intermediate_size // pack_factor]``.
    - ``w2_weight_packed``: ``torch.int32``,
      ``[num_experts, moe_intermediate_size,
      hidden_sizes // pack_factor]``.
    - ``w13_weight_scale``: ``torch.bfloat16``,
      ``[num_experts, hidden_sizes // group_size,
      2 * moe_intermediate_size]``.
    - ``w2_weight_scale``: ``torch.bfloat16``,
      ``[num_experts, moe_intermediate_size // group_size,
      hidden_sizes]``.
    - ``w13_weight_offset``: ``torch.bfloat16``, all zeros,
      ``[num_experts, hidden_sizes // group_size,
      2 * moe_intermediate_size]``.
    - ``w2_weight_offset``: ``torch.bfloat16``, all zeros,
      ``[num_experts, moe_intermediate_size // group_size,
      hidden_sizes]``.
    """

    quant_type: QuantType = QuantType.W4A16

    def __init__(self) -> None:
        self.num_bits = 4  # dtype = torch.int4
        self.pack_factor = 8  # pack 8 of torch.int4 tensors to torch.int32

        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        assert intermediate_size_per_partition % self.pack_factor == 0, (
            f"Expecting `intermediate_size_per_partition` {intermediate_size_per_partition} "
            f"can be divided by `pack_factor` {self.pack_factor}"
        )
        assert hidden_sizes % self.pack_factor == 0, (
            f"Expecting `hidden_sizes` {hidden_sizes} can be divided by `pack_factor` {self.pack_factor}"
        )

        param_dict = {}

        param_dict["w13_weight_packed"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.pack_factor, dtype=torch.int32
        )
        param_dict["w2_weight_packed"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.pack_factor, dtype=torch.int32
        )

        return param_dict

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        assert intermediate_size_per_partition % self.group_size == 0, (
            f"Expecting `intermediate_size_per_partition` {intermediate_size_per_partition} "
            f"can be divided by `group_size` {self.group_size}"
        )
        assert hidden_sizes % self.group_size == 0, (
            f"Expecting `hidden_sizes` {hidden_sizes} can be divided by `group_size` {self.group_size}"
        )

        param_dict = {}

        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.group_size, dtype=params_dtype
        )
        param_dict["w2_weight_scale"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=params_dtype
        )
        param_dict["w13_weight_shape"] = torch.empty(num_experts, 2, dtype=torch.int32)
        param_dict["w2_weight_shape"] = torch.empty(num_experts, 2, dtype=torch.int32)
        # NOTE: weight_offset is NOT pre-allocated. W4A16 is symmetric
        # (offset == 0); a zero tensor is created on-the-fly in apply() to
        # avoid holding a multi-GB all-zero tensor per card.

        return param_dict

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
        enable_force_load_balance: bool = True,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_shared_experts = getattr(layer, "n_shared_experts", 0)
        if num_shared_experts is None:
            num_shared_experts = 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
        assert router_logits.shape[1] == num_logical_experts, (
            "Number of global experts mismatch (excluding redundancy): "
            f"router_logits.shape[1]={router_logits.shape[1]}, num_logical_experts={num_logical_experts}"
        )

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_logical_experts,
            tid2eid=tid2eid,
        )

        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight_packed,
                w2=layer.w2_weight_packed,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w1_offset=_get_zero_offset(layer.w13_weight_scale),
                w2_offset=_get_zero_offset(layer.w2_weight_scale),
                swiglu_limit=layer.swiglu_limit,
            )
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fused triton unpack+transpose+repack: bitwise-equivalent to the old
        # unpack_from_int32 -> transpose(1,2) -> npu_convert_weight_to_int4pack
        # path, but allocates only the output (no 8x int32 intermediate) so it
        # no longer OOMs on large MoE layers (e.g. 256 experts on 64G HBM).
        layer.w13_weight_packed.data = _transpose_and_repack_int4(layer.w13_weight_packed.data)
        layer.w2_weight_packed.data = _transpose_and_repack_int4(layer.w2_weight_packed.data)

        layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(1, 2).contiguous()
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(1, 2).contiguous()
