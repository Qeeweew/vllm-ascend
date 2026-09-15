# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from tests.e2e.single_node.ops.test_w4a16_moe_kernel import assert_accurate, make_case, moe_reference
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl
from vllm_ascend.quantization.methods.wna16.w4a16 import AscendW4A16FusedMoEMethod

MODULE = "vllm_ascend.quantization.methods.wna16.w4a16"


def test_real_dispatch_and_graph_with_high_expert_ids():
    args, (q13, q2) = make_case(2)
    x, w13, s13, w2, s2, ids, routing = args
    ids.copy_(torch.tensor([0, 63, 127, 191, 255, 383], dtype=torch.int32).expand_as(ids))
    layer = SimpleNamespace(
        w13_weight_packed=w13.npu().repeat(64, 1, 1),
        w13_weight_scale=s13.npu().repeat(64, 1, 1),
        w2_weight_packed=w2.npu().repeat(64, 1, 1),
        w2_weight_scale=s2.npu().repeat(64, 1, 1),
        ascend_expert_map=None,
        apply_router_weight_on_input=False,
        global_redundant_expert_num=0,
        ascend_pertoken_scale=None,
    )
    method = object.__new__(AscendW4A16FusedMoEMethod)
    method.enable_native_decode, method.dynamic_eplb, method.group_size = True, False, 32
    comm = object.__new__(AllGatherCommImpl)
    comm.moe_config = SimpleNamespace(ep_size=1, activation=MoEActivation.SILU, swiglu_limit=10.0)
    metadata = SimpleNamespace(attn_metadata={"attn": SimpleNamespace(num_prefills=0, num_decode_tokens=2)})
    xn, ids_n, routing_n = x.npu(), ids.npu(), routing.npu()
    with (
        patch(f"{MODULE}._EXTRA_CTX", SimpleNamespace(moe_comm_method=comm)),
        patch(f"{MODULE}.get_forward_context", return_value=metadata),
    ):
        for _ in range(3):
            actual = method.apply(layer, xn, routing_n, ids_n, None, None).routed_out
        expected = moe_reference(x, q13, s13, q2, s2, ids % 6, routing, 10.0)
        assert_accurate(actual, expected)
        torch_npu.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = method.apply(layer, xn, routing_n, ids_n, None, None).routed_out
        x.mul_(-0.75)
        xn.copy_(x)
        graph.replay()
        expected = moe_reference(x, q13, s13, q2, s2, ids % 6, routing, 10.0)
        assert_accurate(captured, expected)
