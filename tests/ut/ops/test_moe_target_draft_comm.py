# SPDX-License-Identifier: Apache-2.0
"""Target/draft expert geometry must survive loading both models."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_ascend import ascend_forward_context as context_module
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe import fused_moe as runner_module
from vllm_ascend.ops.fused_moe import moe_comm_method as comm_module
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.quantization.quant_type import QuantType


@pytest.fixture
def runners(monkeypatch):
    def init_base(runner, layer_name, config, router, experts, *args):
        nn.Module.__init__(runner)
        runner.layer_name = layer_name
        runner.moe_config = config
        runner.routed_experts = experts

    monkeypatch.setattr(runner_module.MoERunner, "__init__", init_base)
    monkeypatch.setattr(runner_module, "get_tp_group", lambda: object())
    monkeypatch.setattr(runner_module, "get_dp_group", lambda: object())
    monkeypatch.setattr(AscendMoERunner, "is_internal_router", property(lambda _: False))

    def build(experts, top_k):
        config = SimpleNamespace(
            hidden_dim=32,
            ep_size=1,
            num_experts=experts,
            num_local_experts=experts,
            experts_per_token=top_k,
        )
        routed = SimpleNamespace(quant_type=QuantType.W4A16)
        runner = AscendMoERunner("test.mlp", config, router=object(), routed_experts=routed)
        runner._sequence_parallel_context = lambda: nullcontext()
        return runner

    with patch.dict(comm_module._MoECommMethods, clear=True):
        target = build(384, 8)
        draft = build(128, 4)
        assert comm_module.get_moe_comm_method(MoECommType.ALLGATHER) is draft._layer_allgather_comm
        yield target, draft


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("raise_in_draft", [False, True])
def test_target_draft_forward_restores_geometry(monkeypatch, runners, use_v2, raise_in_draft):
    target, draft = runners
    original = draft._layer_allgather_comm
    ctx = SimpleNamespace(
        moe_comm_type=MoECommType.ALLGATHER,
        moe_comm_method=original,
        additional_kwargs={"moe_comm_type": MoECommType.ALLGATHER, "moe_comm_method": original},
    )
    monkeypatch.setattr(context_module, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(context_module.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", use_v2)
    visits = []
    hidden = torch.zeros(2, 32)

    def check(experts, top_k):
        comm = _EXTRA_CTX.moe_comm_method
        assert comm.moe_config.num_experts == experts
        assert comm.token_dispatcher.num_experts_local == experts
        assert comm.token_dispatcher.top_k == top_k
        visits.append(experts)

    def draft_forward(**kwargs):
        check(128, 4)
        if raise_in_draft:
            raise RuntimeError("draft failed")
        return kwargs["hidden_states"]

    def target_forward(**kwargs):
        check(384, 8)
        if raise_in_draft:
            with pytest.raises(RuntimeError, match="draft failed"):
                draft._forward_impl(hidden, torch.zeros(2, 128), None)
        else:
            draft._forward_impl(hidden, torch.zeros(2, 128), None)
        check(384, 8)
        return kwargs["hidden_states"]

    draft.routed_experts.forward_impl = draft_forward
    target.routed_experts.forward_impl = target_forward
    for _ in range(2):
        assert target._forward_impl(hidden, torch.zeros(2, 384), None) is hidden
        assert _EXTRA_CTX.moe_comm_method is original
    assert visits == [384, 128, 384] * 2


def test_layer_scope_preserves_other_communication_modes(monkeypatch, runners):
    target, _ = runners
    original = object()
    ctx = SimpleNamespace(moe_comm_type=MoECommType.MC2, moe_comm_method=original)
    monkeypatch.setattr(runner_module, "_EXTRA_CTX", ctx)
    with target._moe_comm_context():
        assert ctx.moe_comm_method is original
    assert ctx.moe_comm_method is original
