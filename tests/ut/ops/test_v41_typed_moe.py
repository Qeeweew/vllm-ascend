# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for explicit modality transport through the complete MoE op."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.fused_moe.runner import moe_runner as upstream

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import fused_moe as plugin


def make_runner(monkeypatch, shared, reduced):
    runner = plugin.AscendMoERunner.__new__(plugin.AscendMoERunner)
    nn.Module.__init__(runner)
    runner.layer_name = "typed.test"
    runner.moe_config = SimpleNamespace(dp_size=1, ep_size=1, is_sequence_parallel=False, hidden_dim_unpadded=0)
    events = []

    def record(name, result):
        events.append(name)
        return result

    runner.apply_routed_input_transform = lambda x: record("input", (x + 1, x))
    runner._maybe_pad_hidden_states = lambda s, x: record("pad", (torch.cat([x, x[:, :1]], -1), 2, 2))
    runner._encode_layer_name = lambda: "from_forward_context"
    runner._maybe_reduce_routed_output_before_transform = lambda x, _: record("routed_reduce", (x * 2, reduced))
    runner._maybe_reduce_shared_expert_output = lambda s, r: record(
        "shared_reduce", s * 2 if s is not None and r else s
    )
    runner._maybe_apply_routed_scale_to_output = lambda s, x: record("scale", (s, x * 3))
    runner.apply_routed_output_transform = lambda x: record("output", x.square())
    runner._maybe_reduce_final_output = lambda x, width, r: record("final_reduce", x if r else x * 4)
    runner._maybe_add_zero_expert_output = lambda x: record("zero", x + 7)

    def forward_impl(x, logits, shared_input, raw_ids, image_token_mask=None):
        events.append("experts")
        result = x + raw_ids[:, None]
        if image_token_mask is not None:
            result = result + image_token_mask[:, None]
        return (shared_input * 5, result) if shared else result

    runner._forward_impl = forward_impl
    runner._forward_entry = lambda x, logits, s, ids, name, width: forward_impl(x, logits, s, ids)
    context = SimpleNamespace(
        no_compile_layers={runner.layer_name: runner}, all_moe_layers=[runner.layer_name], moe_layer_index=0
    )
    monkeypatch.setattr(upstream, "_USE_LAYERNAME", False)
    monkeypatch.setattr(upstream, "get_forward_context", lambda: context)
    monkeypatch.setattr(plugin, "get_forward_context", lambda: context)
    monkeypatch.setattr(plugin, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER))
    # The upstream forward inspects this property solely for output shape.
    monkeypatch.setattr(
        plugin.AscendMoERunner, "_quant_method", property(lambda _: SimpleNamespace(has_unpadded_output=False))
    )
    return runner, context, events


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("pretransformed", [False, True])
def test_typed_complete_matches_upstream_order_and_advances_layer_index(monkeypatch, shared, reduced, pretransformed):
    runner, context, events = make_runner(monkeypatch, shared, reduced)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    raw = torch.tensor([129264, 129265])
    original = raw.clone()
    shared_input = hidden * 2 if pretransformed else None
    expected = upstream.MoERunner.forward(runner, hidden, hidden, input_ids=raw, shared_experts_input=shared_input)
    expected_events = events.copy()
    events.clear()
    actual = plugin._ascend_moe_forward_complete(
        hidden, hidden, shared_input, raw, runner.layer_name, torch.zeros(2, dtype=torch.bool)
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(raw, original)
    assert events == expected_events
    assert context.moe_layer_index == 1
    assert actual.shape == hidden.shape and isinstance(actual, torch.Tensor)


@pytest.mark.parametrize("field,value", [("dp_size", 2), ("ep_size", 8), ("is_sequence_parallel", True)])
def test_typed_complete_rejects_unsupported_parallelism_before_dispatch(monkeypatch, field, value):
    runner, context, events = make_runner(monkeypatch, False, False)
    setattr(runner.moe_config, field, value)
    hidden = torch.ones(2, 2)
    with pytest.raises(NotImplementedError, match="TP-only"):
        plugin._ascend_moe_forward_complete(
            hidden, hidden, None, None, runner.layer_name, torch.zeros(2, dtype=torch.bool)
        )
    assert not events and context.moe_layer_index == 0


def test_typed_complete_rejects_non_allgather(monkeypatch):
    runner, context, events = make_runner(monkeypatch, False, False)
    monkeypatch.setattr(plugin, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.MC2))
    hidden = torch.ones(2, 2)
    with pytest.raises(NotImplementedError, match="ALLGATHER"):
        plugin._ascend_moe_forward_complete(
            hidden, hidden, None, None, runner.layer_name, torch.zeros(2, dtype=torch.bool)
        )
    assert not events and context.moe_layer_index == 0


def test_compiled_complete_keeps_mask_explicit_and_replays_changed_values(monkeypatch):
    from torch.fx.experimental.proxy_tensor import make_fx

    runner, context, _ = make_runner(monkeypatch, True, False)
    library = torch.library.Library("vllm", "IMPL", "CPU")
    library.impl("ascend_moe_forward_complete", plugin._ascend_moe_forward_complete)
    hidden, raw = torch.ones(2, 2), torch.tensor([129264, 129264])
    mask = torch.tensor([True, False])
    try:
        graph = make_fx(lambda x, ids, images: runner(x, x, input_ids=ids, image_token_mask=images))(hidden, raw, mask)
        calls = [
            node for node in graph.graph.nodes if node.target == torch.ops.vllm.ascend_moe_forward_complete.default
        ]
        assert len(calls) == 1 and len(calls[0].args) == 6
        for values in ([True, False], [False, True], [False, False]):
            mask.copy_(torch.tensor(values))
            context.moe_layer_index = 0
            expected = plugin._ascend_moe_forward_complete(hidden, hidden, None, raw, runner.layer_name, mask)
            context.moe_layer_index = 0
            torch.testing.assert_close(graph(hidden, raw, mask), expected)
            assert context.moe_layer_index == 1
        assert raw.tolist() == [129264, 129264]
    finally:
        library._destroy()
