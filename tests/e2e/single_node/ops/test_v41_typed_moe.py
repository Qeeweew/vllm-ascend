# SPDX-License-Identifier: Apache-2.0
"""Bounded NPU modality-transport test; toy experts, no HCCL or checkpoint."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.fused_moe.runner import moe_runner as upstream

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import fused_moe as plugin
from vllm_ascend.ops.fused_moe.router import fused_topk_router


@torch.inference_mode()
@pytest.mark.parametrize("with_shared", [False, True])
def test_typed_complete_real_router_changed_masks_graph(monkeypatch, with_shared):
    torch.npu.set_device(2)
    device = torch.device("npu:2")
    tokens, width, experts, topk = 8, 16, 4, 2
    hidden_cpu = torch.arange(tokens * width, dtype=torch.float32).reshape(tokens, width).remainder(17) / 32
    logits_cpu = torch.tensor([-1.0, 0.0, 1.0, 2.0]).expand(tokens, -1).contiguous()
    expert_cpu = (
        torch.arange(width * experts * width, dtype=torch.float32).reshape(width, experts * width).remainder(7) / 64
    )
    shared_cpu = torch.eye(width) / 8
    raw_cpu = torch.full((tokens,), 129264, dtype=torch.int64)
    table_cpu = torch.zeros((129265, topk), dtype=torch.int32)
    table_cpu[129264] = torch.tensor([1, 0])
    vision_bias = torch.tensor([0.0, 0.0, 20.0, 19.0], device=device)
    hidden, logits, raw = hidden_cpu.to(device), logits_cpu.to(device), raw_cpu.to(device)
    expert_weight, shared_weight = expert_cpu.to(device), shared_cpu.to(device)
    images = torch.zeros(tokens, dtype=torch.bool, device=device)
    captured_ids = torch.zeros((tokens, topk), dtype=torch.int32, device=device)
    router = fused_topk_router.AscendFusedTopKRouter(
        top_k=topk,
        global_num_experts=experts,
        scoring_func="sqrtsoftplus",
        tid2eid=table_cpu.to(device),
        bias_vl=vision_bias,
        image_sentinel_lo=129264,
        require_image_token_mask=True,
    )
    router.capture_fn = lambda value: captured_ids.copy_(value)
    comm = SimpleNamespace(
        moe_comm_type=MoECommType.ALLGATHER,
        moe_comm_method=SimpleNamespace(
            prepare_finalize=SimpleNamespace(all_gather_input_id_with_dp_group=lambda x: x)
        ),
    )
    monkeypatch.setattr(plugin, "_EXTRA_CTX", comm)
    monkeypatch.setattr(fused_topk_router, "_EXTRA_CTX", comm)
    runner = plugin.AscendMoERunner.__new__(plugin.AscendMoERunner)
    nn.Module.__init__(runner)
    runner.layer_name = "typed.npu.transport"
    runner.moe_config = SimpleNamespace(dp_size=1, ep_size=1, is_sequence_parallel=False)
    runner.apply_routed_input_transform = lambda x: (x, x)
    runner._maybe_pad_hidden_states = lambda shared, x: (x, None, None)
    runner._encode_layer_name = lambda: "from_forward_context"
    runner._maybe_reduce_routed_output_before_transform = lambda x, reduced: (x, reduced)
    runner._maybe_reduce_shared_expert_output = lambda x, reduced: x
    runner._maybe_apply_routed_scale_to_output = lambda shared, routed: (shared, routed * 1.5)
    runner.apply_routed_output_transform = lambda x: x
    runner._maybe_reduce_final_output = lambda x, width, reduced: x
    runner._maybe_add_zero_expert_output = lambda x: x

    def routed_compute(x, scores, shared, ids, image_token_mask=None):
        weights, indices = router._select_experts(x, scores, input_ids=ids, image_token_mask=image_token_mask)
        all_experts = (x @ expert_weight).view(tokens, experts, width)
        selected = all_experts.gather(1, indices.long()[:, :, None].expand(-1, -1, width))
        routed = (selected * weights[:, :, None]).sum(1)
        return (shared @ shared_weight, routed) if with_shared else routed

    runner._forward_impl = routed_compute
    context = SimpleNamespace(
        no_compile_layers={runner.layer_name: runner}, all_moe_layers=[runner.layer_name], moe_layer_index=0
    )
    monkeypatch.setattr(plugin, "get_forward_context", lambda: context)
    monkeypatch.setattr(upstream, "get_forward_context", lambda: context)
    monkeypatch.setattr(upstream, "_USE_LAYERNAME", False)

    def execute():
        context.moe_layer_index = 0
        value = runner(hidden, logits, input_ids=raw, image_token_mask=images)
        assert context.moe_layer_index == 1
        return value

    pointers = tuple(x.data_ptr() for x in (hidden, logits, raw, images, captured_ids))
    for _ in range(3):
        execute()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = execute()
    output_pointer = output.data_ptr()
    snapshots = []
    # Image, literal sentinel, generated sentinel, and padded positions all
    # have the same raw ID. Only processor-owned image positions are true.
    patterns = [
        [True, False, False, False, False, False, False, False],
        [False, True, True, False, False, False, False, False],
        [False] * tokens,
    ]
    for iteration in range(12):
        values = patterns[iteration % len(patterns)]
        images.copy_(torch.tensor(values))
        graph.replay()
        snapshots.append((output.clone(), captured_ids.clone(), values))
        assert output.data_ptr() == output_pointer
        assert tuple(x.data_ptr() for x in (hidden, logits, raw, images, captured_ids)) == pointers
        # Python resolution runs at capture, not replay; it must not be
        # incorrectly counted again when replay launches recorded kernels.
        assert context.moe_layer_index == 1
    torch.npu.synchronize()
    unbiased = torch.nn.functional.softplus(logits_cpu).sqrt()
    all_cpu = (hidden_cpu @ expert_cpu).view(tokens, experts, width)
    for actual, ids, values in snapshots:
        expected_ids = torch.tensor([[2, 3] if image else [1, 0] for image in values])
        scores = unbiased.gather(1, expected_ids)
        scores /= scores.sum(-1, keepdim=True)
        selected = all_cpu.gather(1, expected_ids[:, :, None].expand(-1, -1, width))
        expected = (selected * scores[:, :, None]).sum(1) * 1.5
        if with_shared:
            expected += hidden_cpu @ shared_cpu
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(ids.cpu().long(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(raw.cpu(), raw_cpu, rtol=0, atol=0)
