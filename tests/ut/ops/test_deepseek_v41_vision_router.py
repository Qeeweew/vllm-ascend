# SPDX-License-Identifier: Apache-2.0
"""Real HF/VllmConfig -> Ascend MoE factory -> router -> CPU selection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.router import fused_topk_router, router_factory
from vllm_ascend.patch.platform import patch_fused_moe


@pytest.mark.parametrize("experts,top_k", [(384, 6), (128, 3)])
@pytest.mark.parametrize("hashed", [False, True])
@pytest.mark.parametrize("indices_dtype", [None, torch.int64])
def test_native_v41_target_and_dspark_dispatch(monkeypatch, experts, top_k, hashed, indices_dtype):
    monkeypatch.setattr(
        fused_topk_router,
        "_EXTRA_CTX",
        SimpleNamespace(
            moe_comm_type=MoECommType.ALLGATHER,
            moe_comm_method=SimpleNamespace(
                prepare_finalize=SimpleNamespace(all_gather_input_id_with_dp_group=lambda ids: ids)
            ),
        ),
    )
    logits = torch.linspace(-5, 5, experts).repeat(3, 1)
    token_ids = torch.tensor([1, 2, -1], dtype=torch.int32)
    image_mask = torch.tensor([False, True, False])
    table = torch.arange(top_k * 3, dtype=torch.int32).reshape(3, top_k) if hashed else None
    text_bias = torch.linspace(0, 1, experts)
    image_bias = torch.linspace(10, -10, experts)
    router = fused_topk_router.AscendFusedTopKRouter(
        top_k=top_k,
        global_num_experts=experts,
        scoring_func="sqrtsoftplus",
        tid2eid=table,
        e_score_correction_bias=text_bias,
        bias_vl=image_bias,
        require_image_token_mask=True,
        enable_v41_router=True,
        renormalize=False,
        routed_scaling_factor=1.5,
    )
    calls = []

    def native(logits_arg, ids_arg, mask_arg, table_arg, text_arg, image_arg, weights, ids, k, renorm, scale):
        assert logits_arg is logits and mask_arg is image_mask
        assert table_arg is table and text_arg is text_bias and image_arg is image_bias
        assert ids_arg.dtype == torch.int64 and ids_arg.tolist() == [1, 2, 0]
        assert k == top_k and renorm is False and scale == 1.5
        assert weights.dtype == torch.float32 and ids.dtype == torch.int32
        expected_weights, expected_ids = fused_topk_router.select_deepseek_v4_vision_experts(
            logits_arg, ids_arg, table_arg, image_arg, text_arg, k, renorm, scale, image_token_mask=mask_arg
        )
        weights.copy_(expected_weights)
        ids.copy_(expected_ids)
        calls.append((weights.clone(), ids.clone()))

    monkeypatch.setattr(fused_topk_router, "v41_moe_router", native)
    weights, ids = router._compute_routing(
        torch.zeros(3, 8), logits, indices_dtype, input_ids=token_ids, image_token_mask=image_mask
    )
    assert len(calls) == 1
    assert ids.dtype == (torch.int32 if indices_dtype is None else indices_dtype)
    torch.testing.assert_close(weights, calls[0][0])
    torch.testing.assert_close(ids, calls[0][1].to(ids.dtype))
    assert token_ids.tolist() == [1, 2, -1]
    # The identical configuration still has a baseline for A/B acceptance.
    router.enable_v41_router = False
    baseline_weights, baseline_ids = router._compute_routing(
        torch.zeros(3, 8), logits, indices_dtype, input_ids=token_ids, image_token_mask=image_mask
    )
    assert len(calls) == 1
    torch.testing.assert_close(weights, baseline_weights)
    torch.testing.assert_close(ids, baseline_ids)


@pytest.mark.parametrize("version,count", [("v4", 5), ("v41", 1)])
@pytest.mark.parametrize("base_id", [129257, 129264])
@pytest.mark.parametrize("renormalize", [False, True])
def test_model_config_controls_image_interval_and_unbiased_weights(monkeypatch, version, count, base_id, renormalize):
    # Crossed base IDs prove model type, not a token-ID heuristic, controls
    # width. Production V4 uses 129257; production V4.1 uses 129264.
    hf = DeepseekV41Config(text_config={"image_token_id": base_id}) if version == "v41" else DeepseekV4Config()
    config = VllmConfig()
    config.model_config = SimpleNamespace(hf_config=hf, hf_text_config=hf)
    monkeypatch.setattr(router_factory, "get_ascend_config", lambda: SimpleNamespace(enable_v41_router=False))
    monkeypatch.setattr(
        patch_fused_moe,
        "get_ascend_config",
        lambda: SimpleNamespace(
            eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_path=None, num_redundant_experts=0)
        ),
    )
    upstream = Mock(side_effect=lambda **kwargs: SimpleNamespace(router=kwargs["router"]))
    monkeypatch.setattr(patch_fused_moe, "_original_FusedMoE", upstream)
    monkeypatch.setattr(
        router_factory, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: False)
    )
    monkeypatch.setattr(
        fused_topk_router,
        "_EXTRA_CTX",
        SimpleNamespace(
            moe_comm_type=MoECommType.ALLGATHER,
            moe_comm_method=SimpleNamespace(
                prepare_finalize=SimpleNamespace(all_gather_input_id_with_dp_group=lambda ids: ids)
            ),
        ),
    )
    text_bias = torch.tensor([20.0, 19.0, 0.0, 0.0])
    vision_bias = torch.tensor([0.0, 0.0, 20.0, 19.0])
    with set_current_vllm_config(config):
        runner = patch_fused_moe._ascend_FusedMoE(
            num_experts=4,
            top_k=2,
            scoring_func="sqrtsoftplus",
            renormalize=renormalize,
            e_score_correction_bias=text_bias,
            bias_vl=vision_bias,
            image_sentinel_lo=base_id,
            routed_scaling_factor=1.5,
        )
    router = runner.router
    assert isinstance(router, fused_topk_router.AscendFusedTopKRouter)
    assert router.image_sentinel_count == count
    assert router.bias_vl is vision_bias
    # The plugin retains vision routing configuration; upstream receives the
    # constructed router, not unsupported extra sentinel arguments.
    assert "bias_vl" not in upstream.call_args.kwargs
    assert "image_sentinel_count" not in upstream.call_args.kwargs
    token_ids = torch.arange(129256, 129270)
    logits = torch.tensor([-1.0, 0.0, 1.0, 2.0]).expand(token_ids.numel(), -1).clone()
    mask = (token_ids >= base_id) & (token_ids < base_id + count)
    weights, ids = router._compute_routing(
        torch.zeros(token_ids.numel(), 8),
        logits,
        torch.int32,
        input_ids=token_ids,
        **({"image_token_mask": mask} if version == "v41" else {}),
    )
    expected_ids = torch.tensor(
        [[2, 3] if base_id <= token < base_id + count else [0, 1] for token in token_ids.tolist()], dtype=torch.int32
    )
    torch.testing.assert_close(ids, expected_ids)
    # Biases select experts only. Routed magnitudes come from the original
    # sqrt-softplus scores, then optional normalization and the model scale.
    unbiased = torch.nn.functional.softplus(logits).sqrt()
    expected = unbiased.gather(1, expected_ids.long())
    if renormalize:
        expected /= expected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(weights, expected * 1.5)
    if version == "v41" and base_id == 129264:
        assert ids[token_ids == 129264].tolist() == [[2, 3]]
        assert ids[(token_ids >= 129265) & (token_ids <= 129268)].tolist() == [[0, 1]] * 4


def test_typed_mask_keeps_literal_and_generated_sentinel_hash_ids(monkeypatch):
    monkeypatch.setattr(
        fused_topk_router,
        "_EXTRA_CTX",
        SimpleNamespace(
            moe_comm_type=MoECommType.ALLGATHER,
            moe_comm_method=SimpleNamespace(
                prepare_finalize=SimpleNamespace(all_gather_input_id_with_dp_group=lambda value: value)
            ),
        ),
    )
    table = torch.zeros((129270, 2), dtype=torch.int32)
    table[129264] = torch.tensor([1, 0])
    router = fused_topk_router.AscendFusedTopKRouter(
        top_k=2,
        global_num_experts=4,
        scoring_func="sqrtsoftplus",
        tid2eid=table,
        bias_vl=torch.tensor([0.0, 0.0, 20.0, 19.0]),
        image_sentinel_lo=129264,
        require_image_token_mask=True,
    )
    raw_ids = torch.full((4,), 129264, dtype=torch.int64)
    original_ids = raw_ids.clone()
    mask = torch.tensor([True, False, False, False])
    hidden, logits = torch.zeros(4, 8), torch.zeros(4, 4)
    captured = []
    router.capture_fn = lambda ids: captured.append(ids.clone())
    monkeypatch.setattr(router, "_apply_eplb_mapping", lambda ids: ids + 4)
    weights, selected = router._select_experts(
        hidden,
        logits,
        torch.int64,
        input_ids=raw_ids,
        image_token_mask=mask,
    )
    logical = torch.tensor([[2, 3], [1, 0], [1, 0], [1, 0]])
    torch.testing.assert_close(captured[0].long(), logical)
    torch.testing.assert_close(selected, logical + 4)
    torch.testing.assert_close(raw_ids, original_ids)
    torch.testing.assert_close(weights, torch.full((4, 2), 0.5))
    # A changed mask must affect the next invocation without mutating router state.
    mask.copy_(torch.tensor([False, True, False, False]))
    router._select_experts(hidden, logits, input_ids=raw_ids, image_token_mask=mask)
    assert captured[1].tolist() == [[1, 0], [2, 3], [1, 0], [1, 0]]
    with pytest.raises(ValueError, match="requires explicit"):
        router._select_experts(hidden, logits, input_ids=raw_ids)
    with pytest.raises(ValueError, match=r"bool\[T\]"):
        router._select_experts(hidden, logits, input_ids=raw_ids, image_token_mask=mask.long())
