# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from tests.ut.attention.test_dsa_v41_metadata import make_builder, make_execution_common
from vllm_ascend.attention.dsa_v41 import AscendV41CacheMetadata
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, FusedExpertsResult
from vllm_ascend.quantization.methods.wna16.w4a16 import AscendW4A16FusedMoEMethod

MODULE = "vllm_ascend.quantization.methods.wna16.w4a16"


def fixture():
    method = object.__new__(AscendW4A16FusedMoEMethod)
    method.enable_native_decode = True
    method.native_decode_max_tokens = 128
    method.group_size = 32
    method.dynamic_eplb = False
    comm = object.__new__(AllGatherCommImpl)
    comm.moe_config = SimpleNamespace(ep_size=1, activation=MoEActivation.SILU, swiglu_limit=10.0)
    comm.fused_experts = Mock(return_value=FusedExpertsResult(torch.empty(1, 5120)))
    layer = SimpleNamespace(
        w13_weight_packed=torch.empty(384, 5120, 72, device="meta", dtype=torch.int32),
        w2_weight_packed=torch.empty(384, 288, 640, device="meta", dtype=torch.int32),
        w13_weight_scale=torch.empty(384, 160, 576, device="meta", dtype=torch.bfloat16),
        w2_weight_scale=torch.empty(384, 9, 5120, device="meta", dtype=torch.bfloat16),
        ascend_expert_map=None,
        apply_router_weight_on_input=False,
        global_redundant_expert_num=0,
        ascend_pertoken_scale=None,
        ascend_mc2_mask=None,
        activation=MoEActivation.SILU,
    )
    context = SimpleNamespace(attn_metadata={"attention": SimpleNamespace(num_prefills=0, num_decode_tokens=1)})
    x = torch.zeros(1, 5120, dtype=torch.bfloat16)
    ids = torch.arange(6, dtype=torch.int32).unsqueeze(0)
    return method, comm, layer, context, x, ids


@pytest.mark.parametrize(
    "batch, expected", [(1, True), (4, True), (6, True), (8, True), (64, True), (128, True), (129, False)]
)
def test_measured_decode_boundary(batch, expected):
    method, comm, layer, context, x, ids = fixture()
    with (
        patch(f"{MODULE}.get_forward_context", return_value=context),
        patch(f"{MODULE}.torch.ops._C_ascend.npu_w4a16_moe", create=True),
    ):
        assert method._can_use_native_decode(layer, x.expand(batch, -1), ids.expand(batch, -1), comm) == expected


@pytest.mark.parametrize("limit", [0, 4, 128])
def test_decode_threshold_environment_is_read_at_construction(monkeypatch, limit):
    monkeypatch.setenv("VLLM_ASCEND_W4A16_DECODE_MAX_TOKENS", str(limit))
    config = SimpleNamespace(
        quant_config=SimpleNamespace(quant_description={"group_size": 32}), use_v2_model_runner=True
    )
    with (
        patch(f"{MODULE}.get_current_vllm_config", return_value=config),
        patch(f"{MODULE}.get_ascend_config", return_value=SimpleNamespace(enable_w4a16_decode=True)),
    ):
        method = AscendW4A16FusedMoEMethod()
    assert method.native_decode_max_tokens == limit
    _, comm, layer, context, x, ids = fixture()
    with (
        patch(f"{MODULE}.get_forward_context", return_value=context),
        patch(f"{MODULE}.torch.ops._C_ascend.npu_w4a16_moe", create=True),
    ):
        assert method._can_use_native_decode(layer, x.expand(8, -1), ids.expand(8, -1), comm) == (limit >= 8)


@pytest.mark.parametrize("value", ["-1", "bad", "1.5"])
def test_invalid_decode_threshold_is_rejected(monkeypatch, value):
    from vllm_ascend import envs

    monkeypatch.setenv("VLLM_ASCEND_W4A16_DECODE_MAX_TOKENS", value)
    with pytest.raises(ValueError):
        _ = envs.VLLM_ASCEND_W4A16_DECODE_MAX_TOKENS


@pytest.mark.parametrize(
    "case",
    [
        "disabled",
        "prefill",
        "mixed",
        "missing_metadata",
        "ep",
        "lora",
        "eplb",
        "group64",
        "gelu",
        "input_weight",
        "expert_map",
        "other_shape",
    ],
)
def test_unsupported_conditions_fall_back(case):
    method, comm, layer, context, x, ids = fixture()
    if case == "disabled":
        method.enable_native_decode = False
    elif case in ("prefill", "mixed"):
        context.attn_metadata["attention"].num_prefills = 1
        if case == "prefill":
            context.attn_metadata["attention"].num_decode_tokens = 0
    elif case == "missing_metadata":
        context.attn_metadata = None
    elif case == "ep":
        comm.moe_config.ep_size = 8
    elif case == "lora":
        layer._ascend_moe_lora_context = object()
    elif case == "eplb":
        method.dynamic_eplb = True
    elif case == "group64":
        method.group_size = 64
    elif case == "gelu":
        comm.moe_config.activation = MoEActivation.GELU
    elif case == "input_weight":
        layer.apply_router_weight_on_input = True
    elif case == "expert_map":
        layer.ascend_expert_map = torch.arange(384)
    elif case == "other_shape":
        x = x[:, :128]
    with (
        patch(f"{MODULE}.get_forward_context", return_value=context),
        patch(f"{MODULE}.torch.ops._C_ascend.npu_w4a16_moe", create=True),
    ):
        assert not method._can_use_native_decode(layer, x, ids, comm)


def test_native_return_preserves_finalize_and_shared_expert_contract():
    method, comm, layer, context, x, ids = fixture()
    weights = torch.full((1, 6), 0.25, dtype=torch.float32)
    expected = torch.ones_like(x)
    with (
        patch(f"{MODULE}._EXTRA_CTX", SimpleNamespace(moe_comm_method=comm)),
        patch(f"{MODULE}.get_forward_context", return_value=context),
        patch(f"{MODULE}.torch.ops._C_ascend.npu_w4a16_moe", return_value=expected, create=True) as op,
    ):
        actual = method.apply(layer, x, weights, ids, None, None)
    assert isinstance(actual, FusedExpertsResult)
    assert actual.routed_out is expected
    assert actual.before_gmm2_evt is None
    assert op.call_args.args[-1] == 10.0
    torch.testing.assert_close(op.call_args.args[-2], weights)
    comm.fused_experts.assert_not_called()


def test_one_token_prefill_keeps_cann_pipeline():
    method, comm, layer, context, x, ids = fixture()
    context.attn_metadata["attention"].num_prefills = 1
    context.attn_metadata["attention"].num_decode_tokens = 0
    weights = torch.full((1, 6), 0.25, dtype=torch.float32)
    with (
        patch(f"{MODULE}._EXTRA_CTX", SimpleNamespace(moe_comm_method=comm)),
        patch(f"{MODULE}.get_forward_context", return_value=context),
        patch(f"{MODULE}.torch.ops._C_ascend.npu_w4a16_moe", create=True) as op,
    ):
        actual = method.apply(layer, x, weights, ids, None, None)
    assert actual is comm.fused_experts.return_value
    op.assert_not_called()
    comm.fused_experts.assert_called_once()


@pytest.mark.parametrize(
    "query_lengths,prefilling,capture,expected",
    [
        ([1], [False], False, True),
        ([1], [True], False, False),
        ([1, 1], [False, True], False, False),
        ([1, 1, 0], [False, False, False], False, True),
        ([1], None, False, False),
        ([1], [True], True, True),
        ([4], [True], True, False),
    ],
)
def test_real_v41_metadata_controls_native_dispatch(query_lengths, prefilling, capture, expected):
    method, comm, layer, context, x, ids = fixture()
    common = make_execution_common(query_lengths, prefilling)
    builder = make_builder("swa")
    with (
        patch(
            "torch.ops._C_ascend.npu_sparse_flash_mla_metadata",
            create=True,
            return_value=torch.zeros(1024, dtype=torch.int32),
        ),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("no dispatch D2H")),
    ):
        metadata = builder.build_for_cudagraph_capture(common) if capture else builder.build(0, common)
    assert isinstance(metadata, AscendV41CacheMetadata)
    # Real hybrid group order can put unclassified compressor state first.
    context.attn_metadata = {"compressor.state_cache": object(), "layer.swa_cache": metadata}
    with (
        patch(f"{MODULE}.get_forward_context", return_value=context),
        patch(f"{MODULE}.torch.ops._C_ascend.npu_w4a16_moe", create=True),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("no dispatch D2H")),
    ):
        assert method._can_use_native_decode(layer, x.expand(4, -1), ids.expand(4, -1), comm) == expected
