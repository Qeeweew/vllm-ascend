# SPDX-License-Identifier: Apache-2.0
"""NPU acceptance for V4.1 vision components; run only on a reserved device.

.venv/bin/python -m pytest --confcutdir=tests/e2e/single_node/ops \
    tests/e2e/single_node/ops/test_v41_vision_components.py -q

These tests exercise real NPU encoder attention with H64 and the released
standalone vision.py oracle. They do not establish TP sharding or full-model
multimodal/graph support. No custom kernel rebuild is needed for this change.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.models.deepseek_v4.model import AscendV41VisionAligner, AscendV41VisionTower


@pytest.fixture(scope="module")
def device():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    # A caller can select an idle NPU before pytest.main(); never override
    # the parent's reservation with a hard-coded device 0.
    with set_current_vllm_config(VllmConfig()):
        yield torch.device("npu", torch.npu.current_device())


@pytest.fixture(scope="module")
def official_vision():
    path = Path("/mnt/models/DeepSeek-V4.1-Flash/inference/vision.py")
    if not path.is_file():
        pytest.skip("released V4.1 inference/vision.py is not available")
    spec = importlib.util.spec_from_file_location("released_v41_vision_npu_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def npu_attention_factory(heads, head_dim):
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention

    # MMEncoderAttention selects its device backend using the model's default
    # dtype at construction, as it does inside the vLLM model loader.
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        return AscendMMEncoderAttention(num_heads=heads, head_size=head_dim)
    finally:
        torch.set_default_dtype(previous)


def config(real_width=False):
    return SimpleNamespace(
        vision_dim=1024 if real_width else 128,
        vision_n_heads=16 if real_width else 2,
        vision_n_layers=1 if real_width else 2,
        vision_inter_dim=2816 if real_width else 192,
        vision_patch_size=14,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=3,
        hidden_size=5120 if real_width else 256,
        dim=5120 if real_width else 256,
    )


def check_error(actual, expected, record_property, label):
    # Evaluate metrics on CPU in FP64: FP32 cosine reductions can slightly
    # exceed one and hide small BF16 output-direction errors in the report.
    actual, expected = actual.cpu().double(), expected.double()
    difference = actual - expected
    nrmse = (difference.square().sum() / expected.square().sum().clamp_min(1e-30)).sqrt().item()
    cosine_error = 1 - torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    record_property(f"{label}_nrmse", nrmse)
    record_property(f"{label}_cosine_error", cosine_error)
    record_property(f"{label}_max_abs", difference.abs().max().item())
    assert torch.isfinite(actual).all()
    assert nrmse < 1e-2, f"{label} NRMSE={nrmse}"
    assert cosine_error < 5e-5, f"{label} cosine error={cosine_error}"
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)


@pytest.mark.parametrize("grid,real_width", [((1, 1), False), ((3, 5), False), ((8, 9), False), ((4, 7), True)])
def test_npu_tower_aligner_against_released_reference(device, official_vision, record_property, grid, real_width):
    torch.manual_seed(4117)
    cfg = config(real_width)
    tower = AscendV41VisionTower(cfg, attention_factory=npu_attention_factory)
    aligner = AscendV41VisionAligner(cfg)
    reference_tower = official_vision.ViT(cfg).bfloat16()
    reference_aligner = official_vision.Aligner(cfg).bfloat16()
    reference_tower.load_state_dict(tower.state_dict(), strict=True)
    reference_aligner.load_state_dict(aligner.state_dict(), strict=True)
    n_h, n_w = grid
    patches = torch.randn(n_h * n_w, 3, 14, 14).bfloat16()
    with torch.inference_mode():
        expected_features = reference_tower(patches, n_h, n_w)
        expected_rows = reference_aligner(expected_features, n_h, n_w)
        tower, aligner = tower.to(device), aligner.to(device)
        actual_features = tower(patches.to(device), n_h, n_w)
        actual_rows = aligner(actual_features, n_h, n_w)
    check_error(actual_features, expected_features, record_property, "tower")
    check_error(actual_rows, expected_rows, record_property, "aligner")
    assert actual_rows.shape == (((n_h + 2) // 3) * ((n_w + 2) // 3), cfg.hidden_size)


def test_npu_vision_attention_is_bidirectional(device):
    attention = npu_attention_factory(2, 64).to(device)
    q = torch.zeros((1, 9, 2, 64), dtype=torch.bfloat16, device=device)
    values = torch.arange(9, dtype=torch.bfloat16, device=device).view(1, 9, 1, 1).expand_as(q).contiguous()
    with torch.inference_mode():
        output = attention(q, q, values)
    # Uniform scores must include future keys even for the first query.
    torch.testing.assert_close(output.cpu(), torch.full_like(output.cpu(), 4), rtol=0, atol=0)


def test_npu_tower_reuses_weights_across_image_shapes(device, official_vision, record_property):
    torch.manual_seed(4118)
    cfg = config()
    tower = AscendV41VisionTower(cfg, attention_factory=npu_attention_factory)
    reference = official_vision.ViT(cfg).bfloat16()
    reference.load_state_dict(tower.state_dict(), strict=True)
    tower = tower.to(device)
    addresses = [parameter.data_ptr() for parameter in tower.parameters()]
    with torch.inference_mode():
        for index, (n_h, n_w) in enumerate([(3, 4), (5, 2), (3, 4)]):
            patches = torch.randn(n_h * n_w, 3, 14, 14).bfloat16()
            check_error(
                tower(patches.to(device), n_h, n_w), reference(patches, n_h, n_w), record_property, f"image{index}"
            )
    assert [parameter.data_ptr() for parameter in tower.parameters()] == addresses
