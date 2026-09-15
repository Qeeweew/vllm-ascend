# SPDX-License-Identifier: Apache-2.0
"""CPU comparisons against the released V4.1 vision.py, not a V4 tower."""

import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v4.model import (
    AscendV41VisionAligner,
    AscendV41VisionRMSNorm,
    AscendV41VisionTower,
    _v41_vision_cos_sin,
    _v41_vision_rotary,
)


def config(**overrides):
    values = dict(
        vision_dim=32,
        vision_n_heads=4,
        vision_n_layers=2,
        vision_inter_dim=48,
        vision_patch_size=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=3,
        hidden_size=40,
        dim=40,  # The released standalone aligner calls the LLM width dim.
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(scope="module")
def official_vision():
    path = Path("/mnt/models/DeepSeek-V4.1-Flash/inference/vision.py")
    if not path.is_file():
        pytest.skip("released V4.1 inference/vision.py is not available")
    spec = importlib.util.spec_from_file_location("released_v41_vision_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("grid", [(1, 1), (2, 5), (3, 3), (4, 7)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tower_and_aligner_match_released_reference(official_vision, grid, dtype):
    torch.manual_seed(4101)
    cfg = config()
    tower = AscendV41VisionTower(cfg, dtype=dtype)
    aligner = AscendV41VisionAligner(cfg, dtype=dtype)
    reference_tower = official_vision.ViT(cfg).to(dtype)
    reference_aligner = official_vision.Aligner(cfg).to(dtype)
    reference_tower.load_state_dict(tower.state_dict(), strict=True)
    reference_aligner.load_state_dict(aligner.state_dict(), strict=True)
    n_h, n_w = grid
    patches = torch.randn(n_h * n_w, 3, cfg.vision_patch_size, cfg.vision_patch_size).to(dtype)
    with torch.inference_mode():
        expected_features = reference_tower(patches, n_h, n_w)
        actual_features = tower(patches, n_h, n_w)
        expected_rows = reference_aligner(expected_features, n_h, n_w)
        actual_rows = aligner(actual_features, n_h, n_w)
    torch.testing.assert_close(actual_features, expected_features, rtol=0, atol=0)
    torch.testing.assert_close(actual_rows, expected_rows, rtol=0, atol=0)
    assert actual_rows.shape == (((n_h + 2) // 3) * ((n_w + 2) // 3), cfg.hidden_size)
    assert actual_rows.dtype == dtype


def test_two_dimensional_rope_coordinates_and_half_split():
    cos, sin = _v41_vision_cos_sin(2, 3, 8, 10000.0, torch.device("cpu"))
    # H/W coordinates interleave as axis blocks, not an interleaved complex
    # rotation of neighboring x elements or the LLM's tail-only rotation.
    frequencies = torch.tensor([[h, h / 100, w, w / 100] for h in range(2) for w in range(3)])
    torch.testing.assert_close(cos[:, 0], frequencies.cos(), atol=0, rtol=0)
    torch.testing.assert_close(sin[:, 0], frequencies.sin(), atol=0, rtol=0)
    x = torch.arange(6 * 2 * 8).reshape(6, 2, 8).float()
    rotated = _v41_vision_rotary(x, cos, sin)
    first = x[..., :4] * frequencies.cos()[:, None] - x[..., 4:] * frequencies.sin()[:, None]
    second = x[..., :4] * frequencies.sin()[:, None] + x[..., 4:] * frequencies.cos()[:, None]
    torch.testing.assert_close(rotated, torch.cat((first, second), -1), atol=0, rtol=0)


@pytest.mark.parametrize("grid", [(1, 1), (3, 3), (4, 5)])
def test_aligner_channel_major_unfold_and_bottom_right_padding(grid):
    cfg = config(vision_dim=2, hidden_size=18)
    aligner = AscendV41VisionAligner(cfg, dtype=torch.float32)
    n_h, n_w = grid
    features = torch.arange(n_h * n_w * 2).reshape(n_h * n_w, 2).float()
    observed = []
    handle = aligner.w1.register_forward_pre_hook(lambda _, arguments: observed.append(arguments[0].detach().clone()))
    aligner(features, n_h, n_w)
    handle.remove()
    expected = []
    for top in range(0, n_h, 3):
        for left in range(0, n_w, 3):
            expected.append(
                [
                    features[(top + h) * n_w + left + w, channel].item() if top + h < n_h and left + w < n_w else 0
                    for channel in range(2)
                    for h in range(3)
                    for w in range(3)
                ]
            )
    torch.testing.assert_close(observed[0], torch.tensor(expected), rtol=0, atol=0)


def test_norm_uses_vision_epsilon_and_does_not_assume_positive_gamma():
    norm = AscendV41VisionRMSNorm(4)
    norm.weight.data.copy_(torch.tensor([1, -2, 0, 0.25]))
    x = torch.tensor([[1e-7, 2e-7, 0, -3e-7]], dtype=torch.bfloat16)
    expected = (
        x.double() / (x.double().square().mean(-1, keepdim=True) + 1e-6).sqrt() * norm.weight.double()
    ).bfloat16()
    torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)
    incorrect = (x.float() * (x.float().square().mean(-1, keepdim=True) + 1e-20).rsqrt() * norm.weight).bfloat16()
    assert not torch.equal(norm(x), incorrect)


def test_attention_factory_instantiated_before_forward_and_receives_full_image():
    calls = []

    class CaptureAttention(torch.nn.Module):
        def forward(self, q, k, v):
            assert q.shape == k.shape == v.shape == (1, 6, 4, 8)
            calls.append("forward")
            return torch.zeros_like(v)

    def factory(heads, head_dim):
        assert (heads, head_dim) == (4, 8)
        calls.append("create")
        return CaptureAttention()

    tower = AscendV41VisionTower(config(), dtype=torch.float32, attention_factory=factory)
    assert calls == ["create", "create"]
    tower(torch.randn(6, 3, 2, 2), 2, 3)
    assert calls == ["create", "create", "forward", "forward"]


def test_local_checkpoint_names_and_strict_loading(official_vision):
    cfg = config()
    for native, official in (
        (AscendV41VisionTower(cfg), official_vision.ViT(cfg)),
        (AscendV41VisionAligner(cfg), official_vision.Aligner(cfg)),
    ):
        assert set(native.state_dict()) == set(official.state_dict())
        # Actual checkpoint values are BF16, including norm gamma. Loading
        # them into FP32 norm storage preserves the exact BF16 values.
        checkpoint = {name: tensor.bfloat16() for name, tensor in official.state_dict().items()}
        native.load_state_dict(checkpoint, strict=True)
        for name, actual in native.state_dict().items():
            torch.testing.assert_close(actual.float(), checkpoint[name].float(), atol=0, rtol=0)
        missing = dict(checkpoint)
        missing.pop(next(iter(missing)))
        with pytest.raises(RuntimeError, match="Missing key"):
            native.load_state_dict(missing, strict=True)
        extra = dict(checkpoint, not_a_vision_weight=torch.ones(1))
        with pytest.raises(RuntimeError, match="Unexpected key"):
            native.load_state_dict(extra, strict=True)
        wrong_shape = dict(checkpoint)
        first = next(iter(wrong_shape))
        wrong_shape[first] = torch.empty(1)
        with pytest.raises(RuntimeError, match="size mismatch"):
            native.load_state_dict(wrong_shape, strict=True)


@pytest.mark.parametrize(
    "overrides",
    [
        {"vision_n_heads": 0},
        {"vision_dim": 30},
        {"vision_n_layers": 0},
        {"vision_patch_size": 0},
        {"vision_rope_theta": float("nan")},
    ],
)
def test_tower_rejects_invalid_geometry(overrides):
    with pytest.raises(ValueError, match="geometry"):
        AscendV41VisionTower(config(**overrides))


def test_components_reject_inconsistent_grids_and_dtype():
    tower, aligner = AscendV41VisionTower(config()), AscendV41VisionAligner(config())
    with pytest.raises(ValueError, match="grid"):
        tower(torch.zeros(5, 3, 2, 2, dtype=torch.bfloat16), 2, 3)
    with pytest.raises(ValueError, match="dtype"):
        tower(torch.zeros(6, 3, 2, 2), 2, 3)
    with pytest.raises(ValueError, match="grid"):
        aligner(torch.zeros(5, 32, dtype=torch.bfloat16), 2, 3)
    with pytest.raises(ValueError, match="dtype"):
        aligner(torch.zeros(6, 32), 2, 3)


def test_released_checkpoint_tower_and_aligner_header_shapes_without_weight_allocation():
    root = Path("/mnt/models/DeepSeek-V4.1-Flash")
    if not (root / "config.json").is_file():
        pytest.skip("released V4.1 checkpoint is not available")
    model_config = json.loads((root / "config.json").read_text())
    vision = model_config["vision_config"]
    cfg = config(
        vision_dim=vision["hidden_size"],
        vision_n_heads=vision["num_attention_heads"],
        vision_n_layers=vision["num_hidden_layers"],
        vision_inter_dim=vision["intermediate_size"],
        vision_patch_size=vision["patch_size"],
        vision_rope_theta=vision["rope_theta"],
        vision_downsample_ratio=vision["downsample_ratio"],
        hidden_size=model_config["text_config"]["hidden_size"],
    )
    with torch.device("meta"):
        tower = AscendV41VisionTower(cfg)
        aligner = AscendV41VisionAligner(cfg)
    expected = {f"vision.{name}": list(value.shape) for name, value in tower.state_dict().items()}
    expected.update({f"aligner.{name}": list(value.shape) for name, value in aligner.state_dict().items()})
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    actual_names = {name for name in index if name.startswith(("vision.", "aligner."))}
    assert actual_names == set(expected)
    assert len(expected) == 263
    headers = {}
    for name, expected_shape in expected.items():
        shard = index[name]
        if shard not in headers:
            with (root / shard).open("rb") as stream:
                header_length = struct.unpack("<Q", stream.read(8))[0]
                headers[shard] = json.loads(stream.read(header_length))
        descriptor = headers[shard][name]
        assert descriptor["shape"] == expected_shape
        assert descriptor["dtype"] == "BF16"
