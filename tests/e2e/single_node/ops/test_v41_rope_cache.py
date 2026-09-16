# SPDX-License-Identifier: Apache-2.0
"""Bitwise rotary acceptance; run only after loading the candidate artifact."""

import importlib

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.v41_rope_cache import (
    v41_index_cache_store,
    v41_main_cache_store,
    v41_main_cache_store_reference,
    v41_rope,
    v41_rope_reference,
)
from vllm_ascend.utils import bootstrap_custom_op_env


@pytest.fixture(scope="module", autouse=True)
def runtime():
    bootstrap_custom_op_env(include_vendor_lib=True)
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    torch.npu.set_device(0)


def case(tokens, heads, width, seed=4141):
    generator = torch.Generator().manual_seed(seed)
    shape = (tokens, width) if heads == 1 else (tokens, heads, width)
    x = torch.randn(shape, generator=generator).mul_(8).bfloat16()
    angles = torch.randn((2048, 32), generator=generator)
    positions = torch.arange(tokens, dtype=torch.int64) * 7 % 2048
    return x, positions, angles.cos(), angles.sin()


def assert_bits(actual, expected):
    assert torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("tokens", [0, 1, 2, 4, 5, 10, 16, 64, 128, 1024])
@pytest.mark.parametrize("heads,width", [(1, 128), (1, 512), (8, 512), (32, 128)])
@pytest.mark.parametrize("inverse", [False, True])
def test_rope_exact(tokens, heads, width, inverse):
    cpu = case(tokens, heads, width)
    tensors = tuple(t.npu() for t in cpu)
    output = torch.empty_like(tensors[0])
    expected = v41_rope_reference(*cpu, inverse=inverse)
    v41_rope(*tensors, output, inverse=inverse)
    assert_bits(output, expected)


def test_rope_boundaries_and_bf16_halfway():
    x, positions, cos, sin = case(4, 1, 128)
    positions[:] = torch.tensor([-1, 0, 2047, 2048])
    x[1, -64::2] = 1
    x[1, -63::2] = 0
    # BF16 midpoint around 1, and neighboring FP32 values on either side.
    midpoint = torch.tensor(1 + 1 / 256, dtype=torch.float32)
    cos[0, :] = midpoint
    cos[0, 0] = torch.nextafter(midpoint, torch.tensor(0.0))
    cos[0, 1] = torch.nextafter(midpoint, torch.tensor(2.0))
    cpu = x, positions, cos, sin
    tensors = tuple(t.npu() for t in cpu)
    output = torch.full_like(tensors[0], float("nan"))
    v41_rope(*tensors, output)
    assert_bits(output, v41_rope_reference(*cpu))


def test_rope_nonzero_storage_offsets():
    cpu = case(5, 8, 512)
    tensors = []
    for source in cpu:
        base = torch.empty(source.numel() + 17, dtype=source.dtype, device="npu:0")
        view = base[3 : 3 + source.numel()].view(source.shape)
        view.copy_(source)
        tensors.append(view)
    output_base = torch.full((cpu[0].numel() + 17,), -7, dtype=torch.bfloat16, device="npu:0")
    output = output_base[5 : 5 + cpu[0].numel()].view(cpu[0].shape)
    v41_rope(*tensors, output)
    assert_bits(output, v41_rope_reference(*cpu))
    assert (output_base[:5].cpu() == -7).all() and (output_base[5 + output.numel() :].cpu() == -7).all()


@pytest.mark.parametrize("inverse", [False, True])
def test_rope_changed_inputs_graph_replay(inverse):
    first = case(4, 8, 512)
    tensors = tuple(t.npu() for t in first)
    output = torch.empty_like(tensors[0])
    for _ in range(3):
        v41_rope(*tensors, output, inverse=inverse)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        v41_rope(*tensors, output, inverse=inverse)
    pointers = [t.data_ptr() for t in (*tensors, output)]
    for seed in (4142, 4143, 4144):
        changed = list(case(4, 8, 512, seed))
        changed[1][:] = torch.tensor([2047, -1, 7, 2048])
        for target, source in zip(tensors, changed):
            target.copy_(source)
        graph.replay()
        assert_bits(output, v41_rope_reference(*changed, inverse=inverse))
        assert pointers == [t.data_ptr() for t in (*tensors, output)]


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("tokens", [127, 128, 129])
def test_rope_heads32_tiled_graph_and_offsets(inverse, tokens):
    cpu = list(case(tokens, 32, 128))
    tensors = []
    for source in cpu:
        storage = torch.empty(source.numel() + 16, dtype=source.dtype, device="npu")
        view = storage[3 : 3 + source.numel()].view_as(source)
        view.copy_(source)
        tensors.append(view)
    storage = torch.full((cpu[0].numel() + 16,), -7, dtype=torch.bfloat16, device="npu")
    output = storage[5 : 5 + cpu[0].numel()].view_as(cpu[0])
    for _ in range(3):
        v41_rope(*tensors, output, inverse=inverse)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        v41_rope(*tensors, output, inverse=inverse)
    for iteration in range(3):
        changed = list(case(tokens, 32, 128, 4150 + iteration))
        changed[1][:8] = torch.tensor([-1, 0, 2047, 2048, 0, 0, 2049, -2])
        for target, source in zip(tensors, changed):
            target.copy_(source)
        graph.replay()
        assert_bits(output, v41_rope_reference(*changed, inverse=inverse))
        assert (storage[:5].cpu() == -7).all() and (storage[5 + output.numel() :].cpu() == -7).all()


def guarded_cache(tokens, width, dtype, dimensions, device):
    page = 32
    blocks = max(1, (tokens + page - 1) // page)
    stride = page * width + 7
    base = torch.full((blocks * stride + 19,), -7, dtype=dtype, device=device)
    shape = (blocks, page, width) if dimensions == 3 else (blocks, page, 1, width)
    strides = (stride, width, 1) if dimensions == 3 else (stride, width, width, 1)
    return base, base.as_strided(shape, strides, storage_offset=5)


def store_case(tokens, index, dimensions, device="cpu"):
    width = 128 if index else 512
    x, positions, cos, sin = case(tokens, 1, width)
    positions = torch.arange(tokens, dtype=torch.int64) % 2048
    slots = torch.arange(tokens, dtype=torch.int64)
    if tokens:
        positions[0] = 1
        x[0].zero_()
    if tokens >= 5:
        positions[1:5] = torch.tensor([-1, 2047, 2048, 2049])
        slots[4] = -1
    base, cache = guarded_cache(tokens, width, torch.int8 if index else torch.bfloat16, dimensions, device)
    scale_base, scales = guarded_cache(tokens, 1, torch.float16, dimensions, device)
    return (x, positions, slots, cos, sin), base, cache, scale_base, scales


def index_store_cann_reference(key, positions, slots, cos, sin, key_cache, scale_cache, *, compress_ratio=1):
    """Strict oracle from both CPU and original NPU RoPE followed by CANN quant.

    Ascend Vector division can differ from CPU division at INT8 half boundaries.
    Keep the original CANN quantizer as the byte-level production reference.
    """
    if key.shape[0] == 0:
        return
    groups = torch.div(positions, compress_ratio, rounding_mode="floor") * compress_ratio
    rounded_cpu = v41_rope_reference(key, groups, cos, sin)
    expected_q, expected_scale = torch_npu.npu_dynamic_quant(rounded_cpu.npu(), dst_type=torch.int8)
    # Also execute the unfused NPU Torch chain. Clipped invalid rows cannot
    # publish, and are excluded from the numerical comparison below.
    native_key, native_cos, native_sin = key.npu(), cos.npu(), sin.npu()
    safe_groups = groups.clamp(0, cos.shape[0] - 1).npu()
    c, s = native_cos[safe_groups], native_sin[safe_groups]
    pairs = native_key[:, -64:].float().unflatten(-1, (-1, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2).bfloat16()
    rounded_npu = torch.cat((native_key[:, :-64], rotated), dim=-1)
    valid = (groups >= 0) & (groups < cos.shape[0])
    assert_bits(rounded_npu.cpu()[valid], rounded_cpu[valid])
    original_q, original_scale = torch_npu.npu_dynamic_quant(rounded_npu, dst_type=torch.int8)
    expected_q, expected_scale = expected_q.cpu(), expected_scale.half().cpu()
    assert torch.equal(original_q.cpu()[valid], expected_q[valid])
    assert_bits(original_scale.half().cpu()[valid], expected_scale[valid])
    for token in range(key.shape[0]):
        position, slot = int(positions[token]), int(slots[token])
        if (
            not valid[token]
            or position < 0
            or (position + 1) % compress_ratio
            or slot < 0
            or slot >= key_cache.shape[0] * key_cache.shape[1]
        ):
            continue
        page, within = divmod(slot, key_cache.shape[1])
        key_cache[page, within].copy_(expected_q[token].view_as(key_cache[page, within]))
        scale_cache[page, within].fill_(expected_scale[token])


@pytest.mark.parametrize("tokens", [0, 1, 2, 4, 5, 10, 16, 64, 128, 1024])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dimensions", [3, 4])
@pytest.mark.parametrize("index", [False, True])
def test_store_exact_and_sentinel_guards(tokens, ratio, dimensions, index):
    cpu, expected_base, expected_cache, expected_scale_base, expected_scales = store_case(tokens, index, dimensions)
    _, base, cache, scale_base, scales = store_case(tokens, index, dimensions, "npu:0")
    tensors = tuple(t.npu() for t in cpu)
    if index:
        index_store_cann_reference(*cpu, expected_cache, expected_scales, compress_ratio=ratio)
        v41_index_cache_store(*tensors, cache, scales, compress_ratio=ratio)
        assert_bits(scale_base, expected_scale_base)
        assert torch.equal(base.cpu(), expected_base)
    else:
        v41_main_cache_store_reference(*cpu, expected_cache, compress_ratio=ratio)
        v41_main_cache_store(*tensors, cache, compress_ratio=ratio)
        assert_bits(base, expected_base)


@pytest.mark.parametrize("index", [False, True])
@pytest.mark.parametrize("ratio", [1, 2])
def test_store_changed_positions_slots_graph(index, ratio):
    # Five virtual DSpark query slots include end-of-context padding. Context
    # stores use the same ABI with their explicit prefix slot mapping.
    cpu, _, _, _, _ = store_case(5, index, 4)
    tensors = tuple(t.npu() for t in cpu)
    _, base, cache, scale_base, scales = store_case(5, index, 4, "npu:0")
    call = v41_index_cache_store if index else v41_main_cache_store
    outputs = (cache, scales) if index else (cache,)
    for _ in range(3):
        call(*tensors, *outputs, compress_ratio=ratio)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call(*tensors, *outputs, compress_ratio=ratio)
    pointers = [t.data_ptr() for t in (*tensors, *outputs)]
    for iteration in range(3):
        changed = list(cpu)
        changed[0] = (cpu[0].float() + iteration * 0.25).bfloat16()
        changed[1] = torch.tensor([2045, 2046, 2047, 2048, 2049]) - iteration
        changed[2] = torch.tensor([31, 7, 5, -1, 32])
        _, expected_base, expected_cache, expected_scale_base, expected_scales = store_case(5, index, 4)
        base.fill_(-7)
        scale_base.fill_(-7)
        for target, source in zip(tensors, changed):
            target.copy_(source)
        graph.replay()
        if index:
            index_store_cann_reference(*changed, expected_cache, expected_scales, compress_ratio=ratio)
            assert torch.equal(base.cpu(), expected_base)
            assert_bits(scale_base, expected_scale_base)
        else:
            v41_main_cache_store_reference(*changed, expected_cache, compress_ratio=ratio)
            assert_bits(base, expected_base)
        assert pointers == [t.data_ptr() for t in (*tensors, *outputs)]


def test_index_store_matches_cann_halfway_zero():
    key = torch.zeros((4, 128), dtype=torch.bfloat16)
    key[0, :7] = torch.tensor([127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5])
    key[1] = key[0] * 0.5
    key[2].fill_(-0.0)
    positions, slots = torch.arange(4), torch.tensor([3, 2, 1, 0])
    cos, sin = torch.ones(4, 32), torch.zeros(4, 32)
    native_key = key.npu()
    expected_key, expected_scale = torch_npu.npu_dynamic_quant(native_key, dst_type=torch.int8)
    _, destination = guarded_cache(4, 128, torch.int8, 4, "npu:0")
    _, scales = guarded_cache(4, 1, torch.float16, 4, "npu:0")
    v41_index_cache_store(native_key, positions.npu(), slots.npu(), cos.npu(), sin.npu(), destination, scales)
    assert torch.equal(destination[0, slots, 0].cpu(), expected_key.cpu())
    assert_bits(scales[0, slots, 0, 0], expected_scale.half().cpu())


@pytest.mark.parametrize("value,maximum", [(9.8125, 19.625), (-11.875, 23.75), (10.6875, 21.375), (9.375, 18.75)])
def test_index_store_cann_division_half_boundaries(value, maximum):
    # These actual r9 failure values distinguish CANN division from the CPU
    # mathematical formula. Preserve strict byte equality with the baseline.
    key = torch.zeros((1, 128), dtype=torch.bfloat16)
    key[0, 0], key[0, 1] = maximum, value
    positions, slots = torch.tensor([1]), torch.tensor([63])
    cos, sin = torch.ones((4, 32)), torch.zeros((4, 32))
    expected_raw, expected_keys, expected_scales = packed_index_page_cache(6, 18, "cpu")
    raw, keys, scales = packed_index_page_cache(6, 18, "npu")
    index_store_cann_reference(key, positions, slots, cos, sin, expected_keys, expected_scales)
    v41_index_cache_store(*(value.npu() for value in (key, positions, slots, cos, sin)), keys, scales)
    assert torch.equal(raw.cpu(), expected_raw)
    cpu_q = (key.float() * (127.0 / key.float().abs().amax(-1, keepdim=True))).round().to(torch.int8)
    assert cpu_q[0, 1] != expected_keys[1, 31, 0, 1]


@pytest.mark.parametrize("maximum", [1e-6, 0.125, 1.0, 63.5, 127.0, 256.0])
def test_index_store_cann_quantization_ranges(maximum):
    key = torch.linspace(-maximum, maximum, 128, dtype=torch.float32).bfloat16()[None].npu()
    expected_key, expected_scale = torch_npu.npu_dynamic_quant(key, dst_type=torch.int8)
    _, destination = guarded_cache(1, 128, torch.int8, 3, "npu:0")
    _, scales = guarded_cache(1, 1, torch.float16, 3, "npu:0")
    v41_index_cache_store(
        key,
        torch.tensor([1], device="npu"),
        torch.tensor([31], device="npu"),
        torch.ones((2, 32), dtype=torch.float32, device="npu"),
        torch.zeros((2, 32), dtype=torch.float32, device="npu"),
        destination,
        scales,
    )
    assert torch.equal(destination[0, 31].cpu(), expected_key[0].cpu())
    assert_bits(scales[0, 31], expected_scale.half().cpu())


def packed_index_page_cache(prefix, gap, device):
    # Same dtype views and ranks as model_runner_v1: 4D INT8 keys, 3D FP16 scales.
    blocks, page = 2, 32
    stride = page * 130 + gap
    raw = torch.full((prefix + blocks * stride + 16,), 0xA5, dtype=torch.uint8, device=device)
    keys = raw.view(torch.int8).as_strided((blocks, page, 1, 128), (stride, 128, 128, 1), prefix)
    scales = raw.view(torch.float16).as_strided((blocks, page, 1), (stride // 2, 1, 1), (prefix + page * 128) // 2)
    return raw, keys, scales


@pytest.mark.parametrize("prefix,gap", [(0, 0), (6, 18)])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_index_store_real_packed_pages(prefix, gap, ratio, graph_mode):
    cpu = list(case(5, 1, 128))
    cpu.insert(2, torch.tensor([63, 0, 32, -1, 64], dtype=torch.int64))
    cpu[1] = torch.tensor([1, 3, 5, 2048, 2049], dtype=torch.int64)
    tensors = tuple(value.npu() for value in cpu)
    raw, keys, scales = packed_index_page_cache(prefix, gap, "npu")

    def call():
        v41_index_cache_store(*tensors, keys, scales, compress_ratio=ratio)

    if graph_mode:
        for _ in range(3):
            call()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            call()
        replay = graph.replay
    else:
        replay = call
    pointers = [value.data_ptr() for value in (*tensors, raw, keys, scales)]
    for iteration in range(3):
        changed = list(cpu)
        changed[0] = (cpu[0].float() + iteration * 0.25).bfloat16()
        changed[1] = cpu[1] - iteration
        changed[2] = torch.tensor([63, 31, 32, -1, 64]) - iteration
        expected_raw, expected_keys, expected_scales = packed_index_page_cache(prefix, gap, "cpu")
        index_store_cann_reference(*changed, expected_keys, expected_scales, compress_ratio=ratio)
        raw.fill_(0xA5)
        for target, source in zip(tensors, changed):
            target.copy_(source)
        replay()
        assert torch.equal(raw.cpu(), expected_raw)
        assert pointers == [value.data_ptr() for value in (*tensors, raw, keys, scales)]
