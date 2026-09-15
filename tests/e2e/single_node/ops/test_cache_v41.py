# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.cache_v41 import write_index_cache_v41, write_main_cache_v41


@pytest.fixture(scope="module")
def device():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.npu.set_device(0)
    return torch.device("npu:0")


def make_case(tokens, ratio, width, dtype, rank=4, gapped=True):
    rng = torch.Generator().manual_seed(tokens + ratio + width)
    pages, block = max(4, (tokens + 31) // 32), 32
    shape = (
        (pages, block * (2 if gapped else 1), 1, width) if rank == 4 else (pages, block * (2 if gapped else 1), width)
    )
    storage = torch.full(shape, -17, dtype=dtype)
    cache = storage[:, :block]
    values = torch.randint(-100, 100, (tokens, width), generator=rng).to(dtype)
    slots = torch.randperm(pages * block, generator=rng)[:tokens].to(torch.int64)
    positions = torch.arange(tokens, dtype=torch.int64) + 7
    if tokens:
        slots[::7] = -1
        positions[::11] = -1
        slots[::13] = pages * block + 19
    return storage, cache, values, slots, positions


def reference(cache, values, slots, positions, ratio):
    for row, (slot, position) in enumerate(zip(slots.tolist(), positions.tolist())):
        if 0 <= slot < cache.shape[0] * cache.shape[1] and position >= 0 and (position + 1) % ratio == 0:
            cache[slot // cache.shape[1], slot % cache.shape[1]] = values[row].view_as(cache[0, 0])


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("tokens", [0, 1, 8, 127, 513, 4096])
@pytest.mark.parametrize("rank", [3, 4])
def test_main_bit_exact_padding_group_end_and_page_stride(device, ratio, tokens, rank):
    storage, cache, values, slots, positions = make_case(tokens, ratio, 512, torch.bfloat16, rank)
    device_storage = storage.to(device)
    actual = device_storage[:, :32]
    result = write_main_cache_v41(
        actual, values.to(device), slots.to(device), positions=positions.to(device), compress_ratio=ratio
    )
    assert result.data_ptr() == actual.data_ptr()
    reference(cache, values, slots, positions, ratio)
    torch.testing.assert_close(device_storage.cpu(), storage, atol=0, rtol=0)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("tokens", [0, 1, 8, 127, 513, 4096])
@pytest.mark.parametrize("scale_rank", [3, 4])
def test_index_key_and_scale_bit_exact(device, ratio, tokens, scale_rank):
    storage, cache, values, slots, positions = make_case(tokens, ratio, 128, torch.int8)
    scale_storage, scale_cache, scales, _, _ = make_case(tokens, ratio, 1, torch.float16, scale_rank)
    device_storage, device_scales = storage.to(device), scale_storage.to(device)
    actual_k, actual_s = device_storage[:, :32], device_scales[:, :32]
    returned_k, returned_s = write_index_cache_v41(
        actual_k,
        actual_s,
        values.to(device),
        scales[:, 0].to(device),
        slots.to(device),
        positions=positions.to(device),
        compress_ratio=ratio,
    )
    assert returned_k.data_ptr() == actual_k.data_ptr() and returned_s.data_ptr() == actual_s.data_ptr()
    reference(cache, values, slots, positions, ratio)
    reference(scale_cache, scales, slots, positions, ratio)
    torch.testing.assert_close(device_storage.cpu(), storage, atol=0, rtol=0)
    torch.testing.assert_close(device_scales.cpu(), scale_storage, atol=0, rtol=0)


@pytest.mark.parametrize("ratio", [1, 2])
def test_graph_dynamic_slots_values_rollback_and_masks(device, ratio):
    storage, cache, values, slots, positions = make_case(16, ratio, 512, torch.bfloat16)
    key_storage, key_cache, keys, _, _ = make_case(16, ratio, 128, torch.int8)
    scale_storage, scale_cache, scales, _, _ = make_case(16, ratio, 1, torch.float16)
    main_d, key_d, scale_d = [x.to(device) for x in (storage, key_storage, scale_storage)]
    values_d, keys_d, scales_d, slots_d, pos_d = [x.to(device) for x in (values, keys, scales, slots, positions)]

    def run():
        write_main_cache_v41(main_d[:, :32], values_d, slots_d, positions=pos_d, compress_ratio=ratio)
        write_index_cache_v41(
            key_d[:, :32], scale_d[:, :32], keys_d, scales_d, slots_d, positions=pos_d, compress_ratio=ratio
        )

    for _ in range(3):
        run()
    main_d.copy_(storage)
    key_d.copy_(key_storage)
    scale_d.copy_(scale_storage)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    # Capture itself may execute; restore initial contents before testing replay.
    main_d.copy_(storage)
    key_d.copy_(key_storage)
    scale_d.copy_(scale_storage)
    for step in range(20):
        slots = (torch.arange(16) + step * 3) % 128
        positions = torch.arange(16) + (step % 5)  # Advances and rolls back.
        if step % 4 == 0:
            slots.fill_(-1)
        else:
            slots[::3] = -1
            positions[::5] = -1
        values = values + 1
        keys = keys + 1
        scales = scales + 1
        for dst, src in zip((values_d, keys_d, scales_d, slots_d, pos_d), (values, keys, scales, slots, positions)):
            dst.copy_(src)
        graph.replay()
        reference(cache, values, slots, positions, ratio)
        reference(key_cache, keys, slots, positions, ratio)
        reference(scale_cache, scales, slots, positions, ratio)
        torch.testing.assert_close(main_d.cpu(), storage, atol=0, rtol=0)
        torch.testing.assert_close(key_d.cpu(), key_storage, atol=0, rtol=0)
        torch.testing.assert_close(scale_d.cpu(), scale_storage, atol=0, rtol=0)


@pytest.mark.parametrize("ratio,position", [(1, 0), (2, 1), (2, 2)])
@pytest.mark.parametrize("slot", [0, 127])
def test_single_token_first_and_last_physical_slots(device, ratio, position, slot):
    cache = torch.full((4, 32, 1, 512), -3, dtype=torch.bfloat16, device=device)
    value = torch.full((1, 512), 7, dtype=torch.bfloat16, device=device)
    write_main_cache_v41(
        cache,
        value,
        torch.tensor([slot], dtype=torch.int64, device=device),
        positions=torch.tensor([position], dtype=torch.int64, device=device),
        compress_ratio=ratio,
    )
    expected = torch.full((4, 32, 1, 512), -3, dtype=torch.bfloat16)
    if (position + 1) % ratio == 0:
        expected[slot // 32, slot % 32] = 7
    torch.testing.assert_close(cache.cpu(), expected, atol=0, rtol=0)


def test_store_preserves_bf16_and_scale_bit_patterns(device):
    cache = torch.zeros((4, 32, 1, 512), dtype=torch.bfloat16, device=device)
    # Exhaust BF16 bit patterns, including signed zero, subnormals and NaNs.
    bits = torch.arange(65536).to(torch.int16).reshape(128, 512)
    values = bits.view(torch.bfloat16)
    slots = torch.arange(128, dtype=torch.int64, device=device)
    write_main_cache_v41(cache, values.to(device), slots)
    torch.testing.assert_close(cache.cpu().view(torch.int16).reshape_as(bits), bits, atol=0, rtol=0)
    key_cache = torch.zeros((4, 32, 1, 128), dtype=torch.int8, device=device)
    scale_cache = torch.zeros((4, 32, 1), dtype=torch.float16, device=device)
    scale_bits = torch.tensor([0, -32768, 1, 1023, 31744, -1024, 32257, 15360], dtype=torch.int16).repeat(16)
    keys = torch.arange(128 * 128).to(torch.int8).reshape(128, 128)
    write_index_cache_v41(key_cache, scale_cache, keys.to(device), scale_bits.view(torch.float16).to(device), slots)
    torch.testing.assert_close(key_cache.cpu().reshape_as(keys), keys, atol=0, rtol=0)
    torch.testing.assert_close(scale_cache.cpu().view(torch.int16).flatten(), scale_bits, atol=0, rtol=0)
