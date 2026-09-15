# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from types import SimpleNamespace

import pytest
import test_dsa_v41 as attention_test
import torch

from tests.ut.attention.test_dsa_v41_metadata import make_builder, make_common
from vllm_ascend.attention.dsa_v41 import make_v41_attention_metadata


@pytest.fixture(scope="module")
def runtime():
    return attention_test.runtime.__wrapped__()


@pytest.mark.parametrize("role,ratio", [("swa", 1), ("main", 1), ("main", 2), ("index", 1), ("index", 2)])
def test_device_metadata_slots_and_rebuild(runtime, role, ratio):
    builder = make_builder(role, ratio, "npu:2")
    first = builder.build(0, make_common("npu:2"))
    expected = [254, 255, 288, 67, 68, 69, -1] if ratio == 1 else [-1, 159, -1, 65, -1, 66, -1]
    assert first.slot_mapping.cpu().tolist() == expected
    assert first.token_to_req_indices.cpu().tolist() == [0, 0, 0, 1, 1, 1, -1]
    assert first.schedule.shape == (1024,)
    addresses = {
        field.name: getattr(first, field.name).data_ptr()
        for field in fields(first)
        if isinstance(getattr(first, field.name), torch.Tensor)
    }
    second = builder.build(0, make_common("npu:2", changed=True))
    for name, address in addresses.items():
        assert getattr(second, name).data_ptr() == address
    if ratio == 2:
        assert first.slot_mapping.cpu().tolist() == [159, -1, 65, -1, 66, -1, -1]
        assert first.cmp_residual_kv.cpu().tolist() == [1, 1]


def common_from_case(case, original_lengths, role):
    return SimpleNamespace(
        positions=torch.tensor([length - 1 for length in original_lengths], dtype=torch.int64, device="npu:2"),
        query_start_loc=case["cu"],
        seq_lens=case["lengths"],
        num_reqs=len(original_lengths),
        slot_mapping=torch.empty(case["q"].shape[0], dtype=torch.int64, device="npu:2"),
        block_table_tensor=case["swa_bt" if role == "swa" else "cmp_bt"],
        max_seq_len=max(original_lengths),
    )


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_graph_attention_consumes_refreshed_builder_buffers(runtime, ratio):
    case = attention_test.make_case(ratio, [1, 1], [127, 65], sparse=True)
    device = attention_test.device_case(case)
    ops = runtime(ratio)
    swa_builder = make_builder("swa", device="npu:2")
    main_builder = make_builder("main", ratio, "npu:2") if ratio else None
    swa = swa_builder.build(0, common_from_case(device, [127, 65], "swa"))
    main = main_builder.build(0, common_from_case(device, [127, 65], "main")) if ratio else None
    metadata = make_v41_attention_metadata(swa, main)
    for _ in range(3):
        attention_test.forward(ops, device, metadata)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual, _ = attention_test.forward(ops, device, metadata)
    graph.replay()
    attention_test.check(actual, attention_test.reference(case, ratio)[0])
    updated = attention_test.make_case(ratio, [1, 1], [128, 66], sparse=True, seed=72)
    changed = attention_test.device_case(updated)
    for name, value in device.items():
        if value is not None:
            value.copy_(changed[name])
    swa_builder.build(0, common_from_case(device, [128, 66], "swa"))
    if ratio:
        main_builder.build(0, common_from_case(device, [128, 66], "main"))
    # The captured metadata object is not replaced: only its fixed-address
    # tensor contents changed, including scheduling and compressed lengths.
    graph.replay()
    attention_test.check(actual, attention_test.reference(updated, ratio)[0])
