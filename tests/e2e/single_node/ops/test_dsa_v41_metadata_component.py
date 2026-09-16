# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

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


@pytest.mark.parametrize("role,ratio", [("swa", 1), ("main", 1), ("main", 2), ("index", 2)])
def test_fused_preparation_matches_strided_fallback_in_graph(runtime, role, ratio):
    from vllm_ascend.worker.v41_metadata import V41MetadataPreparation

    assert hasattr(torch.ops._C_ascend, "v41_cache_metadata")
    fused = make_builder(role, ratio, "npu:2")
    fallback = make_builder(role, ratio, "npu:2")
    common = make_common("npu:2")
    strided = make_common("npu:2")
    storage = torch.empty((2, 8), dtype=torch.int32, device="npu:2")
    strided.block_table_tensor = storage[:, ::2]
    owner = V41MetadataPreparation()
    for step in range(3):
        changed = make_common("npu:2", changed=step == 1)
        for name in ("positions", "query_start_loc", "seq_lens", "block_table_tensor"):
            getattr(common, name).copy_(getattr(changed, name))
            getattr(strided, name).copy_(getattr(changed, name))
        if step == 2:
            common.positions.fill_(-1)
            strided.positions.fill_(-1)
        batch = owner.batch(capture=step == 0, use_graph=True)
        with patch.object(fused, "_refresh_slots", side_effect=AssertionError("Expected fused preparation")):
            actual = fused.build(0, common, preparation=batch)
            batch.run()
        with patch.object(fallback, "_refresh_slots", wraps=fallback._refresh_slots) as split:
            expected = fallback.build(0, strided)
            split.assert_called_once()
        for name in (
            "positions",
            "cu_seqlens_q",
            "seqused_kv",
            "block_table",
            "slot_mapping",
            "token_to_req_indices",
            "seqused_cmp_kv",
            "cmp_residual_kv",
        ):
            value = getattr(actual, name)
            if value is not None:
                torch.testing.assert_close(value.cpu(), getattr(expected, name).cpu(), rtol=0, atol=0)
    assert len(owner.graphs) == 1


def common_from_case(case, original_lengths, role):
    query_lengths = (case["cu"][1:] - case["cu"][:-1]).cpu().tolist()
    return SimpleNamespace(
        positions=torch.tensor(
            [
                position
                for length, count in zip(original_lengths, query_lengths)
                for position in range(length - count, length)
            ],
            dtype=torch.int64,
            device="npu:2",
        ),
        query_start_loc=case["cu"],
        seq_lens=case["lengths"],
        num_reqs=len(original_lengths),
        num_actual_tokens=case["q"].shape[0],
        slot_mapping=torch.empty(case["q"].shape[0], dtype=torch.int64, device="npu:2"),
        block_table_tensor=case["swa_bt" if role == "swa" else "cmp_bt"],
        max_seq_len=max(original_lengths),
    )


@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("query_lengths,contexts", [([1, 1], [127, 65]), ([3, 2], [33, 67]), ([6, 6], [127, 65])])
def test_graph_attention_consumes_refreshed_builder_buffers(runtime, ratio, query_lengths, contexts):
    from vllm_ascend.worker.v41_metadata import V41MetadataPreparation

    case = attention_test.make_case(ratio, query_lengths, contexts, sparse=True)
    device = attention_test.device_case(case)
    ops = runtime(ratio)
    swa_builder = make_builder("swa", device="npu:2")
    main_builder = make_builder("main", ratio, "npu:2") if ratio else None
    owner = V41MetadataPreparation()
    batch = owner.batch(capture=True)
    swa_common = common_from_case(device, contexts, "swa")
    main_common = common_from_case(device, contexts, "main") if ratio else None
    swa = swa_builder.build(0, swa_common, preparation=batch)
    main = main_builder.build(0, main_common, preparation=batch) if ratio else None
    batch.run()
    metadata = make_v41_attention_metadata(swa, main)
    for _ in range(3):
        attention_test.forward(ops, device, metadata)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual, _ = attention_test.forward(ops, device, metadata)
    graph.replay()
    attention_test.check(actual, attention_test.reference(case, ratio)[0])
    updated = attention_test.make_case(ratio, query_lengths, [length + 1 for length in contexts], sparse=True, seed=72)
    changed = attention_test.device_case(updated)
    for name, value in device.items():
        if value is not None:
            value.copy_(changed[name])
    swa_common.positions.add_(1)
    batch = owner.batch(use_graph=True)
    swa_builder.build(0, swa_common, preparation=batch)
    if ratio:
        main_common.positions.add_(1)
        main_builder.build(0, main_common, preparation=batch)
    batch.run()
    # The captured metadata object is not replaced: only its fixed-address
    # tensor contents changed, including scheduling and compressed lengths.
    graph.replay()
    attention_test.check(actual, attention_test.reference(updated, ratio)[0])


def test_combined_preparation_graph_refreshes_pages_boundaries_and_schedules(runtime):
    from copy import copy

    from vllm_ascend.worker.v41_metadata import V41MetadataPreparation

    owner = V41MetadataPreparation()
    roles = [("swa", 1), ("swa", 1), ("main", 2), ("main", 2), ("index", 1), ("index", 1)]
    builders = [make_builder(role, ratio, "npu:2") for role, ratio in roles]
    references = [make_builder(role, ratio, "npu:2") for role, ratio in roles]
    common = make_common("npu:2")
    inputs = [copy(common) for _ in roles]
    for index, item in enumerate(inputs):
        item.block_table_tensor = common.block_table_tensor + index * 10

    for step in range(3):
        if step:
            changed = make_common("npu:2", changed=step == 1)
            for name in ("positions", "query_start_loc", "seq_lens"):
                getattr(common, name).copy_(getattr(changed, name))
            for index, item in enumerate(inputs):
                item.block_table_tensor.copy_(changed.block_table_tensor + index * 11)
                item.block_table_tensor[0, 0] = -1
        batch = owner.batch(capture=step == 0, use_graph=True)
        actual = [builder.build(0, item, preparation=batch) for builder, item in zip(builders, inputs)]
        if step:
            # No Python metadata operations may run on a known graph binding.
            with patch.object(batch, "_refresh", side_effect=AssertionError("Unexpected eager preparation")):
                batch.run()
        else:
            batch.run()
        assert len(owner.graphs) == 1
        for reference, item, output in zip(references, inputs, actual):
            expected = reference.build(0, item)
            for field in fields(output):
                value = getattr(output, field.name)
                if isinstance(value, torch.Tensor):
                    expected_value = getattr(expected, field.name)
                    if field.name == "schedule":
                        # The native ABI defines 36 FA + 72 FD records. The
                        # remainder of the 1024-element allocation is unused.
                        size = 36 * (8 if output.role == "index" else 9) + 72 * 8
                        value, expected_value = value[:size], expected_value[:size]
                    torch.testing.assert_close(value.cpu(), expected_value.cpu())

    # A new input binding must read the new tensor, never the captured address.
    original_positions = inputs[0].positions
    assert next(iter(owner.graphs.values())).inputs[0] is original_positions
    inputs[0].positions = inputs[0].positions.clone()
    inputs[0].positions.fill_(-1)
    batch = owner.batch(use_graph=True)
    outputs = [builder.build(0, item, preparation=batch) for builder, item in zip(builders, inputs)]
    batch.run()
    assert torch.all(outputs[0].slot_mapping.cpu() == -1)
    assert len(owner.graphs) == 1

    # Graphs share a scratch pool, but every replay must recompute its own
    # intermediates. Capturing another binding must not corrupt the first one.
    batch = owner.batch(capture=True)
    for builder, item in zip(builders, inputs):
        builder.build(0, item, preparation=batch)
    batch.run()
    assert len(owner.graphs) == 2
    inputs[0].positions = original_positions
    batch = owner.batch(use_graph=True)
    outputs = [builder.build(0, item, preparation=batch) for builder, item in zip(builders, inputs)]
    with patch.object(batch, "_refresh", side_effect=AssertionError("Unexpected eager preparation")):
        batch.run()
    expected = references[0].build(0, inputs[0])
    torch.testing.assert_close(outputs[0].slot_mapping.cpu(), expected.slot_mapping.cpu())
