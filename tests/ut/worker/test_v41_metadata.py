# SPDX-License-Identifier: Apache-2.0

from copy import copy
from unittest.mock import patch

import torch

from tests.ut.attention.test_dsa_v41_metadata import make_builder, make_common, make_execution_common
from vllm_ascend.worker.v41_metadata import V41MetadataPreparation


def test_share_schedule_with_independent_pages_and_refresh_next_step():
    owner = V41MetadataPreparation()
    builders = [make_builder("main", 2), make_builder("main", 2)]
    common = make_common()
    other = copy(common)
    other.block_table_tensor = common.block_table_tensor + 10

    def schedule(**kwargs):
        return torch.full((1024,), int(kwargs["seqused_ori_kv"].sum()), dtype=torch.int32)

    with patch("torch.ops._C_ascend.npu_sparse_flash_mla_metadata", create=True, side_effect=schedule) as native:
        for step in range(2):
            batch = owner.batch()
            first = builders[0].build(0, common, preparation=batch)
            second = builders[1].build(0, other, preparation=batch)
            batch.run()
            assert native.call_count == step + 1
            torch.testing.assert_close(first.schedule, second.schedule)
            assert first.schedule[0] == common.seq_lens.sum()
            valid = first.slot_mapping >= 0
            torch.testing.assert_close(second.slot_mapping[valid], first.slot_mapping[valid] + 320)
            assert first.schedule.data_ptr() != second.schedule.data_ptr()
            common.seq_lens.add_(2)


def test_scheduling_does_not_share_different_boundaries_or_geometry():
    owner = V41MetadataPreparation()
    common = make_common()
    changed = copy(common)
    changed.query_start_loc = torch.tensor([0, 2, 6], dtype=torch.int32)
    builders = [make_builder("main", 2), make_builder("main", 2), make_builder("main", 1)]
    with patch(
        "torch.ops._C_ascend.npu_sparse_flash_mla_metadata",
        create=True,
        return_value=torch.zeros(1024, dtype=torch.int32),
    ) as native:
        batch = owner.batch()
        for builder, inputs in zip(builders, [common, changed, common]):
            builder.build(0, inputs, preparation=batch)
        batch.run()
        assert native.call_count == 3


def test_graph_key_tracks_new_inputs_and_strides_but_not_device_values():
    owner = V41MetadataPreparation()
    common = make_common()
    builder = make_builder("main", 2)

    def key(inputs):
        batch = owner.batch()
        builder.build(0, inputs, preparation=batch)
        return batch._key()

    original = key(common)
    common.seq_lens.add_(1)
    assert key(common) == original
    common.seq_lens = common.seq_lens.clone()
    assert key(common) != original
    common.seq_lens = torch.arange(4, dtype=torch.int32)
    contiguous = key(common)
    common.seq_lens = common.seq_lens[::2]
    assert key(common) != contiguous


def test_host_execution_counts_shared_only_within_current_batch():
    owner = V41MetadataPreparation()
    common = make_execution_common([1, 1], [False, False])
    builders = [make_builder("swa"), make_builder("main", 2)]
    with patch.object(builders[0], "_execution_counts", wraps=builders[0]._execution_counts) as classify:
        batch = owner.batch()
        outputs = [builder.build(0, common, preparation=batch) for builder in builders]
        assert classify.call_count == 1
        assert len(batch.counts) == 1
        assert all(output.num_decode_tokens == 2 and output.num_prefills == 0 for output in outputs)
        common.is_prefilling[0] = True
        batch = owner.batch()
        outputs = [builder.build(0, common, preparation=batch) for builder in builders]
        assert classify.call_count == 2
        assert all(output.num_decode_tokens == 1 and output.num_prefills == 1 for output in outputs)
