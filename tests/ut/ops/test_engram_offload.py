# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.ops.engram_offload import EngramTableShard


def make_shard():
    # Rank owns heads 0 and 2; head 1's bucket is never stored locally.
    full = torch.arange(13 * 4).reshape(13, 4).bfloat16()
    return full, EngramTableShard(torch.cat((full[:3], full[8:])), [0, 2], [(0, 3), (8, 13)])


def test_head_sharding_global_offsets_and_dead_rows():
    full, shard = make_shard()
    ids = torch.tensor([[2, 7, 12], [-1, 4, 8], [0, 5, -1]])
    out = torch.full((3, 2, 4), 99, dtype=torch.bfloat16)
    shard.gather_into(ids, out)
    expected = torch.stack(
        (full[[2, 12]], torch.stack((torch.zeros(4), full[8])), torch.stack((full[0], torch.zeros(4))))
    ).bfloat16()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    # A zero-token step must be legal for padded graph invocations.
    shard.gather_into(ids[:0], out[:0])


@pytest.mark.parametrize("ids", [[[3, 4, 8]], [[0, 4, 7]], [[-2, 4, 8]]])
def test_wrong_bucket_rejected(ids):
    _, shard = make_shard()
    with pytest.raises(ValueError, match="bucket range"):
        shard.gather_into(torch.tensor(ids), torch.empty((1, 2, 4), dtype=torch.bfloat16))


@pytest.mark.parametrize("indices,ranges", [([0, 0], [(0, 3), (8, 13)]), ([0, 2], [(0, 3), (8, 12)]), ([0], [(-1, 7)])])
def test_invalid_shard_storage(indices, ranges):
    with pytest.raises(ValueError):
        EngramTableShard(torch.zeros((8, 4), dtype=torch.bfloat16), indices, ranges)


def test_streaming_load_only_selected_heads(tmp_path):
    full, _ = make_shard()
    path = tmp_path / "table.safetensors"
    save_file({"embedding": full}, path)
    shard = EngramTableShard.from_safetensors(
        path, "embedding", [0, 2], [(0, 3), (8, 13)], pin_memory=False, rows_per_copy=2
    )
    assert shard.weight.shape == (8, 4)
    torch.testing.assert_close(shard.weight, torch.cat((full[:3], full[8:])), rtol=0, atol=0)
    with pytest.raises(ValueError, match="exceeds"):
        EngramTableShard.from_safetensors(path, "embedding", [0], [(0, 14)], pin_memory=False)
    save_file({"embedding": full.float()}, path)
    with pytest.raises(ValueError, match="BF16"):
        EngramTableShard.from_safetensors(path, "embedding", [0], [(0, 3)], pin_memory=False)
