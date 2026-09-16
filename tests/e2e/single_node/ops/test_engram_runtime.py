# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Final device token snapshots -> request history -> host rows -> replay."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.engram_hash import HostEngramHasher, HostEngramLayout
from vllm_ascend.ops.engram_offload import EngramOffloadManager, EngramTableShard
from vllm_ascend.worker.engram_history import EngramRequestHistory
from vllm_ascend.worker.engram_runtime import EngramRuntime


@torch.inference_mode()
@pytest.mark.parametrize("seeded_prompt", [False, True])
@pytest.mark.parametrize("split_snapshot", [False, True])
def test_final_device_inputs_prefix_rollback_mask_and_graph(seeded_prompt, split_snapshot):
    config = SimpleNamespace(
        engram_layer_ids=[1, 14],
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_vocab_size=11,
        engram_num_embeddings=[1000, 1000],
        engram_head_dim=256,
    )
    layout = HostEngramLayout.from_config(config)
    hasher = HostEngramHasher(layout, torch.arange(32), 32, 0)
    history = EngramRequestHistory(hasher)
    prompt = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
    prompt_mask = torch.tensor([True, True, False, True, True, True])
    history.reset_request("request", prompt, prompt_mask=prompt_mask, prompt_image_mask=~prompt_mask)
    shards, tables = [], []
    for layer in range(2):
        heads, ranges = layout.head_shard(layer, 0, 1)
        table = torch.arange(ranges[-1][1] * 256).reshape(-1, 256).remainder(127).bfloat16()
        pinned = torch.empty_like(table, pin_memory=True).copy_(table)
        shards.append(EngramTableShard(pinned, heads, ranges))
        tables.append(table)
    runtime = EngramRuntime(history, EngramOffloadManager(shards, 8, torch.device("npu")))
    ptrs = tuple(row.data_ptr() for row in runtime.offload.device_rows) + (
        runtime.token_mask.data_ptr(),
        runtime.image_token_mask.data_ptr(),
    )
    device_ids = torch.full((8,), -1, dtype=torch.int64, device="npu")
    positions = torch.zeros_like(device_ids)
    mask = torch.zeros(8, dtype=torch.bool, device="npu")
    boundaries = torch.tensor([0, 1], dtype=torch.int32, device="npu")

    def stage(tokens, start, masks):
        # Simulate final device correction after the host input buffer still
        # contained placeholders; runtime must snapshot these corrected IDs.
        device_ids.fill_(-1)
        device_ids[: len(tokens)].copy_(torch.tensor(tokens))
        positions[: len(tokens)].copy_(torch.arange(start, start + len(tokens)))
        mask[: len(tokens)].copy_(torch.tensor(masks))
        boundaries[1] = len(tokens)
        step_mask = None if seeded_prompt and start < prompt.numel() else mask
        if not split_snapshot:
            return runtime.prepare(["request"], device_ids, positions, boundaries, 8, token_mask=step_mask)
        requests = ["request"]
        runtime.begin_prepare(requests, device_ids, positions, boundaries, 8, token_mask=step_mask)
        assert runtime.snapshot_pending
        with pytest.raises(RuntimeError, match="consumed"):
            runtime.begin_prepare(requests, device_ids, positions, boundaries, 8)
        # Subsequent same-stream work can overwrite producer buffers: the
        # snapshot must retain final values at begin_prepare, including masks.
        requests[0] = "not-the-snapshotted-request"
        device_ids.fill_(31)
        positions.zero_()
        mask.zero_()
        boundaries.zero_()
        result = runtime.finish_prepare()
        assert not runtime.snapshot_pending
        return result

    rows, live_mask = stage([1], 0, [True])
    runtime.wait_ready()

    def run():
        return torch.where(live_mask[:, None, None], rows[0] + rows[1], 0), runtime.image_token_mask.clone()

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    runtime.mark_consumed()
    known, known_mask = prompt.tolist(), prompt_mask.tolist()
    snapshots, expected, image_snapshots = [], [], []
    for tokens, start, masks in [
        ([2, 3], 1, [True, False]),  # packed image row uses the seeded prompt mask
        ([4, 5, 6], 3, [True] * 3),  # prefix hit across a masked token
        ([7, 8, 9], 6, [True] * 3),  # actual executed speculative inputs
        ([10], 7, [True]),  # reject former positions 7 and 8
        ([11, 12], 8, [False, True]),
    ]:
        stage(tokens, start, masks)
        with pytest.raises(RuntimeError, match="consumed"):
            runtime.prepare(["request"], device_ids, positions, boundaries, 8, token_mask=mask)
        runtime.wait_ready()
        graph.replay()
        snapshots.append(output[0].clone())
        image_snapshots.append(output[1].clone())
        runtime.mark_consumed()
        known[start:] = tokens
        known_mask[start:] = masks
        hashes = hasher.hash_chunk(
            torch.tensor(known),
            [0, len(known)],
            [0],
            torch.full((1, 3), -1, dtype=torch.int64),
            token_mask=torch.tensor(known_mask),
        )
        want = torch.zeros((8, 6, 256), dtype=torch.bfloat16)
        want[: len(tokens)] = tables[0][hashes[0][start:]] + tables[1][hashes[1][start:]]
        want[: len(tokens)].masked_fill_(~torch.tensor(masks)[:, None, None], 0)
        expected.append(want)
        assert (
            tuple(row.data_ptr() for row in runtime.offload.device_rows)
            + (
                runtime.token_mask.data_ptr(),
                runtime.image_token_mask.data_ptr(),
            )
            == ptrs
        )
    runtime.close()
    for actual, want in zip(snapshots, expected):
        torch.testing.assert_close(actual.cpu(), want, rtol=0, atol=0)
    # Only the processor-owned prompt position is image, including after
    # replay switches to generated DEAD tokens and padded buckets.
    assert image_snapshots[0].cpu().tolist() == [False, True, False, False, False, False, False, False]
    for actual in image_snapshots[1:]:
        assert not actual.cpu().any()


@torch.inference_mode()
def test_split_snapshot_multibatch_reorder_and_rejection():
    config = SimpleNamespace(
        engram_layer_ids=[1],
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_vocab_size=11,
        engram_num_embeddings=[1000],
        engram_head_dim=256,
    )
    layout = HostEngramLayout.from_config(config)
    hasher = HostEngramHasher(layout, torch.arange(32), 32, 0)
    heads, ranges = layout.head_shard(0, 0, 1)
    table = torch.arange(ranges[-1][1] * 256).reshape(-1, 256).remainder(127).bfloat16()
    shard = EngramTableShard(torch.empty_like(table, pin_memory=True).copy_(table), heads, ranges)
    runtimes = [
        EngramRuntime(EngramRequestHistory(hasher), EngramOffloadManager([shard], 8, torch.device("npu")))
        for _ in range(2)
    ]
    for runtime in runtimes:
        for request in ("a", "b", "c"):
            runtime.history.reset_request(request, torch.tensor([1, 2, 3, 4, 5, 6]))
    # Different query counts, request reordering/removal, and rejection of a
    # previously executed speculative suffix. Remaining rows are graph padding.
    cases = [
        (["a", "b", "c"], [[7, 8], [9], [10, 11, 12]], [6, 6, 6]),
        (["c", "a"], [[13], [14, 15]], [7, 7]),
        (["a", "c"], [[16], [17, 18]], [9, 8]),
    ]
    for requests, chunks, starts in cases:
        ids = torch.full((8,), -1, dtype=torch.int64, device="npu")
        positions = torch.full_like(ids, -1)
        flat = [token for chunk in chunks for token in chunk]
        ids[: len(flat)].copy_(torch.tensor(flat))
        positions[: len(flat)].copy_(
            torch.tensor([p for chunk, start in zip(chunks, starts) for p in range(start, start + len(chunk))])
        )
        boundaries = torch.tensor(
            [0, *torch.tensor([len(c) for c in chunks]).cumsum(0).tolist()], dtype=torch.int32, device="npu"
        )
        expected, expected_mask = runtimes[0].prepare(requests, ids, positions, boundaries, 8)
        runtimes[1].begin_prepare(requests, ids, positions, boundaries, 8)
        ids.fill_(31)
        positions.zero_()
        boundaries.zero_()
        actual, actual_mask = runtimes[1].finish_prepare()
        for runtime in runtimes:
            runtime.wait_ready()
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual_mask, expected_mask)
        for runtime in runtimes:
            runtime.mark_consumed()
    for runtime in runtimes:
        runtime.close()
