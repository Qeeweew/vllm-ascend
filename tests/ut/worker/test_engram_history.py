# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.engram_hash import HostEngramHasher, HostEngramLayout
from vllm_ascend.worker.engram_history import EngramRequestHistory


def make_hasher():
    config = SimpleNamespace(
        engram_layer_ids=[1, 14],
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_vocab_size=11,
        engram_num_embeddings=[1000, 1000],
        engram_head_dim=4,
    )
    return HostEngramHasher(HostEngramLayout.from_config(config), torch.arange(32) % 17, 17, 0)


def ids(values):
    return torch.tensor(values, dtype=torch.int64)


def mask(values):
    return torch.tensor(values, dtype=torch.bool)


def reference(hasher, tokens, masks):
    """Independent scalar n-gram/XOR implementation, without hash_chunk."""
    layers = []
    for layer in range(len(hasher.layout.layer_ids)):
        rows = []
        for position in range(len(tokens)):
            rolling, blocked, values, offset = 0, False, [], 0
            for shift in range(hasher.max_ngram):
                at = position - shift
                blocked |= at < 0 or not masks[at]
                token = hasher.pad_id if blocked else int(hasher.token_map[tokens[at]])
                rolling ^= token * int(hasher.multipliers[layer, shift])
                if shift:
                    for prime in hasher.layout.primes[layer][shift - 1]:
                        values.append(rolling % prime + offset)
                        offset += prime
            rows.append(values)
        layers.append(ids(rows))
    return layers


def check(batch, hasher, tokens, masks, start, stop):
    for actual, expected in zip(batch.hash_ids, reference(hasher, tokens, masks)):
        torch.testing.assert_close(actual, expected[start:stop], atol=0, rtol=0)
        assert actual.device.type == "cpu" and actual.is_contiguous()
    torch.testing.assert_close(batch.token_mask, mask(masks[start:stop]), atol=0, rtol=0)


def run(history, request, tokens, start, masks=None):
    return history.prepare(
        [request],
        ids(tokens),
        torch.arange(start, start + len(tokens)),
        [0, len(tokens)],
        token_mask=mask(masks) if masks is not None else None,
    )


def test_typed_images_follow_prompt_positions_not_ids_or_dead_masks():
    history = EngramRequestHistory(make_hasher())
    # Identical IDs at text, image and generated positions must remain distinct.
    prompt = ids([2, 2, 2, 2])
    keep = mask([True, False, False, True])
    images = mask([False, True, False, False])
    history.reset_request("image", prompt, prompt_mask=keep, prompt_image_mask=images)
    history.reset_request("text", prompt)
    images.zero_()  # reset owns its immutable prompt classification
    packed = history.prepare(
        ["text", "image"],
        ids([2, 2, 2, 2, 2]),
        ids([0, 1, 1, 2, 3]),
        [0, 2, 5],
        use_seeded_prompt_mask=True,
    )
    assert packed.image_token_mask.tolist() == [False, False, True, False, False]
    assert packed.token_mask.tolist() == [True, True, False, False, True]
    generated = run(history, "image", [2, 2], 4, [False, True])
    assert generated.image_token_mask.tolist() == [False, False]
    # Reordered/resumed prompt chunks preserve processor identity.
    resumed = history.prepare(["image"], ids([2]), ids([1]), [0, 1], use_seeded_prompt_mask=True)
    assert resumed.image_token_mask.tolist() == [True]
    history.reset_request("image", prompt)
    assert not run(history, "image", [2], 1).image_token_mask.any()


@pytest.mark.parametrize("bad_image", [mask([True, False]), ids([0, 1]), mask([False])])
def test_invalid_typed_prompt_mask_does_not_replace_history(bad_image):
    history = EngramRequestHistory(make_hasher())
    history.reset_request("a", ids([1, 2]))
    with pytest.raises(ValueError):
        history.reset_request("a", ids([3, 4]), prompt_mask=mask([True, False]), prompt_image_mask=bad_image)
    assert run(history, "a", [1, 2], 0).image_token_mask.tolist() == [False, False]


def test_cpu_history_contract_survives_non_cpu_default_device():
    hasher = make_hasher()
    prompt, positions = ids([1, 2]), ids([0, 1])
    keep, images = mask([True, False]), mask([False, True])
    with torch.device("meta"):
        history = EngramRequestHistory(hasher)
        history.reset_request("a", prompt, prompt_mask=keep, prompt_image_mask=images)
        batch = history.prepare(["a"], prompt, positions, [0, 2], use_seeded_prompt_mask=True)
    assert batch.image_token_mask.device.type == "cpu"
    assert batch.image_token_mask.tolist() == [False, True]
    check(batch, hasher, [1, 2], [True, False], 0, 2)


def test_chunks_prefix_hit_and_masked_lookback_match_independent_reference():
    hasher = make_hasher()
    history = EngramRequestHistory(hasher)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    masks = [True, True, False, True, True, True, False, True, True]
    history.reset_request("a", ids(tokens), prompt_mask=mask(masks))
    for start, end in [(0, 2), (2, 3), (3, 6), (8, 9)]:
        batch = run(history, "a", tokens[start:end], start, masks[start:end])
        check(batch, hasher, tokens, masks, start, end)
    # A prefix cache hit can skip directly into any part of the seeded prompt.
    history.reset_request("prefix", ids(tokens), prompt_mask=mask(masks))
    check(run(history, "prefix", tokens[5:8], 5, masks[5:8]), hasher, tokens, masks, 5, 8)


def test_executed_draft_rollback_overwrites_and_truncates_generated_tail():
    hasher = make_hasher()
    history = EngramRequestHistory(hasher)
    prompt = [1, 2, 3]
    history.reset_request("a", ids(prompt))
    tokens = prompt + [4, 5, 6, 7, 8]
    masks = [True] * len(tokens)
    check(run(history, "a", tokens[3:], 3), hasher, tokens, masks, 3, len(tokens))
    # Reject positions >=5, replace two executed inputs, and cut the old tail.
    tokens = tokens[:5] + [11, 12]
    masks = [True] * len(tokens)
    check(run(history, "a", [11, 12], 5), hasher, tokens, masks, 5, 7)
    with pytest.raises(ValueError, match="Missing actual"):
        run(history, "a", [20], 8)
    tokens += [13]
    masks += [True]
    check(run(history, "a", [13], 7), hasher, tokens, masks, 7, 8)


@pytest.mark.parametrize("accepted", range(6))
def test_dspark_k5_acceptance_reorders_requests_and_hashes_corrected_zero(accepted):
    hasher = make_hasher()
    history = EngramRequestHistory(hasher)
    prompts = {"a": [1, 2, 3], "b": [11, 12, 13, 14]}
    old_queries = {"a": [4, 5, 6, 7, 8, 9], "b": [15, 16, 17, 18, 19, 20]}
    new_queries = {"a": [0, 21, 22, 23, 24, 25], "b": [0, 26, 27, 28, 29, 30]}
    for request, prompt in prompts.items():
        history.reset_request(request, ids(prompt))
        # The target verifies the anchor plus all five proposed tokens.
        run(history, request, old_queries[request], len(prompt))
    order = ["b", "a"]
    starts = {request: len(prompts[request]) + 1 + accepted for request in order}
    # After verification, the next real input is the corrected/bonus token.
    # Token zero is valid even when the scheduler had an unresolved placeholder.
    actual = history.prepare(
        order,
        ids(new_queries["b"] + new_queries["a"]),
        ids([position for request in order for position in range(starts[request], starts[request] + 6)]),
        [0, 6, 12],
    )
    for layer, values in enumerate(actual.hash_ids):
        expected = []
        for request in order:
            committed = prompts[request] + old_queries[request][: 1 + accepted] + new_queries[request]
            expected.append(reference(hasher, committed, [True] * len(committed))[layer][-6:])
        torch.testing.assert_close(values, torch.cat(expected), rtol=0, atol=0)
    assert actual.token_mask.all() and not actual.image_token_mask.any()
    for request in order:
        committed = prompts[request] + old_queries[request][: 1 + accepted] + new_queries[request] + [0]
        check(
            run(history, request, [0], starts[request] + 6),
            hasher,
            committed,
            [True] * len(committed),
            len(committed) - 1,
            len(committed),
        )


def test_rollback_keeps_complete_prompt_and_corrects_generated_masks():
    hasher = make_hasher()
    history = EngramRequestHistory(hasher)
    prompt = [1, 2, 3, 4, 5, 6, 7]
    history.reset_request("a", ids(prompt))
    run(history, "a", [8, 9, 10], 7)
    run(history, "a", [2], 1)
    # Rollback removes generated tail, but untouched prompt suffix is retained.
    check(run(history, "a", [7], 6), hasher, prompt, [True] * 7, 6, 7)
    with pytest.raises(ValueError, match="Missing actual"):
        run(history, "a", [11], 9)
    run(history, "a", [8, 9], 7, [True, True])
    run(history, "a", [9], 8, [False])
    tokens, masks = prompt + [8, 9, 10], [True] * 8 + [False, True]
    check(run(history, "a", [10], 9), hasher, tokens, masks, 9, 10)


def test_preemption_restore_reordering_finished_and_explicit_reuse():
    hasher = make_hasher()
    history = EngramRequestHistory(hasher)
    a, b = [1, 2, 3], [11, 12, 13, 14]
    history.reset_request("a", ids(a))
    history.reset_request("b", ids(b))
    run(history, "a", [4, 5], 3)
    a += [4, 5]
    # Preemption removes a from the scheduled batch, not from history.
    run(history, "b", [15], 4)
    b += [15]
    batch = history.prepare(["b", "a"], ids([16, 6]), ids([5, 5]), torch.tensor([0, 1, 2], dtype=torch.int32))
    a += [6]
    b += [16]
    for layer, actual in enumerate(batch.hash_ids):
        expected = torch.cat((reference(hasher, b, [True] * 6)[layer][5:], reference(hasher, a, [True] * 6)[layer][5:]))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Explicit restoration after losing the host object uses actual tail only.
    restored = EngramRequestHistory(hasher)
    restored.reset_request("a", ids(a[:3]), executed_tail=ids(a[3:]))
    check(run(restored, "a", [7], 6), hasher, a + [7], [True] * 7, 6, 7)
    history.drop_request("a")
    history.drop_request("a")
    assert "a" not in history and "b" in history
    with pytest.raises(ValueError, match="Missing Engram history"):
        run(history, "a", [1], 0)
    history.reset_request("a", ids([21, 22]))
    check(run(history, "a", [23], 2), hasher, [21, 22, 23], [True] * 3, 2, 3)


def test_masked_actual_tokens_are_required_and_prompt_masks_are_immutable():
    hasher = make_hasher()
    history = EngramRequestHistory(hasher)
    history.reset_request("a", ids([1, 2]), prompt_mask=mask([True, False]))
    with pytest.raises(ValueError, match="prompt tokens/masks"):
        run(history, "a", [2], 1)  # Omitting mask would incorrectly unblock it.
    with pytest.raises(ValueError, match="placeholders"):
        run(history, "a", [-1], 1, [False])
    with pytest.raises(ValueError, match="prompt tokens/masks"):
        run(history, "a", [3], 1, [False])
    check(run(history, "a", [3], 2), hasher, [1, 2, 3], [True, False, True], 2, 3)


@pytest.mark.parametrize("start,stop", [(0, 3), (2, 5), (4, 7)])
def test_seeded_prompt_mask_uses_actual_positions_and_keeps_generated_tokens_live(start, stop):
    hasher = make_hasher()
    history = EngramRequestHistory(hasher)
    # The same raw ID is masked in the prompt and live after it. The token ID
    # itself must never decide whether a row belongs to an image.
    prompt = [1, 2, 2, 2, 3]
    prompt_masks = [True, False, False, False, True]
    tokens, masks = prompt + [2, 4], prompt_masks + [True, True]
    history.reset_request("a", ids(prompt), prompt_mask=mask(prompt_masks))
    batch = history.prepare(
        ["a"], ids(tokens[start:stop]), torch.arange(start, stop), [0, stop - start], use_seeded_prompt_mask=True
    )
    check(batch, hasher, tokens, masks, start, stop)


def test_seeded_prompt_masks_follow_reordered_requests_and_empty_rows():
    hasher = make_hasher()
    history = EngramRequestHistory(hasher)
    a, am = [1, 2, 3, 4], [True, False, False, True]
    b, bm = [11, 12, 13], [True, True, True]
    history.reset_request("a", ids(a), prompt_mask=mask(am))
    history.reset_request("b", ids(b))
    history.reset_request("empty", ids([20]))
    batch = history.prepare(
        ["b", "empty", "a"], ids([12, 13, 2, 3]), ids([1, 2, 1, 2]), [0, 2, 2, 4], use_seeded_prompt_mask=True
    )
    for actual, expected_b, expected_a in zip(batch.hash_ids, reference(hasher, b, bm), reference(hasher, a, am)):
        torch.testing.assert_close(actual, torch.cat((expected_b[1:3], expected_a[1:3])), rtol=0, atol=0)
    assert batch.token_mask.tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match="combined"):
        history.prepare(["a"], ids([2]), ids([1]), [0, 1], token_mask=mask([False]), use_seeded_prompt_mask=True)
    with pytest.raises(ValueError, match="prompt tokens/masks"):
        history.prepare(["a"], ids([9]), ids([1]), [0, 1], use_seeded_prompt_mask=True)


def test_failed_batch_does_not_partially_commit_or_truncate_history(monkeypatch):
    hasher = make_hasher()
    history = EngramRequestHistory(hasher)
    history.reset_request("a", ids([1, 2]), executed_tail=ids([3, 4]))
    history.reset_request("b", ids([11, 12]))
    with pytest.raises(ValueError, match="Missing actual"):
        history.prepare(["a", "b"], ids([20, 21]), ids([2, 9]), [0, 1, 2])
    check(run(history, "a", [5], 4), hasher, [1, 2, 3, 4, 5], [True] * 5, 4, 5)
    original = hasher.hash_chunk

    def fail_hash(*args, **kwargs):
        raise RuntimeError("hash failure")

    monkeypatch.setattr(hasher, "hash_chunk", fail_hash)
    with pytest.raises(RuntimeError, match="hash failure"):
        run(history, "a", [22], 2)
    monkeypatch.setattr(hasher, "hash_chunk", original)
    check(run(history, "a", [6], 5), hasher, [1, 2, 3, 4, 5, 6], [True] * 6, 5, 6)
    with pytest.raises(ValueError, match="placeholders"):
        history.reset_request("a", ids([11]), executed_tail=ids([-1]))
    check(run(history, "a", [7], 6), hasher, [1, 2, 3, 4, 5, 6, 7], [True] * 7, 6, 7)


def test_seed_and_batch_are_independent_of_caller_buffer_reuse():
    hasher = make_hasher()
    history = EngramRequestHistory(hasher)
    prompt, prompt_mask = ids([1, 2, 3]), mask([True, False, True])
    history.reset_request("a", prompt, prompt_mask=prompt_mask)
    prompt.fill_(9)
    prompt_mask.fill_(True)
    actual, actual_mask = ids([4]), mask([True])
    batch = history.prepare(["a"], actual, ids([3]), [0, 1], token_mask=actual_mask)
    actual.fill_(10)
    actual_mask.fill_(False)
    assert batch.token_mask.tolist() == [True]
    check(run(history, "a", [5], 4), hasher, [1, 2, 3, 4, 5], [True, False, True, True, True], 4, 5)


def test_empty_batches_do_not_erase_history():
    hasher = make_hasher()
    history = EngramRequestHistory(hasher)
    history.reset_request("a", ids([1, 2]))
    history.reset_request("b", ids([]))
    for requests, boundaries in [([], [0]), (["a", "b"], [0, 0, 0])]:
        result = history.prepare(requests, ids([]), ids([]), boundaries)
        assert result.token_mask.shape == (0,)
        assert all(layer.shape == (0, 6) for layer in result.hash_ids)
    check(run(history, "a", [3], 2), hasher, [1, 2, 3], [True] * 3, 2, 3)


@pytest.mark.parametrize(
    "positions,boundaries,requests",
    [
        ([0, 2], [0, 2], ["a"]),
        ([-1, 0], [0, 2], ["a"]),
        ([0, 1], [0, 1], ["a"]),
        ([0, 1], [1, 2], ["a"]),
        ([0, 1], [0, 1, 2], ["a", "a"]),
        ([0, 1], [0, -1, 2], ["a", "b"]),
    ],
)
def test_invalid_positions_boundaries_and_duplicate_requests(positions, boundaries, requests):
    history = EngramRequestHistory(make_hasher())
    history.reset_request("a", ids([1, 2]))
    history.reset_request("b", ids([1, 2]))
    with pytest.raises(ValueError):
        history.prepare(requests, ids([1, 2]), ids(positions), boundaries)


def test_unknown_generation_prefix_and_tensor_contract_fail_closed():
    history = EngramRequestHistory(make_hasher())
    history.reset_request("a", ids([1, 2]))
    with pytest.raises(ValueError, match="Missing actual"):
        run(history, "a", [4], 3)
    for invalid in (ids([-1]), ids([32]), torch.tensor([1], dtype=torch.int32)):
        with pytest.raises(ValueError):
            history.reset_request("bad", invalid)
        with pytest.raises(ValueError):
            history.prepare(["a"], invalid, ids([2]), [0, 1])
    for invalid in (torch.tensor([0.0, 1.0]), torch.tensor([[0, 1]]), [0, 0.5]):
        with pytest.raises(ValueError, match="boundaries"):
            history.prepare(["a"], ids([3]), ids([2]), invalid)
    with pytest.raises(ValueError, match="positions"):
        history.prepare(["a"], ids([3]), torch.tensor([2], dtype=torch.int32), [0, 1])
    with pytest.raises(ValueError, match="masks"):
        history.prepare(["a"], ids([3]), ids([2]), [0, 1], token_mask=torch.ones(1))


def test_explicit_reset_replaces_live_id_and_restores_masked_tail():
    hasher = make_hasher()
    history = EngramRequestHistory(hasher)
    history.reset_request("a", ids([1, 2]), executed_tail=ids([3, 4, 5]))
    history.reset_request("a", ids([11, 12]), executed_tail=ids([13]), tail_mask=mask([False]))
    with pytest.raises(ValueError, match="Missing actual"):
        run(history, "a", [15], 4)
    check(run(history, "a", [14], 3), hasher, [11, 12, 13, 14], [True, True, False, True], 3, 4)


def test_non_cpu_inputs_are_rejected_without_device_access():
    history = EngramRequestHistory(make_hasher())
    history.reset_request("a", ids([1]))
    with pytest.raises(ValueError, match="CPU"):
        history.prepare(["a"], torch.empty(1, dtype=torch.int64, device="meta"), ids([1]), [0, 1])
    with pytest.raises(ValueError, match="positions"):
        history.prepare(["a"], ids([2]), torch.empty(1, dtype=torch.int64, device="meta"), [0, 1])
