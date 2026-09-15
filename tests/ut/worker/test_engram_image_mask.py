# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange

from vllm_ascend.ops.engram_hash import HostEngramHasher, HostEngramLayout
from vllm_ascend.worker.engram_history import EngramRequestHistory
from vllm_ascend.worker.engram_image_mask import (
    V41_IMAGE_TOKEN_ID,
    V41EngramImageSpans,
    pack_v41_engram_token_mask,
)


def ids(values):
    return torch.tensor(values, dtype=torch.int64)


def image_prompt():
    tokens = ids([7] + [V41_IMAGE_TOKEN_ID] * 10 + [8, 9, V41_IMAGE_TOKEN_ID, 129265])
    ranges = [PlaceholderRange(1, 4), PlaceholderRange(5, 6)]
    roles = [ids([0, 1, 2, 3]), ids([0, 1, 1, 1, 2, 3])]
    metadata = V41EngramImageSpans.from_prompt(tokens, ranges, image_roles=roles)
    return tokens, metadata


def hasher():
    config = SimpleNamespace(
        engram_layer_ids=[1, 14],
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_vocab_size=11,
        engram_num_embeddings=[1000, 1000],
        engram_head_dim=4,
    )
    return HostEngramHasher(HostEngramLayout.from_config(config), torch.arange(129270) % 17, 17, 0)


def scalar_hashes(engine, tokens, keep):
    layers = []
    for layer in range(len(engine.layout.layer_ids)):
        rows = []
        for position in range(len(tokens)):
            rolling, blocked, offset, row = 0, False, 0, []
            for shift in range(engine.max_ngram):
                at = position - shift
                blocked |= at < 0 or not keep[at]
                token = engine.pad_id if blocked else int(engine.token_map[tokens[at]])
                rolling ^= token * int(engine.multipliers[layer, shift])
                if shift:
                    for prime in engine.layout.primes[layer][shift - 1]:
                        row.append(rolling % prime + offset)
                        offset += prime
            rows.append(row)
        layers.append(ids(rows))
    return layers


def test_delimiters_adjacent_images_and_literal_sentinel_remain_distinct():
    tokens, metadata = image_prompt()
    assert metadata.image_spans == ((1, 5), (5, 11))
    expected = [True] + [False] * 10 + [True] * 4
    assert metadata.prompt_keep_mask().tolist() == expected
    assert (
        pack_v41_engram_token_mask(
            ["a"], torch.arange(len(tokens)), [0, len(tokens)], {"a": metadata}, input_ids=tokens
        ).tolist()
        == expected
    )
    assert metadata.prompt_keep_mask()[13]  # Literal image ID outside image metadata.


def test_request_adapter_works_with_cached_features_and_roles():
    tokens, expected = image_prompt()
    request = SimpleNamespace(
        prompt_token_ids=tokens.tolist(),
        mm_features=[
            MultiModalFeatureSpec(
                data=None, modality="image", identifier="cached-a", mm_position=PlaceholderRange(1, 4)
            ),
            MultiModalFeatureSpec(
                data={"types": SimpleNamespace(data=ids([0, 1, 1, 1, 2, 3]))},
                modality="image",
                identifier="new-b",
                mm_position=PlaceholderRange(5, 6),
            ),
        ],
    )
    assert V41EngramImageSpans.from_request(request) == expected


def test_cpu_contract_survives_non_cpu_default_device():
    tokens, expected = image_prompt()
    positions = torch.arange(tokens.numel() + 2)
    request = SimpleNamespace(
        prompt_token_ids=tokens.tolist(),
        mm_features=[
            MultiModalFeatureSpec(
                data=None, modality="image", identifier=str(i), mm_position=PlaceholderRange(a, b - a)
            )
            for i, (a, b) in enumerate(expected.image_spans)
        ],
    )
    with torch.device("meta"):
        metadata = V41EngramImageSpans.from_request(request)
        keep = metadata.prompt_keep_mask()
        packed = pack_v41_engram_token_mask(["a"], positions, [0, tokens.numel()], {"a": metadata})
    assert metadata == expected
    assert keep.device.type == packed.device.type == "cpu"
    assert packed.tolist() == keep.tolist() + [False, False]


@pytest.mark.parametrize("start,end", [(0, 1), (1, 3), (3, 6), (5, 11), (10, 12), (11, 15), (14, 15)])
def test_prefix_and_chunk_masks_preserve_exact_dead_lookback(start, end):
    tokens, metadata = image_prompt()
    engine = hasher()
    history = EngramRequestHistory(engine)
    keep = metadata.prompt_keep_mask()
    history.reset_request("a", tokens, prompt_mask=keep)
    packed = pack_v41_engram_token_mask(
        ["a"], torch.arange(start, end), [0, end - start], {"a": metadata}, input_ids=tokens[start:end]
    )
    batch = history.prepare(["a"], tokens[start:end], torch.arange(start, end), [0, end - start], token_mask=packed)
    for actual, expected in zip(batch.hash_ids, scalar_hashes(engine, tokens, keep)):
        torch.testing.assert_close(actual, expected[start:end], rtol=0, atol=0)
    torch.testing.assert_close(batch.token_mask, keep[start:end], rtol=0, atol=0)


def test_multibatch_reorder_empty_request_padding_and_stable_output():
    tokens, image_metadata = image_prompt()
    text = ids([1, 2, V41_IMAGE_TOKEN_ID, 129268])
    spans = {
        "a": image_metadata,
        "b": V41EngramImageSpans.from_prompt(text, []),
        "empty": V41EngramImageSpans.from_prompt(ids([]), []),
    }
    out = torch.ones(8, dtype=torch.bool)
    pointer = out.data_ptr()
    positions = ids([2, 3, 10, 11, 12, -1, -1, -1])
    inputs = torch.cat((text[2:4], tokens[10:13], ids([-1, -1, -1])))
    result = pack_v41_engram_token_mask(["b", "a", "empty"], positions, [0, 2, 5, 5], spans, input_ids=inputs, out=out)
    assert result.data_ptr() == pointer
    assert result.tolist() == [True, True, False, True, True, False, False, False]
    pack_v41_engram_token_mask([], ids([-1] * 8), [0], spans, out=out)
    assert not out.any() and out.data_ptr() == pointer


def test_generated_sentinel_is_text_and_image_boundary_stays_dead():
    tokens = ids([7] + [V41_IMAGE_TOKEN_ID] * 4)
    metadata = V41EngramImageSpans.from_prompt(tokens, [PlaceholderRange(1, 4)])
    engine = hasher()
    history = EngramRequestHistory(engine)
    history.reset_request("a", tokens, prompt_mask=metadata.prompt_keep_mask())
    tail = ids([V41_IMAGE_TOKEN_ID, 129265, 129268, 8, 9])
    all_tokens = torch.cat((tokens, tail))
    keep = torch.cat((metadata.prompt_keep_mask(), torch.ones(len(tail), dtype=torch.bool)))
    for position in range(5, 10):
        mask = pack_v41_engram_token_mask(
            ["a"], ids([position]), [0, 1], {"a": metadata}, input_ids=all_tokens[position : position + 1]
        )
        assert mask.item()
        batch = history.prepare(["a"], all_tokens[position : position + 1], ids([position]), [0, 1], token_mask=mask)
        for actual, expected in zip(batch.hash_ids, scalar_hashes(engine, all_tokens, keep)):
            torch.testing.assert_close(actual, expected[position : position + 1], rtol=0, atol=0)


def test_request_finish_reuse_and_preemption_have_no_hidden_mask_state():
    tokens, metadata = image_prompt()
    spans = {"a": metadata}
    assert not pack_v41_engram_token_mask(["a"], ids([3]), [0, 1], spans).item()
    # Preemption retains the same request-owned immutable ranges.
    assert not pack_v41_engram_token_mask(["a"], ids([3]), [0, 1], spans).item()
    del spans["a"]
    with pytest.raises(ValueError, match="missing image-span"):
        pack_v41_engram_token_mask(["a"], ids([3]), [0, 1], spans)
    spans["a"] = V41EngramImageSpans.from_prompt(tokens, [])
    assert pack_v41_engram_token_mask(["a"], ids([3]), [0, 1], spans).item()


@pytest.mark.parametrize(
    "ranges",
    [
        [PlaceholderRange(-1, 4)],
        [PlaceholderRange(13, 4)],
        [PlaceholderRange(1, 0)],
        [PlaceholderRange(1, 4), PlaceholderRange(4, 4)],
        [PlaceholderRange(1.5, 4)],
    ],
)
def test_invalid_ranges_fail_closed(ranges):
    tokens, _ = image_prompt()
    with pytest.raises(ValueError):
        V41EngramImageSpans.from_prompt(tokens, ranges)


@pytest.mark.parametrize("roles", [[0, 1, 2, 0], [0, 1, 1, 3], [0, 2, 1, 3], [0, 1, 2]])
def test_invalid_roles_fail_closed(roles):
    with pytest.raises(ValueError):
        V41EngramImageSpans.from_prompt(
            ids([V41_IMAGE_TOKEN_ID] * 4), [PlaceholderRange(0, 4)], image_roles=[ids(roles)]
        )


def test_partial_delimiter_embeddings_and_wrong_image_ids_rejected():
    tokens = ids([V41_IMAGE_TOKEN_ID] * 4)
    with pytest.raises(ValueError, match="including delimiters"):
        V41EngramImageSpans.from_prompt(
            tokens, [PlaceholderRange(0, 4, is_embed=torch.tensor([False, True, True, False]))]
        )
    tokens[0] = 129265
    with pytest.raises(ValueError, match="raw token"):
        V41EngramImageSpans.from_prompt(tokens, [PlaceholderRange(0, 4)])


def test_final_correction_validation_is_transactional_for_output():
    _, metadata = image_prompt()
    out = torch.ones(3, dtype=torch.bool)
    with pytest.raises(ValueError, match="final image token"):
        pack_v41_engram_token_mask(
            ["a"], ids([1, 2, -1]), [0, 2], {"a": metadata}, input_ids=ids([V41_IMAGE_TOKEN_ID, 0, -1]), out=out
        )
    assert out.all()


@pytest.mark.parametrize("positions,bounds", [([1, 3], [0, 2]), ([-1, 0], [0, 2]), ([0, 1], [0, 3])])
def test_bad_final_positions_or_boundaries_fail_closed(positions, bounds):
    _, metadata = image_prompt()
    with pytest.raises(ValueError):
        pack_v41_engram_token_mask(["a"], ids(positions), bounds, {"a": metadata})
