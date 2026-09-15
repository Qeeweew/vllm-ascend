# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.engram_hash import HostEngramHasher, HostEngramLayout, compressed_token_map


def make_hasher():
    config = SimpleNamespace(
        engram_layer_ids=[1, 14],
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_vocab_size=11,
        engram_num_embeddings=[1000, 1000],
        engram_head_dim=4,
    )
    layout = HostEngramLayout.from_config(config)
    return HostEngramHasher(layout, torch.tensor([0, 1, 2, 3, 1, 2]), 4, 0)


def scalar_reference(hasher, ids, mask):
    outputs = []
    for layer in range(len(hasher.layout.layer_ids)):
        rows = []
        for position in range(len(ids)):
            rolling, blocked, values = 0, False, []
            offset = 0
            for shift in range(hasher.max_ngram):
                p = position - shift
                blocked |= p < 0 or not mask[p]
                token = hasher.pad_id if blocked else int(hasher.token_map[ids[p]])
                rolling ^= token * int(hasher.multipliers[layer, shift])
                if shift:
                    for prime in hasher.layout.primes[layer][shift - 1]:
                        values.append(rolling % prime + offset)
                        offset += prime
            rows.append(values)
        outputs.append(torch.tensor(rows))
    return outputs


def test_chunk_prefix_and_rollback_match_scalar_hash():
    hasher = make_hasher()
    ids = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4])
    mask = torch.tensor([True, True, False, True, True, True, True, True, True])
    expected = scalar_reference(hasher, ids, mask)
    for first, end in [(0, 9), (0, 2), (2, 6), (6, 9), (4, 7), (8, 9)]:
        prior = torch.tensor([[int(ids[p]) if p >= 0 else -1 for p in range(first - 1, first - 4, -1)]])
        prior_mask = torch.tensor([[bool(mask[p]) if p >= 0 else False for p in range(first - 1, first - 4, -1)]])
        result = hasher.hash_chunk(
            ids[first:end], [0, end - first], [first], prior, token_mask=mask[first:end], lookback_mask=prior_mask
        )
        for actual, want in zip(result, expected):
            torch.testing.assert_close(actual, want[first:end], rtol=0, atol=0)
    # Rejected draft token is replaced; hashes depend only on corrected history.
    ids[5] = 4
    result = hasher.hash_chunk(
        ids[5:8], [0, 3], [5], ids[2:5].flip(0)[None], token_mask=mask[5:8], lookback_mask=mask[2:5].flip(0)[None]
    )
    for actual, want in zip(result, scalar_reference(hasher, ids, mask)):
        torch.testing.assert_close(actual, want[5:8], rtol=0, atol=0)


def test_request_boundaries_reordering_and_empty_requests():
    hasher = make_hasher()
    first, second = torch.tensor([1, 2, 3]), torch.tensor([4, 5])
    result = hasher.hash_chunk(
        torch.cat((second, first)), [0, 2, 2, 5], [0, 77, 0], torch.full((3, 3), -1, dtype=torch.int64)
    )
    a, b = scalar_reference(hasher, first, [True] * 3), scalar_reference(hasher, second, [True] * 2)
    for actual, first_hash, second_hash in zip(result, a, b):
        torch.testing.assert_close(actual, torch.cat((second_hash, first_hash)), rtol=0, atol=0)
    empty = hasher.hash_chunk(first[:0], [0], [], torch.empty((0, 3), dtype=torch.int64))
    assert all(x.shape == (0, 6) for x in empty)


def test_unknown_history_fails_instead_of_hashing_async_placeholder():
    hasher = make_hasher()
    with pytest.raises(ValueError, match="Actual lookback"):
        hasher.hash_chunk(torch.tensor([1]), [0, 1], [3], torch.tensor([[2, -1, 0]]))
    with pytest.raises(ValueError, match="boundaries"):
        hasher.hash_chunk(torch.tensor([1]), [0, 2], [0], torch.tensor([[-1, -1, -1]]))


def test_head_ranges_partition_global_buckets():
    hasher = make_hasher()
    assert hasher.layout.primes[0][0] == (11, 13)
    for layer in range(2):
        ranges = []
        for rank in range(3):
            heads, selected = hasher.layout.head_shard(layer, rank, 3)
            assert heads == (2 * rank, 2 * rank + 1)
            ranges.extend(selected)
        assert ranges[0][0] == 0
        assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
        assert ranges[-1][1] == int(hasher.primes[layer].sum())
    with pytest.raises(ValueError, match="divide evenly"):
        hasher.layout.head_shard(0, 0, 4)


def test_token_normalization_preserves_spaces_empty_and_byte_tokens():
    texts = [" The", "the", "THE", "é", "e", " ", "", "\ufffd", "\ufffd"]

    class Backend:
        def decode(self, ids, skip_special_tokens):
            assert not skip_special_tokens
            return texts[ids[0]]

        def id_to_token(self, token):
            return f"byte{token}"

    class Tokenizer:
        backend_tokenizer = Backend()

        def __len__(self):
            return len(texts)

    mapping, size = compressed_token_map(Tokenizer())
    assert mapping.tolist() == [0, 0, 0, 1, 1, 2, 3, 4, 5]
    assert size == 6
