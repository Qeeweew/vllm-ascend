# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU Engram hashing for lookup outside graph replay.

Normalization and prime/RNG construction follow DeepSeek's inference/engram.py.
No CUDA model package is imported. Chunk hashing is stateless: the runner must
supply actual lookback tokens, including corrected speculative history, so a
prefix hit or rollback cannot accidentally reuse a different request's state.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from tokenizers import Regex, normalizers


def compressed_token_map(tokenizer) -> tuple[torch.Tensor, int]:
    sentinel = "\ue000"
    normalize = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    vocabulary: dict[str, int] = {}
    mapped = []
    for token in range(len(tokenizer)):
        text = backend.decode([token], skip_special_tokens=False)
        key = backend.id_to_token(token) if "\ufffd" in text else (normalize.normalize_str(text) or text)
        if key not in vocabulary:
            vocabulary[key] = len(vocabulary)
        mapped.append(vocabulary[key])
    return torch.tensor(mapped, dtype=torch.int64, device="cpu"), len(vocabulary)


def _is_prime(n: int) -> bool:
    # Deterministic Miller-Rabin for n < 2**32, as in upstream's neutral code.
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


@dataclass(frozen=True)
class HostEngramLayout:
    layer_ids: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    head_dim: int

    @classmethod
    def from_config(cls, config) -> "HostEngramLayout":
        if not config.engram_layer_ids or config.engram_max_ngram_size < 2 or config.engram_n_heads < 1:
            raise ValueError("Engram requires layers and positive n-gram/head counts")
        if not 2 <= config.engram_vocab_size < 2**32 - 1:
            raise ValueError("Unsupported Engram bucket size")
        primes, seen = [], set()
        for _ in config.engram_layer_ids:
            orders = []
            for _ in range(config.engram_max_ngram_size - 1):
                heads, candidate = [], config.engram_vocab_size - 1
                for _ in range(config.engram_n_heads):
                    candidate += 1
                    while candidate in seen or not _is_prime(candidate):
                        candidate += 1
                    if candidate >= 2**32:
                        raise ValueError("Engram prime exceeds supported range")
                    heads.append(candidate)
                    seen.add(candidate)
                orders.append(tuple(heads))
            primes.append(tuple(orders))
        if len(primes) != len(config.engram_num_embeddings) or any(
            sum(p for order in layer for p in order) > rows for layer, rows in zip(primes, config.engram_num_embeddings)
        ):
            raise ValueError("Engram hash buckets exceed checkpoint table sizes")
        return cls(tuple(config.engram_layer_ids), tuple(primes), config.engram_head_dim)

    def head_shard(self, layer: int, rank: int, world_size: int) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
        sizes = [p for order in self.primes[layer] for p in order]
        if world_size < 1 or not 0 <= rank < world_size or len(sizes) % world_size:
            raise ValueError("Engram hash heads must divide evenly across TP ranks")
        width = len(sizes) // world_size
        start, end = rank * width, (rank + 1) * width
        offsets = np.cumsum([0, *sizes]).tolist()
        return tuple(range(start, end)), tuple(zip(offsets[start:end], offsets[start + 1 : end + 1]))


class HostEngramHasher:
    def __init__(self, layout: HostEngramLayout, token_map: torch.Tensor, vocab_size: int, pad_token_id: int):
        if token_map.device.type != "cpu" or token_map.dtype != torch.int64 or token_map.ndim != 1:
            raise ValueError("Compressed token map must be CPU int64")
        if vocab_size <= 0 or (token_map < 0).any() or (token_map >= vocab_size).any():
            raise ValueError("Invalid compressed vocabulary")
        if not 0 <= pad_token_id < token_map.numel():
            raise ValueError("Invalid Engram padding token")
        self.layout = layout
        self.token_map = token_map
        self.pad_id = int(token_map[pad_token_id])
        self.max_ngram = len(layout.primes[0]) + 1
        self.primes = torch.tensor(layout.primes, dtype=torch.int64, device="cpu")
        flat = self.primes.flatten(1)
        self.offsets = flat.cumsum(1) - flat
        bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
        self.multipliers = torch.from_numpy(
            np.stack(
                [
                    np.random.default_rng(10007 * layer).integers(0, bound, size=self.max_ngram, dtype=np.int64) * 2 + 1
                    for layer in layout.layer_ids
                ]
            )
        )

    @classmethod
    def from_tokenizer(cls, config, tokenizer) -> "HostEngramHasher":
        mapping, size = compressed_token_map(tokenizer)
        if size != config.engram_compressed_vocab_size:
            raise ValueError(f"Compressed vocabulary mismatch: {size} != {config.engram_compressed_vocab_size}")
        return cls(HostEngramLayout.from_config(config), mapping, size, config.engram_pad_token_id)

    def hash_chunk(
        self,
        input_ids: torch.Tensor,
        query_start_loc: Sequence[int],
        start_positions: Sequence[int],
        lookback_ids: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        lookback_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Hash flattened request chunks using newest-first lookback tokens.

        Every needed position before a chunk must have a real token ID. -1 is
        allowed only before sequence start. Image tokens keep their real IDs
        and use False in the masks; they block all older n-gram lookbacks.
        Results are one contiguous CPU int64 [T,hash_heads] matrix per layer.
        """
        depth, requests = self.max_ngram - 1, len(start_positions)
        for ids in (input_ids, lookback_ids):
            if ids.device.type != "cpu" or ids.dtype != torch.int64:
                raise ValueError("Host hashing requires CPU int64 token IDs")
        if input_ids.ndim != 1 or tuple(lookback_ids.shape) != (requests, depth):
            raise ValueError("Invalid token/lookback shape")
        starts = tuple(query_start_loc)
        if (
            len(starts) != requests + 1
            or starts[0] != 0
            or starts[-1] != input_ids.numel()
            or any(b < a for a, b in zip(starts, starts[1:]))
            or any(p < 0 for p in start_positions)
        ):
            raise ValueError("Invalid query boundaries or positions")
        for mask, ids in ((token_mask, input_ids), (lookback_mask, lookback_ids)):
            if mask is not None and (mask.device.type != "cpu" or mask.dtype != torch.bool or mask.shape != ids.shape):
                raise ValueError("Token masks must be CPU bool with matching shape")
        # Small host batches are dominated by per-request Torch dispatch.
        # NumPy reads the same CPU storage and constructs all request windows
        # together. Hash arithmetic remains signed int64, including remainder.
        ids = input_ids.numpy()
        mapping = self.token_map.numpy()
        if np.any(ids < 0) or np.any(ids >= mapping.size):
            raise ValueError("Invalid input token ID")
        tokens = mapping[ids]
        if token_mask is not None:
            tokens = np.where(token_mask.numpy(), tokens, -1)
        boundaries = np.asarray(starts, dtype=np.int64)
        prior = lookback_ids.numpy()
        needed = np.minimum(np.asarray(start_positions, dtype=np.int64), depth)
        used = (np.arange(depth)[None, :] < needed[:, None]) & (np.diff(boundaries)[:, None] > 0)
        if np.any(used & ((prior < 0) | (prior >= mapping.size))):
            raise ValueError("Actual lookback token IDs are required; async placeholders are not history")
        previous = np.full(prior.shape, self.pad_id, dtype=np.int64)
        previous[used] = mapping[prior[used]]
        if lookback_mask is not None:
            previous[used & ~lookback_mask.numpy()] = -1

        rows = np.arange(ids.size, dtype=np.int64)
        request = np.searchsorted(boundaries[1:], rows, side="right")
        first = boundaries[request, None]
        indices = rows[:, None] - np.arange(self.max_ngram)[None, :]
        prior_indices = np.clip(first - indices - 1, 0, depth - 1)
        windows = np.where(
            indices >= first,
            tokens[np.maximum(indices, 0)],
            previous[request[:, None], prior_indices],
        )
        windows[np.logical_or.accumulate(windows == -1, axis=-1)] = self.pad_id
        products = windows[:, None, :] * self.multipliers.numpy()
        rolling, hashes = products[:, :, 0].copy(), []
        primes = self.primes.numpy()
        for shift in range(1, self.max_ngram):
            rolling ^= products[:, :, shift]
            hashes.append(rolling[:, :, None] % primes[:, shift - 1])
        output = np.concatenate(hashes, axis=-1) + self.offsets.numpy()
        return tuple(
            torch.from_numpy(np.ascontiguousarray(output[:, layer])) for layer in range(len(self.layout.layer_ids))
        )
