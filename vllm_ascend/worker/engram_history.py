# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-keyed, CPU-only Engram history for preparation before graph replay.

Only complete prompt tokens and actual executed model inputs belong here.
Scheduler placeholders and optimistic speculative positions are not history.
Preemption retains entries; finished requests are dropped; request-ID reuse
requires an explicit reset. Storage is a compact integer array plus mask bytes.
"""

from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import torch

from vllm_ascend.ops.engram_hash import HostEngramHasher


@dataclass(frozen=True)
class EngramHistoryBatch:
    hash_ids: tuple[torch.Tensor, ...]
    token_mask: torch.Tensor
    image_token_mask: torch.Tensor


@dataclass
class _RequestHistory:
    tokens: array
    mask: bytearray
    prompt_length: int
    prompt_image_mask: bytearray


class EngramRequestHistory:
    """Own actual token history independently of changing scheduler row slots.

    ``prepare`` hashes before committing all request updates. A failure leaves
    every history unchanged. Successful calls overwrite inputs at their real
    positions and truncate the generated tail, while retaining the full known
    prompt. A later rejected draft is corrected by the next call's positions.
    """

    def __init__(self, hasher: HostEngramHasher) -> None:
        self.hasher = hasher
        self._requests: dict[str, _RequestHistory] = {}

    def _validate_ids(self, ids: torch.Tensor) -> None:
        if ids.device.type != "cpu" or ids.dtype != torch.int64 or ids.ndim != 1:
            raise ValueError("Engram history requires one-dimensional CPU int64 token IDs")
        if (ids < 0).any() or (ids >= self.hasher.token_map.numel()).any():
            raise ValueError("Actual token IDs are required; async placeholders are not history")

    @staticmethod
    def _mask(ids: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return torch.ones(ids.shape, dtype=torch.bool, device="cpu")
        if mask.device.type != "cpu" or mask.dtype != torch.bool or mask.shape != ids.shape:
            raise ValueError("Engram history masks must be CPU bool with matching token shape")
        return mask

    def reset_request(
        self,
        request_id: str,
        prompt_ids: torch.Tensor,
        *,
        prompt_mask: torch.Tensor | None = None,
        prompt_image_mask: torch.Tensor | None = None,
        executed_tail: torch.Tensor | None = None,
        tail_mask: torch.Tensor | None = None,
    ) -> None:
        """Seed a full prompt, explicitly replacing any previous ID incarnation.

        For restoration after state loss, ``executed_tail`` must contain the
        contiguous actual inputs after the prompt, never unsampled placeholders
        or merely proposed output tokens. Normal preemption needs no reset.
        """
        self._validate_ids(prompt_ids)
        prompt_mask = self._mask(prompt_ids, prompt_mask)
        if prompt_image_mask is None:
            prompt_image_mask = torch.zeros(prompt_ids.shape, dtype=torch.bool, device="cpu")
        else:
            prompt_image_mask = self._mask(prompt_ids, prompt_image_mask)
        if (prompt_image_mask & prompt_mask).any():
            raise ValueError("Image-span positions must be excluded from Engram prompt tokens")
        if executed_tail is None:
            if tail_mask is not None:
                raise ValueError("tail_mask requires executed_tail")
            executed_tail = torch.empty(0, dtype=torch.int64, device="cpu")
        self._validate_ids(executed_tail)
        tail_mask = self._mask(executed_tail, tail_mask)
        tokens = array("q", prompt_ids.tolist())
        tokens.extend(executed_tail.tolist())
        mask = bytearray(prompt_mask.tolist())
        mask.extend(tail_mask.tolist())
        self._requests[request_id] = _RequestHistory(
            tokens, mask, prompt_ids.numel(), bytearray(prompt_image_mask.tolist())
        )

    def drop_request(self, request_id: str) -> None:
        """Idempotent finished-request cleanup; do not call for preemption."""
        self._requests.pop(request_id, None)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._requests

    def prepare(
        self,
        request_ids: Sequence[str],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: Sequence[int] | torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        use_seeded_prompt_mask: bool = False,
    ) -> EngramHistoryBatch:
        """Hash actual unpadded input rows and record their executed history.

        ``input_ids`` and ``positions`` must be final CPU copies after all
        asynchronous input/position corrections. ``query_start_loc`` uses the
        same packed request order. Each request's positions must be contiguous;
        a missing request or a gap after known history fails closed.

        ``use_seeded_prompt_mask`` reuses validated prompt masks at actual
        positions when no explicit step mask is supplied. Generated positions
        remain text. This is intended for the runtime's final CPU snapshot;
        it never guesses image spans from raw sentinel IDs.
        """
        self._validate_ids(input_ids)
        if use_seeded_prompt_mask and token_mask is not None:
            raise ValueError("An explicit Engram step mask cannot be combined with seeded-mask selection")
        token_mask = self._mask(input_ids, token_mask).clone()
        if positions.device.type != "cpu" or positions.dtype != torch.int64 or positions.shape != input_ids.shape:
            raise ValueError("Engram history positions must be CPU int64 with matching token shape")
        if isinstance(query_start_loc, torch.Tensor):
            if (
                query_start_loc.device.type != "cpu"
                or query_start_loc.dtype not in (torch.int32, torch.int64)
                or query_start_loc.ndim != 1
            ):
                raise ValueError("Engram query boundaries must be one-dimensional CPU integers")
            query_start_loc = query_start_loc.tolist()
        boundaries = tuple(query_start_loc)
        if (
            len(boundaries) != len(request_ids) + 1
            or not all(isinstance(x, Integral) for x in boundaries)
            or boundaries[0] != 0
            or boundaries[-1] != input_ids.numel()
            or any(b < a for a, b in zip(boundaries, boundaries[1:]))
            or len(set(request_ids)) != len(request_ids)
        ):
            raise ValueError("Invalid Engram query boundaries or duplicate request IDs")
        depth = self.hasher.max_ngram - 1
        lookback = torch.full((len(request_ids), depth), -1, dtype=torch.int64, device="cpu")
        lookback_mask = torch.zeros_like(lookback, dtype=torch.bool)
        image_token_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device="cpu")
        starts, updates = [], []
        for row, request_id in enumerate(request_ids):
            history = self._requests.get(request_id)
            if history is None:
                raise ValueError(f"Missing Engram history for request {request_id!r}; explicitly reset or restore it")
            first, end = boundaries[row : row + 2]
            if first == end:
                starts.append(0)
                continue
            start = int(positions[first])
            stop = start + end - first
            if start < 0 or not torch.equal(
                positions[first:end], torch.arange(start, stop, dtype=torch.int64, device="cpu")
            ):
                raise ValueError("Engram request positions must be nonnegative and contiguous")
            if start > len(history.tokens):
                raise ValueError(f"Missing actual Engram history before position {start} for request {request_id!r}")
            tokens = array("q", input_ids[first:end].tolist())
            prompt_overlap = max(0, min(stop, history.prompt_length) - start)
            if prompt_overlap:
                image_token_mask[first : first + prompt_overlap] = torch.tensor(
                    history.prompt_image_mask[start : start + prompt_overlap], dtype=torch.bool, device="cpu"
                )
            if use_seeded_prompt_mask and prompt_overlap:
                token_mask[first : first + prompt_overlap] = torch.tensor(
                    history.mask[start : start + prompt_overlap], dtype=torch.bool, device="cpu"
                )
            masks = bytearray(token_mask[first:end].tolist())
            if (
                tokens[:prompt_overlap] != history.tokens[start : start + prompt_overlap]
                or masks[:prompt_overlap] != history.mask[start : start + prompt_overlap]
            ):
                raise ValueError(
                    "Executed prompt tokens/masks differ from the seeded prompt; reset the request explicitly"
                )
            needed = min(start, depth)
            if needed:
                lookback[row, :needed] = torch.tensor(
                    history.tokens[start - needed : start][::-1], dtype=torch.int64, device="cpu"
                )
                lookback_mask[row, :needed] = torch.tensor(
                    history.mask[start - needed : start][::-1], dtype=torch.bool, device="cpu"
                )
            starts.append(start)
            updates.append((history, start, stop, tokens, masks))
        hashes = self.hasher.hash_chunk(
            input_ids, boundaries, starts, lookback, token_mask=token_mask, lookback_mask=lookback_mask
        )
        # All validation and hashing have succeeded. No full-prefix copy or
        # per-token Python object is retained for a long-running request.
        for history, start, stop, tokens, masks in updates:
            history.tokens[start:stop] = tokens
            history.mask[start:stop] = masks
            keep = max(history.prompt_length, stop)
            del history.tokens[keep:]
            del history.mask[keep:]
        return EngramHistoryBatch(hashes, token_mask, image_token_mask)
