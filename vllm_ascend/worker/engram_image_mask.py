# SPDX-License-Identifier: Apache-2.0
"""CPU image-span masks for Engram preparation outside graph execution.

The wrapper owns this immutable per-request metadata. Existing Engram history
continues to own tokens, DEAD boundaries, preemption and request cleanup. Pack
only after final inputs/positions have reached the runtime's CPU snapshot;
this module performs no device transfers or model/processor registration.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.multimodal.inputs import PlaceholderRange

V41_IMAGE_TOKEN_ID = 129264
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


def _cpu_integers(value: torch.Tensor, name: str) -> None:
    if value.device.type != "cpu" or value.dtype not in (torch.int32, torch.int64) or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional CPU integer tensor")


def _validate_roles(roles: torch.Tensor, length: int) -> None:
    _cpu_integers(roles, "image roles")
    values = roles.tolist()
    if len(values) != length or values[0] != IMAGE_START or values[-1] != IMAGE_END:
        raise ValueError("image roles must cover the full START through END span")
    widths, width = [], 0
    for role in values[1:-1]:
        if role == IMAGE:
            width += 1
        elif role == IMAGE_NEW_LINE and width:
            widths.append(width)
            width = 0
        else:
            raise ValueError("invalid V4.1 IMAGE/NEW_LINE row roles")
    if width or not widths or len(set(widths)) != 1:
        raise ValueError("image roles require equally wide rows terminated by NEW_LINE")


@dataclass(frozen=True)
class V41EngramImageSpans:
    """Prompt length and per-image half-open ranges; generated tokens are text."""

    prompt_length: int
    image_spans: tuple[tuple[int, int], ...]
    image_token_id: int = V41_IMAGE_TOKEN_ID

    def __post_init__(self):
        if not isinstance(self.prompt_length, Integral) or self.prompt_length < 0:
            raise ValueError("prompt length must be nonnegative")
        if not isinstance(self.image_token_id, Integral) or self.image_token_id < 0:
            raise ValueError("image token ID must be nonnegative")
        if not isinstance(self.image_spans, tuple):
            raise ValueError("image spans must be an immutable tuple")
        previous_end = 0
        for span in self.image_spans:
            if not isinstance(span, tuple) or len(span) != 2 or not all(isinstance(x, Integral) for x in span):
                raise ValueError("image spans must be integer (start, end) tuples")
            start, end = span
            if start < previous_end or end - start < 4 or end > self.prompt_length:
                raise ValueError("image spans must be ordered, nonoverlapping and within the prompt")
            previous_end = end

    @classmethod
    def from_prompt(
        cls,
        prompt_ids: torch.Tensor,
        image_ranges: Sequence["PlaceholderRange"],
        *,
        image_roles: Sequence[torch.Tensor | None] | None = None,
        image_token_id: int = V41_IMAGE_TOKEN_ID,
    ) -> "V41EngramImageSpans":
        """Validate processor-owned ranges; literal sentinel IDs outside stay text.

        Cached processor features can omit role tensors. Their complete ranges
        still define all image delimiters and patches unambiguously.
        """
        _cpu_integers(prompt_ids, "prompt IDs")
        if (prompt_ids < 0).any():
            raise ValueError("prompt IDs must contain actual tokens")
        if image_roles is not None and len(image_roles) != len(image_ranges):
            raise ValueError("one role tensor is required per supplied image range")
        spans = []
        for index, placeholder in enumerate(image_ranges):
            start, length = placeholder.offset, placeholder.length
            if not isinstance(start, Integral) or not isinstance(length, Integral):
                raise ValueError("image offsets and lengths must be integers")
            end = start + length
            if start < 0 or length < 4 or end > prompt_ids.numel():
                raise ValueError("image span is outside the prompt or shorter than one image row")
            embedded = placeholder.is_embed
            if embedded is not None and (
                embedded.device.type != "cpu"
                or embedded.dtype != torch.bool
                or embedded.shape != (length,)
                or not embedded.all()
            ):
                raise ValueError("V4.1 embeds every image-span position, including delimiters")
            if not torch.all(prompt_ids[start:end] == image_token_id):
                raise ValueError("every image-span raw token must equal the configured V4.1 image ID")
            if image_roles is not None and image_roles[index] is not None:
                _validate_roles(image_roles[index], length)
            spans.append((int(start), int(end)))
        return cls(prompt_ids.numel(), tuple(sorted(spans)), image_token_id)

    @classmethod
    def from_request(cls, request, *, image_token_id: int = V41_IMAGE_TOKEN_ID) -> "V41EngramImageSpans":
        """Adapt scheduler new-request data, including processor-cache hits."""
        if request.prompt_token_ids is None:
            raise ValueError("Engram image masks require actual full prompt IDs")
        ranges, roles = [], []
        for feature in request.mm_features or ():
            if feature.modality != "image":
                raise ValueError("V4.1 Engram image mask helper supports image features only")
            ranges.append(feature.mm_position)
            roles.append(feature.data["types"].data if feature.data is not None and "types" in feature.data else None)
        return cls.from_prompt(
            torch.tensor(request.prompt_token_ids, dtype=torch.int64, device="cpu"),
            ranges,
            image_roles=roles,
            image_token_id=image_token_id,
        )

    def prompt_keep_mask(self) -> torch.Tensor:
        keep = torch.ones(self.prompt_length, dtype=torch.bool, device="cpu")
        for start, end in self.image_spans:
            keep[start:end] = False
        return keep


def pack_v41_engram_token_mask(
    request_ids: Sequence[str],
    positions: torch.Tensor,
    query_start_loc: Sequence[int] | torch.Tensor,
    request_spans: Mapping[str, V41EngramImageSpans],
    *,
    input_ids: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack real requests in final order; graph padding is always false.

    Positions cover the graph bucket; the last real query boundary may end
    earlier. ``input_ids`` optionally cross-checks final image tokens after
    asynchronous corrections. This helper intentionally does not infer images
    from token IDs or implement Engram history/gap/rollback validation.
    """
    _cpu_integers(positions, "final positions")
    if input_ids is not None:
        _cpu_integers(input_ids, "final input IDs")
        if input_ids.shape != positions.shape:
            raise ValueError("final input IDs and positions must have matching shapes")
    if isinstance(query_start_loc, torch.Tensor):
        _cpu_integers(query_start_loc, "real query boundaries")
        query_start_loc = query_start_loc.tolist()
    bounds = tuple(query_start_loc)
    if (
        len(bounds) != len(request_ids) + 1
        or not all(isinstance(x, Integral) for x in bounds)
        or not bounds
        or bounds[0] != 0
        or bounds[-1] > positions.numel()
        or any(a > b for a, b in zip(bounds, bounds[1:]))
        or len(set(request_ids)) != len(request_ids)
    ):
        raise ValueError("invalid real request boundaries or duplicate request IDs")
    if out is not None and (
        out.device.type != "cpu" or out.dtype != torch.bool or out.shape != positions.shape or not out.is_contiguous()
    ):
        raise ValueError("output mask must be contiguous CPU bool covering the position bucket")
    result = torch.zeros(positions.shape, dtype=torch.bool, device="cpu")
    for row, request_id in enumerate(request_ids):
        if request_id not in request_spans:
            raise ValueError(f"missing image-span metadata for request {request_id!r}")
        spans = request_spans[request_id]
        first, end = bounds[row : row + 2]
        if first == end:
            continue
        current = positions[first:end]
        start_position = int(current[0])
        if start_position < 0 or not torch.equal(
            current, torch.arange(start_position, start_position + end - first, dtype=current.dtype, device="cpu")
        ):
            raise ValueError("real request positions must be nonnegative and contiguous")
        result[first:end] = True
        for image_start, image_end in spans.image_spans:
            # Contiguous final positions let us intersect slices without
            # allocating per-image token comparison tensors.
            left = max(0, image_start - start_position)
            right = min(end - first, image_end - start_position)
            if left >= right:
                continue
            target = slice(first + left, first + right)
            if input_ids is not None and not torch.all(input_ids[target] == spans.image_token_id):
                raise ValueError("final image token IDs differ from processor-owned image spans")
            result[target] = False
    if out is not None:
        out.copy_(result)
        return out
    return result
