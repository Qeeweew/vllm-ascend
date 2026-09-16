# SPDX-License-Identifier: Apache-2.0
"""Frozen CPU oracle for the independent B1 candidate experiment.

This enumerates set membership in original-position space. It does not use
the new kernel's sorted-slot representation, tiling or gather implementation.
"""

import torch


def candidate_reference(case, candidates):
    assert all(value.device.type == "cpu" for value in case.values() if isinstance(value, torch.Tensor))
    length = max(0, int(case["sk"][0]))
    page_size = case["k"].shape[1]
    length = min(length, case["bt"].shape[1] * page_size)
    positions = torch.arange(length)
    membership = set(candidates.flatten().tolist())
    allowed = torch.tensor([int(position) // 8 in membership for position in positions], dtype=torch.bool)
    if case["cu"].tolist() != [0, 1]:
        allowed.fill_(False)
    physical = case["bt"][0, positions // page_size].long()
    allowed &= (physical >= 0) & (physical < case["k"].shape[0])
    positions = positions[allowed]
    physical = physical[allowed]
    keys = case["k"][physical, positions % page_size, 0].float()
    scales = case["ks"][physical, positions % page_size, 0].float()
    qk = ((case["q"][0].float() @ keys.T) / 1024).relu().half().float()
    weights = (case["w"][0] * case["qs"][0]).half().float()
    products = qk * weights[:, None]
    scores = products.sum(0) * scales
    # Each term/product is fixed by the documented FP16 rounding. The device
    # uses a depth-five tree; a CPU sum may use up to 31 dependent additions.
    # Include both final-scale roundings: gamma_(5+31+2), not a recall budget.
    eps = torch.finfo(torch.float32).eps
    errors = products.abs().sum(0) * scales.abs() * (38 * eps / (1 - 38 * eps))
    return dict(positions=positions, scores=scores, errors=errors)


def assert_candidate_selection(indices, reference):
    indices = indices.reshape(-1).long()
    valid = indices[indices >= 0]
    positions, scores, errors = (reference[name] for name in ("positions", "scores", "errors"))
    count = min(512, positions.numel())
    assert valid.numel() == count
    assert valid.unique().numel() == count
    assert torch.equal(valid, valid.sort().values)
    assert torch.all(indices[count:] == -1)
    if not count:
        return
    lookup = torch.searchsorted(positions, valid)
    assert torch.all(lookup < positions.numel())
    assert torch.equal(positions[lookup], valid)
    cutoff_index = scores.topk(count).indices[-1]
    cutoff = scores[cutoff_index]
    allowance = errors + errors[cutoff_index]
    assert torch.all(scores[lookup] >= cutoff - allowance[lookup])
    required = positions[scores > cutoff + allowance]
    assert set(required.tolist()) <= set(valid.tolist())


def candidate_reference_vectorized(case, candidates):
    """Same oracle with vectorized CPU membership for long prefill.

    candidate_reference above remains unchanged as the frozen cross-check.
    """
    assert all(value.device.type == "cpu" for value in case.values() if isinstance(value, torch.Tensor))
    length = max(0, int(case["sk"][0]))
    page_size = case["k"].shape[1]
    length = min(length, case["bt"].shape[1] * page_size)
    positions = torch.arange(length)
    allowed = torch.isin(positions // 8, candidates.flatten().long())
    if case["cu"].tolist() != [0, 1]:
        allowed.fill_(False)
    physical = case["bt"][0, positions // page_size].long()
    allowed &= (physical >= 0) & (physical < case["k"].shape[0])
    positions = positions[allowed]
    physical = physical[allowed]
    keys = case["k"][physical, positions % page_size, 0].float()
    scales = case["ks"][physical, positions % page_size, 0].float()
    qk = ((case["q"][0].float() @ keys.T) / 1024).relu().half().float()
    weights = (case["w"][0] * case["qs"][0]).half().float()
    products = qk * weights[:, None]
    scores = products.sum(0) * scales
    eps = torch.finfo(torch.float32).eps
    errors = products.abs().sum(0) * scales.abs() * (38 * eps / (1 - 38 * eps))
    return dict(positions=positions, scores=scores, errors=errors)
