# SPDX-License-Identifier: Apache-2.0
"""CPU analysis of actual DSpark routing observations, outside graph capture."""

import torch


def compare_routes(unpadded, padded, prefix, bias):
    """Separate order changes from expert membership and measure selection margins.

    This fixture uses dynamic text routing without expert groups or EPLB.
    CPU scores diagnose the margin, while recorded NPU IDs are authoritative.
    """
    left = unpadded[f"{prefix}_ids"].long()
    right = padded[f"{prefix}_ids"].long()
    k = left.shape[1]
    membership_changed = (left.sort(dim=-1).values != right.sort(dim=-1).values).any(dim=-1)
    rows = membership_changed.nonzero().flatten()
    scores = [
        torch.nn.functional.softplus(stages[f"{prefix}_logits"].float()).sqrt() + bias.float()
        for stages in (unpadded, padded)
    ]

    def describe(score, ids):
        boundary = score.topk(k + 1, dim=-1).values
        # Negative values mean an observed expert lies below an unselected one
        # in the CPU score reference; report instead of assuming equivalence.
        selected_min = score.gather(1, ids).min(dim=-1).values
        excluded = score.clone().scatter_(1, ids, -torch.inf)
        actual_margin = selected_min - excluded.max(dim=-1).values
        return {
            "ids_on_changed_rows": ids[rows].tolist(),
            "boundary_margin_on_changed_rows": (boundary[:, k - 1] - boundary[:, k])[rows].tolist(),
            "actual_selection_margin_on_changed_rows": actual_margin[rows].tolist(),
            "minimum_actual_selection_margin": float(actual_margin.min()),
        }

    return {
        "changed_membership_rows": rows.tolist(),
        "changed_order_or_membership_rows": (left != right).any(dim=-1).nonzero().flatten().tolist(),
        "selection_score_max_abs": float((scores[0] - scores[1]).abs().max()),
        "selection_score_max_abs_on_changed_rows": (scores[0] - scores[1]).abs().amax(dim=-1)[rows].tolist(),
        "unpadded": describe(scores[0], left),
        "padded": describe(scores[1], right),
    }
