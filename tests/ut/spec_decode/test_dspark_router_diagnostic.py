# SPDX-License-Identifier: Apache-2.0
"""Evidence analysis must distinguish expert reordering and boundary crossings."""

import importlib.util
from pathlib import Path

import torch

_SOURCE = Path(__file__).resolve().parents[3] / "benchmarks/deepseek_v41/dspark_router_diagnostic.py"
_SPEC = importlib.util.spec_from_file_location("dspark_router_diagnostic", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_router_report_uses_actual_ids_and_distinguishes_membership():
    logits = torch.tensor([[3.0, 2.0, 1.0], [3.0, 1.001, 1.0]])
    padded_logits = logits.clone()
    padded_logits[1, 2] = 1.002
    left = {"r_ids": torch.tensor([[0, 1], [0, 1]]), "r_logits": logits}
    right = {"r_ids": torch.tensor([[1, 0], [0, 2]]), "r_logits": padded_logits}
    report = _MODULE.compare_routes(left, right, "r", torch.zeros(3))
    assert report["changed_membership_rows"] == [1]
    assert report["changed_order_or_membership_rows"] == [0, 1]
    assert report["unpadded"]["ids_on_changed_rows"] == [[0, 1]]
    assert report["padded"]["ids_on_changed_rows"] == [[0, 2]]
    assert 0 < report["unpadded"]["minimum_actual_selection_margin"] < 0.001


def test_router_report_exposes_inconsistent_recorded_selection():
    stages = {"r_ids": torch.tensor([[0, 2]]), "r_logits": torch.tensor([[3.0, 2.0, 1.0]])}
    report = _MODULE.compare_routes(stages, stages, "r", torch.zeros(3))
    assert report["changed_membership_rows"] == []
    assert report["unpadded"]["minimum_actual_selection_margin"] < 0
