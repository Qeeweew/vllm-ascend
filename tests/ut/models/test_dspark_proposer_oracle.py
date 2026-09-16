# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for the independently written Markov selection oracle."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import torch


def reference_function():
    folder = Path(__file__).resolve().parents[3] / "benchmarks/deepseek_v41"
    spec = importlib.util.spec_from_file_location("v41_proposer_oracle", folder / "check_dspark_proposer_tp8.py")
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(folder), *sys.path]):
        spec.loader.exec_module(module)
    return module.greedy_reference


def test_each_selected_token_conditions_the_next_markov_transition():
    reference = reference_function()
    embed = torch.tensor([[1, 0], [0, 1], [-1, 0]], dtype=torch.bfloat16)
    head = torch.tensor([[0, -2], [2, 0], [0, 2]], dtype=torch.bfloat16)
    logits = torch.zeros((3, 3), dtype=torch.bfloat16)
    assert reference(logits, embed, head, 0) == [1, 2, 0]
    assert torch.count_nonzero(logits) == 0


def test_bf16_addition_rounding_changes_argmax_and_must_not_be_promoted():
    reference = reference_function()
    embed = torch.ones((2, 1), dtype=torch.bfloat16)
    head = torch.tensor([[0.001], [0.002]], dtype=torch.bfloat16)
    logits = torch.ones((1, 2), dtype=torch.bfloat16)
    assert reference(logits, embed, head, 0) == [0]
    assert reference(logits.float(), embed, head, 0) == [1]
