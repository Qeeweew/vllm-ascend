# SPDX-License-Identifier: Apache-2.0
"""The frozen performance gate must reject each failure independently."""

import ast
from pathlib import Path

import pytest


@pytest.fixture
def evaluate():
    path = Path(__file__).parents[3] / "tests/e2e/single_node/ops/benchmark_indexer_v41_fused.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gates")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["gates"]


def samples():
    return {
        "fused": dict(median_us=90.0, p95_us=99.0, spread=0.03),
        "split": dict(median_us=100.0, p95_us=110.0, spread=0.02),
        "original_native": dict(median_us=110.0, p95_us=120.0, spread=0.01),
        "dense": dict(median_us=100.0, p95_us=110.0, spread=0.01),
    }


def test_exact_frozen_boundaries(evaluate):
    assert all(evaluate(samples(), dict(median_us=90.0, p95_us=100.0)).values())


@pytest.mark.parametrize(
    "name,key,value,failed_gate",
    [
        ("fused", "median_us", 90.001, "split_latency_gate"),
        ("fused", "p95_us", 99.001, "split_latency_gate"),
        ("dense", "median_us", 89.999, "live_dense_gate"),
        ("dense", "p95_us", 90.0, "live_dense_gate"),
        ("original_native", "median_us", 89.999, "live_native_gate"),
        ("split", "spread", 0.030001, "noise_gate"),
    ],
)
def test_no_other_metric_can_waive_failure(evaluate, name, key, value, failed_gate):
    values = samples()
    values[name][key] = value
    result = evaluate(values, dict(median_us=1000.0, p95_us=1000.0))
    assert not result[failed_gate]
    assert not all(result.values())


def test_archived_limit_is_not_scaled_twice(evaluate):
    result = evaluate(samples(), dict(median_us=90.0, p95_us=90.0))
    assert not result["frozen_dense_gate"]
