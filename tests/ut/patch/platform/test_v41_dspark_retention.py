# SPDX-License-Identifier: Apache-2.0
"""Exercise real startup retention policy and real block eviction on CPU."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import vllm.v1.core.kv_cache_utils as upstream
from vllm.config.speculative import SpeculativeConfig
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

from tests.ut.patch.platform.test_v41_cache_planner import make_config, make_specs
from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs


def configured_swa_specs(monkeypatch, tokens=5):
    """Run the real public entry through merge/policy/grouping, before sizing.

    Only the subsequent memory planning is intercepted. The exact group specs
    emitted by Ascend's production hook are then passed to a real manager.
    """
    register_ascend_kv_cache_specs()
    config = make_config()
    spec_config = SimpleNamespace(method="dspark", num_speculative_tokens=tokens, draft_model_config=None)
    spec_config.use_multi_module_mtp = lambda: SpeculativeConfig.use_multi_module_mtp(spec_config)
    config.speculative_config = spec_config
    assert spec_config.use_multi_module_mtp() is False
    swa = replace(make_specs()["swa.0"], extra_retained_tokens=tokens)
    captured = {}
    real_group = upstream.get_kv_cache_groups

    class GroupingComplete(Exception):
        pass

    def group_then_stop(cfg, specs):
        groups = real_group(cfg, specs)
        captured.update({name: group.kv_cache_spec for group in groups for name in group.layer_names})
        raise GroupingComplete

    monkeypatch.setattr(upstream, "get_kv_cache_groups", group_then_stop)
    with pytest.raises(GroupingComplete):
        upstream.get_kv_cache_configs(config, [{"target.swa": swa, "draft.swa": swa}], [1 << 30])
    return captured


def manager_for(spec):
    pool = BlockPool(num_gpu_blocks=16, enable_caching=False, hash_block_size=32)
    manager = SlidingWindowManager(
        spec, block_pool=pool, enable_caching=False, kv_cache_group_id=0, scheduler_block_size=32
    )
    manager.req_to_blocks["request"] = pool.get_new_blocks(8)
    return manager


def test_real_config_entry_preserves_target_and_draft_retention(monkeypatch):
    specs = configured_swa_specs(monkeypatch)
    assert {name: spec.extra_retained_tokens for name, spec in specs.items()} == {"target.swa": 5, "draft.swa": 5}


def test_real_manager_keeps_draft_left_edge_at_page_boundary(monkeypatch):
    spec = configured_swa_specs(monkeypatch)["draft.swa"]
    manager = manager_for(spec)
    # Draft prefix159 needs token31 plus the full block159..163. If startup
    # cleared retention, get_num_skipped_tokens(159)==32 frees token31's page.
    manager.remove_skipped_blocks("request", processed_computed_tokens=159)
    assert manager.req_to_blocks["request"][31 // 32] is not manager.block_pool.null_block


@pytest.mark.parametrize("method", [None, "mtp", "eagle3"])
def test_v41_grouping_does_not_add_dspark_retention_to_other_modes(method):
    register_ascend_kv_cache_specs()
    config = make_config()
    config.speculative_config = None if method is None else SimpleNamespace(method=method, num_speculative_tokens=5)
    spec = make_specs()["swa.0"]
    groups = upstream.get_kv_cache_groups(config, {"target.swa": spec})
    assert all(group.kv_cache_spec.extra_retained_tokens == 0 for group in groups)
    assert spec.extra_retained_tokens == 0


@pytest.mark.parametrize("accepted", range(6))
def test_processed_boundary_preserves_all_rejection_windows(accepted):
    # Before target verifies p..p+5, cache freeing uses the settled boundary p,
    # NOT the optimistic p+6. Scheduler later subtracts rejected tokens before
    # using the next settled boundary q=p+1+accepted.
    spec = replace(make_specs()["swa.0"], extra_retained_tokens=5)
    for p in range(128, 192):
        manager = manager_for(spec)
        manager.remove_skipped_blocks("request", processed_computed_tokens=p)
        q = p + 1 + accepted
        for position in range(max(q - 128, 0), q + 5):
            assert manager.req_to_blocks["request"][position // 32] is not manager.block_pool.null_block
        manager.remove_skipped_blocks("request", processed_computed_tokens=q)
        assert manager.req_to_blocks["request"][max(q - 128, 0) // 32] is not manager.block_pool.null_block


@pytest.mark.parametrize("tokens", range(1, 9))
def test_variable_dspark_real_entry_and_all_rejection_windows(monkeypatch, tokens):
    specs = configured_swa_specs(monkeypatch, tokens)
    assert all(spec.extra_retained_tokens == tokens for spec in specs.values())
    for p in range(128, 192):
        for accepted in range(tokens + 1):
            manager = manager_for(specs["draft.swa"])
            manager.remove_skipped_blocks("request", processed_computed_tokens=p)
            q = p + 1 + accepted
            for position in range(max(q - 128, 0), q + tokens):
                assert manager.req_to_blocks["request"][position // 32] is not manager.block_pool.null_block
            manager.remove_skipped_blocks("request", processed_computed_tokens=q)
            assert manager.req_to_blocks["request"][max(q - 128, 0) // 32] is not manager.block_pool.null_block
