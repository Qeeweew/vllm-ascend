# SPDX-License-Identifier: Apache-2.0
"""Production registry/proposer sharing, without allocating distributed weights."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm import ModelRegistry
from vllm.config import CUDAGraphMode
from vllm.plugins import load_general_plugins

from vllm_ascend.models.deepseek_v4 import dspark, model
from vllm_ascend.spec_decode import llm_base_proposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer


def test_official_draft_architecture_resolves_to_ascend_class():
    load_general_plugins()
    registered = ModelRegistry.models["DSparkV41DraftModel"]
    assert registered.load_model_cls() is dspark.DSparkDeepseekV41ForCausalLM
    info = registered.inspect_model_cls()
    assert not info.supports_multimodal
    assert ModelRegistry.models["DSparkDraftModel"].load_model_cls() is dspark.DSparkDeepseekV4ForCausalLM


def shell(cls):
    instance = cls.__new__(cls)
    nn.Module.__init__(instance)
    return instance


@pytest.mark.parametrize("multimodal", [False, True])
def test_proposer_load_shares_v41_target_embedding_and_head(monkeypatch, multimodal):
    # Real model/proposer classes expose the protocols. Only allocations and
    # checkpoint IO are replaced; load_model and its sharing branches execute.
    language = shell(model.AscendDeepseekV41ForCausalLM)
    language.model = nn.Module()
    language.model.embed_tokens = nn.Embedding(16, 8, dtype=torch.bfloat16)
    language.lm_head = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
    language.config = SimpleNamespace(image_token_id=129264)
    target = language
    if multimodal:
        target = shell(model.AscendDeepseekV41ForConditionalGeneration)
        target.language_model = language
        target.config = language.config

    draft = shell(dspark.DSparkDeepseekV41ForCausalLM)
    draft.model = shell(dspark.DeepseekV41DSparkModel)
    draft.model.embed_tokens = nn.Embedding(16, 8, dtype=torch.bfloat16)
    draft.lm_head = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
    draft.config = SimpleNamespace()
    old_embedding, old_head = draft.model.embed_tokens, draft.lm_head
    with torch.no_grad():
        old_embedding.weight.fill_(float("nan"))
        old_head.weight.fill_(float("nan"))

    proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
    proposer.method = "dspark"
    proposer.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )
    proposer.maybe_eager_context = nullcontext()
    proposer._get_model = Mock(return_value=draft)
    proposer.supports_mm_inputs = multimodal
    proposer.input_ids = torch.zeros(1, dtype=torch.long)
    proposer.parallel_drafting = False
    proposer.use_cuda_graph = False
    proposer.sliding_window = None
    monkeypatch.setattr(llm_base_proposer, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True, world_size=1))
    monkeypatch.setattr("vllm_ascend.ascend_config.get_ascend_config", lambda: SimpleNamespace(draft_window_size=None))
    cache = SimpleNamespace(
        get_kv_cache_spec=lambda _: object(),
        get_attn_backend=lambda: SimpleNamespace(get_supported_kernel_block_sizes=lambda: [32]),
    )
    layers = iter([{}, {"draft.swa": cache}, {}, {"draft.swa": cache}])
    monkeypatch.setattr(llm_base_proposer, "get_layers_from_vllm_config", lambda *args: next(layers))

    proposer.load_model(target)

    proposer._get_model.assert_called_once_with()
    assert proposer.model is draft
    assert proposer.attn_layer_names == ["draft.swa"]
    assert proposer.kernel_block_size == 32
    assert not proposer.supports_mm_inputs
    assert draft.model.embed_tokens is language.model.embed_tokens
    assert draft.lm_head is language.lm_head
    assert old_embedding is not draft.model.embed_tokens and old_head is not draft.lm_head
    ids = torch.tensor([0, 3, 0, 7])
    torch.testing.assert_close(draft.embed_input_ids(ids), language.model.embed_tokens(ids), rtol=0, atol=0)
    assert torch.isfinite(draft.embed_input_ids(ids)).all()
    assert torch.isfinite(draft.lm_head.weight).all()
