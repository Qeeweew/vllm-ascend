# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backbone wiring contract; individual blocks have separate NPU references."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.models.deepseek_v4.model import (
    AscendDeepseekV41ForCausalLM,
    AscendDeepseekV41ForConditionalGeneration,
    DeepseekV41Model,
)


class FirstBlock(nn.Module):
    def forward(self, positions, hidden, pre, *, input_ids, engram_rows, token_mask, image_token_mask):
        torch.testing.assert_close(pre, torch.tensor([[1.0, 0, 0, 0]]).expand(hidden.shape[0], -1))
        assert engram_rows is None
        assert image_token_mask.tolist() == self.expected_images
        factors = torch.tensor([1, 2, 3, 4], dtype=hidden.dtype)[None, :, None]
        return hidden * factors, torch.tensor([[0.0, 1, 0, 0]]).expand(hidden.shape[0], -1)


class SecondBlock(nn.Module):
    def __init__(self, expected_rows):
        super().__init__()
        self.expected_rows = expected_rows

    def forward(self, positions, hidden, pre, *, input_ids, engram_rows, token_mask, image_token_mask):
        torch.testing.assert_close(pre, torch.tensor([[0.0, 1, 0, 0]]).expand(hidden.shape[0], -1))
        assert engram_rows is self.expected_rows
        assert token_mask.tolist() == [True, False]
        assert image_token_mask.tolist() == self.expected_images
        return hidden, torch.tensor([[0.0, 0, 0.25, 0.75]]).expand(hidden.shape[0], -1)


@pytest.mark.parametrize("typed_images", [None, [False, True]])
def test_initial_mix_final_collapse_and_layer_specific_staged_rows(typed_images):
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(model)
    model.engram_layer_ids = (1,)
    rows = torch.zeros((2, 3, 256), dtype=torch.bfloat16)
    model.layers = nn.ModuleList([FirstBlock(), SecondBlock(rows)])
    for layer in model.layers:
        layer.expected_images = typed_images or [False, False]
    model.norm = nn.Identity()
    embedded = torch.tensor([[1, 2, 4, 8], [-2, -4, -8, -16]], dtype=torch.bfloat16)
    ids, positions = torch.tensor([1, 2]), torch.tensor([0, 1])
    result = model(
        ids,
        positions,
        inputs_embeds=embedded,
        engram_rows=(rows,),
        engram_token_mask=torch.tensor([True, False]),
        image_token_mask=None if typed_images is None else torch.tensor(typed_images),
    )
    torch.testing.assert_close(result, embedded * 3.75, rtol=0, atol=0)
    with pytest.raises(ValueError, match="staged rows"):
        model(ids, positions, inputs_embeds=embedded)


class AuxiliaryBlock(nn.Module):
    def __init__(self, second=False):
        super().__init__()
        self.second = second

    def forward(self, positions, hidden, pre, **kwargs):
        if self.second:
            # Simulate a following layer changing its input in place. The
            # preceding auxiliary snapshot must retain the pre-change mean.
            hidden.add_(16)
        else:
            hidden = hidden * torch.tensor([1, 2, 3, 4], dtype=hidden.dtype)[None, :, None]
        pre = torch.tensor([[0.0, 0, 0.25, 0.75]]).expand(hidden.shape[0], -1)
        return hidden, pre


def auxiliary_model():
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(model)
    model.engram_layer_ids = ()
    model.start_layer, model.end_layer = 0, 2
    model.layers = nn.ModuleList([AuxiliaryBlock(), AuxiliaryBlock(second=True)])
    model.norm = nn.Identity()
    return model


def test_auxiliary_mean_precedes_next_layer_change_and_final_collapse():
    model = auxiliary_model()
    embedded = torch.tensor([[2, 4], [6, 8]], dtype=torch.bfloat16)
    ids, positions = torch.tensor([7, 8]), torch.tensor([0, 1])
    regular = model(ids, positions, inputs_embeds=embedded)
    model._set_aux_hidden_state_layers((2, 1, 2))
    assert model.aux_hidden_state_layers == (1, 2)
    logits_hidden, auxiliary = model(ids, positions, inputs_embeds=embedded)
    torch.testing.assert_close(logits_hidden, regular, rtol=0, atol=0)
    torch.testing.assert_close(auxiliary[0], embedded * 2.5, rtol=0, atol=0)
    torch.testing.assert_close(auxiliary[1], embedded * 2.5 + 16, rtol=0, atol=0)
    assert not torch.equal(auxiliary[-1], logits_hidden)
    model._set_aux_hidden_state_layers(())
    assert isinstance(model(ids, positions, inputs_embeds=embedded), torch.Tensor)


@pytest.mark.parametrize("layers", [(0,), (-1,), (3,), (True,), (1.0,)])
def test_auxiliary_layers_reject_non_backbone_or_non_integer_ids(layers):
    with pytest.raises(ValueError, match="one-based"):
        auxiliary_model()._set_aux_hidden_state_layers(layers)


def test_target_and_multimodal_wrapper_relay_eagle_auxiliary_interface():
    language = AscendDeepseekV41ForCausalLM.__new__(AscendDeepseekV41ForCausalLM)
    nn.Module.__init__(language)
    language.config = SimpleNamespace(dspark_target_layer_ids=[0, 1])
    language.model = auxiliary_model()
    wrapper = AscendDeepseekV41ForConditionalGeneration.__new__(AscendDeepseekV41ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.language_model = language
    assert wrapper.get_eagle3_default_aux_hidden_state_layers() == (1, 2)
    wrapper.set_aux_hidden_state_layers((2,))
    assert language.model.aux_hidden_state_layers == (2,)
