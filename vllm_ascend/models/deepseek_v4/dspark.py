# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 DSpark draft model for Ascend.

DSpark weights are stored under the target checkpoint's ``mtp.*`` namespace,
but the draft path is a block drafter rather than the ordinary serial MTP
module. The target model provides selected layer hidden states; this model
projects them into the draft attention context and emits a full draft block.
"""

import copy
import typing
from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn
import vllm.envs as envs
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsEagle3
from vllm.model_executor.models.qwen3_dspark import DSparkConfidenceHead, DSparkMarkovHead
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix, process_eagle_weight
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_ascend.models.common.ops.sequence_parallel import sp_padding_mask, sp_shard
from vllm_ascend.models.deepseek_v4.model import (
    DeepseekV2MixtureOfExperts,
    DeepseekV4DecoderLayer,
    DeepseekV4MoE,
    DeepseekV41Attention,
    DeepseekV41DecoderLayer,
)
from vllm_ascend.ops.cache_v41 import write_main_cache_v41
from vllm_ascend.ops.mhc_v41 import mhc_collapse
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.utils import enable_dsa_cp


def _apply_dsv4_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    x: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    cos, sin = get_cos_and_sin_dsa(positions)
    layer_name = rotary_emb.layername
    cos_t = cos[layer_name]
    sin_t = sin[layer_name]
    if inverse:
        sin_t = -sin_t
    return rotary_emb(x, cos_t, sin_t)


def _get_dspark_num_mtp_layers(config: PretrainedConfig) -> int:
    num_layers = getattr(config, "n_mtp_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "dspark_num_mtp_layers", 3)
    return int(num_layers or 3)


class DeepseekV4DSparkModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.vllm_config = vllm_config
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.block_size = int(config.dspark_block_size)
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.num_dspark_layers = _get_dspark_num_mtp_layers(config)
        self.mtp_start_layer_idx = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + idx): DeepseekV4DecoderLayer(
                    vllm_config,
                    prefix=f"mtp.{idx}",
                    is_draft_layer=True,
                )
                for idx in range(self.num_dspark_layers)
            }
        )

        first_layer = self.layers[str(self.mtp_start_layer_idx)]
        self.use_sequence_parallel_moe = first_layer.use_sequence_parallel_moe

        _model_quant_cfg = getattr(config, "quantization_config", None)
        _main_proj_qconfig = (
            vllm_config.quant_config
            if _model_quant_cfg is not None and _model_quant_cfg.get("quant_method") == "fp8"
            else None
        )
        self.main_proj = ColumnParallelLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=_main_proj_qconfig,
            prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx}.main_proj"),
            gather_output=True,
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        first_layer.main_proj = self.main_proj
        first_layer.main_norm = self.main_norm

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        last_layer_idx = self.mtp_start_layer_idx + self.num_dspark_layers - 1
        draft_vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(
                prefix,
                f"layers.{last_layer_idx}.markov_head",
            ),
        )

        self.confidence_head = DSparkConfidenceHead(
            input_dim=config.hidden_size + config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "confidence_head"),
            bias=False,
            with_markov=True,
        )
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        last_layer = self.layers[str(last_layer_idx)]
        last_layer.norm = self.norm
        last_layer.markov_head = self.markov_head
        last_layer.hc_head_fn = self.hc_head_fn
        last_layer.hc_head_base = self.hc_head_base
        last_layer.hc_head_scale = self.hc_head_scale

        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.dsa_attn.swa_cache_layer.prefix for layer in self.layers.values()]

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.main_norm(self.main_proj(aux_hidden_states))

    def _project_shared_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn: type[nn.Module] | None = None,
    ) -> torch.Tensor:
        assert attn is not None
        kv = attn.kv_norm(attn.wkv(hidden_states))
        k_nope, k_pe = kv.split([attn.nope_head_dim, attn.rope_head_dim], dim=-1)
        k_pe = _apply_dsv4_rope(attn.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        return torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, attn.head_dim).contiguous()

    def _store_standard_swa_kv(
        self,
        shared_kv: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        attn: type[nn.Module] | None = None,
    ) -> None:
        if slot_mapping is None or slot_mapping.numel() == 0:
            return

        assert attn is not None
        swa_cache_layer = attn.dsa_attn.swa_cache_layer
        swa_kv_cache = getattr(swa_cache_layer, "kv_cache", None)
        if swa_kv_cache is None:
            return
        while isinstance(swa_kv_cache, (list, tuple)) and len(swa_kv_cache) == 1:
            swa_kv_cache = swa_kv_cache[0]

        from vllm_ascend.attention.dsa_attn_kv_plan import get_dsa_attn_kv_plan

        if slot_mapping.ndim == 1:
            slot_mapping = get_dsa_attn_kv_plan(self.vllm_config).format_dsa_slot_mapping(
                slot_mapping, swa_cache_layer.block_size
            )
        get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(swa_kv_cache, shared_kv, slot_mapping)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: list[torch.Tensor | None] | None = None,
    ) -> None:
        if context_states.numel() == 0 or context_slot_mapping is None:
            return
        for layer_idx, layer in enumerate(self.layers.values()):
            layer_context_slot_mapping = None if context_slot_mapping is None else context_slot_mapping[layer_idx]
            if context_positions.numel() == 0:
                return
            attn = layer.self_attn
            shared_kv = self._project_shared_kv(context_states, context_positions, attn)
            self._store_standard_swa_kv(shared_kv, layer_context_slot_mapping, attn)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids).unsqueeze(-2).repeat(1, self.hc_mult, 1)
        full_num_tokens = positions.shape[0]
        use_sp = self.use_sequence_parallel_moe
        orig_is_padding = None
        forward_context = None
        if use_sp:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                orig_is_padding = forward_context.is_padding
                forward_context.is_padding = sp_padding_mask(orig_is_padding, hidden_states)
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)

        residual = None
        for layer in self.layers.values():
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling=None,
                input_ids=input_ids,
            )
        if use_sp:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]

        if forward_context is not None:
            forward_context.is_padding = orig_is_padding
        head_hidden = self.hc_head(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        return head_hidden

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = torch.nn.functional.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor, logits_processor: LogitsProcessor) -> torch.Tensor:
        return self.markov_head.bias(markov_embed, logits_processor)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        return logits_processor(lm_head, self.norm(hidden_states))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )


@support_torch_compile
class DSparkDeepseekV4ForCausalLM(nn.Module, DeepseekV2MixtureOfExperts, SupportsEagle3):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config

        # check if quant config exist
        from vllm_ascend.utils import get_rotation_path

        self.rotation_path = get_rotation_path(vllm_config) if vllm_config.quant_config is not None else None

        self.model = DeepseekV4DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.expert_weights: typing.MutableSequence[typing.Sequence[torch.Tensor]] = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekV4MoE] = []
        example_moe = None
        for layer in self.model.layers.values():
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, DeepseekV4DecoderLayer)
            if isinstance(layer.mlp, DeepseekV4MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
        )

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Full-vocab draft: base logits, no d2t scatter.
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids  # full-vocab: draft ids are target ids

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        del spec_step_idx
        return self.model.compute_logits(
            hidden_states,
            self.lm_head,
            self.logits_processor,
        )

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_bias(markov_embed, self.logits_processor)

    def compute_confidence(self, head_hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states,
            context_positions,
            context_slot_mapping,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``mtp.{i}.*`` draft weights from the target checkpoint.

        Non-MTP weights belong to the target model and are skipped, except for
        standalone embedding/head weights used by the Ascend draft loader.
        """
        expert_mapping = self.model.get_expert_mapping()

        # (param_name, checkpoint shard name, shard_id) for non-expert
        # stacked parameters. Ascend keeps wq_a and wkv as separate parameters.
        stacked_params_mapping = [
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
            ("shared_experts.gate_up_proj", "shared_experts.gate_proj", 0),
            ("shared_experts.gate_up_proj", "shared_experts.up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        for name, loaded_weight in weights:
            if name == "embed.weight" and not self.rotation_path:
                name = "model.embed_tokens.weight"
            elif name == "head.weight" and not self.rotation_path:
                name = "lm_head.weight"
            elif name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                name = f"model.{name}"
            else:
                mapped_name = self._remap_dspark_name(name)
                if mapped_name is None:
                    continue
                name = mapped_name

            # Detect whether the checkpoint ships its own embed_tokens / lm_head
            # for the draft model.
            process_eagle_weight(self, name)

            # Expert scale parameters use Ascend's ``weight_scale`` convention.
            if name.endswith(".scale"):
                name = name.replace(".scale", ".weight_scale")

            # The multimodal checkpoint also contains one vision-router bias
            # for each MTP/DSpark layer.  DSpark runs only during text decode,
            # so draft MoE gates intentionally do not expose ``bias_vl``.
            # Do not alias it to the text correction bias: that would change
            # text routing whenever speculative decoding is enabled.
            if name.endswith(".e_score_correction_bias_vl") and name not in params_dict:
                logger.info_once("Ignoring vision-only router bias while loading the text-only DSpark drafter")
                continue

            if ".experts." in name:
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(typing.Callable[..., bool], param.weight_loader)
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                continue

            # Stacked rules only apply to decoder-layer weights. Head-stack
            # parameters load directly through the fallback below.
            is_layer_param = name.startswith("model.layers.")
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if not is_layer_param or f".{weight_name}." not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(name)
                break
            else:
                if "attn_sink" in name:
                    if enable_dsa_cp():
                        narrow = loaded_weight
                    else:
                        narrow = loaded_weight[head_start:head_end]
                    with torch.no_grad():
                        params_dict[name].copy_(narrow)
                    loaded_params.add(name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def _remap_dspark_name(self, name: str) -> str | None:
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m is None:
            return None
        stage = int(m.group(1))
        rest = m.group(2)

        if stage == self.model.num_dspark_layers - 1 and rest.startswith("confidence_head."):
            return f"model.{rest}"

        if stage == 0 and rest == "embed.weight":
            return "model.embed_tokens.weight"
        if stage == self.model.num_dspark_layers - 1 and rest == "head.weight":
            return "lm_head.weight"
        if rest.startswith(("hc_head_fn", "hc_head_base", "hc_head_scale")):
            return f"model.{rest}"

        first_layer_idx = self.config.num_hidden_layers
        last_layer_idx = first_layer_idx + self.model.num_dspark_layers - 1
        if rest.startswith(("main_proj.", "main_norm.")):
            layer_idx = first_layer_idx
        elif rest.startswith(("norm.", "markov_head.")):
            layer_idx = last_layer_idx
        else:
            layer_idx = first_layer_idx + stage
        name = f"model.layers.{layer_idx}.{rest}"

        replacements = (
            (".attn.", ".self_attn."),
            (".ffn_norm.", ".post_attention_layernorm."),
            (".attn_norm.", ".input_layernorm."),
            (".ffn.", ".mlp."),
            (".w1.", ".gate_proj."),
            (".w2.", ".down_proj."),
            (".w3.", ".up_proj."),
            (".mlp.gate.bias", ".mlp.gate.e_score_correction_bias"),
        )
        for checkpoint_name, param_name in replacements:
            name = name.replace(checkpoint_name, param_name)
        return name


def _v41_dspark_config(config: PretrainedConfig) -> PretrainedConfig:
    """Make a text-only draft config without mutating the target's expert bank."""
    result = copy.deepcopy(config)
    if result.hidden_size != 5120 or result.hc_mult != 4:
        raise ValueError("Ascend V4.1 DSpark requires H5120 and four HC streams")
    if (getattr(result, "draft_vocab_size", None) or result.vocab_size) != result.vocab_size:
        raise ValueError("V4.1 DSpark requires the full target vocabulary")
    result.n_routed_experts = int(result.dspark_n_routed_experts)
    result.num_experts_per_tok = int(result.dspark_num_experts_per_tok)
    if (result.n_routed_experts, result.num_experts_per_tok) != (128, 3):
        raise ValueError("Ascend V4.1 DSpark requires the released 128-expert/top3 layout")
    result.vision_n_layers = 0
    result.num_hash_layers = 0
    result.engram_layer_ids = []
    return result


class DeepseekV41DSparkModel(nn.Module):
    """Three CR0 V4.1 blocks; target context and draft queries stay separate."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if vllm_config.speculative_config is None:
            raise ValueError("V4.1 DSpark requires speculative_config")
        self.config = config = _v41_dspark_config(vllm_config.speculative_config.draft_model_config.hf_config)
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 8 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Ascend V4.1 DSpark currently requires TP8/PP1")
        if parallel.enable_expert_parallel or parallel.use_sequence_parallel_moe or parallel.enable_eplb:
            raise ValueError("Ascend V4.1 DSpark does not support EP, sequence parallel or EPLB")
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.max_position = vllm_config.model_config.max_model_len
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.num_dspark_layers = int(getattr(config, "n_mtp_layers", None) or config.num_nextn_predict_layers)
        if self.num_dspark_layers != 3 or len(self.target_layer_ids) != 3:
            raise ValueError("V4.1 DSpark requires three draft blocks and three target auxiliary states")
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=torch.bfloat16)
        self.layers = nn.ModuleList()
        rope_cache = None
        for stage in range(self.num_dspark_layers):
            layer_id = config.num_hidden_layers + stage
            if config.compress_ratios[layer_id] != 0:
                raise ValueError("Every V4.1 DSpark draft block must use CR0")
            layer_prefix = maybe_prefix(prefix, f"layers.{layer_id}")
            attention = DeepseekV41Attention(
                config,
                layer_id,
                vllm_config.model_config.max_model_len,
                f"{layer_prefix}.self_attn",
                rope_cache=rope_cache,
                vllm_config=vllm_config,
                is_draft_layer=True,
            )
            rope_cache = (attention.rope_cos, attention.rope_sin)
            moe = DeepseekV4MoE(
                config,
                parallel,
                vllm_config.quant_config,
                prefix=f"{layer_prefix}.mlp",
                is_draft_layer=True,
                image_sentinel_lo=getattr(config, "image_token_id", 129264),
            )
            self.layers.append(DeepseekV41DecoderLayer(config, attention, moe, engram=None))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=torch.bfloat16)
        with set_default_torch_dtype(torch.bfloat16):
            self.markov_head = DSparkMarkovHead(
                config.vocab_size,
                config.vocab_size,
                config.dspark_markov_rank,
                prefix=maybe_prefix(prefix, "markov_head"),
                quant_config=None,
            )
        self.confidence_head = (
            DSparkConfidenceHead(
                config.hidden_size + config.dspark_markov_rank,
                prefix=maybe_prefix(prefix, "confidence_head"),
                bias=False,
                with_markov=True,
            )
            if getattr(config, "enable_confidence_head", True)
            else None
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        if aux_hidden_states.ndim != 2 or aux_hidden_states.shape[1] != self.hidden_size * len(self.target_layer_ids):
            raise ValueError("V4.1 DSpark context must concatenate the three target auxiliary states")
        return self.main_norm(self.main_proj(aux_hidden_states))

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.swa_cache_layer.prefix for layer in self.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: list[torch.Tensor | None] | None = None,
    ) -> None:
        if context_states.ndim != 2 or context_states.shape != (context_positions.numel(), self.hidden_size):
            raise ValueError("V4.1 DSpark projected context and positions must have matching rows")
        if context_positions.ndim != 1:
            raise ValueError("V4.1 DSpark context positions must be one-dimensional")
        if context_slot_mapping is not None and len(context_slot_mapping) != len(self.layers):
            raise ValueError("V4.1 DSpark requires one context slot mapping per draft layer")
        # Proposer metadata uses INT32; the fused rotary kernel requires INT64.
        # Cast once for all draft layers, within the captured context graph.
        context_positions = context_positions.to(torch.int64)
        for stage, layer in enumerate(self.layers):
            attn = layer.self_attn
            projected = attn.fused_wqa_wkv(context_states)
            kv = attn.kv_norm(projected[..., attn.q_rank :].contiguous())
            slots = None if context_slot_mapping is None else context_slot_mapping[stage]
            if slots is not None:
                write_main_cache_v41(attn.swa_cache_layer.kv_cache, attn.rotate(kv, context_positions), slots)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        if embedded.shape != (positions.numel(), self.hidden_size) or input_ids.shape != positions.shape:
            raise ValueError("V4.1 DSpark queries require matching token, position and embedding rows")
        hidden = embedded[:, None, :].expand(-1, self.hc_mult, -1).contiguous()
        pre = torch.zeros((hidden.shape[0], self.hc_mult), dtype=torch.float32, device=hidden.device)
        pre[:, 0] = 1
        image_mask = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        # Keep virtual positions in cache metadata; only the RoPE lookup for
        # padded end-of-context queries uses a legal placeholder. The draft
        # builder independently masks their cache slots and attention rows.
        rope_positions = torch.where((positions >= 0) & (positions < self.max_position), positions, 0).to(torch.int64)
        for layer in self.layers:
            hidden, pre = layer(rope_positions, hidden, pre, input_ids=input_ids, image_token_mask=image_mask)
        return mhc_collapse(hidden, pre)


class DSparkDeepseekV41ForCausalLM(nn.Module, DeepseekV2MixtureOfExperts):
    """V4.1 draft contract, separate from the learned-HC-head V4 drafter.

    Admission remains controlled by the V4.1 integration guard. Packed MoE
    tensors use the same TP loaders as the target; no CUDA model is imported.
    """

    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if vllm_config.speculative_config is None:
            raise ValueError("V4.1 DSpark requires speculative_config")
        config = vllm_config.speculative_config.draft_model_config.hf_config
        weight_format = getattr(config, "ascend_weight_format", {})
        if (
            weight_format.get("group_size"),
            weight_format.get("signed_scale"),
            weight_format.get("checkpoint_packing"),
        ) != (
            32,
            True,
            "offset_binary_q_plus_8",
        ):
            raise ValueError("Convert V4.1 DSpark weights to signed-scale INT4 group32 before loading")
        self.model = DeepseekV41DSparkModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.config = self.model.config
        self.quant_config = vllm_config.quant_config
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.num_moe_layers = self.model.num_dspark_layers
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.expert_weights = []
        self.moe_mlp_layers = [layer.mlp for layer in self.model.layers]
        self.moe_layers = [layer.experts for layer in self.moe_mlp_layers]
        self.extract_moe_parameters(self.moe_mlp_layers[-1])

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, inputs_embeds=None):
        return self.model(input_ids, positions, inputs_embeds)

    def combine_hidden_states(self, aux_hidden_states):
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self):
        return self.model.get_draft_kv_cache_layer_names()

    def get_draft_attn_causal(self) -> list[bool]:
        return [False] * self.model.num_dspark_layers

    def precompute_and_store_context_kv(self, context_states, context_positions, context_slot_mapping=None):
        self.model.precompute_and_store_context_kv(context_states, context_positions, context_slot_mapping)

    def compute_logits(self, hidden_states, spec_step_idx=0):
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def compute_draft_logits(self, hidden_states):
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids):
        return draft_ids

    def markov_embed(self, token_ids):
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed):
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def compute_confidence(self, head_hidden, markov_embed):
        if self.model.confidence_head is None:
            raise ValueError("V4.1 DSpark confidence head is disabled")
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def _remap_dspark_name(self, name: str) -> str | None:
        match = re.fullmatch(r"mtp\.(\d+)\.(.+)", name.removeprefix("model."))
        if match is None:
            return None
        stage, rest = int(match[1]), match[2]
        if not 0 <= stage < self.model.num_dspark_layers:
            raise ValueError(f"Unexpected V4.1 DSpark stage: {name}")
        if rest.startswith(("main_proj.", "main_norm.")):
            if stage != 0:
                raise ValueError(f"DSpark context projection must belong to stage 0: {name}")
            return f"model.{rest}"
        if rest.startswith(("norm.", "markov_head.", "confidence_head.")):
            if stage != self.model.num_dspark_layers - 1:
                raise ValueError(f"DSpark heads must belong to the final stage: {name}")
            if rest.startswith("confidence_head.") and self.model.confidence_head is None:
                return None
            rest = rest.replace("markov_head.embed.", "markov_head.markov_w1.")
            rest = rest.replace("markov_head.head.", "markov_head.markov_w2.")
            return f"model.{rest}"
        # The released MM bias is not a text correction bias. The draft has
        # no image spans or vision router; this is the sole layer skip rule.
        if rest == "ffn.gate.bias_vl":
            return None
        replacements = (
            ("attn.", "self_attn."),
            ("ffn.", "mlp."),
            ("attn_norm.", "input_layernorm."),
            ("ffn_norm.", "post_attention_layernorm."),
            (".w1.", ".gate_proj."),
            (".w2.", ".down_proj."),
            (".w3.", ".up_proj."),
        )
        for source, target in replacements:
            rest = rest.replace(source, target)
        if rest == "mlp.gate.bias":
            rest = "mlp.gate.e_score_correction_bias"
        return f"model.layers.{stage}.{rest}"

    def _weight_load_plan(self, params):
        """Enumerate every required fusion/expert slice, not just parameters."""
        plan = {}
        for name in params:
            if name in {"model.embed_tokens.weight", "lm_head.weight"}:
                continue
            expert = re.fullmatch(
                r"(.+\.experts)\.routed_experts\.(w13|w2)_(weight_packed|weight_scale|weight_shape)", name
            )
            if expert:
                projections = (("gate_proj", "w1"), ("up_proj", "w3")) if expert[2] == "w13" else (("down_proj", "w2"),)
                for expert_id in range(self.config.n_routed_experts):
                    for projection, shard in projections:
                        plan[f"{expert[1]}.{expert_id}.{projection}.{expert[3]}"] = (name, shard, expert_id)
            elif name.endswith(".fused_wqa_wkv.weight"):
                for shard, component in enumerate(("wq_a", "wkv")):
                    plan[name.replace("fused_wqa_wkv", component)] = (name, shard, None)
            elif name.endswith(".shared_experts.gate_up_proj.weight"):
                for shard, component in enumerate(("gate_proj", "up_proj")):
                    plan[name.replace("gate_up_proj", component)] = (name, shard, None)
            else:
                plan[name] = (name, None, None)
        return plan

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load converted mtp tensors completely; reject unknown/missing slices."""
        params = dict(self.named_parameters())
        plan = self._weight_load_plan(params)
        seen, loaded = set(), set()
        for source, weight in weights:
            mapped = self._remap_dspark_name(source)
            if mapped is None:
                continue
            if mapped not in plan:
                raise ValueError(f"Unexpected or unconverted V4.1 DSpark weight: {source}")
            if mapped in seen:
                raise ValueError(f"Duplicate V4.1 DSpark weight: {source}")
            expected_dtype = (
                torch.int32
                if mapped.endswith((".weight_packed", ".weight_shape"))
                else torch.bfloat16
                if mapped.endswith(".weight_scale")
                else None
            )
            if (expected_dtype is not None and weight.dtype != expected_dtype) or (
                expected_dtype is None and weight.dtype not in (torch.bfloat16, torch.float32)
            ):
                raise ValueError(f"Unexpected converted V4.1 DSpark dtype: {source}: {weight.dtype}")
            target, shard, expert_id = plan[mapped]
            parameter = params[target]
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            if expert_id is not None:
                if not loader(parameter, weight, target, shard_id=shard, expert_id=expert_id, return_success=True):
                    raise ValueError(f"TP draft loader did not consume expert weight: {source}")
            elif shard is not None:
                loader(parameter, weight, shard)
            else:
                if target.endswith(".attn_sink"):
                    tp, rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
                    if weight.shape != (self.config.num_attention_heads,):
                        raise ValueError(f"Invalid draft attention sink shape: {source}")
                    width = self.config.num_attention_heads // tp
                    weight = weight[rank * width : (rank + 1) * width]
                loader(parameter, weight)
            seen.add(mapped)
            loaded.add(target)
        missing = plan.keys() - seen
        if missing:
            raise ValueError(f"Missing {len(missing)} V4.1 DSpark weight slices: {sorted(missing)[:8]}")
        return loaded
