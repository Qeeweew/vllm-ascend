# SPDX-License-Identifier: Apache-2.0
"""Test-only worker: initialize packed dummy weights and observe MoE dispatch."""

import torch
from smoke_runner_tp8 import EngramSmokeProbe

from vllm_ascend.worker.worker import NPUWorker


class EngramSmokeWorker(NPUWorker, EngramSmokeProbe):
    def finish_engram_smoke(self):
        """Terminal RPC: release pinned tables after the final execution."""
        runtime = self.model_runner.engram_runtime
        runtime.shutdown()
        released = [shard.weight is None for shard in runtime.offload.shards]
        assert runtime._closed and not runtime._prepared and all(released)
        return {"closed": runtime._closed, "shard_weights_released": released}

    def capture_layer_trace(self, directory: str, all_ranks: bool = False, layer_limit: int = 3) -> dict:
        """RPC: bounded eager activation traces from TP0 or all eight ranks.

        D2H copies deliberately synchronize every observed stage. This is a
        correctness diagnostic and must never be used for timing or capture.
        """
        from pathlib import Path

        from vllm.forward_context import get_forward_context

        if self.rank != 0 and not all_ranks:
            return {"rank": self.rank, "installed": False}
        if not 1 <= layer_limit <= 3:
            raise ValueError("Layer trace limit must be between 1 and 3")
        if not self.model_runner.model_config.enforce_eager:
            raise ValueError("capture_layer_trace requires enforce_eager=True")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Layer trace hooks cannot be installed during graph capture")
        for handle in getattr(self, "_layer_trace_handles", ()):
            handle.remove()
        restore_logits = getattr(self, "_layer_trace_restore_logits", None)
        if restore_logits is not None:
            restore_logits()
        for restore in reversed(getattr(self, "_layer_trace_restore_moe", ())):
            restore()
        for restore in reversed(getattr(self, "_layer_trace_restore_attention", ())):
            restore()
        destination = Path(directory)
        if all_ranks:
            destination = destination / f"rank{self.rank}"
        destination.mkdir(parents=True, exist_ok=True)
        model = self.model_runner.get_model()
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        backbone = model.model
        traced_layers = list(backbone.layers)[:layer_limit]
        state = {"next_id": 0, "current": None, "last": None}
        handles = []
        restore_moe = []
        restore_attention = []

        def snapshot(value, max_rows=32):
            if isinstance(value, torch.Tensor):
                shape = tuple(value.shape)
                bounded = value[:max_rows] if value.ndim else value
                if bounded.numel() > 32 * 4 * 5120:
                    raise ValueError(f"Unexpected large activation in layer trace: {shape}")
                return {
                    "tensor": bounded.detach().cpu().clone(),
                    "shape": shape,
                    "truncated": bool(value.ndim and value.shape[0] > max_rows),
                }
            if isinstance(value, (tuple, list)):
                return tuple(snapshot(item) for item in value)
            return value

        def record(name, value):
            if state["current"] is not None and not torch.npu.is_current_stream_capturing():
                state["current"]["stages"][name] = snapshot(value)

        def start_forward(module, args, kwargs):
            state["current"] = state["last"] = None
            if torch.npu.is_current_stream_capturing():
                return
            context = get_forward_context()
            metadata = context.attn_metadata
            if getattr(context, "in_profile_run", False) or not isinstance(metadata, dict):
                return
            positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            counts = next((value for value in metadata.values() if hasattr(value, "num_prefills")), None)
            if positions is None or counts is None:
                return
            sequence = state["next_id"]
            state["next_id"] += 1
            payload = {
                "forward_id": sequence,
                "tp_rank": self.rank,
                "max_saved_rows": 32,
                "input_ids": snapshot(input_ids),
                "positions": snapshot(positions),
                "req_ids": list(self.model_runner.input_batch.req_ids),
                "num_prefills": counts.num_prefills,
                "num_decode_tokens": counts.num_decode_tokens,
                "cu_seqlens_q": snapshot(counts.cu_seqlens_q),
                "seqused_kv": snapshot(counts.seqused_kv),
                "compressor_state": {},
                "attention_cache": {},
                "stages": {},
            }
            for layer in traced_layers:
                attention = layer.self_attn
                names = [attention.swa_cache_layer.prefix]
                if attention.compress_ratio:
                    names.append(f"{attention._kv_source_prefix}.main_cache")
                for name in names:
                    cache = metadata[name]
                    payload["attention_cache"][name] = {
                        "block_table": snapshot(cache.block_table),
                        "slot_mapping": snapshot(cache.slot_mapping),
                        "schedule": snapshot(cache.schedule, max_rows=1024),
                    }
            for group_id, group in enumerate(self.model_runner.kv_cache_config.kv_cache_groups):
                names = [name for name in group.layer_names if ".compressor.state_cache" in name]
                if not names:
                    continue
                table = self.model_runner.input_batch.block_table[group_id].get_device_tensor()
                for name in names:
                    ring = metadata[name]
                    payload["compressor_state"][name] = {
                        "group_id": group_id,
                        "block_table_first_col": snapshot(table[: counts.seqused_kv.numel(), :1]),
                        "slot_mapping": snapshot(ring.slot_mapping),
                        "query_start_loc": snapshot(ring.query_start_loc),
                    }
            state["current"] = payload

        def finish_forward(module, args, kwargs, output):
            if state["current"] is None or torch.npu.is_current_stream_capturing():
                return
            if not all_ranks:
                record("backbone.output", output)
            payload = state["current"]
            torch.save(payload, destination / f"forward_{payload['forward_id']:05d}.pt")
            state["last"] = {
                key: payload[key] for key in ("forward_id", "tp_rank", "input_ids", "positions", "req_ids")
            }
            state["current"] = None

        def add_stage(module, name, *, save_input=False):
            if save_input:
                handles.append(
                    module.register_forward_pre_hook(
                        lambda owner, args, kwargs: record(f"{name}.input", args), with_kwargs=True
                    )
                )
            handles.append(
                module.register_forward_hook(
                    lambda owner, args, kwargs, output: record(f"{name}.output", output), with_kwargs=True
                )
            )

        handles.append(backbone.register_forward_pre_hook(start_forward, with_kwargs=True))
        handles.append(backbone.register_forward_hook(finish_forward, with_kwargs=True))
        for layer_id, layer in enumerate(traced_layers):
            prefix = f"layer{layer_id}"

            def layer_input(owner, args, kwargs, name=prefix):
                record(f"{name}.hidden_in", args[1])
                record(f"{name}.pre_mix_in", args[2])
                record(f"{name}.engram_rows", kwargs.get("engram_rows"))

            handles.append(layer.register_forward_pre_hook(layer_input, with_kwargs=True))
            if not all_ranks:
                add_stage(layer, prefix)
            if getattr(layer, "engram", None) is not None:
                add_stage(layer.engram, f"{prefix}.engram")
            add_stage(layer.input_layernorm, f"{prefix}.input_norm", save_input=True)
            add_stage(layer.self_attn, f"{prefix}.attention")
            for owner, attribute, stage in (
                (layer.self_attn, "project_inputs", f"{prefix}.attention.project_inputs"),
                (layer.self_attn.sparse, "forward", f"{prefix}.attention.sparse"),
            ):
                original_call = getattr(owner, attribute)

                def traced_call(*args, _original=original_call, _stage=stage, **kwargs):
                    output = _original(*args, **kwargs)
                    record(f"{_stage}.output", output)
                    return output

                setattr(owner, attribute, traced_call)
                restore_attention.append(
                    lambda owner=owner, attribute=attribute, original=original_call: setattr(owner, attribute, original)
                )
            add_stage(layer.self_attn.wo_b, f"{prefix}.attention.wo_b", save_input=True)
            output_linear = layer.self_attn.wo_b
            quant_method = output_linear.quant_method
            original_linear = quant_method.apply

            def traced_linear(*args, _original=original_linear, _layer=output_linear, _prefix=prefix, **kwargs):
                output = _original(*args, **kwargs)
                owner = kwargs.get("layer", args[0] if args else None)
                if owner is _layer:
                    record(f"{_prefix}.attention.wo_b.local_matmul", output)
                return output

            quant_method.apply = traced_linear
            restore_attention.append(
                lambda method=quant_method, original=original_linear: setattr(method, "apply", original)
            )
            if all_ranks:
                # Focus the eight-worker diagnostic on the first attention
                # divergence; avoid synchronizing later MoE/final-norm stages.
                continue
            add_stage(layer.post_attention_layernorm, f"{prefix}.post_attention_norm", save_input=True)
            add_stage(layer.mlp, f"{prefix}.moe")
            method = layer.mlp.experts.routed_experts.quant_method.quant_method
            original_apply = method.apply

            def traced_apply(*args, _original=original_apply, _prefix=prefix, **kwargs):
                record(f"{_prefix}.moe.dispatch_input", kwargs.get("x"))
                record(f"{_prefix}.moe.topk_ids", kwargs.get("topk_ids"))
                record(f"{_prefix}.moe.routing_weights", kwargs.get("topk_weights"))
                result = _original(*args, **kwargs)
                record(f"{_prefix}.moe.local_routed_output", result.routed_out)
                return result

            method.apply = traced_apply
            restore_moe.append(lambda method=method, original=original_apply: setattr(method, "apply", original))
        if not all_ranks:
            add_stage(backbone.norm, "final_norm", save_input=True)
        original_logits = model.compute_logits

        def traced_logits(*args, **kwargs):
            if not all_ranks and state["last"] is not None and not torch.npu.is_current_stream_capturing():
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                payload = dict(state["last"], hidden_states=snapshot(hidden))
                torch.save(payload, destination / f"forward_{payload['forward_id']:05d}_logits_input.pt")
                state["last"] = None
            return original_logits(*args, **kwargs)

        model.compute_logits = traced_logits
        self._layer_trace_handles = handles
        self._layer_trace_restore_moe = restore_moe
        self._layer_trace_restore_attention = restore_attention
        self._layer_trace_restore_logits = lambda: setattr(model, "compute_logits", original_logits)
        return {
            "rank": self.rank,
            "installed": True,
            "directory": str(destination),
            "layers": len(traced_layers),
            "attention_only": all_ranks,
        }

    def capture_moe_inputs(self, directory: str) -> None:
        """Save first real decode inputs on TP0; this is never a timing run."""
        from pathlib import Path

        from vllm.forward_context import get_forward_context

        if self.rank != 0:
            return
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        model = self.model_runner.get_model()
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        for layer_id, layer in enumerate(model.model.layers):
            method = layer.mlp.experts.routed_experts.quant_method.quant_method
            original = method.apply
            saved = []

            def dump_apply(_original=original, _saved=saved, _layer=layer_id, **kwargs):
                result = _original(**kwargs)
                if _saved or torch.npu.is_current_stream_capturing():
                    return result
                metadata = get_forward_context().attn_metadata
                if not isinstance(metadata, dict):
                    return result
                counts = next((value for value in metadata.values() if hasattr(value, "num_prefills")), None)
                if getattr(counts, "num_prefills", None) != 0 or getattr(counts, "num_decode_tokens", 0) <= 0:
                    return result
                torch.save(
                    {
                        "x": kwargs["x"].cpu(),
                        "ids": kwargs["topk_ids"].cpu(),
                        "routing": kwargs["topk_weights"].cpu(),
                        "output": result.routed_out.cpu(),
                        "layer": _layer,
                        "tp_rank": 0,
                    },
                    destination / f"layer{_layer}_tp0.pt",
                )
                _saved.append(True)
                return result

            method.apply = dump_apply

    @torch.inference_mode()
    def load_model(self) -> None:
        super().load_model()
        # Initialize before warmup/capture as well as before real requests.
        self.prepare_engram_smoke()
        self.w4_dispatch = {
            "native_capture": 0,
            "native_eager": 0,
            "fallback_capture": 0,
            "fallback_eager": 0,
        }
        self.engram_tokens = []
        self.candidate_dispatch = {"capture": 0, "eager": 0}
        offload = self.model_runner.engram_runtime.offload
        prepare = offload.prepare

        def record_tokens(hash_ids, bucket_tokens):
            self.engram_tokens.append(hash_ids[0].shape[0])
            return prepare(hash_ids, bucket_tokens)

        offload.prepare = record_tokens
        model = self.model_runner.get_model()
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        for layer in model.model.layers:
            candidate = getattr(getattr(layer.self_attn, "selector", None), "_candidate_selector", None)
            if candidate is not None:
                original_select = candidate.select_scores

                def counted_candidate(_original=original_select):
                    phase = "capture" if torch.npu.is_current_stream_capturing() else "eager"
                    self.candidate_dispatch[phase] += 1
                    return _original()

                candidate.select_scores = counted_candidate
            method = layer.mlp.experts.routed_experts.quant_method.quant_method
            original = method._can_use_native_decode

            def counted(*args, _original=original, **kwargs):
                native = _original(*args, **kwargs)
                phase = "capture" if torch.npu.is_current_stream_capturing() else "eager"
                self.w4_dispatch[f"{'native' if native else 'fallback'}_{phase}"] += 1
                return native

            method._can_use_native_decode = counted
