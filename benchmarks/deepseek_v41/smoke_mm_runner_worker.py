# SPDX-License-Identifier: Apache-2.0
"""Test-only MM worker. Synchronous mask observations are not timing data."""

import torch
from smoke_runner_tp8 import EngramSmokeProbe

from vllm_ascend.worker.worker import NPUWorker


class V41MMSmokeWorker(NPUWorker, EngramSmokeProbe):
    def load_model(self) -> None:
        super().load_model()
        self.w4_dispatch = {
            "native_capture": 0,
            "native_eager": 0,
            "fallback_capture": 0,
            "fallback_eager": 0,
        }
        model = self.model_runner.get_model()
        for layer in model.get_language_model().model.layers:
            method = layer.mlp.experts.routed_experts.quant_method.quant_method
            original = method._can_use_native_decode

            def count_dispatch(*args, _original=original, **kwargs):
                native = _original(*args, **kwargs)
                phase = "capture" if torch.npu.is_current_stream_capturing() else "eager"
                self.w4_dispatch[f"{'native' if native else 'fallback'}_{phase}"] += 1
                return native

            method._can_use_native_decode = count_dispatch

    def start_mm_smoke(self):
        model = self.model_runner.get_model()
        if type(model).__name__ != "AscendDeepseekV41ForConditionalGeneration":
            raise RuntimeError("Production model registry has not enabled the V4.1 MM wrapper")
        if self.model_runner.encoder_cudagraph_manager is not None:
            raise RuntimeError("The initial V4.1 MM smoke requires eager encoder execution")
        self.encoder_calls = 0
        self.encoder_span_lengths = []
        self.image_prefills = []
        self.router_observations = []
        self.graph_replays = 0
        self.forward_signatures = []
        self.mm_last_bucket = 0
        runtime = self.model_runner.engram_runtime
        original_prepare = runtime.prepare

        def remember_bucket(*args, **kwargs):
            result = original_prepare(*args, **kwargs)
            self.mm_last_bucket = kwargs.get("bucket_tokens", args[4] if len(args) > 4 else 0)
            return result

        runtime.prepare = remember_bucket
        original = model.embed_multimodal

        def count_encoder(**kwargs):
            assert not torch.npu.is_current_stream_capturing()
            spans = original(**kwargs)
            self.encoder_calls += 1
            self.encoder_span_lengths.extend(span.shape[0] for span in spans)
            return spans

        model.embed_multimodal = count_encoder

        def observe_prefill(module, positional, kwargs):
            # The hook is installed after warmup/capture. Replayed graphs do
            # not execute Python hooks. Device reads only instrument eager
            # steps and must not be used in a performance measurement.
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("Unexpected graph recapture after MM smoke instrumentation")
            ids = kwargs.get("input_ids", positional[0] if positional else None)
            image_mask = kwargs.get("image_token_mask")
            engram_mask = kwargs.get("engram_token_mask")
            if ids is None or image_mask is None or engram_mask is None:
                raise AssertionError("Raw IDs and both typed masks must reach the MM wrapper")
            assert image_mask.dtype == engram_mask.dtype == torch.bool
            image_cpu, keep_cpu = image_mask.cpu(), engram_mask.cpu()
            assert not torch.any(image_cpu & keep_cpu)
            embeddings = kwargs.get("inputs_embeds", positional[3] if len(positional) > 3 else None)
            self.forward_signatures.append({"raw_ids": True, "embeddings": embeddings is not None})
            image_count = int(image_cpu.sum())
            if image_count:
                assert embeddings is not None and embeddings.ndim == 2
                assert torch.all(ids.cpu()[image_cpu] == module.config.image_token_id)
                self.image_prefills.append({"image_tokens": image_count, "text_tokens": int(keep_cpu.sum())})

        self._mm_smoke_hook = model.register_forward_pre_hook(observe_prefill, with_kwargs=True)
        for layer_index, layer in enumerate(model.get_language_model().model.layers):
            router = layer.mlp.experts.routed_experts.router
            original_routing = router._compute_routing

            def observe_routing(*args, _router=router, _original=original_routing, _layer=layer_index, **kwargs):
                if torch.npu.is_current_stream_capturing():
                    raise RuntimeError("Router observations cannot run in graph capture")
                ids = kwargs["input_ids"]
                mask = kwargs["image_token_mask"]
                assert torch.equal(mask, runtime.image_token_mask[: mask.numel()])
                assert torch.equal(ids, self.model_runner.input_ids.gpu[: ids.numel()])
                result = _original(*args, **kwargs)
                text_mask = ~mask
                hash_checked = _router.tid2eid is not None
                if hash_checked:
                    assert torch.equal(result[1][text_mask].long(), _router.tid2eid[ids[text_mask].long()].long())
                self.router_observations.append(
                    {
                        "layer": _layer,
                        "image_tokens": int(mask.sum()),
                        "literal_image_id_text_tokens": int((text_mask & (ids == model.config.image_token_id)).sum()),
                        "text_hash_checked": hash_checked,
                    }
                )
                return result

            router._compute_routing = observe_routing

        # Count successful calls that actually take an existing graph entry.
        # This is host-only observation; it introduces no capture-time ops.
        from vllm.forward_context import get_forward_context

        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        original_graph_call = ACLGraphWrapper.__call__

        def count_replay(wrapper, *args, **kwargs):
            context = get_forward_context()
            entry = wrapper.concrete_aclgraph_entries.get(context.batch_descriptor)
            replay = context.cudagraph_runtime_mode == wrapper.runtime_mode and entry is not None
            replay = replay and entry.aclgraph is not None
            result = original_graph_call(wrapper, *args, **kwargs)
            self.graph_replays += int(replay)
            return result

        ACLGraphWrapper.__call__ = count_replay
        return self.inspect_mm_smoke()

    def inspect_mm_smoke(self):
        result = self.inspect_engram_smoke()
        runtime = self.model_runner.engram_runtime
        model = self.model_runner.get_model()
        result.update(
            supports_mm_inputs=self.model_runner.supports_mm_inputs,
            tower_allocated=model.vision is not None or model.aligner is not None,
            mm_parameter_elements=sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if not name.startswith("language_model.")
            ),
            encoder_calls=self.encoder_calls,
            encoder_span_lengths=list(self.encoder_span_lengths),
            image_prefills=list(self.image_prefills),
            router_observations=list(self.router_observations),
            graph_replays=self.graph_replays,
            forward_signatures=list(self.forward_signatures),
            image_mask_ptr=runtime.image_token_mask.data_ptr(),
            image_mask_sum=int(runtime.image_token_mask[: self.mm_last_bucket].cpu().sum()),
            last_bucket=self.mm_last_bucket,
        )
        return result

    def finish_mm_smoke(self):
        """Checked host unregister after the last request; never resume."""
        runtime = self.model_runner.engram_runtime
        runtime.shutdown()
        released = [shard.weight is None for shard in runtime.offload.shards]
        assert runtime._closed and not runtime._prepared and all(released)
        # Runner teardown invokes shutdown again; verify that is idempotent.
        runtime.shutdown()
        return {"closed": runtime._closed, "prepared": runtime._prepared, "shard_weights_released": released}
