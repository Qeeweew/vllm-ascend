# SPDX-License-Identifier: Apache-2.0
"""Observation-only worker for the unmodified full production checkpoint."""

import dataclasses

from full_engram_audit import audit_loading, inspect_tables, process_memory

from vllm_ascend.worker.worker import NPUWorker


class V41FullModelWorker(NPUWorker):
    def load_model(self):
        with audit_loading() as records:
            self.full_load_records = records
            super().load_model()
        self._observe_native_dispatch()

    def _observe_native_dispatch(self):
        """Associate successful submissions with concrete graphs before warmup."""
        import torch
        from vllm.forward_context import get_forward_context

        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        self.full_operator_dispatch = {}
        self.full_graph_replays = 0
        self._full_graph_records = {}
        self._full_graph_owners = {}  # Keep identities alive, including replaced entries.
        self._full_capture_stack = []
        self._full_observer_restore = []

        def install(owner, name, observer):
            self._full_observer_restore.append((owner, name, getattr(owner, name)))
            setattr(owner, name, observer)

        def record(name):
            phase = "capture" if torch.npu.is_current_stream_capturing() else "eager"
            key = f"{name}_{phase}"
            self.full_operator_dispatch[key] = self.full_operator_dispatch.get(key, 0) + 1
            if phase == "capture" and self._full_capture_stack:
                counts = self._full_capture_stack[-1]
                counts[name] = counts.get(name, 0) + 1

        # Observe actual native submissions, including W4A16; a true eligibility
        # predicate alone does not establish that the operator was called.
        for name, label in (
            ("v41_rope", "v41_rope"),
            ("v41_main_cache_store", "v41_main_cache_store"),
            ("v41_index_cache_store", "v41_index_cache_store"),
            ("v41_moe_router", "v41_moe_router"),
            ("npu_w4a16_moe", "w4_native"),
        ):
            if not hasattr(torch.ops._C_ascend, name):
                continue
            original = getattr(torch.ops._C_ascend, name)

            def observe(*args, _original=original, _name=label, **kwargs):
                result = _original(*args, **kwargs)
                record(_name)
                return result

            install(torch.ops._C_ascend, name, observe)

        original_graph_call = ACLGraphWrapper.__call__

        def observe_graph(wrapper, *args, **kwargs):
            context = get_forward_context()
            descriptor = context.batch_descriptor
            if context.cudagraph_runtime_mode != wrapper.runtime_mode:
                return original_graph_call(wrapper, *args, **kwargs)
            entry = wrapper.concrete_aclgraph_entries.get(descriptor)
            graph = entry.aclgraph if entry is not None else None
            if graph is not None:
                result = original_graph_call(wrapper, *args, **kwargs)
                self.full_graph_replays += 1
                key = (id(wrapper), id(entry), id(graph))
                if key in self._full_graph_records:
                    self._full_graph_records[key]["request_replays"] += 1
                return result

            counts = {}
            self._full_capture_stack.append(counts)
            try:
                result = original_graph_call(wrapper, *args, **kwargs)
            finally:
                self._full_capture_stack.pop()
            # Publish evidence only after the complete capture succeeded. A
            # failed capture must not authorize a later, unrelated replay.
            entry = wrapper.concrete_aclgraph_entries.get(descriptor)
            if entry is not None and entry.aclgraph is not None:
                key = (id(wrapper), id(entry), id(entry.aclgraph))
                self._full_graph_owners[key] = (wrapper, entry, entry.aclgraph)
                self._full_graph_records[key] = {
                    "wrapper_id": key[0],
                    "entry_id": key[1],
                    "graph_id": key[2],
                    "descriptor": dataclasses.asdict(descriptor),
                    "runtime_mode": str(wrapper.runtime_mode),
                    "captured_native_ops": counts,
                    "request_replays": 0,
                }
            return result

        install(ACLGraphWrapper, "__call__", observe_graph)

    def _start_request_dispatch_audit(self):
        """Retain warmup capture provenance, reset only request replay counts."""
        self.full_graph_replays = 0
        for record in self._full_graph_records.values():
            record["request_replays"] = 0

    def _inspect_graph_dispatch(self):
        return [
            {
                **record,
                "descriptor": dict(record["descriptor"]),
                "captured_native_ops": dict(record["captured_native_ops"]),
            }
            for record in self._full_graph_records.values()
        ]

    def start_full_model_audit(self, source, with_vision=False):
        model = self.model_runner.get_model()
        language = model.get_language_model()
        assert len(language.model.layers) == 40
        assert language.config.engram_num_embeddings == [384006168, 384016682]
        assert (model.vision is not None) == with_vision
        assert (model.aligner is not None) == with_vision
        self.full_encoder_spans = []
        if with_vision:
            import torch

            original_encoder = model.embed_multimodal

            def observe_encoder(**kwargs):
                assert not torch.npu.is_current_stream_capturing()
                spans = original_encoder(**kwargs)
                self.full_encoder_spans.extend(int(span.shape[0]) for span in spans)
                return spans

            model.embed_multimodal = observe_encoder
        runtime = self.model_runner.engram_runtime
        reports, _ = inspect_tables(runtime, source, self.vllm_config.model_config.model, self.full_load_records)
        self._start_request_dispatch_audit()
        return {"tables": reports, "load_audit": self.full_load_records, "state": self.inspect_full_model_audit()}

    def inspect_full_model_audit(self):
        import torch

        runtime = self.model_runner.engram_runtime
        return {
            "rows": [row.data_ptr() for row in runtime.offload.device_rows],
            "mask": runtime.token_mask.data_ptr(),
            "image_mask": runtime.image_token_mask.data_ptr(),
            "prepared": runtime._prepared,
            "offload_steps": runtime.offload._step,
            "graph_replays": self.full_graph_replays,
            "operator_dispatch": dict(self.full_operator_dispatch),
            "graph_dispatch": self._inspect_graph_dispatch(),
            "encoder_spans": list(self.full_encoder_spans),
            "memory": process_memory(),
            "device_allocated_bytes": torch.npu.memory_allocated(),
            "device_reserved_bytes": torch.npu.memory_reserved(),
            "device_peak_allocated_bytes_since_worker_start": torch.npu.max_memory_allocated(),
            "device_peak_reserved_bytes_since_worker_start": torch.npu.max_memory_reserved(),
        }

    def finish_full_model_audit(self):
        import gc

        runtime = self.model_runner.engram_runtime
        runtime.shutdown()
        released = [shard.weight is None and shard._pinned_owner is None for shard in runtime.offload.shards]
        events = self.full_load_records["registration_events"]
        unregisters = sum(event["event"] == "unregistered" for event in events)
        assert runtime._closed and all(released) and len(released) == unregisters == 2
        runtime.shutdown()
        for owner, name, original in reversed(self._full_observer_restore):
            setattr(owner, name, original)
        self._full_observer_restore.clear()
        self._full_graph_owners.clear()
        gc.collect()
        return {
            "closed": True,
            "owners_released": released,
            "checked_unregisters": unregisters,
            "registration_events": events,
            "memory": process_memory(),
        }
