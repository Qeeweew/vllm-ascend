# SPDX-License-Identifier: Apache-2.0
"""Observation-only worker for the unmodified full production checkpoint."""

from full_engram_audit import audit_loading, inspect_tables, process_memory

from vllm_ascend.worker.worker import NPUWorker


class V41FullModelWorker(NPUWorker):
    def load_model(self):
        with audit_loading() as records:
            self.full_load_records = records
            super().load_model()

    def start_full_model_audit(self, source):
        model = self.model_runner.get_model()
        language = model.get_language_model()
        assert len(language.model.layers) == 40
        assert language.config.engram_num_embeddings == [384006168, 384016682]
        assert model.vision is None and model.aligner is None
        runtime = self.model_runner.engram_runtime
        reports, _ = inspect_tables(runtime, source, self.vllm_config.model_config.model, self.full_load_records)
        self.full_graph_replays = 0
        from vllm.forward_context import get_forward_context

        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        original = ACLGraphWrapper.__call__

        def observe_replay(wrapper, *args, **kwargs):
            context = get_forward_context()
            entry = wrapper.concrete_aclgraph_entries.get(context.batch_descriptor)
            replay = context.cudagraph_runtime_mode == wrapper.runtime_mode and entry is not None
            replay = replay and entry.aclgraph is not None
            result = original(wrapper, *args, **kwargs)
            self.full_graph_replays += int(replay)
            return result

        ACLGraphWrapper.__call__ = observe_replay
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
            "memory": process_memory(),
            "device_allocated_bytes": torch.npu.memory_allocated(),
            "device_reserved_bytes": torch.npu.memory_reserved(),
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
        gc.collect()
        return {
            "closed": True,
            "owners_released": released,
            "checked_unregisters": unregisters,
            "registration_events": events,
            "memory": process_memory(),
        }
