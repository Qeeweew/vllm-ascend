# SPDX-License-Identifier: Apache-2.0
"""Worker-owned preparation graphs for V4.1's fixed-address cache metadata."""

from dataclasses import dataclass

import torch


def _tensor_binding(tensor):
    if tensor is None:
        return None
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


@dataclass
class _PreparationGraph:
    graph: object
    # Common metadata is mutable, so retain the actual captured tensors too.
    tasks: tuple
    inputs: tuple


class V41MetadataPreparation:
    """Capture at startup; refresh dynamic contents on the model's stream.

    Graphs are keyed by all input bindings, not just batch size. An uncaptured
    binding runs the same operations eagerly, so a new tensor can never replay
    a graph that still reads its predecessor. No device values are cached.
    Builders already require one in-flight batch; preparation and model replay
    use the same stream and the same lifetime/serialization contract.
    """

    def __init__(self):
        self.graphs: dict[tuple, _PreparationGraph] = {}
        self.pool = None

    def batch(self, *, capture=False, use_graph=False):
        return V41MetadataBatch(self, capture=capture, use_graph=use_graph)


class V41MetadataBatch:
    def __init__(self, owner, *, capture, use_graph):
        self.owner = owner
        self.capture = capture
        self.use_graph = use_graph
        self.tasks = []
        self.counts = {}

    def execution_counts(self, builder, common):
        key = (
            common.num_reqs,
            common.num_actual_tokens,
            _tensor_binding(getattr(common, "query_start_loc_cpu", None)),
            _tensor_binding(getattr(common, "is_prefilling", None)),
        )
        if key not in self.counts:
            self.counts[key] = builder._execution_counts(common)
        return self.counts[key]

    def add(self, builder, metadata, common):
        self.tasks.append((builder, metadata, common))

    def _key(self):
        return tuple(
            (
                id(builder),
                metadata.positions.numel(),
                common.num_reqs,
                *(
                    _tensor_binding(getattr(common, name))
                    for name in ("positions", "query_start_loc", "seq_lens", "block_table_tensor")
                ),
            )
            for builder, metadata, common in self.tasks
        )

    def _refresh(self):
        schedules = {}
        for builder, metadata, common in self.tasks:
            builder._refresh_device(metadata, common, schedules)

    def run(self):
        if not self.tasks:
            return
        on_npu = self.tasks[0][1].positions.device.type == "npu"
        if not on_npu or not (self.capture or self.use_graph):
            self._refresh()
            return
        key = self._key()
        entry = self.owner.graphs.get(key)
        if entry is not None:
            entry.graph.replay()
        elif self.capture and not torch.npu.is_current_stream_capturing():
            # Warm up operator compilation/tiling before graph capture. All
            # outputs are overwritten; preparation never writes model caches.
            self._refresh()
            graph = torch.npu.NPUGraph()
            if self.owner.pool is None:
                self.owner.pool = torch.npu.graph_pool_handle()
            with torch.npu.graph(graph, pool=self.owner.pool):
                self._refresh()
            inputs = tuple(
                getattr(common, name)
                for _, _, common in self.tasks
                for name in ("positions", "query_start_loc", "seq_lens", "block_table_tensor")
            )
            self.owner.graphs[key] = _PreparationGraph(graph, tuple(self.tasks), inputs)
            graph.replay()
        else:
            self._refresh()
