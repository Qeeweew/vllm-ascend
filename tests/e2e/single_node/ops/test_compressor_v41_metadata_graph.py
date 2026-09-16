# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from vllm.v1.kv_cache_interface import CircularBufferSpec

from vllm_ascend.models.deepseek_v4.compressor import CompressorV41MetadataBuilder
from vllm_ascend.worker.v41_metadata import V41MetadataPreparation


def test_ring_preparation_graph_replays_updated_boundaries_and_pages():
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=16))
    spec = CircularBufferSpec(block_size=8, num_kv_heads=1, head_size=1024, head_size_v=0, dtype=torch.float32)
    builders = [CompressorV41MetadataBuilder(spec, ["ring"], config, device) for _ in range(3)]
    reference = CompressorV41MetadataBuilder(spec, ["ring"], config, torch.device("cpu"))
    owner = V41MetadataPreparation()
    inputs = [
        SimpleNamespace(
            positions=torch.zeros(8, dtype=torch.int64, device=device),
            query_start_loc=torch.zeros(5, dtype=torch.int32, device=device),
            seq_lens=torch.zeros(4, dtype=torch.int32, device=device),
            block_table_tensor=torch.zeros((4, 1), dtype=torch.int32, device=device),
            slot_mapping=torch.empty(8, dtype=torch.int64, device=device),
            num_reqs=4,
        )
        for _ in builders
    ]
    consumer = torch.npu.NPUGraph()
    for step in range(8):
        expected = []
        for group, common in enumerate(inputs):
            cpu = SimpleNamespace(
                positions=torch.arange(8, dtype=torch.int64) + step,
                query_start_loc=torch.tensor([0, 2, 2, 5, 8] if step % 2 else [0, 1, 3, 3, 6], dtype=torch.int32),
                block_table_tensor=torch.tensor([[group + step], [7], [-1], [2]], dtype=torch.int32),
                slot_mapping=torch.empty(8, dtype=torch.int64),
            )
            cpu.positions[4] = -1
            for name in ("positions", "query_start_loc", "block_table_tensor"):
                getattr(common, name).copy_(getattr(cpu, name))
            result = reference.build(0, cpu)
            expected.append((result.slot_mapping.clone(), result.token_to_req_indices.clone()))
        batch = owner.batch(capture=step == 0, use_graph=True)
        outputs = [builder.build(0, common, preparation=batch) for builder, common in zip(builders, inputs)]
        if step == 0:
            batch.run()
            with torch.npu.graph(consumer, stream=torch.npu.Stream(device=device)):
                observed = [(x.slot_mapping.clone(), x.token_to_req_indices.clone()) for x in outputs]
        else:
            with patch.object(batch, "_refresh", side_effect=AssertionError("Unexpected eager metadata refresh")):
                batch.run()
        consumer.replay()
        for actual, want in zip(observed, expected):
            for value, reference_value in zip(actual, want):
                torch.testing.assert_close(value.cpu(), reference_value, rtol=0, atol=0)
    assert len(owner.graphs) == 1
