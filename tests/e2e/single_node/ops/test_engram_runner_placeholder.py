# SPDX-License-Identifier: Apache-2.0
"""Real NPU snapshot before inherited clamp/embedding; no model admission change."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ops.engram_hash import HostEngramHasher, HostEngramLayout
from vllm_ascend.ops.engram_offload import EngramOffloadManager, EngramTableShard
from vllm_ascend.worker.engram_history import EngramRequestHistory
from vllm_ascend.worker.engram_runtime import EngramRuntime
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@torch.inference_mode()
def test_npu_placeholder_rejected_before_embedding_and_corrected_zero_graph_replays():
    # Explicitly use the allocated NPU 1; this test does not initialize TP/HCCL.
    torch.npu.set_device(1)
    device = torch.device("npu:1")
    config = SimpleNamespace(
        engram_layer_ids=[1],
        engram_max_ngram_size=3,
        engram_n_heads=2,
        engram_vocab_size=11,
        engram_num_embeddings=[1000],
        engram_head_dim=256,
    )
    layout = HostEngramLayout.from_config(config)
    hasher = HostEngramHasher(layout, torch.arange(32), 32, 0)
    history = EngramRequestHistory(hasher)
    history.reset_request("a", torch.tensor([1, 2]))
    heads, ranges = layout.head_shard(0, 0, 1)
    table = torch.arange(ranges[-1][1] * 256).reshape(-1, 256).remainder(127).bfloat16()
    shard = EngramTableShard(torch.empty_like(table, pin_memory=True).copy_(table), heads, ranges)
    runtime = EngramRuntime(history, EngramOffloadManager([shard], 2, device))
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.engram_runtime = runtime
    runner.input_ids = SimpleNamespace(gpu=torch.full((2,), -1, dtype=torch.int64, device=device))
    runner.positions = torch.tensor([2, 99], device=device)
    runner.query_start_loc = SimpleNamespace(gpu=torch.tensor([0, 1, 2], device=device))
    runner.input_batch = SimpleNamespace(req_ids=["a"], num_reqs=1)
    runner.inputs_embeds = SimpleNamespace(gpu=torch.zeros(2, 8, device=device))
    runner.speculative_config = object()
    runner.supports_mm_inputs = True
    runner.enable_prompt_embeds = False
    runner.uses_mrope = False
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.encoder_cache = {}
    runner.maybe_get_ec_connector_output = lambda *args, **kwargs: nullcontext(None)
    runner._execute_mm_encoder = Mock()
    runner._gather_mm_embeddings = Mock(return_value=(None, None))
    runner._init_model_kwargs = Mock(return_value={})
    runner._extract_mm_kwargs = Mock(return_value={})
    embedding = torch.arange(32 * 8, dtype=torch.float32, device=device).reshape(32, 8)
    runner.model = SimpleNamespace(
        requires_raw_input_tokens=True,
        embed_input_ids=Mock(side_effect=lambda ids, **kwargs: torch.nn.functional.embedding(ids, embedding)),
    )
    scheduler = SimpleNamespace(total_num_scheduled_tokens=1, scheduled_spec_decode_tokens={"a": [-1]})
    outputs, expected = [], []
    ptrs = (runtime.offload.device_rows[0].data_ptr(), runtime.token_mask.data_ptr())
    try:
        with patch("vllm.v1.worker.gpu_model_runner.get_pp_group", return_value=SimpleNamespace(is_first_rank=True)):
            runner._sanitize_placeholder_input_ids_for_forward(scheduler, 2)
            with pytest.raises(ValueError, match="Actual token IDs|placeholders"):
                runner._preprocess(scheduler, 2)
            runner.model.embed_input_ids.assert_not_called()
            assert runner.input_ids.gpu.cpu().tolist() == [-1, -1]
            assert not runtime._prepared

            graph = torch.npu.NPUGraph()
            # torch.npu.graph caches one default stream across devices. This
            # test may follow tests on NPU 0, so capture on this device explicitly.
            capture_stream = torch.npu.Stream(device=device)
            for step, token in enumerate((0, 3, 0, 4)):
                # Queue final device correction immediately before the snapshot.
                # Scheduler placeholders remain unchanged; graph padding is -1.
                runner.input_ids.gpu.fill_(-1)
                runner.input_ids.gpu[0] = token
                runner.positions.copy_(torch.tensor([2, 99]))
                runner._sanitize_placeholder_input_ids_for_forward(scheduler, 2)
                result = runner._preprocess(scheduler, 2)
                assert runtime._prepared
                runtime.wait_ready()
                if step == 0:
                    for _ in range(3):
                        torch.where(runtime.token_mask[:, None, None], result[4]["engram_rows"][0], 0)
                    torch.npu.synchronize()
                    with torch.npu.graph(graph, stream=capture_stream):
                        output = torch.where(runtime.token_mask[:, None, None], result[4]["engram_rows"][0], 0)
                graph.replay()
                outputs.append(output.clone())
                runtime.mark_consumed()
                hashes = hasher.hash_chunk(
                    torch.tensor([1, 2, token]), [0, 3], [0], torch.full((1, 2), -1, dtype=torch.int64)
                )[0]
                want = torch.zeros_like(output, device="cpu")
                want[0] = table[hashes[-1]]
                expected.append(want)
                assert (runtime.offload.device_rows[0].data_ptr(), runtime.token_mask.data_ptr()) == ptrs
                assert runner.input_ids.gpu.cpu().tolist() == [token, 0]
                torch.testing.assert_close(runner.inputs_embeds.gpu[0], embedding[token], rtol=0, atol=0)
            assert runner.model.embed_input_ids.call_count == 4
        runtime.close()
        for actual, want in zip(outputs, expected):
            torch.testing.assert_close(actual.cpu(), want, rtol=0, atol=0)
    finally:
        runtime.shutdown()
