# SPDX-License-Identifier: Apache-2.0
"""Exercise the real common-metadata constructor through the NPU runner.

Only scheduling/buffer owners and native schedule output are lightweight test
doubles. AscendCommonAttentionMetadata, its upstream base and V4.1 metadata
builders are real, so upstream dataclass ABI changes cannot hide behind mocks.
"""

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheGroupSpec

from vllm_ascend.attention.dsa_v41 import AscendV41CacheMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import AscendV41SWACacheSpec
from vllm_ascend.models.deepseek_v4.compressor import CompressorV41Metadata
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def make_runner(asynchronous):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.max_model_len = 128
    runner.optimistic_seq_lens_cpu = torch.tensor([7, 11, 0], dtype=torch.int32)
    runner.seq_lens = torch.tensor([5, 9, 0] if asynchronous else [7, 11, 0], dtype=torch.int32)
    runner.query_start_loc = SimpleNamespace(
        gpu=torch.tensor([0, 2, 4, 4], dtype=torch.int32),
        cpu=torch.tensor([0, 1, 4, 4] if asynchronous else [0, 2, 4, 4], dtype=torch.int32),
    )
    table = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0]], dtype=torch.int32)
    slots = torch.full((8,), 123, dtype=torch.int64)
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu_tensor=torch.tensor([5, 9, 0], dtype=torch.int32),
        num_prompt_tokens_cpu_tensor=torch.tensor([7, 11, 0], dtype=torch.int32),
        block_table=[SimpleNamespace(slot_mapping=SimpleNamespace(gpu=slots), get_device_tensor=lambda: table)],
    )
    spec = AscendV41SWACacheSpec(
        block_size=32, num_kv_heads=1, head_size=512, head_size_v=0, dtype=torch.bfloat16, sliding_window=128
    )
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[KVCacheGroupSpec(["layer.swa"], spec)])
    # Returning the real common object exposes exactly what the runner passed
    # to its backend. No constructor monkeypatch is involved.
    builder = SimpleNamespace(build=lambda **kwargs: kwargs["common_attn_metadata"])
    runner.attn_groups = [[SimpleNamespace(layer_names=["layer.swa"], get_metadata_builder=lambda _: builder)]]
    runner.model_config = SimpleNamespace(enable_return_routed_experts=False)
    runner.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8, max_num_seqs=3),
        model_config=SimpleNamespace(max_model_len=128, hf_config=SimpleNamespace(num_attention_heads=64)),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
    )
    runner.cache_config = SimpleNamespace(kv_sharing_fast_prefill=False)
    runner.sparse_kv_offload_enabled = False
    runner.device_metadata_executor = runner.device_metadata_providers = None
    runner.dcp_size = 1
    runner.is_mm_prefix_lm = runner.use_compress = False
    runner.use_async_spec_decode = asynchronous
    runner.actual_seq_lengths_q = [2, 4]
    runner.positions = torch.tensor([3, 4, 7, 8, 0, 0, 0, 0], dtype=torch.int64)
    runner.attn_state = None
    runner.decode_token_per_req = 1
    runner.group_len = runner.group_key_idx = runner.group_key_cache_idx = SimpleNamespace(
        gpu=torch.zeros(3, dtype=torch.int32)
    )
    runner._offload_req_ids_tensor = runner._offload_token_to_req = None
    runner._get_encoder_seq_lens = lambda *args: (None, None)
    runner._has_gdn = False
    # Select the generic speculative-common metadata return path so padding
    # also exercises the real Ascend unpadded() implementation.
    runner.speculative_config = SimpleNamespace()
    runner.drafter = SimpleNamespace()
    return runner


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_runner_constructs_real_metadata_and_keeps_host_bounds_separate(asynchronous, padded):
    runner = make_runner(asynchronous)
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("metadata must not copy device lengths to host")):
        metadata, draft = runner._build_attention_metadata(
            num_tokens=4,
            num_reqs=2,
            max_query_len=2,
            num_tokens_padded=8 if padded else 4,
            num_reqs_padded=3 if padded else 2,
        )
    common = metadata["layer.swa"]
    assert isinstance(common, AscendCommonAttentionMetadata)
    assert isinstance(common, CommonAttentionMetadata)
    assert common.num_reqs == (3 if padded else 2)
    assert common.seq_lens.data_ptr() == runner.seq_lens.data_ptr()
    assert common._seq_lens_cpu.data_ptr() == runner.optimistic_seq_lens_cpu.data_ptr()
    assert common.seq_lens_cpu_upper_bound.data_ptr() == runner.optimistic_seq_lens_cpu.data_ptr()
    assert draft.num_reqs == 2 and draft.seq_lens.shape == (2,)
    assert draft._seq_lens_cpu.tolist() == draft.seq_lens_cpu_upper_bound.tolist() == [7, 11]
    if asynchronous:
        assert common.seq_lens_cpu is common.num_computed_tokens_cpu is None
        assert draft.seq_lens_cpu is draft.num_computed_tokens_cpu is None
        assert common.seq_lens[:2].tolist() == [5, 9]
    else:
        assert common.seq_lens_cpu[:2].tolist() == [7, 11]
        assert common.num_computed_tokens_cpu[:2].tolist() == [5, 9]
    if padded:
        assert common.slot_mapping[4:].tolist() == [-1] * 4
        assert draft.block_table_tensor is common.block_table_tensor
        assert draft.slot_mapping is common.slot_mapping
    # Current upstream computes exact values from DEVICE boundaries, not the
    # optimistic host arrays or the stale CPU per-request draft partition.
    if hasattr(draft, "compute_num_computed_tokens"):
        assert draft.compute_num_computed_tokens().tolist() == ([3, 7] if asynchronous else [5, 9])


def test_v41_builder_consumes_real_common_device_lengths_and_boundaries():
    runner = make_runner(asynchronous=True)
    common = runner._build_attention_metadata(4, 2, 2, num_tokens_padded=8, num_reqs_padded=3)[0]["layer.swa"]
    spec = runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec
    builder = AscendV41CacheMetadataBuilder(spec, ["layer.swa"], runner.vllm_config, runner.device)
    with patch(
        "torch.ops._C_ascend.npu_sparse_flash_mla_metadata",
        create=True,
        return_value=torch.zeros(1024, dtype=torch.int32),
    ) as schedule:
        metadata = builder.build(0, common)
    assert metadata.seqused_kv.tolist() == [5, 9, 0]
    assert metadata.cu_seqlens_q.tolist() == [0, 2, 4, 4]
    assert metadata.token_to_req_indices.tolist() == [0, 0, 1, 1, -1, -1, -1, -1]
    assert metadata.slot_mapping.tolist() == [35, 36, 167, 168, -1, -1, -1, -1]
    assert schedule.call_args.kwargs["seqused_ori_kv"] is metadata.seqused_kv


@pytest.mark.parametrize("has_image", [False, True])
@pytest.mark.parametrize("config_kind", ["explicit", "deepseek_v41", "deepseek_v41_text"])
def test_mm_ranges_leave_frozen_compressor_metadata_untouched(has_image, config_kind):
    runner = make_runner(asynchronous=False)
    runner.is_mm_prefix_lm = True
    runner.model_config.hf_text_config = (
        SimpleNamespace(mm_prefix_span_leading_pad_modulus=0)
        if config_kind == "explicit"
        else SimpleNamespace(model_type=config_kind, vision_n_layers=32)
    )
    runner.input_batch.req_ids = ["first", "second"]
    runner.input_batch.req_id_to_index = {"first": 0, "second": 1}
    feature = SimpleNamespace(modality="image", mm_position=SimpleNamespace(extract_embeds_range=lambda: [(1, 3)]))
    runner.requests = {
        "first": SimpleNamespace(mm_features=[feature] if has_image else []),
        "second": SimpleNamespace(mm_features=[]),
    }
    state = CompressorV41Metadata(torch.zeros(4), torch.zeros(3), torch.zeros(4))
    legacy_mask = SimpleNamespace(mm_prefix_range=None)
    for name, value in (("layer.state", state), ("layer.legacy_mask", legacy_mask)):
        builder = SimpleNamespace(build=lambda _value=value, **kwargs: _value)
        runner.attn_groups[0].append(
            SimpleNamespace(layer_names=[name], get_metadata_builder=lambda _, _builder=builder: _builder)
        )
    metadata, _ = runner._build_attention_metadata(4, 2, 2)
    expected = {0: [(1, 3)] if has_image else [], 1: []}
    assert metadata["layer.swa"].mm_req_doc_ranges == expected
    assert metadata["layer.state"] is state
    assert not hasattr(state, "mm_prefix_range")
    assert metadata["layer.legacy_mask"].mm_prefix_range == expected


def test_unpadded_preserves_new_upstream_fields_and_drops_stale_device_caches():
    runner = make_runner(asynchronous=True)
    common = runner._build_attention_metadata(4, 2, 2, num_tokens_padded=8, num_reqs_padded=3)[0]["layer.swa"]
    names = {item.name for item in fields(CommonAttentionMetadata)}
    if "dcp_local_seq_lens_cpu_upper_bound" not in names:
        pytest.skip("Installed vLLM predates explicit DCP CPU upper bounds")
    common.dcp_local_seq_lens = torch.tensor([3, 5, 0])
    common.dcp_local_seq_lens_cpu = torch.tensor([3, 5, 0])
    common.dcp_local_seq_lens_cpu_upper_bound = torch.tensor([4, 6, 0])
    common._num_computed_tokens_cpu = torch.tensor([5, 9, 0])
    common.encoder_seq_lens = torch.tensor([10, 20, 0])
    common.encoder_seq_lens_cpu = np.array([10, 20, 0])
    common.req_idx = np.array([11, 22, -1])
    common.rswa_prefix_lens = torch.tensor([10, 20, 0])
    common.replayssm_decode_base_cpu = torch.tensor([1, 2, 0])
    common.max_logits_per_req = 4
    common.mm_req_doc_ranges = {0: [(2, 4)]}
    common.causal = torch.tensor([True, False, False])
    common._num_computed_tokens_cache = torch.tensor([3, 7, 0])
    common._token_to_req_indices_cache = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.int32)
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected host copy")):
        unpadded = common.unpadded(4, 2)
    assert unpadded.dcp_local_seq_lens.tolist() == unpadded.dcp_local_seq_lens_cpu.tolist() == [3, 5]
    assert unpadded.dcp_local_seq_lens_cpu_upper_bound.tolist() == [4, 6]
    assert unpadded._num_computed_tokens_cpu.tolist() == [5, 9]
    assert unpadded.encoder_seq_lens.tolist() == unpadded.encoder_seq_lens_cpu.tolist() == [10, 20]
    assert unpadded.req_idx.tolist() == [11, 22]
    assert unpadded.rswa_prefix_lens.tolist() == [10, 20]
    assert unpadded.replayssm_decode_base_cpu.tolist() == [1, 2]
    assert unpadded.causal.tolist() == [True, False]
    assert unpadded.max_logits_per_req == 4
    assert unpadded.mm_req_doc_ranges is common.mm_req_doc_ranges
    assert unpadded._num_computed_tokens_cache is unpadded._token_to_req_indices_cache is None
    assert common._num_computed_tokens_cache.shape == (3,)
    assert common._token_to_req_indices_cache.shape == (8,)
