# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

from vllm_ascend.ops.engram_hash import HostEngramHasher, HostEngramLayout
from vllm_ascend.worker.engram_history import EngramRequestHistory
from vllm_ascend.worker.engram_runtime import EngramRuntime
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def runner_with_runtime():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.engram_runtime = Mock()
    runner.engram_runtime.image_token_mask = torch.zeros(16, dtype=torch.bool)
    runner.model = SimpleNamespace()
    return runner


@pytest.mark.parametrize("has_runtime", [False, True])
def test_runner_shutdown_releases_host_tables_before_upstream_device_cleanup(has_runtime):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    events = []
    runtime = Mock()
    runtime.shutdown.side_effect = lambda: events.append("host")
    if has_runtime:
        runner.engram_runtime = runtime
    with patch(
        "vllm_ascend.worker.model_runner_v1.GPUModelRunner.shutdown",
        side_effect=lambda: events.append("upstream"),
    ):
        runner.shutdown()
        runner.shutdown()
    assert events == (["host", "upstream", "upstream"] if has_runtime else ["upstream", "upstream"])
    if has_runtime:
        assert runner.engram_runtime is None


def test_runner_shutdown_keeps_unregister_error_visible_and_runs_upstream_cleanup():
    runner = runner_with_runtime()
    runtime = runner.engram_runtime
    runtime.shutdown.side_effect = RuntimeError("checked unregister failed")
    with (
        patch("vllm_ascend.worker.model_runner_v1.GPUModelRunner.shutdown") as upstream,
        pytest.raises(RuntimeError, match="checked unregister"),
    ):
        runner.shutdown()
    upstream.assert_called_once()
    assert runner.engram_runtime is runtime


@pytest.mark.parametrize(
    "supports_mm,raw,encoder,prompt_embeds,want_ids,want_embeds",
    [
        (True, True, False, False, True, True),
        (True, False, False, False, False, True),
        (False, False, False, True, False, True),
        (False, False, False, False, True, False),
        (True, True, True, False, True, False),
    ],
)
def test_dummy_capture_matches_runtime_input_buffer_signature(
    supports_mm, raw, encoder, prompt_embeds, want_ids, want_embeds
):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.supports_mm_inputs = supports_mm
    runner.model_config = SimpleNamespace(is_encoder_decoder=encoder)
    runner.model = SimpleNamespace(requires_raw_input_tokens=raw)
    runner.enable_prompt_embeds = prompt_embeds
    runner.input_ids = SimpleNamespace(gpu=torch.arange(8))
    runner.inputs_embeds = SimpleNamespace(gpu=torch.arange(32).reshape(8, 4))
    token_ids, embeddings = runner._dummy_input_buffers(4)
    assert (token_ids is not None) == want_ids
    assert (embeddings is not None) == want_embeds
    if want_ids:
        assert token_ids.shape == (4,)
        assert token_ids.data_ptr() == runner.input_ids.gpu.data_ptr()
    if want_embeds:
        assert embeddings.shape == (4, 4)
        assert embeddings.data_ptr() == runner.inputs_embeds.gpu.data_ptr()
    if supports_mm and not encoder:
        actual_ids, actual_embeds = runner._prepare_mm_inputs(4)
        assert (actual_ids is None) == (token_ids is None)
        assert actual_embeds.data_ptr() == embeddings.data_ptr()
        runner.inputs_embeds.gpu[0].fill_(99)
        assert embeddings[0].tolist() == [99] * 4


def request(request_id, tokens, mm_features=()):
    return SimpleNamespace(req_id=request_id, prompt_token_ids=tokens, mm_features=mm_features)


def scheduler_output(new_requests=(), finished=(), preempted=()):
    return SimpleNamespace(
        scheduled_new_reqs=new_requests,
        finished_req_ids=finished,
        preempted_req_ids=preempted,
    )


def actual_history():
    config = SimpleNamespace(
        engram_layer_ids=[1],
        engram_max_ngram_size=3,
        engram_n_heads=2,
        engram_vocab_size=11,
        engram_num_embeddings=[1000],
        engram_head_dim=4,
    )
    hasher = HostEngramHasher(HostEngramLayout.from_config(config), torch.arange(32), 32, 0)
    return EngramRequestHistory(hasher)


@pytest.mark.parametrize("has_attribute", [False, True])
def test_old_models_do_not_access_scheduler_or_input_buffers(has_attribute):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    if has_attribute:
        runner.engram_runtime = None
    untouched = {"other": object()}
    original = untouched.copy()
    runner._update_engram_requests(None)
    runner._prepare_engram_model_kwargs(None, -1, untouched)
    assert untouched == original


def test_finished_requests_are_dropped_before_same_id_is_reseeded():
    runner = runner_with_runtime()
    runner._update_engram_requests(
        scheduler_output([request("reused", [4, 5, 6])], finished=["finished", "reused"], preempted=["paused"])
    )
    calls = runner.engram_runtime.history.mock_calls
    assert calls[:2] == [call.drop_request("finished"), call.drop_request("reused")]
    assert len(calls) == 3 and calls[2][0] == "reset_request"
    args, kwargs = runner.engram_runtime.history.reset_request.call_args
    assert args[0] == "reused"
    assert args[1].dtype == torch.int64 and args[1].device.type == "cpu"
    assert args[1].tolist() == [4, 5, 6]
    assert kwargs == {"prompt_mask": None, "prompt_image_mask": None}


def test_preemption_retains_executed_history_and_finished_id_reuse_resets_it():
    runner = runner_with_runtime()
    history = actual_history()
    runner.engram_runtime.history = history
    runner._update_engram_requests(scheduler_output([request("a", [1, 2]), request("b", [10, 11])]))
    history.prepare(["a"], torch.tensor([3, 4]), torch.tensor([2, 3]), [0, 2])
    runner._update_engram_requests(scheduler_output(preempted=["a"]))
    # Preserved actual tail supports resumed decode; absent history would fail.
    history.prepare(["a"], torch.tensor([5]), torch.tensor([4]), [0, 1])
    runner._update_engram_requests(scheduler_output([request("a", [20, 21])], finished=["a", "b"]))
    assert "a" in history and "b" not in history
    with pytest.raises(ValueError, match="Missing actual"):
        history.prepare(["a"], torch.tensor([6]), torch.tensor([5]), [0, 1])
    history.prepare(["a"], torch.tensor([22]), torch.tensor([2]), [0, 1])


def test_missing_actual_prompt_is_rejected_before_history_reset():
    runner = runner_with_runtime()
    with pytest.raises(ValueError, match="actual prompt token IDs"):
        runner._update_engram_requests(scheduler_output([request("embedding_only", None)]))
    runner.engram_runtime.history.reset_request.assert_not_called()


def test_multimodal_prompt_requires_mask_provider():
    runner = runner_with_runtime()
    with pytest.raises(NotImplementedError, match="image-span Engram masks"):
        runner._update_engram_requests(scheduler_output([request("image", [1, 2], [object()])]))
    runner.engram_runtime.history.reset_request.assert_not_called()


def test_multimodal_mask_is_forwarded_unchanged_and_text_does_not_call_provider():
    runner = runner_with_runtime()
    mask = torch.tensor([True, False, True])
    provider = Mock(return_value=mask)
    runner.model.engram_prompt_mask = provider
    image = request("image", [1, 2, 3], [object()])
    runner._update_engram_requests(scheduler_output([image, request("text", [4, 5])]))
    provider.assert_called_once_with(image)
    resets = runner.engram_runtime.history.reset_request.call_args_list
    assert resets[0].kwargs["prompt_mask"] is mask
    assert resets[0].kwargs["prompt_image_mask"].tolist() == [False, True, False]
    assert resets[1].kwargs["prompt_mask"] is None
    assert resets[1].kwargs["prompt_image_mask"] is None


def test_invalid_multimodal_mask_is_rejected_by_actual_history():
    runner = runner_with_runtime()
    runner.engram_runtime.history = actual_history()
    runner.model.engram_prompt_mask = lambda _: torch.tensor([True])
    with pytest.raises(ValueError):
        runner._update_engram_requests(scheduler_output([request("image", [1, 2], [object()])]))
    assert "image" not in runner.engram_runtime.history


@pytest.mark.parametrize("bucket,query_ends", [(4, [0, 1, 2]), (8, [0, 3, 5]), (8, [0, 0, 5])])
def test_prepare_uses_final_device_rows_real_request_order_and_boundaries(bucket, query_ends):
    runner = runner_with_runtime()
    # CPU tensors stand in for .gpu buffers; deliberately wrong CPU mirrors
    # prove the runner never reads optimistic asynchronous placeholders.
    final_ids = torch.arange(100, 116)
    final_positions = torch.arange(200, 200 + bucket)
    final_boundaries = torch.tensor(query_ends + [bucket, bucket], dtype=torch.int32)
    runner.input_ids = SimpleNamespace(gpu=final_ids, cpu=torch.full((16,), -1))
    runner.query_start_loc = SimpleNamespace(gpu=final_boundaries, cpu=torch.tensor([0, 99, 199]))
    runner.input_batch = SimpleNamespace(req_ids=["b", "a"], num_reqs=2, token_ids_cpu=None)
    rows = (torch.empty(bucket, 2, 4), torch.empty(bucket, 2, 4))
    out_mask = torch.zeros(bucket, dtype=torch.bool)
    in_mask = torch.ones(bucket, dtype=torch.bool)
    kwargs = {"engram_token_mask": in_mask, "other": object()}
    original_other = kwargs["other"]

    def prepare(req_ids, input_ids, positions, boundaries, bucket_tokens, *, token_mask):
        assert req_ids is runner.input_batch.req_ids
        assert req_ids == ["b", "a"]
        assert input_ids.shape == (bucket,) and input_ids.data_ptr() == final_ids.data_ptr()
        assert input_ids.tolist() == list(range(100, 100 + bucket))
        assert positions is final_positions
        assert boundaries.tolist() == query_ends
        assert boundaries.data_ptr() == final_boundaries.data_ptr()
        assert bucket_tokens == bucket and token_mask is in_mask
        return rows, out_mask

    runner.engram_runtime.prepare.side_effect = prepare
    runner.engram_runtime.image_token_mask[1] = True

    runner._prepare_engram_model_kwargs(final_positions, bucket, kwargs)
    assert kwargs["engram_rows"] is rows and kwargs["engram_token_mask"] is out_mask
    assert kwargs["image_token_mask"].data_ptr() == runner.engram_runtime.image_token_mask.data_ptr()
    assert kwargs["image_token_mask"].shape == (bucket,)
    assert kwargs["image_token_mask"][1]
    assert kwargs["other"] is original_other
    assert [c[0] for c in runner.engram_runtime.mock_calls] == ["prepare"]
    runner.engram_runtime.mark_consumed.assert_not_called()


def test_prepare_without_input_mask_forwards_none():
    runner = runner_with_runtime()
    runner.input_ids = SimpleNamespace(gpu=torch.tensor([5, 0]))
    runner.query_start_loc = SimpleNamespace(gpu=torch.tensor([0, 1, 2]))
    runner.input_batch = SimpleNamespace(req_ids=["a"], num_reqs=1)
    runner.engram_runtime.prepare.return_value = ((), torch.tensor([True, False]))
    kwargs = {}
    runner._prepare_engram_model_kwargs(torch.tensor([7, 0]), 2, kwargs)
    assert runner.engram_runtime.prepare.call_args.kwargs["token_mask"] is None
    assert kwargs["engram_token_mask"].tolist() == [True, False]


def test_prepare_failure_does_not_publish_rows_or_wait():
    runner = runner_with_runtime()
    runner.input_ids = SimpleNamespace(gpu=torch.tensor([5]))
    runner.query_start_loc = SimpleNamespace(gpu=torch.tensor([0, 1]))
    runner.input_batch = SimpleNamespace(req_ids=["a"], num_reqs=1)
    runner.engram_runtime.prepare.side_effect = ValueError("missing actual history")
    kwargs = {"other": object()}
    original = kwargs.copy()
    with pytest.raises(ValueError, match="missing actual history"):
        runner._prepare_engram_model_kwargs(torch.tensor([7]), 1, kwargs)
    assert kwargs == original
    runner.engram_runtime.wait_ready.assert_not_called()
    runner.engram_runtime.mark_consumed.assert_not_called()


@pytest.mark.parametrize("has_runtime", [False, True])
def test_sanitizer_preserves_engram_actual_ids_and_legacy_behavior(has_runtime):
    runner = runner_with_runtime()
    if not has_runtime:
        runner.engram_runtime = None
    runner.input_ids = SimpleNamespace(gpu=torch.tensor([3, -1, 0, 7, -1, -1]))
    scheduler = SimpleNamespace(scheduled_spec_decode_tokens={"a": [-1, -1]})
    runner._sanitize_placeholder_input_ids_for_forward(scheduler, 5)
    expected = [3, -1, 0, 7, -1, -1] if has_runtime else [3, 0, 0, 7, 0, -1]
    assert runner.input_ids.gpu.tolist() == expected


def test_engram_unresolved_actual_placeholder_is_rejected_without_history_commit():
    runner = runner_with_runtime()
    history = actual_history()
    history.reset_request("a", torch.tensor([1, 2]))
    runner.input_ids = SimpleNamespace(gpu=torch.tensor([-1, -1]))
    runner.query_start_loc = SimpleNamespace(gpu=torch.tensor([0, 1, 2]))
    runner.input_batch = SimpleNamespace(req_ids=["a"], num_reqs=1)
    scheduler = SimpleNamespace(scheduled_spec_decode_tokens={"a": [-1]})
    positions = torch.tensor([2, 0])

    def prepare(req_ids, input_ids, positions, boundaries, bucket_tokens, *, token_mask):
        count = int(boundaries[-1])
        batch = history.prepare(req_ids, input_ids[:count], positions[:count], boundaries, token_mask=token_mask)
        return (), batch.token_mask

    runner.engram_runtime.prepare.side_effect = prepare
    runner._sanitize_placeholder_input_ids_for_forward(scheduler, 2)
    with pytest.raises(ValueError, match="Actual token IDs|placeholders"):
        runner._prepare_engram_model_kwargs(positions, 2, {})
    runner.engram_runtime.wait_ready.assert_not_called()
    # The scheduler still has an async placeholder after the real device ID
    # arrives. Accept the actual row; padding -1 remains outside query bounds.
    runner.input_ids.gpu[0] = 3
    runner._sanitize_placeholder_input_ids_for_forward(scheduler, 2)
    kwargs = {}
    runner._prepare_engram_model_kwargs(positions, 2, kwargs)
    assert runner.input_ids.gpu.tolist() == [3, -1]
    assert kwargs["engram_token_mask"].tolist() == [True]
    runner.engram_runtime.wait_ready.assert_not_called()
    history.prepare(["a"], torch.tensor([4]), torch.tensor([3]), [0, 1])


def preprocess_runner(active_token, *, has_runtime=True, embedding_error=False):
    """Use the real upstream speculative clamp and multimodal embedding path."""
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.input_ids = SimpleNamespace(gpu=torch.tensor([active_token, -1]), cpu=torch.tensor([-1, -1]))
    runner.positions = torch.tensor([2, 99])
    runner.query_start_loc = SimpleNamespace(gpu=torch.tensor([0, 1, 2]))
    runner.input_batch = SimpleNamespace(req_ids=["a"], num_reqs=1)
    runner.inputs_embeds = SimpleNamespace(gpu=torch.zeros(2, 4))
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
    runner.model = SimpleNamespace(
        requires_raw_input_tokens=True,
        embed_input_ids=Mock(
            side_effect=RuntimeError("embedding failed") if embedding_error else None,
            return_value=torch.ones(1, 4),
        ),
    )
    runner.engram_runtime = None
    if has_runtime:
        history = actual_history()
        history.reset_request("a", torch.tensor([1, 2]))
        offload = Mock(shards=[object()], max_tokens=2, device=torch.device("cpu"))
        offload.prepare.return_value = (torch.zeros(2, 2, 4),)
        runner.engram_runtime = EngramRuntime(history, offload)
    scheduler = SimpleNamespace(total_num_scheduled_tokens=1, scheduled_spec_decode_tokens={"a": [-1]})
    return runner, scheduler


def test_real_preprocess_rejects_placeholder_before_clamp_embedding_or_hash(monkeypatch):
    runner, scheduler = preprocess_runner(-1)
    runtime = runner.engram_runtime
    hash_chunk = Mock(wraps=runtime.history.hasher.hash_chunk)
    monkeypatch.setattr(runtime.history.hasher, "hash_chunk", hash_chunk)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    runner._sanitize_placeholder_input_ids_for_forward(scheduler, 2)
    with (
        patch("vllm.v1.worker.gpu_model_runner.get_pp_group", return_value=SimpleNamespace(is_first_rank=True)),
        pytest.raises(ValueError, match="Actual token IDs|placeholders"),
    ):
        runner._preprocess(scheduler, 2)
    assert runner.input_ids.gpu.tolist() == [-1, -1]
    runner.model.embed_input_ids.assert_not_called()
    hash_chunk.assert_not_called()
    runtime.offload.prepare.assert_not_called()
    assert not runtime._prepared
    # Rejection did not commit token 0 or an invalid history position.
    runtime.history.prepare(["a"], torch.tensor([3]), torch.tensor([2]), [0, 1])


@pytest.mark.parametrize("active_token", [0, 3])
def test_real_preprocess_uses_one_snapshot_before_clamp_and_one_forward_lifecycle(active_token, monkeypatch):
    runner, scheduler = preprocess_runner(active_token)
    runtime = runner.engram_runtime
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    snapshots = []
    original_cpu = torch.Tensor.cpu

    def snapshot_cpu(tensor, *args, **kwargs):
        snapshots.append(tensor.clone())
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", snapshot_cpu)
    with patch("vllm.v1.worker.gpu_model_runner.get_pp_group", return_value=SimpleNamespace(is_first_rank=True)):
        runner._sanitize_placeholder_input_ids_for_forward(scheduler, 2)
        result = runner._preprocess(scheduler, 2)
    assert len(snapshots) == 1
    assert snapshots[0].tolist() == [active_token, -1, 2, 99, 0, 1]
    assert runner.input_ids.gpu.tolist() == [active_token, 0]
    assert runner.positions.tolist() == [2, 0]
    assert runner.model.embed_input_ids.call_args.args[0].tolist() == [active_token]
    assert result[4]["engram_token_mask"].tolist() == [True, False]
    assert runtime._prepared
    runtime.offload.prepare.assert_called_once()
    runtime.offload.wait_ready.assert_not_called()
    runtime.wait_ready()
    runtime.mark_consumed()
    runtime.offload.wait_ready.assert_called_once()
    runtime.offload.mark_consumed.assert_called_once()
    assert not runtime._prepared


def test_real_preprocess_keeps_legacy_clamp_without_engram(monkeypatch):
    runner, scheduler = preprocess_runner(-1, has_runtime=False)
    with patch("vllm.v1.worker.gpu_model_runner.get_pp_group", return_value=SimpleNamespace(is_first_rank=True)):
        runner._sanitize_placeholder_input_ids_for_forward(scheduler, 2)
        result = runner._preprocess(scheduler, 2)
    assert runner.input_ids.gpu.tolist() == [0, 0]
    assert runner.model.embed_input_ids.call_args.args[0].tolist() == [0]
    assert "engram_rows" not in result[4]


def test_preprocess_embedding_failure_preserves_pending_runtime_for_checked_shutdown(monkeypatch):
    runner, scheduler = preprocess_runner(0, embedding_error=True)
    runtime = runner.engram_runtime
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    with (
        patch("vllm.v1.worker.gpu_model_runner.get_pp_group", return_value=SimpleNamespace(is_first_rank=True)),
        pytest.raises(RuntimeError, match="embedding failed"),
    ):
        runner._preprocess(scheduler, 2)
    assert runtime._prepared and not runtime._closed
    runtime.offload.mark_consumed.assert_not_called()
    with patch("vllm_ascend.worker.model_runner_v1.GPUModelRunner.shutdown"):
        runner.shutdown()
    runtime.offload.shutdown.assert_called_once()
    assert runtime._closed and not runtime._prepared
    assert runner.engram_runtime is None
