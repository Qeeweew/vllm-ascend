# SPDX-License-Identifier: Apache-2.0
"""Host factory contract with real config, tokenizer normalization and safetensors.

Only NPU staging and the pinned allocator are replaced: these tests never
initialize a device. The real table loader still copies selected file slices.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

from vllm_ascend.models.deepseek_v4 import model as model_module
from vllm_ascend.ops import engram_offload
from vllm_ascend.ops.engram_hash import HostEngramLayout


def make_config(*, real_sizes=False):
    # Field names and production dimensions come from the release text_config.
    return DeepseekV41Config(
        text_config={
            "engram_layer_ids": [1, 14],
            "engram_num_embeddings": [384006168, 384016682] if real_sizes else [10000, 10000],
            "engram_max_ngram_size": 4,
            "engram_vocab_size": 16000000 if real_sizes else 3,
            "engram_n_heads": 8,
            "engram_head_dim": 256 if real_sizes else 4,
            "engram_pad_token_id": 2,
            "engram_compressed_vocab_size": 99092 if real_sizes else 4,
        }
    )


@pytest.fixture
def factory(tmp_path, monkeypatch):
    model = model_module.AscendDeepseekV41ForCausalLM.__new__(model_module.AscendDeepseekV41ForCausalLM)
    nn.Module.__init__(model)
    model.config = make_config()
    model.model = nn.Linear(1, 1, bias=False)
    model.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path), trust_remote_code=False),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=37),
    )
    backend = Tokenizer(WordLevel({"one": 0, "two": 1, "[PAD]": 2, "three": 3}, unk_token="[PAD]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]")
    tokenizer_loader = Mock(return_value=tokenizer)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(model_module, "get_ascend_config", lambda: SimpleNamespace(engram_numa_nodes=None))
    manager = Mock(
        side_effect=lambda shards, max_tokens, device: SimpleNamespace(
            shards=tuple(shards), max_tokens=max_tokens, device=device
        )
    )
    monkeypatch.setattr(engram_offload, "EngramOffloadManager", manager)

    # Record the requested allocator contract while avoiding NPU initialization.
    requested_pin = []
    original_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        if "pin_memory" in kwargs:
            requested_pin.append(kwargs.pop("pin_memory"))
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)
    layout = HostEngramLayout.from_config(model.config)
    tables, weight_map = [], {}
    for layer, layer_id in enumerate(layout.layer_ids):
        rows = sum(p for order in layout.primes[layer] for p in order)
        table = (torch.arange(rows * layout.head_dim).reshape(rows, layout.head_dim) % 127).bfloat16()
        name = f"layers.{layer_id}.engram.embed.weight"
        filename = f"converted-engram-{layer_id}.safetensors"
        save_file({name: table}, tmp_path / filename)
        weight_map[name] = filename
        tables.append(table)
        model.config.engram_num_embeddings[layer] = rows
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return SimpleNamespace(
        model=model,
        root=tmp_path,
        tables=tables,
        layout=layout,
        tokenizer_loader=tokenizer_loader,
        manager=manager,
        requested_pin=requested_pin,
        weight_map=weight_map,
    )


@pytest.mark.parametrize("rank", range(8))
def test_factory_loads_tp_local_heads_from_indexed_converted_files(factory, monkeypatch, rank):
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_rank", lambda: rank)
    runtime = factory.model.create_engram_runtime()
    assert runtime.history.hasher.layout == factory.layout
    assert runtime.token_mask.shape == (37,)
    factory.tokenizer_loader.assert_called_once_with(factory.root, trust_remote_code=False)
    assert factory.requested_pin == [True, True]
    factory.manager.assert_called_once()
    assert factory.manager.call_args.args[1:] == (37, torch.device("cpu"))
    for layer, shard in enumerate(runtime.offload.shards):
        assert shard.head_indices == tuple(range(rank * 3, rank * 3 + 3))
        heads, ranges = factory.layout.head_shard(layer, rank, 8)
        assert shard.head_indices == heads and shard.head_ranges == ranges
        expected = torch.cat([factory.tables[layer][start:end] for start, end in ranges])
        torch.testing.assert_close(shard.weight, expected, rtol=0, atol=0)


def test_production_head_ranges_cover_each_table_once_across_tp8():
    config = make_config(real_sizes=True)
    layout = HostEngramLayout.from_config(config)
    rank_bytes = [0] * 8
    for layer, rows in enumerate(config.engram_num_embeddings):
        cursor, seen_heads = 0, []
        for rank in range(8):
            heads, ranges = layout.head_shard(layer, rank, 8)
            seen_heads.extend(heads)
            for start, end in ranges:
                assert start == cursor
                cursor = end
                rank_bytes[rank] += (end - start) * config.engram_head_dim * 2
        assert cursor == rows
        assert seen_heads == list(range(24))
    assert sum(rank_bytes) == 393227699200
    assert max(rank_bytes) - min(rank_bytes) < 2 * 1024**2


def test_no_engram_returns_before_checkpoint_or_tokenizer_loading(factory):
    factory.model.config.engram_layer_ids = []
    (factory.root / "model.safetensors.index.json").unlink()
    assert factory.model.create_engram_runtime() is None
    factory.tokenizer_loader.assert_not_called()
    factory.manager.assert_not_called()
    assert factory.requested_pin == []


@pytest.mark.parametrize("layer_id", [1, 14])
def test_missing_table_is_fatal_before_runtime_construction(factory, layer_id):
    name = f"layers.{layer_id}.engram.embed.weight"
    del factory.weight_map[name]
    (factory.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": factory.weight_map}))
    with pytest.raises(ValueError, match=f"Missing converted Engram table {name}"):
        factory.model.create_engram_runtime()
    factory.manager.assert_not_called()
    factory.tokenizer_loader.assert_not_called()
    assert factory.requested_pin == []


def test_missing_converted_file_never_uses_source_checkpoint(factory):
    (factory.root / factory.weight_map["layers.1.engram.embed.weight"]).unlink()
    with pytest.raises(FileNotFoundError):
        factory.model.create_engram_runtime()
    factory.manager.assert_not_called()


def test_wrong_dtype_fails_before_pinned_allocation(factory):
    name = "layers.1.engram.embed.weight"
    save_file({name: factory.tables[0].float()}, factory.root / factory.weight_map[name])
    with pytest.raises(ValueError, match="must be BF16"):
        factory.model.create_engram_runtime()
    assert factory.requested_pin == []
    factory.manager.assert_not_called()


def test_pin_failure_is_fatal_without_pageable_fallback(factory, monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: False)
    with pytest.raises(RuntimeError, match="not pinned"):
        factory.model.create_engram_runtime()
    assert factory.requested_pin == [True]
    factory.manager.assert_not_called()


def test_pin_allocator_error_propagates(factory, monkeypatch):
    def exhausted(*args, **kwargs):
        raise RuntimeError("host pinned allocation exhausted")

    monkeypatch.setattr(torch, "empty", exhausted)
    with pytest.raises(RuntimeError, match="pinned allocation exhausted"):
        factory.model.create_engram_runtime()
    factory.manager.assert_not_called()


def test_factory_never_reads_unowned_heads_or_materializes_full_tensor(factory, monkeypatch):
    original_open = engram_offload.safe_open
    reads = []

    class Slice:
        def __init__(self, source, name):
            self.source, self.name = source, name

        def get_shape(self):
            return self.source.get_shape()

        def get_dtype(self):
            return self.source.get_dtype()

        def __getitem__(self, rows):
            reads.append((self.name, rows.start, rows.stop))
            return self.source[rows]

    class Reader:
        def __init__(self, *args, **kwargs):
            self.reader = original_open(*args, **kwargs)

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            return self.reader.__exit__(*args)

        def get_slice(self, name):
            return Slice(self.reader.get_slice(name), name)

        def get_tensor(self, name):
            pytest.fail(f"Full-table materialization requested for {name}")

    monkeypatch.setattr(engram_offload, "safe_open", Reader)
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_rank", lambda: 3)
    factory.model.create_engram_runtime()
    assert reads == [
        (f"layers.{layer_id}.engram.embed.weight", start, end)
        for layer, layer_id in enumerate(factory.layout.layer_ids)
        for start, end in factory.layout.head_shard(layer, 3, 8)[1]
    ]


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("bad_shape", ["rows", "head_dim"])
def test_all_table_shapes_are_checked_before_any_pin_or_tokenizer(factory, layer, bad_shape):
    name = f"layers.{factory.layout.layer_ids[layer]}.engram.embed.weight"
    table = factory.tables[layer]
    wrong = table[:-1] if bad_shape == "rows" else table[:, :-1].contiguous()
    save_file({name: wrong}, factory.root / factory.weight_map[name])
    with pytest.raises(ValueError, match="must be BF16"):
        factory.model.create_engram_runtime()
    assert factory.requested_pin == []
    factory.tokenizer_loader.assert_not_called()
    factory.manager.assert_not_called()


@pytest.mark.parametrize("rank", range(8))
def test_explicit_numa_node_follows_tp_rank_and_loaded_parameter_device(factory, monkeypatch, rank):
    nodes = [6, 7, 4, 5, 0, 1, 2, 3]
    # Deliberately remap devices: NUMA nodes index TP ranks, not NPU ordinals.
    device = torch.device(f"npu:{7 - rank}")
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(model_module, "get_ascend_config", lambda: SimpleNamespace(engram_numa_nodes=nodes))
    monkeypatch.setattr(factory.model.model, "parameters", lambda: iter([SimpleNamespace(device=device)]))
    loader = Mock(side_effect=[Mock(), Mock()])
    monkeypatch.setattr(engram_offload.EngramTableShard, "from_safetensors", loader)
    # CPU wiring test; device allocation belongs to the offload integration suite.
    runtime_factory = Mock()
    monkeypatch.setattr("vllm_ascend.worker.engram_runtime.EngramRuntime", runtime_factory)
    factory.model.create_engram_runtime()
    assert loader.call_count == 2
    assert all(call.kwargs == {"numa_node": nodes[rank], "device": device} for call in loader.call_args_list)
    assert factory.manager.call_args.args[2] == device
    runtime_factory.assert_called_once()


@pytest.mark.parametrize("failure", ["second_table", "manager", "runtime"])
def test_factory_closes_loaded_shards_after_partial_initialization_failure(factory, monkeypatch, failure):
    first, second = Mock(), Mock()
    error = RuntimeError("initialization failed")
    loader = Mock(side_effect=[first, error if failure == "second_table" else second])
    monkeypatch.setattr(engram_offload.EngramTableShard, "from_safetensors", loader)
    manager = Mock()
    factory.manager.side_effect = error if failure == "manager" else None
    factory.manager.return_value = manager
    if failure == "runtime":
        monkeypatch.setattr("vllm_ascend.worker.engram_runtime.EngramRuntime", Mock(side_effect=error))
    with pytest.raises(RuntimeError, match="initialization failed"):
        factory.model.create_engram_runtime()
    if failure == "runtime":
        manager.close.assert_called_once()
    else:
        first.close.assert_called_once()
        assert second.close.call_count == (failure == "manager")
