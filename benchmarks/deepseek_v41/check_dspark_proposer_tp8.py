# SPDX-License-Identifier: Apache-2.0
"""Real runner/proposer integration with real draft weights and synthetic target aux.

CPU preparation is the default. The TP8 run loads no target layers or Engram
tables. It exercises real proposer construction, loading, cache builders,
initial-input kernel, context KV, draft forward and sequential Markov selection.
It does not validate target verification, scheduling, acceptance or serving.
"""

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import torch
from check_dspark_v41_tp8 import PEAK_BUDGET_BYTES, component_config
from check_dspark_v41_tp8 import prepare as prepare_components
from dspark_v41_reference import ConvertedWeights


def cases():
    return [(n, None) for n in (9, 33, 129)] + [(n, r) for n in (255, 256) for r in range(6)]


def prepare(args):
    """Make a draft-only loader view without modifying source/converted files."""
    if (args.output / "prepared.json").exists():
        raise ValueError("Choose a new output directory")
    args.context_tokens = 9
    prepare_components(args)
    weights = ConvertedWeights(args.checkpoint)
    index = {name: shard for name, shard in weights.index.items() if name.startswith("mtp.")}
    fixture = args.output / "config"
    for shard in sorted(set(index.values())):
        (fixture / shard).symlink_to((args.checkpoint / shard).resolve())
    (fixture / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}, indent=2) + "\n")
    config = component_config(args)
    result = {
        "status": "prepared_only",
        "cases": cases(),
        "source": str(args.source),
        "checkpoint": str(args.checkpoint),
        "draft_shards": sorted(set(index.values())),
        "draft_architecture": config.speculative_config.draft_model_config.architectures,
        "target_aux": "synthetic deterministic BF16; no target layers executed",
        "real_runner_constructor": True,
        "real_proposer_load_model": True,
        "real_proposer_propose": True,
        "scheduler_acceptance_validated": False,
        "graph_validated": False,
        "peak_budget_bytes": PEAK_BUDGET_BYTES,
        "manifest_sha256": hashlib.sha256((args.checkpoint / "conversion_manifest.json").read_bytes()).hexdigest(),
        "npu_initialized": torch.npu.is_initialized(),
    }
    assert not result["npu_initialized"]
    (args.output / "prepared.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def vocabulary(config, weights, rank):
    from torch import nn
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

    target = nn.Module()
    target.model = nn.Module()
    target.model.embed_tokens = VocabParallelEmbedding(
        config.vocab_size, config.hidden_size, params_dtype=torch.bfloat16
    )
    target.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, params_dtype=torch.bfloat16)
    width = config.vocab_size // 8
    for module, name in ((target.model.embed_tokens, "embed.weight"), (target.lm_head, "head.weight")):
        module.weight.copy_(weights.read(name, rows=slice(rank * width, (rank + 1) * width)))
    return target


def initialize_caches(proposer, config, device):
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

    caches = [layer.self_attn.swa_cache_layer for layer in proposer.model.model.layers]
    for cache in caches:
        cache.bind_kv_cache(torch.zeros((8, 32, 1, 512), dtype=torch.bfloat16, device=device))
    names = [cache.prefix for cache in caches]
    spec = caches[0].get_kv_cache_spec(config)
    kv_config = KVCacheConfig(8, [], [KVCacheGroupSpec(names, spec, is_eagle_group=True)])
    proposer.initialize_attn_backend(kv_config, [32])
    return caches


def observe_metadata(proposer, output):
    """Diagnostic D2H changes synchronization; never use it as the final gate."""
    import torch_npu

    records = []
    for group_index, group in enumerate(proposer.draft_attn_groups):
        builder = group.get_metadata_builder()
        original = builder._refresh_schedule

        def observed(
            cu_q,
            lengths,
            cmp_lengths,
            residual,
            batch,
            draft_lengths=None,
            *,
            owner=builder,
            call=original,
            index=group_index,
        ):
            record = {
                "group": index,
                "num_heads": owner.num_heads,
                "max_tokens": owner.max_tokens,
                "max_sequence": owner.max_sequence,
                "batch": batch,
                "diagnostic_synchronization": True,
                "stream": torch.npu.current_stream().npu_stream,
                "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
                "mapped_operator_libraries": sorted(
                    {
                        line.split()[-1]
                        for line in Path("/proc/self/maps").read_text().splitlines()
                        if any(name in line for name in ("libcust_opapi", "libopapi", "vllm_ascend_C"))
                    }
                ),
                "inputs": {},
            }
            for name, tensor in (
                ("cu_q", cu_q),
                ("lengths", lengths),
                ("cmp_lengths", cmp_lengths),
                ("residual", residual),
                ("draft_lengths", draft_lengths),
            ):
                record["inputs"][name] = (
                    None
                    if tensor is None
                    else {
                        "shape": list(tensor.shape),
                        "stride": list(tensor.stride()),
                        "storage_offset": tensor.storage_offset(),
                        "storage_bytes": tensor.untyped_storage().nbytes(),
                        "data_ptr": tensor.data_ptr(),
                        "contiguous": tensor.is_contiguous(),
                        "npu_format": torch_npu.get_npu_format(tensor),
                        "dtype": str(tensor.dtype),
                        "device": str(tensor.device),
                        "values": tensor.cpu().tolist(),
                    }
                )
            records.append(record)
            output.write_text(json.dumps(records, indent=2) + "\n")
            result = call(cu_q, lengths, cmp_lengths, residual, batch, draft_lengths)
            torch.npu.synchronize()
            record["completed"] = True
            output.write_text(json.dumps(records, indent=2) + "\n")
            return result

        builder._refresh_schedule = observed


def one_case(proposer, config, caches, sequence, rejected, seed, device, *, perturb_rejected=False):
    """Observe actual inputs/results; scalar CPU checks do not replace execution."""
    from vllm.forward_context import BatchDescriptor

    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

    effective = sequence - (rejected or 0)
    table = torch.arange(7, -1, -1, dtype=torch.int32, device=device)[None]
    positions = torch.arange(sequence, dtype=torch.int64, device=device)
    slots = (table[0, positions // 32] * 32 + positions % 32).long()
    offsets_cpu = torch.tensor([0, sequence], dtype=torch.int32)
    common = AscendCommonAttentionMetadata(
        query_start_loc=offsets_cpu.to(device),
        query_start_loc_cpu=offsets_cpu,
        seq_lens=torch.tensor([sequence], dtype=torch.int32, device=device),
        num_reqs=1,
        num_actual_tokens=sequence,
        max_query_len=sequence,
        max_seq_len=sequence,
        block_table_tensor=table,
        slot_mapping=slots,
        _seq_lens_cpu=torch.tensor([effective], dtype=torch.int32),
        seq_lens_cpu=None,
    )
    proposer.set_per_group_attn_metadata(0, table, slots)
    for cache in caches:
        cache.kv_cache.zero_()
    generator = torch.Generator().manual_seed(41182 + sequence)
    aux = torch.randn((sequence, 15360), generator=generator, dtype=torch.float32).mul_(0.125).bfloat16().to(device)
    if perturb_rejected:
        assert rejected
        aux[effective:].mul_(-3).add_(2)
    record = {"sequence": sequence, "rejected": rejected, "seed": seed}
    original_logits = proposer.model.compute_draft_logits

    def capture_logits(hidden):
        logits = original_logits(hidden)
        record["raw_logits"] = logits.detach().cpu().clone()
        return logits

    proposer.model.compute_draft_logits = capture_logits
    try:
        proposals = (
            proposer._propose(
                num_speculative_tokens=5,
                target_token_ids=torch.arange(sequence, dtype=torch.int64, device=device),
                target_positions=positions,
                target_hidden_states=aux,
                next_token_ids=torch.tensor([seed], dtype=torch.int64, device=device),
                token_indices_to_sample=None,
                common_attn_metadata=common,
                target_model_batch_desc=BatchDescriptor(num_tokens=sequence, uniform=False),
                sampling_metadata=None,
                num_prefill_reqs=1,
                num_scheduled_tokens=sequence,
                num_rejected_tokens_gpu=None
                if rejected is None
                else torch.tensor([rejected], dtype=torch.int32, device=device),
            )
            .detach()
            .cpu()
            .clone()
        )
    finally:
        proposer.model.compute_draft_logits = original_logits
    assert proposals.shape == (1, 5) and torch.isfinite(record["raw_logits"]).all()
    assert proposer.positions[:5].cpu().tolist() == list(range(effective, effective + 5))
    assert torch.equal(proposer._context_positions_buffer[:sequence].cpu().long(), torch.arange(sequence))
    for context_slots in proposer._context_slot_mapping_buffers:
        assert torch.equal(context_slots[:sequence].cpu().long(), slots.cpu())
    noise = config.speculative_config.draft_model_config.hf_config.dspark_noise_token_id
    assert proposer.input_ids[:5].cpu().tolist() == [seed, noise, noise, noise, noise]
    builder = proposer.draft_attn_groups[0].get_metadata_builder()
    expected_slots = [(7 - p // 32) * 32 + p % 32 if p < 256 else -1 for p in range(effective, effective + 5)]
    assert builder.slots[:5].cpu().tolist() == expected_slots
    visible = list(range(max(0, effective - 128), min(effective + 5, 256)))
    indices, lengths = builder.draft_swa_indices[:5].cpu(), builder.draft_swa_lengths[:5].cpu()
    for row, position in enumerate(range(effective, effective + 5)):
        if position < 256:
            assert indices[row, 0, : len(visible)].tolist() == visible
            assert lengths[row, 0] == len(visible)
        else:
            assert torch.all(indices[row] == -1) and lengths[row, 0] == 0
    record.update(proposals=proposals, positions=proposer.positions[:5].cpu().clone(), slots=expected_slots)
    return record


@torch.inference_mode()
def run(args):
    """Use production runner construction and loading, then checked TP8 teardown."""
    if int(os.environ.get("WORLD_SIZE", "0")) != 8 or os.environ.get("HCCL_DETERMINISTIC") != "strict":
        raise ValueError("Require TP8 torchrun and HCCL_DETERMINISTIC=strict")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if (args.output / f"rank{rank}.json").exists():
        raise ValueError("Use a new prepared output directory")
    status, initialized = (
        {
            "rank": rank,
            "status": "failed",
            "cases": [],
            "diagnostic_metadata_sync": args.observe_metadata,
        },
        False,
    )
    try:
        from vllm.config import set_current_vllm_config
        from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment

        from vllm_ascend import ops
        from vllm_ascend.ascend_config import init_ascend_config
        from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel
        from vllm_ascend.models import register_model
        from vllm_ascend.utils import adapt_patch, enable_custom_op, register_ascend_customop

        torch.set_num_threads(4)
        torch.npu.set_device(local_rank)
        device = torch.device("npu", local_rank)
        config = component_config(args)
        adapt_patch()
        # Import model users only after platform and worker patches are active.
        from vllm.model_executor.layers import fused_moe

        from vllm_ascend.models.deepseek_v4 import model as target_model_module
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        assert target_model_module.FusedMoEFactory is fused_moe.FusedMoEFactory
        register_model()
        ops.register_dummy_fusion_op()
        register_ascend_customop(config)
        init_ascend_config(config)
        assert enable_custom_op()
        with set_current_vllm_config(config):
            init_distributed_environment(8, rank, "env://", local_rank, "hccl")
            initialized = True
            ensure_model_parallel_initialized(8, 1, 1, 1)
            init_ascend_model_parallel(config.parallel_config)
            torch.set_default_dtype(torch.bfloat16)
            status["phase"] = "constructing_runner"
            print(f"rank {rank}: constructing actual NPUModelRunner", flush=True)
            runner = NPUModelRunner(config, device)
            proposer = runner.drafter
            with torch.device(device):
                target = vocabulary(config.model_config.hf_text_config, ConvertedWeights(args.checkpoint), rank)
            print(f"rank {rank}: actual proposer.load_model", flush=True)
            status["phase"] = "loading_draft"
            proposer.load_model(target)
            assert proposer.model.model.embed_tokens is target.model.embed_tokens
            assert proposer.model.lm_head is target.lm_head
            assert not proposer.use_cuda_graph and proposer.parallel_drafting
            if torch.npu.max_memory_allocated() >= PEAK_BUDGET_BYTES:
                raise RuntimeError("Draft loading exceeded fixed 8 GiB per-rank budget")
            caches = initialize_caches(proposer, config, device)
            if args.observe_metadata:
                observe_metadata(proposer, args.output / f"metadata_rank{rank}.json")
            records = []
            for index, (sequence, rejected) in enumerate(cases()):
                status["phase"] = f"case_{index}"
                record = one_case(proposer, config, caches, sequence, rejected, (100, 0, 129264)[index % 3], device)
                if rejected:
                    perturbed = one_case(
                        proposer, config, caches, sequence, rejected, record["seed"], device, perturb_rejected=True
                    )
                    assert torch.equal(record["raw_logits"], perturbed["raw_logits"])
                    assert torch.equal(record["proposals"], perturbed["proposals"])
                    record["rejected_aux_perturbation_exact"] = True
                records.append(record)
                status["cases"].append({key: record[key] for key in ("sequence", "rejected", "seed", "slots")})
                status["cases"][-1]["proposals"] = record["proposals"].tolist()
                print(f"rank {rank}: case {index} sequence={sequence} rejected={rejected} completed", flush=True)
            torch.npu.synchronize()
            status["peak_allocated_bytes"] = torch.npu.max_memory_allocated()
            status["reserved_bytes"] = torch.npu.memory_reserved()
            if status["peak_allocated_bytes"] >= PEAK_BUDGET_BYTES:
                raise RuntimeError("Proposer integration exceeded fixed 8 GiB per-rank budget")
            if rank == 0:
                torch.save(records, args.output / "rank0.pt")
            status["status"] = "executed_pending_cpu_selection_oracle"
            torch.distributed.barrier()
    except Exception:
        status["error"] = traceback.format_exc()
        raise
    finally:
        if initialized:
            from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

            try:
                torch.npu.synchronize()
                destroy_model_parallel()
                destroy_distributed_environment()
                status["distributed_cleanup"] = True
            except Exception:
                status.update(status="failed_cleanup", cleanup_error=traceback.format_exc())
        (args.output / f"rank{rank}.json").write_text(json.dumps(status, indent=2) + "\n")


def greedy_reference(raw_logits, embed, head, seed):
    """Respect the actual logits dtype when adding each sequential Markov bias."""
    previous, predicted = seed, []
    head_float = head.float()
    for logits in raw_logits:
        bias = torch.nn.functional.linear(embed[previous].float(), head_float).bfloat16()
        adjusted = logits.clone()
        adjusted.add_(bias)
        previous = int(adjusted.argmax())
        predicted.append(previous)
    return predicted


def compare(args):
    """Independent sequential greedy oracle, conditioned on actual raw logits."""
    torch.set_num_threads(8)
    ranks = [json.loads((args.output / f"rank{rank}.json").read_text()) for rank in range(8)]
    assert all(r["status"] == "executed_pending_cpu_selection_oracle" and r["distributed_cleanup"] for r in ranks)
    assert all(r["cases"] == ranks[0]["cases"] for r in ranks)
    weights = ConvertedWeights(args.checkpoint)
    embed = weights.read("mtp.2.markov_head.embed.weight")
    head = weights.read("mtp.2.markov_head.head.weight")
    results = []
    for record in torch.load(args.output / "rank0.pt", map_location="cpu", weights_only=True):
        predicted = greedy_reference(record["raw_logits"], embed, head, record["seed"])
        actual = record["proposals"][0].tolist()
        results.append(
            {
                "sequence": record["sequence"],
                "rejected": record["rejected"],
                "actual": actual,
                "reference": predicted,
                "exact": actual == predicted,
            }
        )
    diagnostic = any(r.get("diagnostic_metadata_sync", False) for r in ranks)
    report = {
        "status": ("diagnostic_passed" if diagnostic else "passed") if all(r["exact"] for r in results) else "failed",
        "diagnostic_metadata_sync": diagnostic,
        "cases": results,
        "all_rank_proposals_exact": True,
        "target_aux_synthetic": True,
        "scheduler_validated": False,
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in ranks),
    }
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observe-metadata", action="store_true", help="Diagnostic only: synchronizes metadata inputs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    result = run(args) if args.run else compare(args) if args.compare else prepare(args)
    if result is not None:
        print(json.dumps(result), flush=True)
        return int(result.get("status") == "failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
