# SPDX-License-Identifier: Apache-2.0
"""Real TP8 DSpark context/query replay comparisons with changing requests."""

import json
from collections import Counter
from unittest.mock import patch

import torch
from dspark_router_diagnostic import compare_routes
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


def graph_cases():
    return [
        ([9], [0]),
        ([33], [0]),
        ([129], [0]),
        ([9, 33, 129], [0, 1, 2]),
        ([17, 41, 113], [1, 2, 3]),
        ([64, 64, 64, 64], [0, 1, 2, 3]),
        ([255, 256], [0, 0]),
        ([255, 256], [4, 5]),
        ([33, 9, 65], [5, 0, 1]),
        ([33], [1]),
        ([9, 33, 129], [2, 3, 4]),
    ]


def inputs(proposer, sequence, rejected, device, seed, *, perturb=False):
    batch = len(sequence)
    host_cu = torch.tensor([0, *torch.tensor(sequence).cumsum(0).tolist()], dtype=torch.int32)
    positions = torch.cat([torch.arange(n, device=device) for n in sequence])
    table = torch.arange(batch * 8, dtype=torch.int32, device=device).reshape(batch, 8).flip(1)
    # Rotate physical pages on every case; logical visibility stays identical.
    table = (table + seed) % (proposer.max_batch_size * 8)
    slots = torch.cat(
        [
            table[b, :][torch.arange(n, device=device) // 32] * 32 + torch.arange(n, device=device) % 32
            for b, n in enumerate(sequence)
        ]
    )
    generator = torch.Generator().manual_seed(91203 + seed)
    aux = torch.randn((sum(sequence), 15360), generator=generator).mul_(0.125).bfloat16().to(device)
    if perturb:
        start = 0
        for length, reject in zip(sequence, rejected):
            if reject:
                aux[start + length - reject : start + length].mul_(-3).add_(2)
            start += length
    effective = torch.tensor(sequence, dtype=torch.int32) - torch.tensor(rejected, dtype=torch.int32)
    common = AscendCommonAttentionMetadata(
        query_start_loc=host_cu.to(device),
        query_start_loc_cpu=host_cu,
        seq_lens=torch.tensor(sequence, dtype=torch.int32, device=device),
        num_reqs=batch,
        num_actual_tokens=sum(sequence),
        max_query_len=max(sequence),
        max_seq_len=max(sequence),
        block_table_tensor=table,
        slot_mapping=slots,
        _seq_lens_cpu=effective,
        seq_lens_cpu=None,
    )
    for group in proposer.draft_attn_groups:
        proposer.set_per_group_attn_metadata(group.kv_cache_group_id, table, slots)
    return dict(
        num_speculative_tokens=5,
        target_token_ids=torch.arange(sum(sequence), dtype=torch.int64, device=device),
        target_positions=positions,
        target_hidden_states=aux,
        next_token_ids=torch.tensor([100 + seed + b for b in range(batch)], dtype=torch.int64, device=device),
        token_indices_to_sample=None,
        common_attn_metadata=common,
        target_model_batch_desc=BatchDescriptor(num_tokens=sum(sequence), uniform=False),
        sampling_metadata=None,
        num_prefill_reqs=batch,
        num_scheduled_tokens=sum(sequence),
        num_rejected_tokens_gpu=torch.tensor(rejected, dtype=torch.int32, device=device),
    )


def run_graph_cases(proposer, config, caches, device, rank, output, *, padding_diagnostic=False):
    """Use the same real model for eager oracle and captured/replayed requests."""
    graph = proposer._v41_graph
    assert graph is not None and proposer.use_cuda_graph
    vocab = config.model_config.hf_text_config.vocab_size
    logits_buffer = torch.empty((proposer.max_query_tokens, vocab), dtype=torch.bfloat16, device=device)
    original_logits = proposer.model.compute_draft_logits
    original_replay = torch.npu.NPUGraph.replay
    replays = Counter()
    journal = []
    stage_buffers, stage_handles = {}, []
    router_patches, router_biases = [], {}
    if padding_diagnostic:
        # Device-only observation also runs inside capture. Copying to CPU is
        # deferred until the complete proposer returns.
        def watch(name, module, rows, width):
            buffer = torch.empty((rows, width), dtype=torch.bfloat16, device=device)
            stage_buffers[name] = buffer

            def save(owner, args, result):
                value = result[0] if isinstance(result, tuple) else result
                buffer[: value.shape[0]].copy_(value.reshape(value.shape[0], -1))

            stage_handles.append(module.register_forward_hook(save))

        draft = proposer.model.model
        watch("context_projection", draft.main_proj, proposer.max_num_tokens, proposer.hidden_size)
        watch("context_norm", draft.main_norm, proposer.max_num_tokens, proposer.hidden_size)
        for index, layer in enumerate(draft.layers):
            watch(f"layer{index}_attention", layer.self_attn, proposer.max_query_tokens, proposer.hidden_size)
            watch(
                f"layer{index}_moe_input",
                layer.post_attention_layernorm,
                proposer.max_query_tokens,
                proposer.hidden_size,
            )
            watch(f"layer{index}_moe", layer.mlp, proposer.max_query_tokens, proposer.hidden_size)
            watch(f"layer{index}_output", layer, proposer.max_query_tokens, proposer.hidden_size * draft.hc_mult)
            # Observe actual choices; recomputing top-k would hide a routing
            # implementation error. Only persistent device copies enter capture.
            router = layer.mlp.experts.routed_experts.router
            assert router.scoring_func == "sqrtsoftplus" and not router.use_grouped_topk
            assert router.tid2eid is None
            name = f"layer{index}_router"
            router_biases[name] = router.e_score_correction_bias.detach().cpu().clone()
            for suffix, width, dtype in (
                ("logits", layer.mlp.n_routed_experts, torch.float32),
                ("weights", router.top_k, torch.float32),
                ("ids", router.top_k, torch.int32),
            ):
                stage_buffers[f"{name}_{suffix}"] = torch.empty(
                    (proposer.max_query_tokens, width), dtype=dtype, device=device
                )

            def observe_router(*args, original=router._select_experts, prefix=name, **kwargs):
                logits = kwargs["router_logits"]
                weights, ids = original(*args, **kwargs)
                for suffix, value in (("logits", logits), ("weights", weights), ("ids", ids)):
                    stage_buffers[f"{prefix}_{suffix}"][: value.shape[0]].copy_(value)
                return weights, ids

            observer = patch.object(router, "_select_experts", observe_router)
            observer.start()
            router_patches.append(observer)

    def observe_replay(owner, *args, **kwargs):
        replays[id(owner)] += 1
        return original_replay(owner, *args, **kwargs)

    def observe_logits(hidden):
        logits = original_logits(hidden)
        logits_buffer[: logits.shape[0]].copy_(logits)
        return logits

    proposer.model.compute_draft_logits = observe_logits
    replay_observer = patch.object(torch.npu.NPUGraph, "replay", observe_replay)
    replay_observer.start()
    try:
        proposer.dummy_run(
            num_tokens=proposer.max_num_tokens,
            num_reqs=proposer.max_batch_size,
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
        )
        context_entries = graph.context.concrete_aclgraph_entries
        query_entries = graph.query.concrete_aclgraph_entries
        identities = (
            [id(e.aclgraph) for e in context_entries.values()],
            [id(e.aclgraph) for e in query_entries.values()],
        )
        assert all(e.aclgraph is not None for e in [*context_entries.values(), *query_entries.values()])
        captured = {
            family: [
                {"tokens": entry.batch_descriptor.num_tokens, "graph_id": id(entry.aclgraph)}
                for entry in entries.values()
            ]
            for family, entries in (("context", context_entries), ("query", query_entries))
        }
        (output / f"graph_capture_rank{rank}.json").write_text(json.dumps(captured, indent=2) + "\n")
        records = []
        for case, (sequence, rejected) in enumerate(graph_cases()):
            results = {}
            for mode in ("unpadded", "eager", "graph", "perturbed"):
                for cache in caches:
                    cache.kv_cache.zero_()
                kwargs = inputs(proposer, sequence, rejected, device, case, perturb=mode == "perturbed")
                original_call = graph._call
                if mode == "unpadded":
                    proposer._v41_graph = None
                    proposer.use_cuda_graph = False
                    proposer._runnable = proposer._run_merged_draft
                elif mode == "eager":
                    # Identical fixed shapes, executed without graph capture.
                    # The original unpadded path remains a separate comparison.
                    graph._call = lambda wrapper, size, mode, call=original_call: call(
                        wrapper, size, CUDAGraphMode.NONE
                    )
                try:
                    proposals = proposer._propose(**kwargs).cpu().clone()
                finally:
                    graph._call = original_call
                    proposer._v41_graph = graph
                    proposer.use_cuda_graph = True
                logits = logits_buffer[: len(sequence) * 5].cpu().clone()
                stages = {
                    name: buffer[: sum(sequence) if name.startswith("context_") else len(sequence) * 5].cpu().clone()
                    for name, buffer in stage_buffers.items()
                }
                results[mode] = (proposals, logits, [cache.kv_cache.cpu().clone() for cache in caches], stages)
            eager, replay, perturbed = results["eager"], results["graph"], results["perturbed"]
            routing = {}
            for name, bias in router_biases.items():
                routing[name] = compare_routes(results["unpadded"][3], eager[3], name, bias)
            if padding_diagnostic and rank == 0:
                torch.save(
                    {"biases": router_biases, "stages": {mode: result[3] for mode, result in results.items()}},
                    output / f"graph_stages_case{case}.pt",
                )
            journal.append(
                {
                    "case": case,
                    "sequence": sequence,
                    "rejected": rejected,
                    "eager_proposals": eager[0].tolist(),
                    "graph_proposals": replay[0].tolist(),
                    "logits_max_abs": float((eager[1].float() - replay[1].float()).abs().max()),
                    "rejected_logits_max_abs": float((replay[1].float() - perturbed[1].float()).abs().max()),
                    "cache_max_abs": [float((a.float() - b.float()).abs().max()) for a, b in zip(eager[2], replay[2])],
                    "unpadded_proposals": results["unpadded"][0].tolist(),
                    "unpadded_logits_max_abs": float((results["unpadded"][1].float() - eager[1].float()).abs().max()),
                    "unpadded_cache_max_abs": [
                        float((a.float() - b.float()).abs().max()) for a, b in zip(results["unpadded"][2], eager[2])
                    ],
                    "actual_graph_replays": dict(replays),
                    "routing": routing,
                    "stages": {
                        name: {
                            "graph_exact": torch.equal(value, replay[3][name]),
                            "unpadded_max_abs": float(
                                (value.float() - results["unpadded"][3][name].float()).abs().max()
                            ),
                            "unpadded_different_values": int((value != results["unpadded"][3][name]).sum()),
                        }
                        for name, value in eager[3].items()
                    },
                }
            )
            (output / f"graph_journal_rank{rank}.json").write_text(json.dumps(journal, indent=2) + "\n")
            assert torch.equal(eager[0], replay[0]) and torch.equal(replay[0], perturbed[0])
            if not padding_diagnostic:
                assert torch.equal(results["unpadded"][0], replay[0])
            assert all(torch.equal(value, replay[3][name]) for name, value in eager[3].items())
            assert torch.equal(eager[1], replay[1]) and torch.equal(replay[1], perturbed[1])
            assert all(torch.equal(a, b) for a, b in zip(eager[2], replay[2]))
            assert identities == (
                [id(e.aclgraph) for e in context_entries.values()],
                [id(e.aclgraph) for e in query_entries.values()],
            )
            assert torch.isfinite(replay[1]).all()
            records.append(
                dict(
                    sequence=sequence,
                    rejected=rejected,
                    seed=case,
                    slots=[],
                    proposals=replay[0],
                    raw_logits=replay[1],
                    graph_exact=True,
                    cache_exact=True,
                    rejected_aux_perturbation_exact=True,
                    unpadded_proposals_exact=torch.equal(results["unpadded"][0], replay[0]),
                )
            )
            print(f"rank {rank}: graph case {case} batch={len(sequence)} context={sequence} exact", flush=True)
        assert sum(replays[id(e.aclgraph)] for e in context_entries.values()) == len(records) * 2
        assert sum(replays[id(e.aclgraph)] for e in query_entries.values()) == len(records) * 2
        return records, dict(
            context_graphs=len(context_entries),
            query_graphs=len(query_entries),
            real_query_replays=len(records) * 2,
            real_context_replays=len(records) * 2,
            graph_ids=identities,
            actual_graph_replays=dict(replays),
        )
    finally:
        for observer in reversed(router_patches):
            observer.stop()
        for handle in stage_handles:
            handle.remove()
        replay_observer.stop()
        proposer.model.compute_draft_logits = original_logits
