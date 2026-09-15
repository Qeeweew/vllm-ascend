# SPDX-License-Identifier: Apache-2.0
"""CPU-only stage oracle for real V4.1 draft captures; no production math imports.

This conditions each comparison on the actual stage input. It is intentionally
not an independent full-model output oracle. Expert weights are decoded one
selected expert at a time, and vocabulary projections are read in row chunks.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


class ConvertedWeights:
    def __init__(self, root):
        self.root = Path(root)
        manifest = json.loads((self.root / "conversion_manifest.json").read_text())
        self.index = {name: shard for shard, record in manifest["shards"].items() for name in record["tensors"]}
        self.headers = {
            name: header for record in manifest["shards"].values() for name, header in record["tensors"].items()
        }

    def read(self, name, rows=None, columns=None):
        with safe_open(self.root / self.index[name], framework="pt", device="cpu") as reader:
            view = reader.get_slice(name)
            if rows is None and columns is None:
                return reader.get_tensor(name).clone()
            if columns is None:
                return view[rows].clone()
            return view[slice(None) if rows is None else rows, columns].clone()

    def linear(self, x, name, *, rows=None, columns=None, dtype=torch.bfloat16):
        weight = self.read(name, rows, columns)
        return F.linear(x.float(), weight.float()).to(dtype)

    def embedding(self, ids, name):
        return torch.cat([self.read(name, rows=slice(token, token + 1)) for token in ids.tolist()])

    def vocab_linear(self, x, name):
        pieces = []
        for first in range(0, self.headers[name]["shape"][0], 4096):
            # LogitsProcessor converts BF16 LM-head GEMM outputs to FP32.
            pieces.append(self.linear(x, name, rows=slice(first, first + 4096)).float())
        return torch.cat(pieces, -1)


def rms(x, weight, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def rotate(x, positions, dim, theta, inverse=False):
    angles = positions.float()[:, None] / (theta ** (torch.arange(0, dim, 2).float()[None] / dim))
    cosine, sine = angles.cos(), angles.sin()
    if inverse:
        sine = -sine
    if x.ndim == 3:
        cosine, sine = cosine[:, None], sine[:, None]
    tail = x[..., -dim:].float().reshape(*x.shape[:-1], dim // 2, 2)
    even = tail[..., 0] * cosine - tail[..., 1] * sine
    odd = tail[..., 0] * sine + tail[..., 1] * cosine
    return torch.cat((x[..., :-dim], torch.stack((even, odd), -1).flatten(-2).to(x.dtype)), -1)


def hc_pre(hidden, incoming, fn, scale, base, eps, hc_eps, iterations):
    flat = hidden.float().flatten(1)
    control = F.linear(flat, fn.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    next_pre = torch.sigmoid(control[:, :4] * scale[0] + base[:4]) + hc_eps
    post = 2 * torch.sigmoid(control[:, 4:8] * scale[1] + base[4:8])
    comb = (control[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4).softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    collapsed = (hidden.float() * incoming[..., None]).sum(1).to(hidden.dtype)
    return collapsed, post, comb, next_pre


def hc_post(x, residual, post, comb):
    return (post[..., None] * x[:, None].float() + torch.einsum("tsi,tsj->tij", comb, residual.float())).to(x.dtype)


def dense_noncausal_attention(query, logical_kv, sinks):
    logits = torch.einsum("qhd,kd->qhk", query.float(), logical_kv.float()) / math.sqrt(query.shape[-1])
    logits = torch.cat((logits, sinks[None, :, None].expand(query.shape[0], -1, 1)), -1)
    probabilities = logits.softmax(-1)[..., :-1]
    return torch.einsum("qhk,kd->qhd", probabilities, logical_kv.float())


def unpack(packed):
    return torch.stack([(packed >> (4 * digit)) & 15 for digit in range(8)], -1).flatten(-2).sub(8).float()


def expert_matrix(weights, name, rank, down=False):
    start, stop = rank * 288, (rank + 1) * 288
    if down:
        q = unpack(weights.read(name + ".weight_packed", columns=slice(start // 8, stop // 8)))
        scale = weights.read(name + ".weight_scale", columns=slice(start // 32, stop // 32))
    else:
        q = unpack(weights.read(name + ".weight_packed", rows=slice(start, stop)))
        scale = weights.read(name + ".weight_scale", rows=slice(start, stop))
    # CANN W4A16 dequantization and GMM have explicit BF16 boundaries.
    return (q * scale.float().repeat_interleave(32, -1)).bfloat16().float()


def routed_moe(weights, stage, rank, x, ids, routing):
    output = torch.zeros_like(x, dtype=torch.float32)
    for expert in ids.unique().tolist():
        tokens, routes = (ids == expert).nonzero(as_tuple=True)
        prefix = f"mtp.{stage}.ffn.experts.{expert}"
        gate = F.linear(x[tokens].float(), expert_matrix(weights, prefix + ".w1", rank)).bfloat16().float()
        up = F.linear(x[tokens].float(), expert_matrix(weights, prefix + ".w3", rank)).bfloat16().float()
        active = (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).bfloat16()
        down = F.linear(active.float(), expert_matrix(weights, prefix + ".w2", rank, down=True)).bfloat16().float()
        output.index_add_(0, tokens, down * routing[tokens, routes, None].bfloat16().float())
    return output.bfloat16()


def metrics(actual, expected):
    actual, expected = actual.double(), expected.double()
    delta = actual - expected
    return dict(
        nrmse=float(delta.norm() / expected.norm().clamp_min(1e-20)),
        peak_relative=float(delta.abs().max() / expected.abs().max().clamp_min(1e-20)),
        max_abs=float(delta.abs().max()),
        finite=bool(torch.isfinite(actual).all()),
    )


def compare(capture, config, weights):
    checks = []
    rank = capture["rank"]
    eps, hc_eps = config["rms_norm_eps"], config["hc_eps"]
    positions = capture["positions"]

    def check(name, actual, expected, kind="dense"):
        result = dict(name=name, kind=kind, **metrics(actual, expected))
        if kind == "exact":
            passed = torch.equal(actual, expected)
        elif kind == "hc_control":
            passed = result["nrmse"] < 2e-4
        elif kind == "hc_post":
            passed = bool(torch.allclose(actual.float(), expected.float(), rtol=0.01, atol=0.01))
        elif kind == "attention":
            passed = (
                actual.float() - expected.float()
            ).square().mean().sqrt() <= expected.float().square().mean().sqrt() * 0.006 + 1e-6
            passed = bool(passed) and bool(torch.allclose(actual.float(), expected.float(), rtol=0.025, atol=0.012))
        else:
            passed = result["nrmse"] < 0.006 and result["peak_relative"] < 0.015
        result["passed"] = bool(passed and result["finite"])
        checks.append(result)

    projected = weights.linear(capture["aux"], "mtp.0.main_proj.weight")
    check("context.main_projection", capture["main_projection"], projected)
    check(
        "context.main_norm",
        capture["context"],
        rms(capture["main_projection"], weights.read("mtp.0.main_norm.weight"), eps),
    )
    for stage, layer in enumerate(capture["layers"]):
        prefix = f"mtp.{stage}"
        context_kv = weights.linear(capture["context"], prefix + ".attn.wkv.weight")
        context_kv = rms(context_kv, weights.read(prefix + ".attn.kv_norm.weight"), eps)
        context_kv = rotate(context_kv, capture["context_positions"], config["qk_rope_head_dim"], config["rope_theta"])
        check(f"{stage}.context_kv_rope", layer["context_kv"], context_kv)
        x = layer["attention_input"]
        check(
            f"{stage}.attention_norm",
            x,
            rms(layer["hc_pre"][0]["outputs"][0], weights.read(prefix + ".attn_norm.weight"), eps),
        )
        q_low = weights.linear(x, prefix + ".attn.wq_a.weight")
        q_low = rms(q_low, weights.read(prefix + ".attn.q_norm.weight"), eps)
        kv = rms(weights.linear(x, prefix + ".attn.wkv.weight"), weights.read(prefix + ".attn.kv_norm.weight"), eps)
        query = weights.linear(q_low, prefix + ".attn.wq_b.weight", rows=slice(rank * 4096, (rank + 1) * 4096)).reshape(
            -1, 8, 512
        )
        query = rotate(query, positions, config["qk_rope_head_dim"], config["rope_theta"])
        kv = rotate(kv, positions, config["qk_rope_head_dim"], config["rope_theta"])
        check(f"{stage}.query", layer["query"], query)
        check(f"{stage}.query_kv", layer["query_kv"], kv)
        logical_kv = torch.cat((layer["context_kv"], layer["query_kv"]))
        visible = logical_kv[max(0, capture["context_positions"].numel() - 128) :]
        sink = weights.read(prefix + ".attn.attn_sink")[rank * 8 : (rank + 1) * 8]
        check(
            f"{stage}.noncausal_attention",
            layer["sparse_output"],
            dense_noncausal_attention(layer["query"], visible, sink),
            "attention",
        )
        inverse = rotate(
            layer["sparse_output"], positions, config["qk_rope_head_dim"], config["rope_theta"], inverse=True
        )
        grouped = inverse.reshape(-1, 4096)
        output_a = weights.linear(grouped, prefix + ".attn.wo_a.weight", rows=slice(rank * 1024, (rank + 1) * 1024))
        check(f"{stage}.output_a", layer["output_b_input"], output_a)
        output_b = weights.linear(
            layer["output_b_input"], prefix + ".attn.wo_b.weight", columns=slice(rank * 1024, (rank + 1) * 1024)
        )
        check(f"{stage}.output_b_local", layer["output_b_local"], output_b)
        for sub, label in enumerate(("attn", "ffn")):
            record = layer["hc_pre"][sub]
            expected = hc_pre(
                record["hidden"],
                record["incoming"],
                *(weights.read(prefix + f".hc_{label}_{field}") for field in ("fn", "scale", "base")),
                eps,
                hc_eps,
                config["hc_sinkhorn_iters"],
            )
            for name, actual, want in zip(("collapsed", "post", "comb", "next_pre"), record["outputs"], expected):
                check(f"{stage}.{label}.{name}", actual, want, "exact" if name == "collapsed" else "hc_control")
            post = layer["hc_post"][sub]
            check(f"{stage}.{label}.post_hidden", post["output"], hc_post(*post["inputs"]), "hc_post")
        moe = layer["moe"]
        check(
            f"{stage}.ffn_norm",
            moe["x"],
            rms(layer["hc_pre"][1]["outputs"][0], weights.read(prefix + ".ffn_norm.weight"), eps),
        )
        check(f"{stage}.shared_input", layer["shared"]["input"], moe["x"], "exact")
        check(
            f"{stage}.ffn_incoming_mix",
            layer["hc_pre"][1]["incoming"],
            layer["hc_pre"][0]["outputs"][3],
            "exact",
        )
        check(
            f"{stage}.ffn_residual",
            layer["hc_pre"][1]["hidden"],
            layer["hc_post"][0]["output"],
            "exact",
        )
        if stage:
            previous = capture["layers"][stage - 1]["block_output"]
            check(f"{stage}.attention_incoming_mix", layer["hc_pre"][0]["incoming"], previous[1], "exact")
            check(f"{stage}.attention_residual", layer["hc_pre"][0]["hidden"], previous[0], "exact")
        else:
            embedding = weights.embedding(capture["ids"], "embed.weight")
            check("0.query_embedding", layer["hc_pre"][0]["hidden"], embedding[:, None].expand(-1, 4, -1), "exact")
            initial_mix = torch.zeros_like(layer["hc_pre"][0]["incoming"])
            initial_mix[:, 0] = 1
            check("0.attention_incoming_mix", layer["hc_pre"][0]["incoming"], initial_mix, "exact")
        route_scores = F.softplus(F.linear(moe["x"].float(), weights.read(prefix + ".ffn.gate.weight").float())).sqrt()
        wanted_ids = (route_scores + weights.read(prefix + ".ffn.gate.bias")).topk(3, dim=-1).indices
        # Top-k ordering can differ while expert/weight associations agree.
        check(f"{stage}.router_ids", moe["ids"].sort(-1).values, wanted_ids.sort(-1).values, "exact")
        route_weights = route_scores.gather(1, moe["ids"].long())
        route_weights /= route_weights.sum(-1, keepdim=True)
        # Factory uses apply_routed_scale_to_output=False: its router owns
        # this factor and the runner applies no additional output scaling.
        route_weights *= config["routed_scaling_factor"]
        check(f"{stage}.router_weights", moe["weights"], route_weights)
        check(
            f"{stage}.routed_moe_local",
            moe["output"],
            routed_moe(weights, stage, rank, moe["x"], moe["ids"], moe["weights"]),
        )
        shared = layer["shared"]
        selected = slice(rank * 288, (rank + 1) * 288)
        gate = weights.linear(shared["input"], prefix + ".ffn.shared_experts.w1.weight", rows=selected).float()
        up = weights.linear(shared["input"], prefix + ".ffn.shared_experts.w3.weight", rows=selected).float()
        active = (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).bfloat16()
        shared_output = weights.linear(active, prefix + ".ffn.shared_experts.w2.weight", columns=selected)
        check(f"{stage}.shared_moe_local", shared["output"], shared_output)
    last = capture["layers"][-1]
    collapsed = (last["block_output"][0].float() * last["block_output"][1][..., None]).sum(1).bfloat16()
    check("head.collapse", capture["head_hidden"], collapsed, "exact")
    normalized = rms(capture["head_hidden"], weights.read("mtp.2.norm.weight"), eps)
    check("head.norm", capture["normalized"], normalized)
    if capture["logits"] is not None:
        check("head.logits", capture["logits"], weights.vocab_linear(capture["normalized"], "head.weight"))
    markov = weights.embedding(capture["markov_ids"], "mtp.2.markov_head.embed.weight")
    check("head.markov_embed", capture["markov_embed"], markov, "exact")
    if capture["markov_bias"] is not None:
        check(
            "head.markov_bias",
            capture["markov_bias"],
            weights.vocab_linear(capture["markov_embed"], "mtp.2.markov_head.head.weight"),
        )
    confidence_input = torch.cat((capture["head_hidden"], capture["markov_embed"]), -1).float()
    confidence = (
        weights.linear(confidence_input, "mtp.2.confidence_head.proj.weight", dtype=torch.float32).squeeze(-1).sigmoid()
    )
    check("head.confidence", capture["confidence"], confidence)
    return dict(
        rank=rank,
        passed=all(item["passed"] for item in checks),
        checks=checks,
        scope="actual-stage-input CPU oracle; not independently propagated full-model output",
    )
