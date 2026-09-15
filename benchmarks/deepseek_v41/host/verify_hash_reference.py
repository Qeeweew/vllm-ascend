# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from vllm_ascend.ops.engram_hash import HostEngramHasher

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
args = parser.parse_args()
root = args.source
cfg = SimpleNamespace(**json.loads((root / "config.json").read_text())["text_config"])
tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
host = HostEngramHasher.from_tokenizer(cfg, tokenizer)
spec = importlib.util.spec_from_file_location("reference_engram", root / "inference/engram.py")
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)
args = SimpleNamespace(**vars(cfg), engram_pad_id=cfg.engram_pad_token_id, max_batch_size=2, max_seq_len=128)
layout = ref.EngramLayout.from_args(args)
assert layout.primes == host.layout.primes
reference = ref.NgramHashState(args, layout, tokenizer)
assert torch.equal(reference.multipliers, host.multipliers)
assert torch.equal(reference.token_map, host.token_map)
generator = torch.Generator().manual_seed(923)
tokens = torch.randint(0, len(tokenizer), (2, 80), generator=generator)
mask = torch.rand((2, 80), generator=generator) > 0.15
want = reference(tokens, 0, mask)
checks = 0
for start, end in [(0, 80), (0, 1), (1, 13), (13, 64), (64, 80), (27, 40)]:
    prior = torch.stack([tokens[:, p] if p >= 0 else torch.full((2,), -1) for p in range(start - 1, start - 4, -1)], -1)
    prior_mask = torch.stack(
        [mask[:, p] if p >= 0 else torch.zeros(2, dtype=torch.bool) for p in range(start - 1, start - 4, -1)], -1
    )
    got = host.hash_chunk(
        tokens[:, start:end].flatten(),
        [0, end - start, 2 * (end - start)],
        [start, start],
        prior,
        token_mask=mask[:, start:end].flatten(),
        lookback_mask=prior_mask,
    )
    for layer, actual in enumerate(got):
        assert torch.equal(actual, want[:, start:end, layer].reshape(-1, 24))
        checks += 1
print(
    json.dumps(
        {
            "tokenizer_vocab": len(tokenizer),
            "compressed_vocab": cfg.engram_compressed_vocab_size,
            "reference_exact_checks": checks,
            "layer_ids": cfg.engram_layer_ids,
            "tp8_head_shards": [[host.layout.head_shard(layer, rank, 8) for rank in range(8)] for layer in range(2)],
        },
        indent=2,
    )
)
