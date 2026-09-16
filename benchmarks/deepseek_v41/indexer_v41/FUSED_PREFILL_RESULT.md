# Fused CR1 consumer prefill: work in progress

No performance acceptance yet. The implementation and protocol are described
in [FUSED_MULTIBATCH_DESIGN.md](FUSED_MULTIBATCH_DESIGN.md). Historical r14
B1 results remain a regression control, not evidence of prefill completion.

## Implementation available

- v3 public interface retained; T, not B, selects split-N width.
- One AIV owns each complete 16384-position query topk512. With split1 the
  paired AIVs alternate query ownership; split-N uses even-AIV completion
  relays and one odd-AIV topk owner.
- AIC scores the next query while the prior query's topk runs. Distinct GM
  query records and paired ACK bound producer lead; all graph invocations
  initialize IB mailboxes internally.
- Candidate sorting/deduplication and scale/address preparation occur once
  per query across all AIVs. Empty requests, ragged query boundaries, causal
  tails and graph padding have explicit semantics.
- r17 adds a static 256 MiB user-workspace capacity gate; larger shapes use
  legacy dispatch. This is not optimized large-prefill coverage. Future
  service configurations needing larger T require ring/chunk work.

## Evidence recorded so far

| Artifact | Result |
|---|---|
| r15 full build | Invalid: unsigned-to-float AscendC compile error; retained, never installed |
| r16 full build + isolated install | Clean build log; source/copy/installed header and kernel SHA matched |
| r16 NPU T1, context4097 | Passed independent selection oracle |
| r16 NPU B1 prefill T32, context4097 | Passed independent per-query causal oracle |
| r16 independent scheduling review | 1360 randomized abstract schedules passed; not a device-ordering proof |
| r17 full build + isolated install | Clean build; same kernel, changed tiling library for capacity gate |
| r17 ragged prefill T11/B5, T103/B6, T393/B5 | All passed; CANN confirms fused=1/split1 |
| r17 graph T32/B5, 16 changing-content replays | Passed with empty requests, padding, changed pages and candidates |
| r17 B1 prefill T1024/context4097 | Passed; CANN confirms fused=1/split1/20 producers |

r16's two tests used the installed binding. The five r17 tests used
the separately built r3 Torch binding with corrected workspace lifetime and
passed in 154.64 seconds, mostly CPU oracle time. Explicit CANN dispatch logs
confirm the new specialization for all five shapes. See
`artifacts/qli-fused/r17/initial-correctness.json` and its JUnit/dispatch logs.
The remaining gates include the full prefill/decode shape matrix, same-shape
split sweep and capacity fallback. Current devices are
shared, so timings can diagnose but cannot establish final performance gates.

## Remaining performance work

Freeze complete-selector prefill median/P95 non-regression and >=1.2x legacy
throughput target at T>=128; report HBM and useful-work FLOPs. Keep <=3% decode
regression target against historical r14 B1 where applicable. Profile real
fused dispatch across Cube, MTE, Vector and waits before N256/N512 tuning,
load-balancing changes, candidate compaction or compatible-query K reuse.
CR1 source and CR2 dense prefill remain independent pending workloads.
