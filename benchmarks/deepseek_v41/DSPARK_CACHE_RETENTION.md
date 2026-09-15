# V4.1 DSpark cache retention

The real upstream `get_kv_cache_configs` entry replaces every sliding-window
spec's `extra_retained_tokens` with zero unless multi-module MTP is enabled.
DSpark is not multi-module MTP, so this discarded the V4.1 model's declared
five-token lookback for both target and draft SWA caches.

The Ascend V4.1 grouping hook now restores at least the configured DSpark
lookback before grouping, startup admission sizing and scheduler manager
construction. Other speculative methods and other models retain their existing
policy. The window itself remains 128; attention visibility is unchanged.
No global replacement of upstream cache planning or dataclass operations is used.

## Reproduction and scope

`tests/ut/patch/platform/test_v41_dspark_retention.py` runs the real public
configuration entry through merge, upstream retention policy and actual Ascend
grouping, intercepting only subsequent memory sizing. It then constructs real
`BlockPool` and `SlidingWindowManager` objects with the resulting spec.
Before the fix, **2 tests failed and 6 passed**; the log is
`/tmp/v41-dspark-retention-repro.log`.

The observed spec changed from extra5 to extra0. With processed prefix159,
the real manager then freed page0, including token31, whereas a draft whose
prefix is159 needs logical tokens31 through163. This establishes a cache
interface failure if that window is reused after cleanup at the same boundary.
It does not establish an observed end-to-end logits failure: the current
synchronous path normally frees at settled boundary p before target execution,
then drafts at q=p+1+accepted. That order may happen to avoid the bad page.

The explicit extra5 tests cover all page phases p=128..191 and acceptance
counts0..5, using both p and the next settled q boundary. The real scheduler's
processed-token boundary remains essential: using the optimistic p+6 can
still free needed pages even with extra5. This patch does not change that
scheduler contract and cannot compensate for violating it.

## Validation

The retention reproduction plus existing planner/spec/metadata suites passed
**100 tests** after the fix (`/tmp/v41-dspark-retention-fixed.log`). Three
additional mode-isolation cases cover no speculation, MTP and EAGLE3;
the expanded retention suite passed **11 tests**
(`/tmp/v41-dspark-retention-isolation.log`).
The exact commands use the workspace Python and these test files:

```bash
../.venv/bin/python -m pytest -q   tests/ut/patch/platform/test_v41_dspark_retention.py   tests/ut/patch/platform/test_v41_cache_planner.py   tests/ut/core/test_cache_v41_spec.py   tests/ut/attention/test_dsa_v41_metadata.py
```

Separately, Engram history now checks all K5 acceptance lengths0..5, reordered
requests, six target-verification rows, corrected token0, and the following
step against an independent scalar n-gram hash oracle. History/runner tests:
**61 passed**, log `/tmp/v41-dspark-history-cpu.log`. These are CPU contracts;
full speculative scheduler, real-weight inference and graph rollback remain
integration gates, and production speculative admission is still disabled.
