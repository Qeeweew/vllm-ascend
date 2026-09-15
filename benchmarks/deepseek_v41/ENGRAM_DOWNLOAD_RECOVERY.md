# Engram shard download recovery

## Live status

Recovery started on 2026-09-15 at 20:31 UTC. This report records a **running
download**, not completed weights or successful full-file verification.
At 20:36:57 UTC each independently owned suffix had grown past 34 MiB.
The original temporary files remained unchanged.

The current process is recorded in
`/mnt/models/DeepSeek-V4.1-Flash/.v41-recovery-3bd368ab0f3d/process.json`.
The shell tool session is 4351; actual Python PID at this report is 3045493.
Log: `/tmp/v41-engram-recovery.log`. Per-shard status JSON and durable
manifest are in the recovery directory. `published` means the complete
SHA256-verified file has been atomically linked into the model directory.
`target_already_exists_preserved` means an existing destination was left
untouched; the verified recovery copy remains available separately.

## Source evidence

The ModelScope `.msc` file contains only downloaded path/revision records.
It has no records for the incomplete shards. Completed shard 46 uses
revision `3bd368ab0f3da472b1adc6e19d37717a6cd0967f`.

Both public repositories report identical expected size and whole-file
SHA256 for shards 47 and 48. Their local safetensors headers independently
agree with the expected sizes.

| Shard | Existing read-only prefix bytes | Complete bytes | Missing bytes |
| --- | ---: | ---: | ---: |
| 47 | 84174438400 | 101535150936 | 17360712536 |
| 48 | 83363889152 | 101537926640 | 18174037488 |

- Shard 47 SHA256:
  `824db4881320407ac340736d14dcee5ecd748c27d0f5836b8127ecc2e3781b0f`.
- Shard 48 SHA256:
  `976330f4954338e1ad8b508c32aa912032c7ad908959fd53c8307650fe4520ed`.

Pinned download URL template:

```text
https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash/resolve/3bd368ab0f3da472b1adc6e19d37717a6cd0967f/{filename}
```

HEAD and 4 KiB Range probes succeeded against both ModelScope and Hugging
Face. Existing-prefix tail bytes matched local bytes exactly; first-missing
Range probes returned the exact expected Content-Range. This sampled
comparison alone does not establish full-prefix integrity; final SHA256
verification is mandatory. Hugging Face's final CDN ETag differs from the
whole-file hash and is not used as a SHA256 checksum.

Sanitized probe artifacts:

- `/tmp/v41-download-public-modelscope.json`
- `/tmp/v41-download-public-huggingface.json`
- `/tmp/v41-download-range-probes.json`
- `/tmp/v41-download-suffix-probes.json`

## Recovery contract

Script: `benchmarks/deepseek_v41/recover_engram_shards.py`.

1. Exclusively lock the recovery directory, record audited original-file
   device/inode/size/mtime/ctime, and verify available space with an 8 GiB
   reserve. Never write, rename, truncate, or delete original temp files.
2. Download each missing suffix to an independently owned `.suffix` file.
   Requests use the pinned revision and bounded 64 MiB ranges. Accept only
   HTTP 206 with exactly matching Content-Range and byte count. Reject
   unexpected content encodings and bytes beyond the requested interval.
3. Resume after the actual existing suffix length, preserving previous
   downloads. Flush and fsync at 16 MiB or 10-second checkpoints. Retry
   transient connection/server failures with bounded backoff. Two shards
   download concurrently. SIGTERM flushes progress and stops safely.
4. Stop if the original prefix identity changes. After suffix completion,
   recheck space and stream original prefix plus suffix into a separate
   `.assembling` file, hashing all bytes written. Require the exact complete
   length and public SHA256 after fsync and a final source identity check.
   Interrupted assembly can restart using the preserved complete suffix.
5. Link the verified assembly into the recovery complete-file name, then
   atomically hard-link it into the model directory. Link creation fails
   instead of overwriting any existing destination. Existing complete
   recovery files are size/SHA256 checked before reuse. Failed integrity
   checks preserve all artifacts for diagnosis and never publish a target.

The model storage is ext4; no reflink savings are assumed. Missing suffixes
need **33.094 GiB**, both complete recovery copies need **189.127 GiB**, and
keeping suffixes through completion gives **222.221 GiB** peak additional
space. Publication hard links add no second data copy. Preflight observed
about **921.585 GiB** free. Suffix cleanup is intentionally not automatic.

To restart after interruption, run the same command; the directory lock
prevents overlapping instances:

```bash
ionice -c 2 -n 7 nice -n 10 ../.venv/bin/python -u \
  benchmarks/deepseek_v41/recover_engram_shards.py \
  >> /tmp/v41-engram-recovery.log 2>&1
```

## Bandwidth limitation

Proxy `http://127.0.0.1:7897` currently provides roughly 120 KiB/s per shard,
about 240 KiB/s aggregate. Eight concurrent 1 MiB ModelScope range probes
transferred 8 MiB in 38.22 seconds, providing no aggregate improvement.
A Hugging Face 1 MiB probe took 29.15 seconds. A direct ModelScope connection
failed after 20 seconds. No proxy configuration was changed.

At the measured rate, completion would take approximately **40 hours**, not
a few minutes. The background downloader remains active while model work
continues. This estimate excludes final disk copying/hash verification and
will change if a faster network path becomes available.

## Tests

```bash
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest \
  tests/ut/models/test_engram_shard_recovery.py -q
```

**11 passed**, 0.18 seconds test time; log:
`/tmp/v41-engram-recovery-tests.log`. Tests use small real files and controlled
HTTP responses, covering exact resume offsets, bounded range continuation,
already-complete suffix reuse, invalid Range responses, changed source
identity, corrupt suffix rejection, disk-space rejection, verified hard-link
publication and existing user-target preservation. Real public HTTP behavior
was established by the separate probes and growing background suffix files.
