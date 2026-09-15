# Final-shard conversion continuation

`watch_conversion_tail.py` is prepared, tested and running in the background
(PID 3204064 at startup; session 63944). Log:
`/tmp/v41-conversion-tail-watcher.log`. It currently awaits final source files.
It resumes the existing converter when published source shards 47/48 appear;
it does not implement or modify downloading or quantization.

| Final source file | Required size (bytes) |
| --- | ---: |
| model-00047-of-00048.safetensors | 101535150936 |
| model-00048-of-00048.safetensors | 101537926640 |

The sizes agree in the previously retrieved public Hugging Face and ModelScope
metadata saved as `/tmp/v41-download-public-huggingface.json` and
`/tmp/v41-download-public-modelscope.json`. This is a file-size precheck;
the existing converter retains its source-identity, tensor-header and output
checksum checks.

The watcher reads only the source index, exact final tail-file paths, existing
conversion manifest and output publication markers. It neither scans nor
opens partial/download files. A final path must be a regular file of the
required size; a symlink or wrong size stops the process.

With no new published shard, the watcher waits without invoking conversion.
When a final shard is present but absent from the manifest, it synchronously
calls `examples/quantization/convert_deepseek_v41.py` with
`--allow-incomplete --threads 8`. The converter's existing nonblocking lock is
preserved. Each arriving shard triggers at most one call; two arriving together
can share one call. The watcher does not skip the converter's verification of
previously converted shards.

Any converter error, lock conflict, malformed manifest, or successful return
without the expected manifest progress stops the watcher with exit code 1.
No automatic retry occurs after such errors. All 48 manifest entries,
`complete=true`, and published output config/index are required for successful
exit. An inconsistent finalization state stops for inspection.

Run from the repository root using the conversion environment, after review:

```bash
../.venv/bin/python benchmarks/deepseek_v41/watch_conversion_tail.py \
  --source /mnt/models/DeepSeek-V4.1-Flash \
  --output /mnt/models/DeepSeek-V4.1-Flash-W4A16-G32
```

Polling defaults to 30 seconds. `--poll-seconds` accepts values in `(0, 60]`.
The process exits after completion; it does not install a persistent service.
It emits progress/error JSON on stdout/stderr and inherits converter output.

CPU validation uses synthetic four-byte files and a fake subprocess runner;
it never calls the converter or changes actual model files:

```bash
../.venv/bin/python -m pytest --confcutdir=tests/ut/quantization \
  tests/ut/quantization/test_watch_conversion_tail.py -q
```
