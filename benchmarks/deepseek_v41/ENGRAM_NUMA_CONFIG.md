# Explicit Engram NUMA placement

Set `engram_numa_nodes` in Ascend `additional_config` to place each TP rank's
host Engram table shards on a chosen NUMA node. The default is `null`, which
preserves the default pinned allocator. No automatic node selection occurs.

For the current eight-card host's audited CPU lookup topology:

```text
--additional-config '{"engram_numa_nodes": [6, 7, 4, 5, 0, 1, 2, 3]}'
```

Entry `r` is the host NUMA node for TP rank `r`, independent of its visible NPU
ordinal. Both Engram layers use that rank's node. The factory separately passes
the actual loaded model parameter device to host registration and staging.
This list is specific to the current host; inspect another host's CPU/NPU
mapping and memory capacity before choosing its explicit list.

The option requires DeepSeek V4.1 with nonempty `engram_layer_ids`. Its length
must equal tensor-parallel size. Every entry must be a nonnegative integer;
booleans, floating-point values and numeric strings are rejected. Repeated
nodes are allowed. The allocator validates that requested nodes exist and are
permitted, populates the final NUMA-bound allocation from selected checkpoint
head slices, and registers it as pinned memory. Invalid placement or failed
registration raises an error; it does not silently fall back to another node.

Factory failures close any previously loaded registered shards. If runtime
construction fails after staging initialization, the manager closes through
its synchronization and unregister path. Successful lifetime management is
owned by the Engram runtime/offload manager.

Placement affects host table storage; it does not configure CPU worker thread
affinity. See `ENGRAM_HOST_FACTORY_AUDIT.md` and the offload NUMA validation
report for measured registration, placement and access behavior. Recheck
complete TP8 decode timing with Engram lookup and H2D staging when evaluating
this setting.

CPU validation: 110 configuration cases and 32 factory cases pass, including
all eight TP ranks with remapped device ordinals and partial-initialization
cleanup. These tests validate configuration/wiring; actual NUMA registration
and transfer correctness are covered by the offload integration tests.
