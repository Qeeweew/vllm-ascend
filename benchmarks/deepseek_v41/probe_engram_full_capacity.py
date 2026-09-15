#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage real-size synthetic Engram registrations, hold all eight ranks, free.

No checkpoint pages are read. Each rank has an isolated process and its own
NPU context. Allocation/first touch/registration are strictly sequential.
Only after all ranks report ready is cumulative capacity considered tested.
"""

import argparse
import ctypes
import gc
import json
import multiprocessing
import os
import time
from pathlib import Path
from types import SimpleNamespace

GIB = 1024**3
NUMA_NODES = (6, 6, 4, 4, 0, 0, 2, 2)
GLOBAL_RESERVE = 256 * GIB
NODE_RESERVE = 32 * GIB


def memory():
    return {
        line.split()[0].rstrip(":"): int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
    }


def rss():
    return {
        parts[0].rstrip(":"): int(parts[1]) * 1024
        for line in Path("/proc/self/status").read_text().splitlines()
        if (parts := line.split()) and parts[0] in ("VmRSS:", "VmLck:", "VmPin:")
    }


def node_available(node):
    values = {}
    for line in (Path("/sys/devices/system/node") / f"node{node}" / "meminfo").read_text().splitlines():
        parts = line.split()
        values[parts[2].rstrip(":")] = int(parts[3]) * 1024
    return values["MemFree"] + values["FilePages"] - values["Shmem"] + values["SReclaimable"]


def sample_placement(tensor, node):
    size = tensor.numel() * tensor.element_size()
    page = os.sysconf("SC_PAGE_SIZE")
    offsets = sorted({0, (size // 2) // page * page, (size - 1) // page * page, *range(0, size, GIB)})
    addresses = (ctypes.c_void_p * len(offsets))(*(tensor.data_ptr() + offset for offset in offsets))
    status = (ctypes.c_int * len(offsets))()
    numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
    numa.move_pages.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    numa.move_pages.restype = ctypes.c_long
    result = numa.move_pages(0, len(offsets), addresses, None, status, 0)
    if result != 0 or any(value != node for value in status):
        raise RuntimeError(f"NUMA placement mismatch: return={result}, nodes={set(status)}")
    from audit_engram_numa import mapping

    return {"sample_pages": len(offsets), "sample_nodes": sorted(set(status)), "mapping": mapping(tensor.data_ptr())}


def rank_worker(record, pipe):
    from unittest.mock import patch

    import torch
    import torch_npu  # noqa: F401

    import vllm_ascend.ops.engram_pinned_host as pinned
    from vllm_ascend.ops.engram_offload import DEAD_HASH_ID, EngramOffloadManager, EngramTableShard

    rank, node = record["rank"], record["node"]
    owners, shards, samples = [], [], []
    manager = None
    owner = None
    success = False

    def send(event, **fields):
        pipe.send({"event": event, "rank": rank, "time": time.time(), **fields})

    class TimedAPI(pinned._HostMemoryAPI):
        def register(self, pointer, size):
            send("register_start", bytes=size)
            start = time.monotonic()
            try:
                super().register(pointer, size)
            except Exception as error:
                send("register_failed", seconds=time.monotonic() - start, error=str(error))
                raise
            send("registered", bytes=size, seconds=time.monotonic() - start)

        def unregister(self, pointer):
            start = time.monotonic()
            super().unregister(pointer)
            send("unregistered", seconds=time.monotonic() - start)

    try:
        torch.set_num_threads(4)
        torch.npu.set_device(rank)
        device = torch.device("npu", rank)
        send("baseline", rss=rss(), device_free_bytes=torch.npu.mem_get_info()[0])
        with patch.object(pinned, "_HostMemoryAPI", TimedAPI):
            for layer, spec in enumerate(record["layers"]):
                if memory()["MemAvailable"] < spec["bytes"] + GLOBAL_RESERVE:
                    raise RuntimeError("global memory reserve would be violated before layer allocation")
                if node_available(node) < spec["bytes"] + NODE_RESERVE:
                    raise RuntimeError("NUMA memory reserve would be violated before layer allocation")
                rows = spec["bytes"] // 512
                selected, values, starts = [], [], []
                cursor = 0
                for head, (begin, end) in enumerate(spec["ranges"]):
                    starts.append(cursor)
                    for position in (0, (end - begin) // 2, end - begin - 1):
                        selected.append(cursor + position)
                        values.append((rank * 13 + layer * 7 + len(values)) % 127 + 1)
                    cursor += end - begin

                def initialize(tensor, layer=layer, spec=spec, selected=selected, values=values):
                    send("first_touch_start", layer=layer, bytes=spec["bytes"])
                    start = time.monotonic()
                    tensor.zero_()
                    for row, value in zip(selected, values):
                        tensor[row].fill_(value)
                    send("first_touch_done", layer=layer, seconds=time.monotonic() - start)

                owner = pinned.EngramPinnedHostTensor((rows, 256), numa_node=node, device=device, initialize=initialize)
                owners.append(owner)
                placement = sample_placement(owner.tensor, node)
                destination = torch.empty((len(selected), 256), dtype=torch.bfloat16, device=device)
                for index, row in enumerate(selected):
                    destination[index].copy_(owner.tensor[row], non_blocking=True)
                done = torch.npu.Event()
                done.record()
                owner.record_event(done)
                done.synchronize()
                expected = torch.tensor(values, dtype=torch.bfloat16)[:, None].expand(len(values), 256)
                torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
                shard = EngramTableShard(owner.tensor, spec["heads"], spec["ranges"])
                shards.append(shard)
                samples.append((selected, values, starts))
                send("layer_ready", layer=layer, bytes=spec["bytes"], placement=placement, rss=rss(), direct_dma=True)

        manager = EngramOffloadManager(shards, max_tokens=4, device=device)
        outputs = tuple(torch.empty_like(rows) for rows in manager.device_rows)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for output, device_rows in zip(outputs, manager.device_rows):
                output.copy_(device_rows + 1)
        torch.npu.synchronize()
        for step in range(20):
            hashes, expected_layers = [], []
            for spec, (selected, values, starts) in zip(record["layers"], samples):
                ids = torch.zeros((3, 24), dtype=torch.int64)
                expected = torch.ones((4, 3, 256), dtype=torch.bfloat16)
                for local_head, (head, (begin, _)) in enumerate(zip(spec["heads"], spec["ranges"])):
                    for token in range(3):
                        sample = local_head * 3 + (token + step) % 3
                        ids[token, head] = begin + selected[sample] - starts[local_head]
                        expected[token, local_head].fill_(values[sample] + 1)
                    if step % 2:
                        ids[2, head] = DEAD_HASH_ID
                        expected[2, local_head].fill_(1)
                hashes.append(ids)
                expected_layers.append(expected)
            manager.prepare(hashes, 4)
            manager.wait_ready()
            graph.replay()
            manager.mark_consumed()
            torch.npu.synchronize()
            for actual, expected in zip(outputs, expected_layers):
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        reserved = torch.npu.memory_reserved()
        if reserved >= GIB:
            raise RuntimeError(f"probe NPU allocation exceeded 1 GiB: {reserved}")
        send(
            "ready",
            bytes=sum(spec["bytes"] for spec in record["layers"]),
            graph_replays=20,
            rss=rss(),
            device_reserved_bytes=reserved,
            device_allocated_bytes=torch.npu.memory_allocated(),
        )
        command = pipe.recv()
        if command != "release":
            raise RuntimeError("unexpected controller command")
        success = True
    except BaseException as error:
        send("failed", error_type=type(error).__name__, error=str(error))
    finally:
        errors = []
        try:
            torch.npu.synchronize()
        except Exception as error:
            errors.append(f"synchronize: {error}")
        if manager is not None:
            try:
                if manager._prepared:
                    if not manager._waited:
                        manager.wait_ready()
                    manager.mark_consumed()
                manager.close()
            except Exception as error:
                errors.append(f"manager.close: {error}")
        for shard in shards:
            shard.weight = None
        for owner in reversed(owners):
            try:
                owner.close()
            except Exception as error:
                errors.append(f"owner.close: {error}")
        owner = None
        owners.clear()
        shards.clear()
        manager = None
        gc.collect()
        send("released", success=success, cleanup_errors=errors, rss=rss())
        pipe.close()


def build_records(config_path):
    from vllm_ascend.ops.engram_hash import HostEngramLayout

    config = json.loads(config_path.read_text())
    config = SimpleNamespace(**config.get("text_config", config))
    layout = HostEngramLayout.from_config(config)
    records = []
    for rank in range(8):
        layers = []
        for layer in range(len(layout.layer_ids)):
            heads, ranges = layout.head_shard(layer, rank, 8)
            layers.append({"heads": heads, "ranges": ranges, "bytes": sum(b - a for a, b in ranges) * 512})
        records.append({"rank": rank, "node": NUMA_NODES[rank], "layers": layers})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash/config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-timeout", type=float, default=600)
    args = parser.parse_args()
    records = build_records(args.config)
    context = multiprocessing.get_context("spawn")
    workers, events = [], []
    complete = False

    def emit(event):
        events.append(event)
        print(json.dumps(event), flush=True)
        temporary = args.output.with_suffix(".json.new")
        temporary.write_text(
            json.dumps(
                {"complete": complete and event["event"] == "finished", "records": records, "events": events},
                indent=2,
            )
        )
        os.replace(temporary, args.output)

    def receive_until(pipe, process, stop_event, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pipe.poll(1):
                event = pipe.recv()
                emit(event)
                if event["event"] == stop_event:
                    return event
                if event["event"] == "failed" and stop_event != "released":
                    raise RuntimeError(f"rank {event['rank']} failed: {event['error']}")
            elif not process.is_alive():
                raise RuntimeError(f"our probe process {process.pid} exited before {stop_event}")
        raise TimeoutError(f"our probe process {process.pid} timed out before {stop_event}")

    try:
        for index, record in enumerate(records):
            remaining = sum(spec["bytes"] for record in records[index:] for spec in record["layers"])
            local_remaining = sum(
                spec["bytes"]
                for candidate in records[index:]
                if candidate["node"] == record["node"]
                for spec in candidate["layers"]
            )
            available = memory()["MemAvailable"]
            local_available = node_available(record["node"])
            if available < remaining + GLOBAL_RESERVE or local_available < local_remaining + NODE_RESERVE:
                raise RuntimeError("controller host/NUMA admission reserve failed")
            emit(
                {
                    "event": "admitted",
                    "rank": index,
                    "available_bytes": available,
                    "node_available_estimate_bytes": local_available,
                    "remaining_bytes": remaining,
                }
            )
            parent, child = context.Pipe()
            process = context.Process(target=rank_worker, args=(record, child))
            process.start()
            child.close()
            workers.append((process, parent))
            receive_until(parent, process, "ready", args.stage_timeout)
        if not all(process.is_alive() for process, _ in workers):
            raise RuntimeError("a previously ready rank exited before simultaneous residency was established")
        emit(
            {
                "event": "all_ranks_resident",
                "bytes": sum(spec["bytes"] for r in records for spec in r["layers"]),
                "mem_available_bytes": memory()["MemAvailable"],
            }
        )
        complete = True
    except BaseException as error:
        emit({"event": "controller_failed", "error_type": type(error).__name__, "error": str(error)})
    finally:
        # Release in reverse order. No signal targets any process not spawned here.
        for process, pipe in reversed(workers):
            try:
                if process.is_alive():
                    pipe.send("release")
                    released = receive_until(pipe, process, "released", args.stage_timeout)
                    if released["cleanup_errors"]:
                        complete = False
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=30)
                    complete = False
                if process.is_alive():
                    process.kill()
                    process.join(timeout=30)
                    complete = False
            except BaseException as error:
                complete = False
                emit({"event": "cleanup_failed", "pid": process.pid, "error": str(error)})
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=30)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=30)
            finally:
                pipe.close()
            emit({"event": "worker_exit", "pid": process.pid, "exitcode": process.exitcode})
            if process.exitcode != 0:
                complete = False
        emit({"event": "finished", "complete": complete, "mem_available_bytes": memory()["MemAvailable"]})
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
