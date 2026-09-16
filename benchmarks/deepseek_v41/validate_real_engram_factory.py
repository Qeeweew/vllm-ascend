# SPDX-License-Identifier: Apache-2.0
"""Actual full Engram factory, 16 real owners, 144 source row oracles, small NPU staging.

CPU preparation is the default. --run requires a coordinated eight-card window.
No target/draft/vision model weights are allocated and no HCCL group is created.
The production factory runs with explicit rank/TP context supplied per process.
"""

import argparse
import gc
import json
import multiprocessing
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from full_engram_audit import audit_loading, inspect_tables, process_memory, replay_sample_rows
from preflight_full_model import GIB, NUMA_NODES, build_preflight
from probe_engram_full_capacity import GLOBAL_RESERVE, NODE_RESERVE, memory, node_available


def factory_admission(preflight):
    blockers = [reason for reason in preflight["blockers"] if "HBM" not in reason]
    for rank in range(8):
        free = preflight["hbm_memory"].get(str(rank), {}).get("free_bytes")
        if free is None or free < 4 * GIB:
            blockers.append(f"Factory rank {rank} requires verified 4 GiB free HBM for context/staging headroom")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "host_pinned_bytes": preflight["host_pinned_bytes"],
        "host_admission_bytes": preflight["host_admission_bytes"],
        "hbm_admission_bytes_per_rank": 4 * GIB,
        "expected_staging_payload_bytes_per_rank": 2 * 4 * 3 * 256 * 2,
        "framework_context_overhead_included_in_reserve": True,
    }


def rank_worker(rank, source, converted, pipe):
    runtime = None
    records = None
    success = False

    def emit(event):
        pipe.send({"rank": rank, "time": time.time(), **event})

    try:
        import torch
        import torch_npu  # noqa: F401

        import vllm_ascend.models.deepseek_v4.model as model_module

        torch.set_num_threads(4)
        torch.npu.set_device(rank)
        config = json.loads((Path(converted) / "config.json").read_text())["text_config"]
        payload = sum(
            (end - begin) * 512
            for owner in record_owners(source)
            if owner["rank"] == rank
            for begin, end in owner["ranges"]
        )
        if memory()["MemAvailable"] < payload + GLOBAL_RESERVE:
            raise RuntimeError("Worker global-memory reserve failed")
        if node_available(NUMA_NODES[rank]) < payload + NODE_RESERVE:
            raise RuntimeError("Worker NUMA-memory reserve failed")
        # Supply only the factory's device anchor, never a full model constructor.
        anchor = torch.empty(1, dtype=torch.bfloat16, device=torch.device("npu", rank))
        instance = SimpleNamespace(
            config=SimpleNamespace(**config),
            model=SimpleNamespace(parameters=lambda: iter([anchor])),
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(model=converted, trust_remote_code=False),
                scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
            ),
        )
        emit({"event": "factory_start", "memory": process_memory()})
        with (
            patch.object(model_module, "get_tensor_model_parallel_rank", return_value=rank),
            patch.object(model_module, "get_tensor_model_parallel_world_size", return_value=8),
            patch.object(model_module, "get_ascend_config", return_value=SimpleNamespace(engram_numa_nodes=NUMA_NODES)),
            audit_loading(callback=emit) as records,
        ):
            runtime = model_module.AscendDeepseekV41ForCausalLM.create_engram_runtime(instance)
        tables, samples = inspect_tables(runtime, source, converted, records)
        replay = replay_sample_rows(runtime, samples)
        if replay["device_reserved_bytes"] >= GIB:
            raise RuntimeError("Independent factory probe exceeded its 1 GiB allocator ceiling")
        emit(
            {
                "event": "ready",
                "tables": tables,
                "load_audit": records,
                "replay": replay,
                "memory": process_memory(),
                "tp_partition_context_injected": True,
                "full_model_or_hccl_loaded": False,
            }
        )
        if pipe.recv() != "release":
            raise RuntimeError("Unexpected controller message")
        success = True
    except BaseException as error:
        emit({"event": "failed", "error": f"{type(error).__name__}: {error}"})
    finally:
        errors = []
        if runtime is not None:
            try:
                runtime.shutdown()
                assert all(shard.weight is None and shard._pinned_owner is None for shard in runtime.offload.shards)
                unregisters = sum(event["event"] == "unregistered" for event in records["registration_events"])
                assert unregisters == 2
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
        gc.collect()
        emit(
            {
                "event": "released",
                "success": success,
                "cleanup_errors": errors,
                "registration_events": [] if records is None else records["registration_events"],
                "memory": process_memory(),
            }
        )
        pipe.close()


def record_owners(source):
    from vllm_ascend.ops.engram_hash import HostEngramLayout

    text = json.loads((Path(source) / "config.json").read_text())["text_config"]
    layout = HostEngramLayout.from_config(SimpleNamespace(**text))
    owners = []
    for rank in range(8):
        for layer in range(2):
            heads, ranges = layout.head_shard(layer, rank, 8)
            owners.append(
                {
                    "rank": rank,
                    "node": NUMA_NODES[rank],
                    "heads": heads,
                    "ranges": ranges,
                    "bytes": sum(end - begin for begin, end in ranges) * 512,
                }
            )
    return owners


def final_status(report):
    if report["status"] != "resident_checks_passed_cleanup_pending":
        return report["status"]
    releases = [event for event in report["events"] if event["event"] == "released"]
    exits = [event for event in report["events"] if event["event"] == "worker_exit"]
    complete = (
        len(releases) == 8
        and {event["rank"] for event in releases} == set(range(8))
        and all(
            event["success"]
            and not event["cleanup_errors"]
            and sum(item["event"] == "unregistered" for item in event["registration_events"]) == 2
            for event in releases
        )
        and len(exits) == 8
        and len({event["pid"] for event in exits}) == 8
        and all(event["exitcode"] == 0 for event in exits)
    )
    return "passed" if complete else "failed_cleanup"


def main():
    """Admit complete real tables, retain all TP owners together, then require every release acknowledgement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash"))
    parser.add_argument("--converted", type=Path, default=Path("/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--stage-timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    preflight = build_preflight(args.source, args.converted)
    admission = factory_admission(preflight)
    report = {
        "status": "prepared_only",
        "preflight": preflight,
        "factory_admission": admission,
        "events": [],
        "real_checkpoint_rows": True,
        "full_model_validation": False,
        "hccl_validation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def emit(event):
        report["events"].append(event)
        print(
            json.dumps({key: value for key, value in event.items() if key not in {"tables", "load_audit"}}), flush=True
        )
        temporary = args.output.with_suffix(".new.json")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temporary, args.output)

    def receive(pipe, process, target):
        deadline = time.monotonic() + args.stage_timeout
        while time.monotonic() < deadline:
            if pipe.poll(1):
                event = pipe.recv()
                emit(event)
                if event["event"] == target:
                    return event
                if event["event"] == "failed" and target != "released":
                    raise RuntimeError(event["error"])
            elif not process.is_alive():
                raise RuntimeError(f"Owned worker {process.pid} exited before {target}")
        raise TimeoutError(f"Owned worker {process.pid} did not reach {target}")

    workers = []
    if args.run and not admission["ready"]:
        report["status"] = "blocked_before_launch"
    elif args.run:
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7") != "0,1,2,3,4,5,6,7":
            parser.error("This host-specific NUMA/HBM admission requires physical devices 0–7 in order")
        context = multiprocessing.get_context("spawn")
        owners = record_owners(args.source)
        report["status"] = "running"
        try:
            for rank in range(8):
                remaining = sum(owner["bytes"] for owner in owners if owner["rank"] >= rank)
                local = sum(owner["bytes"] for owner in owners if owner["rank"] == rank)
                if memory()["MemAvailable"] < remaining + GLOBAL_RESERVE:
                    raise RuntimeError("Controller global host reserve failed")
                if node_available(NUMA_NODES[rank]) < local + NODE_RESERVE:
                    raise RuntimeError("Controller per-node reserve failed")
                parent, child = context.Pipe()
                process = context.Process(target=rank_worker, args=(rank, str(args.source), str(args.converted), child))
                process.start()
                child.close()
                workers.append((process, parent))
                receive(parent, process, "ready")
            assert all(process.is_alive() for process, _ in workers)
            report["status"] = "resident_checks_passed_cleanup_pending"
            emit({"event": "all_16_real_owners_resident", "payload_bytes": preflight["host_pinned_bytes"]})
        except BaseException as error:
            report["status"] = "failed"
            emit({"event": "controller_failed", "error": f"{type(error).__name__}: {error}"})
        finally:
            for process, pipe in reversed(workers):
                try:
                    if process.is_alive():
                        pipe.send("release")
                    released = receive(pipe, process, "released")
                    if released["cleanup_errors"] or not released["success"]:
                        report["status"] = "failed_cleanup"
                    process.join(timeout=60)
                    if process.is_alive():
                        report["status"] = "failed_cleanup"
                        process.terminate()
                        process.join(timeout=30)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=30)
                    if process.exitcode != 0:
                        report["status"] = "failed_cleanup"
                    emit({"event": "worker_exit", "pid": process.pid, "exitcode": process.exitcode})
                except Exception as error:
                    report["status"] = "failed_cleanup"
                    emit({"event": "cleanup_failed", "pid": process.pid, "error": str(error)})
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=30)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=30)
                finally:
                    pipe.close()
    report["status"] = final_status(report)
    emit({"event": "finished", "status": report["status"]})
    return 0 if report["status"] in {"passed", "prepared_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
