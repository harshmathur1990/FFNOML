#!/usr/bin/env python3
"""Minimal CUDA/NCCL probe used by the Olivia runtime test harness."""

import datetime
import argparse
import faulthandler
import os
from pathlib import Path
import threading
import time

import torch
import torch.distributed as dist


parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("success", "failure", "stall"))
parser.add_argument("--local-rank", "--local_rank", type=int)
arguments, unknown_arguments = parser.parse_known_args()
MODE = arguments.mode

diagnostics = Path(os.environ.get("OLIVIA_CASE_DIAGNOSTICS", "."))
diagnostics.mkdir(parents=True, exist_ok=True)
global_rank = int(os.environ.get("RANK", "-1"))
local_rank = arguments.local_rank
if local_rank is None:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
log_path = diagnostics / f"gpu-rank-{global_rank}.log"
log_stream = log_path.open("a", buffering=1)
faulthandler.enable(file=log_stream, all_threads=True)
faulthandler.dump_traceback_later(15, repeat=True, file=log_stream)


def record(event, **fields):
    values = " ".join(f"{key}={value}" for key, value in fields.items())
    log_stream.write(
        f"{time.time():.6f} rank={global_rank} local_rank={local_rank} pid={os.getpid()} "
        f"host={os.uname().nodename} event={event} {values}\n"
    )


stop_watchdog = threading.Event()


def watchdog():
    while not stop_watchdog.wait(5):
        record("watchdog_alive")


threading.Thread(target=watchdog, name="olivia-gpu-watchdog", daemon=True).start()

try:
    record("python_started", torch=torch.__version__, cuda=torch.version.cuda,
           ignored_arguments=repr(unknown_arguments))
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    torch.cuda.set_device(local_rank)
    record(
        "cuda_device_selected",
        device=torch.cuda.get_device_name(local_rank).replace(" ", "_"),
        visible=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    global_rank = dist.get_rank()
    world = dist.get_world_size()
    record("nccl_initialized", world=world)

    value = torch.tensor([float(global_rank + 1)], device="cuda")
    torch.cuda.synchronize()
    record("cuda_smoke_complete")
    dist.barrier()
    record("initial_collective_complete")

    if MODE == "failure":
        if global_rank == 0:
            record("intentional_gpu_process_failure")
            os._exit(23)
        time.sleep(2)
        record("peer_waiting_for_launcher_cleanup")
        while True:
            time.sleep(60)

    if MODE == "stall" and global_rank == 0:
        record("intentional_gpu_stall_enter")
        while True:
            time.sleep(60)

    record("allreduce_enter")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    expected = world * (world + 1) / 2
    actual = value.item()
    if actual != expected:
        raise RuntimeError(f"NCCL allreduce returned {actual}, expected {expected}")
    record("allreduce_complete", value=actual)
    dist.barrier()
    if global_rank == 0:
        print(f"OLIVIA_GPU_PROBE_OK world={world} sum={actual}", flush=True)
finally:
    stop_watchdog.set()
    faulthandler.cancel_dump_traceback_later()
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as exception:
            record("destroy_process_group_failed", error=repr(exception))
    record("python_exiting")
    log_stream.close()
