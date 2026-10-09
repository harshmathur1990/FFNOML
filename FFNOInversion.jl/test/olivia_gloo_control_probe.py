#!/usr/bin/env python3
"""Short mixed NCCL/Gloo control-plane smoke test for Olivia."""

import datetime
import os
import time

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    timeout = datetime.timedelta(seconds=60)
    dist.init_process_group("nccl", timeout=timeout)
    rank = dist.get_rank()
    world = dist.get_world_size()
    command_group = None

    try:
        command_group = dist.new_group(backend="gloo", timeout=timeout)
        dist.barrier(group=command_group)

        idle_seconds = float(os.environ.get("FFNO_GLOO_IDLE_SECONDS", "3"))
        command = [{"operation": "GLOO_SMOKE", "sequence": 1} if rank == 0 else None]
        if rank == 0:
            time.sleep(idle_seconds)
        wait_started = time.monotonic()
        dist.broadcast_object_list(command, src=0, group=command_group)
        wait_seconds = time.monotonic() - wait_started
        if command[0] != {"operation": "GLOO_SMOKE", "sequence": 1}:
            raise RuntimeError(f"rank {rank} received an invalid Gloo command: {command[0]!r}")

        waits = [None] * world
        dist.all_gather_object(waits, wait_seconds, group=command_group)
        worker_waits = waits[1:]
        if worker_waits and min(worker_waits) < idle_seconds * 0.75:
            raise RuntimeError(
                f"Gloo workers did not remain blocked for the root delay: waits={waits}"
            )

        value = torch.tensor([rank + 1.0], device="cuda")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        expected = world * (world + 1) / 2
        if value.item() != expected:
            raise RuntimeError(
                f"NCCL all-reduce returned {value.item()}, expected {expected}"
            )

        if rank == 0:
            formatted_waits = ",".join(f"{value:.3f}" for value in waits)
            print(
                f"OLIVIA_GLOO_CONTROL_OK world={world} "
                f"worker_wait_seconds={formatted_waits} nccl_sum={value.item()}",
                flush=True,
            )
    finally:
        if command_group is not None:
            dist.destroy_process_group(command_group)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
