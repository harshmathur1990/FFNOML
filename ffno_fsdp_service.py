#!/usr/bin/env python3
"""Persistent socket service for multi-GPU FSDP FFNO prediction and VJPs."""

import argparse
import faulthandler
import os
from pathlib import Path
import socket
import sys
import tomllib
import traceback

import numpy as np
import torch
import torch.distributed as dist

from distributed_inference import partition_range
from ffno_fsdp_runtime import DistributedFSDPBackend


PROTOCOL_VERSION = 2


def read_exact(stream, count):
    chunks = bytearray(count)
    view = memoryview(chunks)
    offset = 0
    while offset < count:
        received = stream.recv_into(view[offset:])
        if received == 0:
            raise EOFError("FSDP client disconnected during array transfer")
        offset += received
    return bytes(chunks)


def read_array(stream, shape):
    count = int(np.prod(shape))
    raw = read_exact(stream, count * np.dtype(np.float32).itemsize)
    return np.frombuffer(raw, dtype="<f4").reshape(shape, order="F").copy()


def write_array(stream, value):
    payload = np.asfortranarray(value, dtype="<f4").tobytes(order="F")
    stream.sendall(payload)


def broadcast_command(command, device):
    values = [command if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(values, src=0, device=device)
    return values[0]


def scatter_h(global_value, global_shape, h_axis, device):
    rank = dist.get_rank()
    world = dist.get_world_size()
    h_global = global_shape[h_axis]
    h0, h1 = partition_range(h_global, rank, world)
    local_shape = list(global_shape)
    local_shape[h_axis] = h1 - h0
    if rank == 0:
        local = None
        for destination in range(world):
            d0, d1 = partition_range(h_global, destination, world)
            selection = [slice(None)] * len(global_shape)
            selection[h_axis] = slice(d0, d1)
            part = torch.from_numpy(
                np.ascontiguousarray(global_value[tuple(selection)], dtype=np.float32)
            ).to(device)
            if destination == 0:
                local = part
            else:
                dist.send(part, dst=destination)
        return local
    local = torch.empty(tuple(local_shape), dtype=torch.float32, device=device)
    dist.recv(local, src=0)
    return local


def gather_h(local_value, global_shape, h_axis, device):
    rank = dist.get_rank()
    world = dist.get_world_size()
    h_global = global_shape[h_axis]
    local = torch.from_numpy(
        np.ascontiguousarray(local_value, dtype=np.float32)
    ).to(device)
    if rank != 0:
        dist.send(local, dst=0)
        return None
    global_value = np.empty(global_shape, dtype=np.float32, order="F")
    for source in range(world):
        h0, h1 = partition_range(h_global, source, world)
        local_shape = list(global_shape)
        local_shape[h_axis] = h1 - h0
        part = local if source == 0 else torch.empty(
            tuple(local_shape), dtype=torch.float32, device=device
        )
        if source != 0:
            dist.recv(part, src=source)
        selection = [slice(None)] * len(global_shape)
        selection[h_axis] = slice(h0, h1)
        global_value[tuple(selection)] = part.detach().cpu().numpy()
    return global_value


def build_backends(configuration):
    backends = {}
    for model in sorted(configuration["models"], key=lambda item: item["name"]):
        name = str(model["name"])
        backends[name] = DistributedFSDPBackend(
            checkpoint_path=model["checkpoint_path"],
            factory_module=model.get("factory_module", "ffno_model_factory"),
            factory_name=model.get("factory_name", "create_ffno3d"),
            level_names=model["level_names"],
            factory_kwargs=model.get("factory_kwargs", {}),
            require_multi_gpu=True,
        )
    return backends


def serve(configuration, backends):
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank)
    service = configuration["service"]
    stream = None
    listener = None
    reader = None
    if rank == 0:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((service.get("bind_host", "0.0.0.0"), int(service["port"])))
        listener.listen(1)
        print(
            f"FFNO_FSDP_SERVICE_LISTENING port={service['port']} world={world} "
            f"models={','.join(sorted(backends))}",
            flush=True,
        )
        stream, address = listener.accept()
        stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        reader = stream.makefile("rb", buffering=0)
        hello = reader.readline().decode("utf-8").strip().split()
        if len(hello) != 2 or hello[0] != "HELLO" or hello[1] != service["token"]:
            raise RuntimeError("invalid FSDP service handshake")
        stream.sendall(
            f"READY {PROTOCOL_VERSION} {world} {len(backends)} "
            "FULL_SHARD_H_SLAB\n".encode("utf-8")
        )
        print(f"FFNO_FSDP_SERVICE_CLIENT_CONNECTED address={address}", flush=True)

    command_count = 0
    while True:
        if rank == 0:
            line = reader.readline()
            if not line:
                command = ("DISCONNECT",)
            else:
                fields = line.decode("utf-8").strip().split()
                command = tuple(fields)
        else:
            command = None
        command = broadcast_command(command, device)
        operation = command[0]
        if operation in ("SHUTDOWN", "DISCONNECT"):
            if rank == 0 and operation == "SHUTDOWN":
                stream.sendall(b"BYE\n")
            break
        if operation not in ("PREDICT", "VJP") or len(command) != 8:
            raise RuntimeError(f"invalid FSDP service command: {command}")
        _, name, nz_raw, nx_raw, ny_raw, levels_raw, dx_raw, dy_raw = command
        if name not in backends:
            raise KeyError(f"unknown FSDP model {name!r}")
        nz, nx, ny, levels = map(int, (nz_raw, nx_raw, ny_raw, levels_raw))
        if nx < world:
            raise ValueError(
                f"global nx={nx} cannot be split over FSDP world size {world}"
            )
        backend = backends[name]
        if levels != len(backend.level_names):
            raise ValueError("request level count differs from FSDP model")
        dx = float(dx_raw)
        dy = float(dy_raw)
        if rank == 0:
            features_global = read_array(stream, (6, nz, nx, ny))
            z_global = read_array(stream, (nz, nx, ny))
            population_bar_global = (
                read_array(stream, (nz, nx, ny, levels))
                if operation == "VJP"
                else None
            )
        else:
            features_global = None
            z_global = None
            population_bar_global = None

        features_tensor = scatter_h(
            features_global, (6, nz, nx, ny), 2, device
        )
        z_tensor = scatter_h(z_global, (nz, nx, ny), 1, device)
        features_local = features_tensor.cpu().numpy()
        z_local = z_tensor.cpu().numpy()
        if operation == "PREDICT":
            populations_local = backend.predict_local(
                features_local, z_local, dx, dy
            )
            populations_global = gather_h(
                populations_local, (nz, nx, ny, levels), 1, device
            )
            if rank == 0:
                stream.sendall(f"OK PREDICT {nz} {nx} {ny} {levels}\n".encode())
                write_array(stream, populations_global)
        else:
            population_bar_tensor = scatter_h(
                population_bar_global, (nz, nx, ny, levels), 1, device
            )
            result = backend.vjp_local(
                features_local,
                z_local,
                dx,
                dy,
                population_bar_tensor.cpu().numpy(),
            )
            features_bar_global = gather_h(
                result["features"], (6, nz, nx, ny), 2, device
            )
            z_bar_global = gather_h(
                result["z_scale"], (nz, nx, ny), 1, device
            )
            if rank == 0:
                stream.sendall(f"OK VJP {nz} {nx} {ny} {levels}\n".encode())
                write_array(stream, features_bar_global)
                write_array(stream, z_bar_global)
        command_count += 1
        if rank == 0:
            print(
                f"FFNO_FSDP_SERVICE_COMMAND_OK count={command_count} "
                f"operation={operation} model={name} shape={nz}x{nx}x{ny}",
                flush=True,
            )

    if reader is not None:
        reader.close()
    if stream is not None:
        stream.close()
    if listener is not None:
        listener.close()
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    arguments = parser.parse_args()
    manifest = Path(arguments.manifest).resolve()
    with manifest.open("rb") as stream:
        configuration = tomllib.load(stream)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group(
            "nccl", device_id=torch.device("cuda", local_rank)
        )
    except TypeError:
        dist.init_process_group("nccl")
    faulthandler.enable(all_threads=True)
    interval = int(os.environ.get("FFNO_FSDP_TRACEBACK_INTERVAL", "60"))
    if interval > 0:
        faulthandler.dump_traceback_later(interval, repeat=True)
    try:
        backends = build_backends(configuration)
        serve(configuration, backends)
    except BaseException:
        print(
            f"FFNO_FSDP_SERVICE_FAILURE rank={dist.get_rank()}\n{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        if interval > 0:
            faulthandler.cancel_dump_traceback_later()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
