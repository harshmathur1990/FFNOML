#!/usr/bin/env python3
"""FFNO service for forward runs on small GPUs: the wire protocol of ffno_fsdp_service.py, answering PREDICT from host RAM
(ffno_batched_backend.py). One process, no torchrun. VJP and JVP are refused: an inversion needs the FSDP service.

Environment: CUDA_VISIBLE_DEVICES picks the GPUs. FFNO_BATCHED_CHUNK_DEPTH (16), FFNO_BATCHED_CHUNK_ROWS (16),
FFNO_BATCHED_VERTICAL_CHUNK (1024) set the batches and do not change the result. FFNO_BATCHED_VERTICAL_NORM_SLABS (1) does:
see ffno_batched_backend.py."""

import argparse
import importlib
import os
from pathlib import Path
import socket
import tomllib

import numpy as np

from ffno_batched_backend import BatchedFFNOBackend
from ffno_fsdp_service import PROTOCOL_VERSION, read_array, write_array


def _log(message):
    print(message, flush=True)


def build_backends(configuration):
    backends = {}
    for model in sorted(configuration["models"], key=lambda item: item["name"]):
        factory = getattr(importlib.import_module(model.get("factory_module", "ffno_model_factory")),
                          model.get("factory_name", "create_ffno3d"))
        backends[str(model["name"])] = BatchedFFNOBackend(
            str(Path(model["checkpoint_path"]).resolve()),
            lambda Cin, Cout, device, kwargs=dict(model.get("factory_kwargs", {})), make=factory: make(
                Cin=Cin, Cout=Cout, device=device, **kwargs),
            chunk_depth=int(os.environ.get("FFNO_BATCHED_CHUNK_DEPTH", 16)),
            chunk_rows=int(os.environ.get("FFNO_BATCHED_CHUNK_ROWS", 16)),
            vertical_chunk=int(os.environ.get("FFNO_BATCHED_VERTICAL_CHUNK", 1024)),
            vertical_norm_slabs=int(os.environ.get("FFNO_BATCHED_VERTICAL_NORM_SLABS", 1)),
            log=_log,
        )
    return backends


def serve(configuration, backends):
    service = configuration["service"]
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((service.get("bind_host", "0.0.0.0"), int(service["port"])))
    listener.listen(1)
    gpus = len(next(iter(backends.values())).devices)
    _log(f"FFNO_BATCHED_SERVICE_LISTENING port={service['port']} gpus={gpus} models={','.join(sorted(backends))}")
    stream, address = listener.accept()
    stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    reader = stream.makefile("rb", buffering=0)
    hello = reader.readline().decode("utf-8").strip().split()
    if len(hello) != 2 or hello[0] != "HELLO" or hello[1] != service["token"]:
        raise RuntimeError("invalid service handshake")
    stream.sendall(f"READY {PROTOCOL_VERSION} {gpus} {len(backends)} BATCHED_FORWARD\n".encode("utf-8"))
    _log(f"FFNO_BATCHED_SERVICE_CLIENT_CONNECTED address={address}")
    while True:
        line = reader.readline()
        command = tuple(line.decode("utf-8").strip().split()) if line else ("DISCONNECT",)
        operation = command[0]
        if operation == "DISCONNECT":
            break
        if operation == "SHUTDOWN":
            stream.sendall(b"BYE\n")
            break
        if len(command) != 8 or operation not in ("PREDICT", "VJP", "JVP"):
            raise RuntimeError(f"invalid service command: {command}")
        _, name, nz, nx, ny, levels, dx, dy = command
        nz, nx, ny, levels = map(int, (nz, nx, ny, levels))
        # the request's arrays are on the wire whatever the answer: read them to keep the stream in step
        features = read_array(stream, (6, nz, nx, ny))
        z = read_array(stream, (nz, nx, ny))
        if operation != "PREDICT":
            read_array(stream, (nz, nx, ny, levels) if operation == "VJP" else (6, nz, nx, ny))
            stream.sendall(f"ERROR {operation} is not available in the batched forward service\n".encode("utf-8"))
            continue
        backend = backends[name]
        if levels != backend.levels:
            raise ValueError("request level count differs from the model")
        populations = backend.predict_si(features, z, float(dx), float(dy))
        stream.sendall(f"OK PREDICT {nz} {nx} {ny} {levels}\n".encode("utf-8"))
        write_array(stream, populations)
        _log(f"FFNO_BATCHED_SERVICE_COMMAND_OK model={name} shape={nz}x{nx}x{ny}")
    reader.close()
    stream.close()
    listener.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    manifest = Path(parser.parse_args().manifest).resolve()
    with manifest.open("rb") as stream:
        configuration = tomllib.load(stream)
    serve(configuration, build_backends(configuration))


if __name__ == "__main__":
    main()
