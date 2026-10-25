"""Opt-in per-rank diagnostics for the persistent FFNO FSDP service.

Enabled by FFNO_FSDP_DIAGNOSTICS (0 off, the default; 1 progress around each
request and each top-level model stage; 2 also every submodule). Each rank
appends timestamped records to its own file, flushed per line so the record
survives a native crash:

    <FFNO_FSDP_RANK_DIAGNOSTICS_DIR>/rank-<rank>.log

The directory defaults to ``fsdp-ranks-<job>`` next to the service manifest.
Records cover library versions and loaded native libraries, input shapes,
strides, dtypes and finiteness, GPU memory before and after each request, and
progress through scatter, model stages and gather. In diagnostic mode the stage
hooks synchronize the GPU so asynchronous failures are attributed to the stage
that caused them; this is slow and is not a production setting.

FFNO_FSDP_CORE_DUMPS=1 raises the core-file limit to its hard maximum and runs
each rank in its own directory under the diagnostics directory, so ranks
sharing a node do not overwrite one another's ``core`` file.
"""

import os
from pathlib import Path
import platform
import resource
import socket
import sys
import time

import numpy as np
import torch


def diagnostics_level():
    raw = os.environ.get("FFNO_FSDP_DIAGNOSTICS", "0").strip()
    return int(raw) if raw else 0


class RankDiagnostics:
    """Line-flushed per-rank progress log; every method is a no-op when disabled."""

    def __init__(self, rank, local_rank, manifest_path, level=None):
        self.level = diagnostics_level() if level is None else level
        self.rank = rank
        self.local_rank = local_rank
        self.stream = None
        self.directory = None
        if self.level <= 0:
            return
        job = os.environ.get("SLURM_JOB_ID", str(os.getppid()))
        default = Path(manifest_path).resolve().parent / f"fsdp-ranks-{job}"
        self.directory = Path(
            os.environ.get("FFNO_FSDP_RANK_DIAGNOSTICS_DIR", default)
        ).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = open(
            self.directory / f"rank-{rank:04d}.log", "a", buffering=1
        )
        self.record(
            "start",
            host=socket.gethostname(),
            pid=os.getpid(),
            local_rank=local_rank,
            level=self.level,
        )

    @property
    def enabled(self):
        return self.stream is not None

    def record(self, event, **fields):
        if self.stream is None:
            return
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        fraction = f"{time.time() % 1:.3f}"[1:]
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        self.stream.write(f"{stamp}{fraction} rank={self.rank} {event} {details}\n")

    def enable_core_dumps(self):
        """Raise RLIMIT_CORE and move into a per-rank working directory."""
        if os.environ.get("FFNO_FSDP_CORE_DUMPS", "0") != "1":
            return
        soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
        resource.setrlimit(resource.RLIMIT_CORE, (hard, hard))
        base = self.directory or Path.cwd()
        directory = base / f"core-rank-{self.rank:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        os.chdir(directory)
        pattern = Path("/proc/sys/kernel/core_pattern")
        self.record(
            "core_dumps",
            rlimit_core=f"{soft}->{hard}",
            cwd=directory,
            core_pattern=(pattern.read_text().strip() if pattern.exists() else "?"),
        )

    def record_versions(self, device):
        if self.stream is None:
            return
        gpu = device.type == "cuda" and torch.cuda.is_available()
        properties = torch.cuda.get_device_properties(device) if gpu else None
        nccl = torch.cuda.nccl.version() if gpu else None
        self.record(
            "versions",
            python=platform.python_version(),
            executable=sys.executable,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            nccl=".".join(map(str, nccl)) if isinstance(nccl, tuple) else nccl,
            numpy=np.__version__,
            device=f"'{properties.name}'" if gpu else device,
            capability=f"{properties.major}.{properties.minor}" if gpu else "-",
        )
        for key in sorted(os.environ):
            if key.startswith(("NCCL_", "FI_", "CUDA_", "TORCH_", "OFI_", "CXI_")) or key in (
                "LD_LIBRARY_PATH",
                "PYTHONPATH",
                "OMP_NUM_THREADS",
            ):
                self.record("env", **{key: os.environ[key]})

    def record_libraries(self, label):
        """Loaded shared objects (NCCL network plugins load at the first communicator)."""
        if self.stream is None:
            return
        maps = Path("/proc/self/maps")
        if not maps.exists():
            return
        libraries = sorted(
            {
                line.split()[-1]
                for line in maps.read_text().splitlines()
                if ".so" in line.split()[-1]
            }
        )
        for library in libraries:
            self.record("library", stage=label, path=library)

    def record_memory(self, label, device, reset_peak=False):
        if self.stream is None or device.type != "cuda":
            return
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(device)
        free, total = torch.cuda.mem_get_info(device)
        self.record(
            "gpu_memory",
            stage=label,
            free_mib=free >> 20,
            total_mib=total >> 20,
            allocated_mib=torch.cuda.memory_allocated(device) >> 20,
            reserved_mib=torch.cuda.memory_reserved(device) >> 20,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) >> 20,
        )

    def record_array(self, label, value):
        """Shape, strides, dtype and finiteness of a host or device array."""
        if self.stream is None:
            return
        if isinstance(value, torch.Tensor):
            finite = bool(torch.isfinite(value).all())
            minimum = float(value.min()) if value.numel() else float("nan")
            maximum = float(value.max()) if value.numel() else float("nan")
            self.record(
                "array",
                name=label,
                kind="tensor",
                shape=tuple(value.shape),
                strides=tuple(value.stride()),
                dtype=value.dtype,
                device=value.device,
                contiguous=value.is_contiguous(),
                finite=finite,
                min=f"{minimum:.6g}",
                max=f"{maximum:.6g}",
            )
            return
        value = np.asarray(value)
        self.record(
            "array",
            name=label,
            kind="numpy",
            shape=value.shape,
            strides=value.strides,
            dtype=value.dtype,
            c_contiguous=value.flags.c_contiguous,
            finite=bool(np.isfinite(value).all()),
            min=f"{float(value.min()):.6g}" if value.size else "nan",
            max=f"{float(value.max()):.6g}" if value.size else "nan",
        )

    def install_model_hooks(self, name, model, device):
        """Record entry and exit of model stages; level 2 hooks every submodule."""
        if self.stream is None:
            return []
        root = getattr(model, "module", model)
        if self.level >= 2:
            targets = [
                (path, module) for path, module in root.named_modules() if path
            ]
        else:
            targets = [
                (path, module)
                for path, module in root.named_children()
                if path != "blocks"
            ]
            targets += [
                (f"blocks.{index}", block)
                for index, block in enumerate(getattr(root, "blocks", ()))
            ]

        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def before(path):
            def hook(module, inputs):
                synchronize()
                self.record("stage_begin", model=name, stage=path)

            return hook

        def after(path):
            def hook(module, inputs, output):
                synchronize()
                self.record("stage_end", model=name, stage=path)

            return hook

        handles = []
        for path, module in targets:
            handles.append(module.register_forward_pre_hook(before(path)))
            handles.append(module.register_forward_hook(after(path)))
        self.record("hooks_installed", model=name, stages=len(targets))
        return handles

    def close(self):
        if self.stream is not None:
            self.record("stop")
            self.stream.close()
            self.stream = None
