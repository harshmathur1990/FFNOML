#!/usr/bin/env python3
"""Minimal CUDA/NCCL probe used by the Olivia runtime test harness."""

import datetime
import argparse
import faulthandler
import os
from pathlib import Path
import threading
import time
import traceback

import torch
import torch.distributed as dist


parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("success", "failure", "stall", "vjp", "ffno_vjp"))
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
current_stage = "startup"


def watchdog():
    while not stop_watchdog.wait(5):
        record("watchdog_alive")


threading.Thread(target=watchdog, name="olivia-gpu-watchdog", daemon=True).start()

try:
    current_stage = "python_started"
    record("python_started", torch=torch.__version__, cuda=torch.version.cuda,
           ignored_arguments=repr(unknown_arguments))
    current_stage = "cuda_availability_check"
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    record(
        "cuda_runtime_available",
        device_count=torch.cuda.device_count(),
        visible=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
    current_stage = "cuda_device_selection"
    torch.cuda.set_device(local_rank)
    record(
        "cuda_device_selected",
        device=torch.cuda.get_device_name(local_rank).replace(" ", "_"),
        visible=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
    current_stage = "nccl_initialization"
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    global_rank = dist.get_rank()
    world = dist.get_world_size()
    record("nccl_initialized", world=world)

    current_stage = "cuda_tensor_allocation"
    value = torch.tensor([float(global_rank + 1)], device="cuda")
    current_stage = "cuda_smoke"
    torch.cuda.synchronize()
    record("cuda_smoke_complete")
    current_stage = "initial_barrier"
    dist.barrier()
    record("initial_collective_complete")

    if MODE == "failure":
        current_stage = "intentional_failure"
        if global_rank == 0:
            record("intentional_gpu_process_failure")
            os._exit(23)
        time.sleep(2)
        record("peer_waiting_for_launcher_cleanup")
        while True:
            time.sleep(60)

    if MODE == "stall" and global_rank == 0:
        current_stage = "intentional_stall"
        record("intentional_gpu_stall_enter")
        while True:
            time.sleep(60)

    if MODE == "vjp":
        current_stage = "cuda_autograd_vjp"
        inputs = torch.arange(1, 9, dtype=torch.float32, device="cuda", requires_grad=True)
        cotangent = torch.linspace(0.25, 2.0, 8, dtype=torch.float32, device="cuda")
        outputs = inputs.square() + 3.0 * inputs
        gradient, = torch.autograd.grad(outputs, inputs, grad_outputs=cotangent)
        expected_gradient = (2.0 * inputs.detach() + 3.0) * cotangent
        maximum_error = (gradient - expected_gradient).abs().max()
        torch.cuda.synchronize()
        if maximum_error.item() > 2.0e-6:
            raise RuntimeError(
                f"CUDA VJP maximum error {maximum_error.item()} exceeds tolerance"
            )
        checksum = gradient.sum()
        expected_checksum = expected_gradient.sum() * world
        dist.all_reduce(checksum, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        if not torch.allclose(checksum, expected_checksum, rtol=1.0e-6, atol=1.0e-5):
            raise RuntimeError(
                f"distributed VJP checksum {checksum.item()} differs from "
                f"expected {expected_checksum.item()}"
            )
        record(
            "cuda_vjp_complete",
            maximum_error=maximum_error.item(),
            distributed_checksum=checksum.item(),
        )
        if global_rank == 0:
            print(
                f"OLIVIA_GPU_VJP_PROBE_OK world={world} "
                f"maximum_error={maximum_error.item()} checksum={checksum.item()}",
                flush=True,
            )

    if MODE == "ffno_vjp":
        current_stage = "production_fsdp_ffno_vjp"
        import sys
        import numpy as np

        repository = Path(os.environ["OLIVIA_REPO_DIR"])
        sys.path.insert(0, str(repository.parent))
        from distributed_inference import partition_range
        from ffno_fsdp_runtime import DistributedFSDPBackend

        run_root = Path(os.environ.get("FFNOML_RUN_DIR", repository.parent)).resolve()
        checkpoint = Path(
            os.environ.get(
                "FFNO_TEST_CHECKPOINT",
                run_root
                / "training_FFNO3D_zscale_expand_lognlte"
                / "3D_sim_train_H.pt",
            )
        ).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"production FFNO checkpoint is missing: {checkpoint}")
        level_names = [f"H level {index}" for index in range(1, 7)]
        backend = DistributedFSDPBackend(
            checkpoint_path=checkpoint,
            factory_module="ffno_model_factory",
            factory_name="create_ffno3d",
            level_names=level_names,
            require_multi_gpu=True,
        )
        # rfft(W) has W // 2 + 1 frequency bins. Ensure there is at least one
        # frequency slab per distributed rank; cuFFT rejects zero-width tensors.
        global_shape = (4, max(8, world), max(8, 2 * (world - 1)))
        h0, h1 = partition_range(global_shape[1], global_rank, world)
        local_shape = (global_shape[0], h1 - h0, global_shape[2])
        features = np.zeros((6,) + local_shape, dtype=np.float32)
        features[0] = 5500.0 + 0.25 * global_rank
        features[4] = 17.0
        features[5] = -7.0
        z_scale = np.broadcast_to(
            np.linspace(-3.0e5, 0.0, local_shape[0], dtype=np.float32)[:, None, None],
            local_shape,
        ).copy()
        populations = backend.predict_local(features, z_scale, 48000.0, 48000.0)
        cotangent = 1.0 / np.maximum(populations, np.finfo(np.float32).tiny)
        cotangent /= np.prod(global_shape) * len(level_names)
        feature_direction = np.zeros_like(features)
        feature_direction[0] = 1.0
        z_direction = np.full_like(z_scale, 100.0)
        # The model evaluates in float32, so one finite-difference step can sit
        # in either the cancellation-dominated or truncation-dominated regime.
        # Sweep several small physical perturbations and retain the best
        # agreement, without relaxing the actual 5% directional tolerance.
        steps = (0.25, 0.5, 1.0, 2.0, 4.0)
        global_population_count = np.prod(global_shape) * len(level_names)

        def directional_finite_difference(feature_delta, z_delta, step):
            plus = backend.predict_local(
                features + step * feature_delta,
                z_scale + step * z_delta,
                48000.0,
                48000.0,
            )
            minus = backend.predict_local(
                features - step * feature_delta,
                z_scale - step * z_delta,
                48000.0,
                48000.0,
            )
            # The VJP cotangent is the derivative of the global mean log
            # population. Difference that same objective in float64 to avoid
            # introducing additional host-side summation cancellation.
            return float(
                np.sum(
                    np.log(plus.astype(np.float64))
                    - np.log(minus.astype(np.float64))
                )
                / (2.0 * step * global_population_count)
            )

        # Evaluate finite differences before backward. Besides being the usual
        # ordering for a directional check, this lets the repeated base forward
        # below detect any FSDP post-backward state corruption explicitly.
        finite_difference_locals = []
        for step in steps:
            finite_difference_locals.append(directional_finite_difference(
                feature_direction, z_direction, step
            ))
        component_step = 4.0
        zero_features = np.zeros_like(features)
        zero_z = np.zeros_like(z_scale)
        feature_finite_difference_local = directional_finite_difference(
            feature_direction, zero_z, component_step
        )
        z_finite_difference_local = directional_finite_difference(
            zero_features, z_direction, component_step
        )

        result = backend.vjp_local(
            features, z_scale, 48000.0, 48000.0, cotangent
        )
        analytic_feature_local = float(
            np.sum(result["features"] * feature_direction)
        )
        analytic_z_local = float(np.sum(result["z_scale"] * z_direction))
        analytic_local = analytic_feature_local + analytic_z_local
        repeated_populations = backend.predict_local(
            features, z_scale, 48000.0, 48000.0
        )
        repeat_log_error_local = float(np.max(np.abs(
            np.log(repeated_populations.astype(np.float64))
            - np.log(populations.astype(np.float64))
        )))
        totals = torch.tensor(
            [
                analytic_local,
                analytic_feature_local,
                analytic_z_local,
                feature_finite_difference_local,
                z_finite_difference_local,
                *finite_difference_locals,
            ],
            dtype=torch.float64,
            device="cuda",
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        repeat_log_error = torch.tensor(
            repeat_log_error_local, dtype=torch.float64, device="cuda"
        )
        dist.all_reduce(repeat_log_error, op=dist.ReduceOp.MAX)
        reduced = [float(value) for value in totals.cpu()]
        analytic = reduced[0]
        analytic_feature = reduced[1]
        analytic_z = reduced[2]
        feature_finite_difference = reduced[3]
        z_finite_difference = reduced[4]
        finite_differences = reduced[5:]
        repeat_log_error = float(repeat_log_error.cpu())
        relative_errors = [
            abs(analytic - finite_difference) / max(
                abs(analytic), abs(finite_difference), 1.0e-8
            )
            for finite_difference in finite_differences
        ]
        best_index = int(np.argmin(relative_errors))
        step = steps[best_index]
        finite_difference = finite_differences[best_index]
        relative_error = relative_errors[best_index]
        description = backend.describe()
        shard_count = torch.tensor(
            [description["local_parameter_count"]], dtype=torch.int64, device="cuda"
        )
        dist.all_reduce(shard_count, op=dist.ReduceOp.MAX)
        maximum_local_parameters = int(shard_count.item())
        checkpoint_readers = torch.tensor(
            [int(description["checkpoint_loaded_locally"])],
            dtype=torch.int64,
            device="cuda",
        )
        dist.all_reduce(checkpoint_readers, op=dist.ReduceOp.SUM)
        checkpoint_reader_count = int(checkpoint_readers.item())
        if not description["fsdp_enabled"]:
            raise RuntimeError("production FFNO backend is not wrapped in FSDP")
        if checkpoint_reader_count != 1:
            raise RuntimeError(
                "the full checkpoint must be read by exactly torchrun rank 0; "
                f"observed readers={checkpoint_reader_count}"
            )
        if maximum_local_parameters >= description["full_parameter_count"]:
            raise RuntimeError(
                "FSDP did not shard model parameters: "
                f"maximum_local={maximum_local_parameters} "
                f"full={description['full_parameter_count']}"
            )
        if not np.isfinite(repeat_log_error) or repeat_log_error > 1.0e-6:
            raise RuntimeError(
                "production FSDP FFNO changed predictions after VJP: "
                f"maximum_log_difference={repeat_log_error}"
            )
        if not np.isfinite(relative_error) or relative_error > 5.0e-2:
            raise RuntimeError(
                "production FSDP FFNO VJP directional check failed: "
                f"analytic={analytic} finite_difference={finite_difference} "
                f"relative_error={relative_error} best_step={step} "
                f"step_sweep={dict(zip(steps, finite_differences))} "
                f"feature_analytic={analytic_feature} "
                f"feature_finite_difference={feature_finite_difference} "
                f"z_analytic={analytic_z} "
                f"z_finite_difference={z_finite_difference} "
                f"post_vjp_log_difference={repeat_log_error}"
            )
        record(
            "production_fsdp_ffno_vjp_complete",
            analytic=analytic,
            finite_difference=finite_difference,
            relative_error=relative_error,
            feature_analytic=analytic_feature,
            feature_finite_difference=feature_finite_difference,
            z_analytic=analytic_z,
            z_finite_difference=z_finite_difference,
            post_vjp_log_difference=repeat_log_error,
            finite_difference_step=step,
            finite_difference_sweep=",".join(
                f"{candidate}:{value}" for candidate, value in zip(
                    steps, finite_differences
                )
            ),
            checkpoint=checkpoint,
            world=world,
            local_h=local_shape[1],
            maximum_local_parameters=maximum_local_parameters,
            full_parameters=description["full_parameter_count"],
            checkpoint_readers=checkpoint_reader_count,
        )
        if global_rank == 0:
            print(
                "OLIVIA_PRODUCTION_FSDP_FFNO_VJP_OK "
                f"world={world} relative_error={relative_error} "
                f"analytic={analytic} finite_difference={finite_difference} "
                f"finite_difference_step={step} "
                f"maximum_local_parameters={maximum_local_parameters} "
                f"full_parameters={description['full_parameter_count']} "
                f"checkpoint_readers={checkpoint_reader_count}",
                flush=True,
            )
        dist.barrier()

    record("allreduce_enter")
    current_stage = "nccl_allreduce"
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    expected = world * (world + 1) / 2
    actual = value.item()
    if actual != expected:
        raise RuntimeError(f"NCCL allreduce returned {actual}, expected {expected}")
    record("allreduce_complete", value=actual)
    current_stage = "final_barrier"
    dist.barrier()
    if global_rank == 0:
        print(f"OLIVIA_GPU_PROBE_OK world={world} sum={actual}", flush=True)
except BaseException as exception:
    record(
        "python_failure",
        stage=current_stage,
        exception_type=type(exception).__name__,
        error=repr(exception),
        traceback=repr(traceback.format_exc()),
    )
    raise
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
