import os
import re
import inspect
import warnings
import torch
import torch.distributed as dist
from tqdm import tqdm

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    StateDictOptions,
    set_model_state_dict,
    set_optimizer_state_dict,
)


def load_checkpoint(checkpoint_path, *, map_location="cpu"):
    """
    Compatibility wrapper for `torch.load` across PyTorch versions.

    PyTorch 2.6 changed the default `weights_only` value from False to True, which
    can fail for checkpoints containing NumPy arrays/dtypes unless those globals
    are allowlisted. We first attempt a safe weights-only load and fall back to a
    full unpickle only if needed.
    """
    supports_weights_only = False
    try:
        supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        supports_weights_only = False

    if not supports_weights_only:
        return torch.load(checkpoint_path, map_location=map_location)

    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" not in msg and "WeightsUnpickler" not in msg:
            raise

    # Retry with a small NumPy allowlist (common in our checkpoints for stats).
    try:
        import numpy as np

        reconstruct = getattr(np.core.multiarray, "_reconstruct", None)
        scalar = getattr(np.core.multiarray, "scalar", None)
        allowed = [x for x in (reconstruct, scalar, np.ndarray, np.dtype) if x is not None]

        try:
            from torch.serialization import safe_globals
        except Exception:
            safe_globals = None

        if safe_globals is not None:
            with safe_globals(allowed):
                return torch.load(
                    checkpoint_path,
                    map_location=map_location,
                    weights_only=True,
                )

        # Older torch: persistently add allowlist if context manager isn't available.
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals(allowed)
            return torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=True,
            )
    except Exception:
        pass

    # Last resort: old behavior (unsafe; only do this for trusted checkpoints).
    warnings.warn(
        "Falling back to torch.load(weights_only=False). "
        "Only use trusted checkpoint files.",
        RuntimeWarning,
    )
    return torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )


def save_checkpoint_fsdp(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    train_comp,
    val_comp,
    save_path,
    normalization_stats=None,
    io_metadata=None,
):
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    )

    # all ranks call these
    model_state = get_model_state_dict(model, options=options)
    opt_state = get_optimizer_state_dict(model, optimizer, options=options)

    if not is_main_process():
        return

    ckpt = {
        "epoch": epoch,
        "model_state": model_state,
        "opt_state": opt_state,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_components": train_comp,
        "val_components": val_comp,
    }
    if normalization_stats is not None:
        ckpt["normalization_stats"] = normalization_stats
    if io_metadata is not None:
        ckpt["io_metadata"] = io_metadata
    torch.save(ckpt, save_path)


def get_resume_checkpoint_path(save_path):
    root, ext = os.path.splitext(save_path)
    if ext:
        return f"{root}.resume{ext}"
    return save_path + ".resume"


def save_resume_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    save_path,
    normalization_stats=None,
    io_metadata=None,
):
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    )

    model_state = get_model_state_dict(model, options=options)
    opt_state = get_optimizer_state_dict(model, optimizer, options=options)

    if not is_main_process():
        return

    ckpt = {
        "epoch": epoch,
        "completed_epochs": epoch,
        "model_state": model_state,
        "opt_state": opt_state,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "current_lr": optimizer.param_groups[0]["lr"],
    }
    if normalization_stats is not None:
        ckpt["normalization_stats"] = normalization_stats
    if io_metadata is not None:
        ckpt["io_metadata"] = io_metadata
    torch.save(ckpt, save_path)


def get_checkpoint_normalization(ckpt):
    if not isinstance(ckpt, dict):
        return None

    stats = ckpt.get("normalization_stats")
    if not isinstance(stats, dict):
        return None

    required = ("mean_X", "std_X", "mean_Y", "std_Y")
    if not all(key in stats for key in required):
        return None

    out = {}
    for key in required:
        value = stats[key]
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        else:
            value = torch.as_tensor(value, dtype=torch.float32).cpu().numpy()
        out[key] = value

    return out


def get_checkpoint_io_metadata(ckpt):
    if not isinstance(ckpt, dict):
        return None

    meta = ckpt.get("io_metadata")
    if not isinstance(meta, dict):
        return None

    if "Cin" not in meta or "Cout" not in meta:
        return None

    return {
        "Cin": int(meta["Cin"]),
        "Cout": int(meta["Cout"]),
    }


def load_training_state(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    *,
    map_location="cpu",
):
    ckpt = load_checkpoint(checkpoint_path, map_location=map_location)
    result = dict(ckpt)
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    )

    if "model_state" in ckpt:
        set_model_state_dict(
            model,
            ckpt["model_state"],
            options=options,
        )

    if optimizer is not None and ckpt.get("opt_state") is not None:
        set_optimizer_state_dict(
            model,
            optimizer,
            ckpt["opt_state"],
            options=options,
        )

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    del ckpt

    return result


def expand_model_from_checkpoint(
    checkpoint_path,
    model,
    *,
    map_location="cpu",
    zero_init_new_blocks=True,
):
    ckpt = load_checkpoint(checkpoint_path, map_location=map_location)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise RuntimeError(f"Invalid checkpoint: {checkpoint_path}")

    current_state = model.state_dict()
    source_state = ckpt["model_state"]

    merged_state = {}
    copied = []
    skipped = []

    for key, current_tensor in current_state.items():
        source_tensor = source_state.get(key)

        if source_tensor is not None and tuple(source_tensor.shape) == tuple(current_tensor.shape):
            merged_state[key] = source_tensor
            copied.append(key)
        elif (
            source_tensor is not None
            and key.endswith("vertical.z_proj.0.weight")
            and source_tensor.ndim == current_tensor.ndim == 3
            and source_tensor.shape[0] == current_tensor.shape[0]
            and source_tensor.shape[1] == 1
            and current_tensor.shape[1] > 1
            and source_tensor.shape[2] == current_tensor.shape[2]
        ):
            upgraded = torch.zeros_like(current_tensor)
            upgraded[:, :1, :] = source_tensor
            merged_state[key] = upgraded
            copied.append(key)
        else:
            merged_state[key] = current_tensor
            if source_tensor is not None:
                skipped.append(key)

    old_block_ids = set()
    block_pattern = re.compile(r"blocks\.(\d+)\.")

    for key in source_state.keys():
        match = block_pattern.search(key)
        if match:
            old_block_ids.add(int(match.group(1)))

    old_n_blocks = (max(old_block_ids) + 1) if old_block_ids else 0
    zeroed = []

    if zero_init_new_blocks:
        for key, tensor in merged_state.items():
            match = block_pattern.search(key)
            if not match:
                continue

            block_idx = int(match.group(1))
            if block_idx < old_n_blocks:
                continue

            if key.endswith(("res_fused", "res_pw", "res_mlp")):
                merged_state[key] = torch.zeros_like(tensor)
                zeroed.append(key)

    model.load_state_dict(merged_state, strict=True)

    return {
        "checkpoint_path": checkpoint_path,
        "copied_keys": copied,
        "skipped_keys": skipped,
        "old_n_blocks": old_n_blocks,
        "zeroed_keys": zeroed,
    }


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main_process():
    return get_rank() == 0


def reduce_sum_scalar(value, device):
    """
    Sum a python float across all ranks.
    """
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def get_total_gpu_mem_used_gb(device):
    """
    Return the local device's used GPU memory in GiB.

    Important: this helper is used from the main-rank-only tqdm/logging path,
    so it must not perform distributed collectives. Calling `all_reduce` here
    would deadlock because non-main ranks do not enter the logging code.
    """
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return (total_bytes - free_bytes) / 1024**3


def reduce_components(comp_sums, count, device):
    """
    Reduce component sums across ranks, then divide by total batch count.
    comp_sums: local sums over batches on this rank
    count: local number of batches on this rank
    """
    out = {}

    total_count = torch.tensor(float(count), device=device, dtype=torch.float64)
    if is_dist():
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)

    total_count = max(total_count.item(), 1.0)

    for k, v in comp_sums.items():
        if isinstance(v, (int, float)):
            t = torch.tensor(float(v), device=device, dtype=torch.float64)
        elif isinstance(v, torch.Tensor):
            t = v.detach().to(device=device, dtype=torch.float64)
        else:
            continue

        if is_dist():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        t = t / total_count

        if t.ndim == 0:
            out[k] = t.item()
        else:
            out[k] = t.cpu()

    return out


def compute_loss(
    pred,
    target,
    weight,
    loss_fn,
    x,
    pred_full=None,
    target_full=None,
    source_true=None,
):
    loss, components = loss_fn(
        x,
        pred,
        target,
        logb_pred_full=pred_full,
        logb_true_full=target_full,
        source_true=source_true,
    )

    if weight is not None:
        loss = (loss * weight).sum() / (weight.sum() + 1e-12)
    else:
        loss = loss.mean()

    return loss, components


def flatten_columns_logb(logb):
    """
    logb: (B, L, Nz, Ny, Nx)
    -> (B*Ny*Nx, L, Nz)
    """
    B, L, Nz, Ny, Nx = logb.shape
    return logb.permute(0, 3, 4, 1, 2).reshape(B * Ny * Nx, L, Nz)


def _accumulate_model_stats(stats_sums, stats_list, weight_factor=1.0):
    if stats_list is None:
        return

    for layer_stats in stats_list:
        if not isinstance(layer_stats, dict):
            continue

        layer = layer_stats.get("layer", -1)

        for k, v in layer_stats.items():
            if k == "layer":
                continue

            key = f"layer{layer}.{k}"

            if isinstance(v, torch.Tensor):
                v = v.detach()
                if v.ndim == 0:
                    v = v.item()
                else:
                    v = v.to(dtype=torch.float64).cpu()
            elif isinstance(v, (int, float)):
                v = float(v)
            else:
                continue

            v = v * weight_factor

            if key not in stats_sums:
                if isinstance(v, torch.Tensor):
                    stats_sums[key] = v.clone()
                else:
                    stats_sums[key] = v
            else:
                stats_sums[key] += v


def _accumulate_components(comp_sums, components, weight_factor=1.0):
    """
    Accumulate numeric component statistics with optional weighting.
    Keeps vectors such as source_per_atom as vectors.
    Skips non-numeric metadata like atom_names.
    """
    if components is None or not isinstance(components, dict):
        return

    for k, v in components.items():
        if isinstance(v, torch.Tensor):
            v = v.detach()

            if v.ndim == 0:
                v = v.item()
            else:
                v = v.to(dtype=torch.float64).cpu()

        elif isinstance(v, (int, float)):
            v = float(v)

        else:
            continue

        v = v * weight_factor

        if k not in comp_sums:
            if isinstance(v, torch.Tensor):
                comp_sums[k] = v.clone()
            else:
                comp_sums[k] = v
        else:
            comp_sums[k] += v


def compute_mean_stats(layer_stats):
    mean_stats = {}
    for k in layer_stats:
        base_key = k.split(".", 1)[1]  # remove layer prefix
        if base_key not in mean_stats:
            mean_stats[base_key] = []
        mean_stats[base_key].append(layer_stats[k])

    return {f"mean.{k}": sum(v)/len(v) for k, v in mean_stats.items()}


def _make_postfix(loss, lr, device, components):
    postfix = {
        "loss": f"{loss:.2e}",
    }

    if lr is not None:
        postfix["lr"] = f"{lr:.1e}"

    if isinstance(device, str) and device.startswith("cuda"):
        mem = get_total_gpu_mem_used_gb(device)
        postfix["gpu_mem"] = f"{mem:.2f}G"

    if components is not None and isinstance(components, dict):
        if "data" in components and torch.is_tensor(components["data"]):
            postfix["L1"] = f"{components['data'].item():.2e}"
        elif "data" in components:
            postfix["L1"] = f"{float(components['data']):.2e}"

        if "source" in components and torch.is_tensor(components["source"]):
            postfix["L2"] = f"{components['source'].item():.2e}"
        elif "source" in components:
            postfix["L2"] = f"{float(components['source']):.2e}"

        if "gradient" in components and torch.is_tensor(components["gradient"]):
            postfix["L3"] = f"{components['gradient'].item():.2e}"
        elif "gradient" in components:
            postfix["L3"] = f"{float(components['gradient']):.2e}"


        if "source_per_atom" in components and "atom_names" in components:
            spa = components["source_per_atom"]
            names = components["atom_names"]

            if torch.is_tensor(spa):
                spa = spa.detach().cpu()

            for j, n in enumerate(names):
                postfix[n] = f"{float(spa[j]):.2e}"

    # --------------------------------------------
    # diagnostics
    # --------------------------------------------
    if components is not None:
        if "rmse" in components:
            postfix["rmse"] = f"{float(components['rmse']):.2e}"

        if "rel_rmse" in components:
            postfix["rel%"] = f"{float(components['rel_rmse'])*100:.2f}"

        if "std_ratio" in components:
            postfix["std_r"] = f"{float(components['std_ratio']):.2f}"

        if "p95_err" in components:
            postfix["p95"] = f"{float(components['p95_err']):.2e}"

    return postfix


def save_epoch_stats(stats, epoch, save_dir):
    if not is_main_process():
        return

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"epoch_{epoch:04d}_stats.pt")

    payload = {
        "epoch": epoch,
        "stats": stats,
    }
    torch.save(payload, path)


def resample_train_dataset(loader, epoch):
    dataset = getattr(loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "resample_train_selection"):
        return

    base_seed = getattr(dataset, "train_select_seed", None)
    if base_seed is None:
        seed = epoch
    else:
        seed = int(base_seed) + int(epoch)

    dataset.resample_train_selection(seed)


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    epoch,
    num_epochs,
    grad_clip=1.0,
):
    model.train()
    running = 0.0
    n_columns = 0
    comp_sums = {}
    running_comp = {}

    lr = optimizer.param_groups[0]["lr"]

    pbar = tqdm(
        loader,
        desc=f"epoch {epoch}/{num_epochs}",
        leave=True,
        disable=not is_main_process(),
    )

    for x, y, z, dx, dy, scale, weight, source_target in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z = z.to(device, non_blocking=True)
        dx = dx.to(device, non_blocking=True)
        dy = dy.to(device, non_blocking=True)

        if weight is not None:
            weight = weight.to(device, non_blocking=True)

        if source_target is not None and source_target.numel() > 0:
            source_target = source_target.to(device, non_blocking=True)
        else:
            source_target = None

        optimizer.zero_grad(set_to_none=True)

        pred = model(x, z, dx, dy)

        pred_full = pred
        y_full = y
        source_true = None

        pred = flatten_columns_logb(pred_full)
        y = flatten_columns_logb(y_full)
        x = flatten_columns_logb(x)

        if source_target is not None and source_target.numel() > 0:
            source_true = flatten_columns_logb(source_target)

        loss, components = compute_loss(
            pred=pred,
            target=y,
            weight=weight,
            loss_fn=loss_fn,
            x=x,
            pred_full=pred_full,
            target_full=y_full,
            source_true=source_true,
        )

        loss.backward()

        if grad_clip is not None and grad_clip > 0:

            if isinstance(model, FSDP):
                model.clip_grad_norm_(grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        weight_factor = pred.shape[0]

        running += loss.item() * weight_factor
        n_columns += weight_factor

        _accumulate_components(comp_sums, components, weight_factor=weight_factor)
        _accumulate_components(running_comp, components, weight_factor=weight_factor)

        if is_main_process():
            avg_so_far = running / max(1, n_columns)
            avg_comp = {
                k: float(v) / max(1, n_columns)
                for k, v in running_comp.items()
                if isinstance(v, (int, float))
            }
            postfix = _make_postfix(avg_so_far, lr, device, avg_comp)
            pbar.set_postfix(postfix)

    global_running = reduce_sum_scalar(running, device)
    global_batches = reduce_sum_scalar(n_columns, device)

    global_avg_loss = global_running / max(1, global_batches)
    global_comp = reduce_components(comp_sums, n_columns, device)

    return global_avg_loss, global_comp


def validate(
    model,
    loader,
    loss_fn,
    device,
    collect_model_stats=False,
    forward_kwargs=None,
):
    model.eval()
    forward_kwargs = forward_kwargs or {}

    running = 0.0
    n_columns = 0
    comp_sums = {}

    running_comp = {}

    model_stats_sums = {}

    pbar = tqdm(
        loader,
        desc="val",
        leave=True,
        disable=not is_main_process(),
    )

    with torch.no_grad():
        for x, y, z, dx, dy, scale, weight, source_target in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            dx = dx.to(device, non_blocking=True)
            dy = dy.to(device, non_blocking=True)

            if weight is not None:
                weight = weight.to(device, non_blocking=True)

            if source_target is not None and source_target.numel() > 0:
                source_target = source_target.to(device, non_blocking=True)
            else:
                source_target = None

            if collect_model_stats:
                pred, stats = model(
                    x,
                    z,
                    dx,
                    dy,
                    collect_stats=True,
                    **forward_kwargs,
                )
            else:
                pred = model(x, z, dx, dy, **forward_kwargs)
                stats = None

            pred_full = pred
            y_full = y
            source_true = None

            pred = flatten_columns_logb(pred_full)
            y = flatten_columns_logb(y_full)
            x = flatten_columns_logb(x)

            if source_target is not None and source_target.numel() > 0:
                source_true = flatten_columns_logb(source_target)

            loss, components = compute_loss(
                pred=pred,
                target=y,
                weight=weight,
                loss_fn=loss_fn,
                x=x,
                pred_full=pred_full,
                target_full=y_full,
                source_true=source_true,
            )

            # ============================================
            # Extra diagnostics (VERY IMPORTANT)
            # ============================================
            with torch.no_grad():
                err = pred - y

                mse = (err ** 2).mean()
                rmse = torch.sqrt(mse)

                target_std = y.std()
                pred_std = pred.std()

                rel_rmse = rmse / (target_std + 1e-8)
                std_ratio = pred_std / (target_std + 1e-8)

                p95_err = err.abs().quantile(0.95)

            running += loss.item() * pred.shape[0]   # number of columns
            n_columns += pred.shape[0]

            weight_factor = pred.shape[0]

            # global (epoch summary)
            _accumulate_components(comp_sums, components, weight_factor=weight_factor)

            # local running (for tqdm)
            _accumulate_components(running_comp, components, weight_factor=weight_factor)

            _accumulate_model_stats(model_stats_sums, stats, weight_factor=pred.shape[0])

            # accumulate diagnostics
            diag = {
                "rmse": rmse,
                "rel_rmse": rel_rmse,
                "std_ratio": std_ratio,
                "p95_err": p95_err,
            }

            _accumulate_components(running_comp, diag, weight_factor=weight_factor)
            _accumulate_components(comp_sums, diag, weight_factor=weight_factor)

            if is_main_process():
                avg_so_far = running / max(1, n_columns)
                avg_comp = {
                    k: float(v) / max(1, n_columns)
                    for k, v in running_comp.items()
                    if isinstance(v, (int, float))
                }
                postfix = _make_postfix(avg_so_far, None, device, avg_comp)
                pbar.set_postfix(postfix)

    global_running = reduce_sum_scalar(running, device)
    global_batches = reduce_sum_scalar(n_columns, device)

    global_avg_loss = global_running / max(1, global_batches)
    global_comp = reduce_components(comp_sums, n_columns, device)

    global_model_stats = reduce_components(model_stats_sums, n_columns, device)

    if collect_model_stats:
        return global_avg_loss, global_comp, global_model_stats

    return global_avg_loss, global_comp


def train(
    model,
    train_loader,
    val_loader,
    scheduler,
    optimizer,
    loss_fn,
    save_path,
    *,
    num_epochs=50,
    device="cuda",
    grad_clip=1.0,
    early_stopping=None,
    resume_last_epoch=None,
    resume_state=None,
    resume_path=None,
    best_val_init=None,
    normalization_stats=None,
    io_metadata=None,
):
    latest_save_path = resume_path or get_resume_checkpoint_path(save_path)
    resume_state = resume_state or {}

    completed_epochs = resume_state.get("completed_epochs")
    if completed_epochs is None:
        completed_epochs = resume_state.get("epoch")
    if completed_epochs is None:
        completed_epochs = resume_last_epoch
    if completed_epochs is None:
        completed_epochs = 0

    start_epoch = int(completed_epochs) + 1

    if best_val_init is None:
        best_val = float("inf")
    else:
        best_val = float(best_val_init)

    if early_stopping is not None:
        es_enabled = early_stopping.get("enabled", False)
        patience = early_stopping.get("patience", 10)
        min_delta = early_stopping.get("min_delta", 0.0)
    else:
        es_enabled = False
        patience = 0
        min_delta = 0.0

    epochs_no_improve = 0

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    for epoch in range(start_epoch, num_epochs + 1):

        resample_train_dataset(train_loader, epoch)

        # Important when using DistributedSampler
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_comp = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
            grad_clip=grad_clip,
        )

        if scheduler is not None:
            scheduler.step()
            # print(f"LR after step: {optimizer.param_groups[0]['lr']:.10f}")

        if val_loader is not None:
            if hasattr(val_loader, "sampler") and hasattr(val_loader.sampler, "set_epoch"):
                val_loader.sampler.set_epoch(epoch)

            # val_loss, val_comp, val_stats = validate(
            #     model=model,
            #     loader=val_loader,
            #     loss_fn=loss_fn,
            #     device=device,
            # )

            val_loss, val_comp = validate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )

            # mean_stats = compute_mean_stats(val_stats)
            # val_stats.update(mean_stats)

            improved = (best_val - val_loss) > min_delta

            # save_epoch_stats(val_stats, epoch, save_dir=save_path + "_stats")

            if improved:
                best_val = val_loss
                epochs_no_improve = 0

                save_checkpoint_fsdp(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_comp=train_comp,
                    val_comp=val_comp,
                    save_path=save_path,
                    normalization_stats=normalization_stats,
                    io_metadata=io_metadata,
                )

                if is_main_process():
                    msg = (
                        f"[Epoch {epoch:03d}] "
                        f"train={train_loss:.6e} "
                        f"val={val_loss:.6e} (saved best) "
                        f"L1={val_comp.get('data', 0):.3e} "
                        f"L2={val_comp.get('source', 0):.3e}"
                    )
                    print(msg)

            else:
                epochs_no_improve += 1

                if is_main_process():
                    msg = (
                        f"[Epoch {epoch:03d}] "
                        f"train={train_loss:.6e} "
                        f"val={val_loss:.6e} "
                        f"L1={val_comp.get('data', 0):.3e} "
                        f"L2={val_comp.get('source', 0):.3e}"
                    )
                    print(msg)

                if es_enabled and epochs_no_improve > patience:
                    if is_main_process():
                        print(f"Early stopping triggered (patience={patience})")
                    break

        else:
            save_checkpoint_fsdp(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=None,
                train_comp=train_comp,
                val_comp=None,
                save_path=save_path,
                normalization_stats=normalization_stats,
                io_metadata=io_metadata,
            )

            if is_main_process():
                print(f"[Epoch {epoch:03d}] train={train_loss:.6e} (saved)")

        save_resume_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            save_path=latest_save_path,
            normalization_stats=normalization_stats,
            io_metadata=io_metadata,
        )
