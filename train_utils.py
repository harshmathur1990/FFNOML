import os
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


def save_checkpoint_fsdp(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    train_comp,
    val_comp,
    save_path,
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
    torch.save(ckpt, save_path)


def load_training_state(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    *,
    map_location="cpu",
):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
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

    return ckpt


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


def compute_loss(pred, target, weight, loss_fn, x):
    loss, components = loss_fn(x, pred, target)

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
        mem = torch.cuda.memory_allocated(device) / 1024**3
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

    for x, y, dx, dy, scale, weight in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        dx = dx.to(device, non_blocking=True)
        dy = dy.to(device, non_blocking=True)

        if weight is not None:
            weight = weight.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        pred = model(x, dx, dy)

        pred = flatten_columns_logb(pred)
        y = flatten_columns_logb(y)
        x = flatten_columns_logb(x)

        loss, components = compute_loss(
            pred=pred,
            target=y,
            weight=weight,
            loss_fn=loss_fn,
            x=x
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

        # accumulate diagnostics
        diag = {
            "rmse": rmse,
            "rel_rmse": rel_rmse,
            "std_ratio": std_ratio,
            "p95_err": p95_err,
        }

        _accumulate_components(running_comp, diag, weight_factor=weight_factor)
        _accumulate_components(comp_sums, diag, weight_factor=weight_factor)

        # all ranks must participate in collectives
        global_running = reduce_sum_scalar(running, device)
        global_n = reduce_sum_scalar(n_columns, device)

        avg_so_far = global_running / max(1, global_n)

        avg_comp = {}
        for k, v in running_comp.items():
            global_v = reduce_sum_scalar(v, device)
            avg_comp[k] = global_v / max(1, global_n)

        if is_main_process():
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
):
    model.eval()

    running = 0.0
    n_columns = 0
    comp_sums = {}

    running_comp = {}

    # model_stats_sums = {}

    pbar = tqdm(
        loader,
        desc="val",
        leave=True,
        disable=not is_main_process(),
    )

    with torch.no_grad():
        for x, y, dx, dy, scale, weight in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            dx = dx.to(device, non_blocking=True)
            dy = dy.to(device, non_blocking=True)

            if weight is not None:
                weight = weight.to(device, non_blocking=True)

            # pred, stats = model(x, dx, dy, collect_stats=True)
            pred = model(x, dx, dy)

            pred = flatten_columns_logb(pred)
            y = flatten_columns_logb(y)
            x = flatten_columns_logb(x)

            loss, components = compute_loss(
                pred=pred,
                target=y,
                weight=weight,
                loss_fn=loss_fn,
                x=x
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

            # _accumulate_model_stats(model_stats_sums, stats, weight_factor=pred.shape[0])

            # accumulate diagnostics
            diag = {
                "rmse": rmse,
                "rel_rmse": rel_rmse,
                "std_ratio": std_ratio,
                "p95_err": p95_err,
            }

            _accumulate_components(running_comp, diag, weight_factor=weight_factor)
            _accumulate_components(comp_sums, diag, weight_factor=weight_factor)

            global_running = reduce_sum_scalar(running, device)
            global_n = reduce_sum_scalar(n_columns, device)

            avg_so_far = global_running / max(1, global_n)

            avg_comp = {}
            for k, v in running_comp.items():
                global_v = reduce_sum_scalar(v, device)
                avg_comp[k] = global_v / max(1, global_n)

            if is_main_process():
                postfix = _make_postfix(avg_so_far, None, device, avg_comp)
                pbar.set_postfix(postfix)

    global_running = reduce_sum_scalar(running, device)
    global_batches = reduce_sum_scalar(n_columns, device)

    global_avg_loss = global_running / max(1, global_batches)
    global_comp = reduce_components(comp_sums, n_columns, device)

    # global_model_stats = reduce_components(model_stats_sums, n_columns, device)

    return global_avg_loss, global_comp  #, global_model_stats


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

    best_val = float("inf")

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
            )

            if is_main_process():
                print(f"[Epoch {epoch:03d}] train={train_loss:.6e} (saved)")

        save_resume_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            save_path=latest_save_path,
        )
