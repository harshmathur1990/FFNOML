# ============================================================
# FFNO Utilities (full-volume / patch-to-patch operator training)
#
# Public API (new):
#   build_dataset_ffno
#   build_solving_set_ffno
#   ffno_train_model
#   ffno_predict_populations
#   ffno_test_model
#
# ============================================================

import sys
import os
import json
import numpy as np
import h5py
import torch
import torch.distributed as dist
from interp_utils import interpolate_everything
from train_utils import (
    train,
    validate,
    compute_mean_stats,
    expand_model_from_checkpoint,
    get_resume_checkpoint_path,
    load_training_state,
)
from scipy.ndimage import gaussian_filter
from model_builder import ModelBuilder
from data_builder import DataLoaderBuilder
from normalize_utils import *
from tqdm import tqdm
import itertools
import math


# ============================================================
# ---------------- PREPROCESSING ------------------------------
# ============================================================

def _prepare_input_features(temp, vx, vy, vz, ne, rho):
    """
    temp: [nx, ny, nz]
    returns features: [nx, ny, nz, Cin]
    """
    return np.stack(
        [
            np.log10(temp),
            vx,
            vy,
            vz,
            np.log10(ne),
            np.log10(rho),
        ],
        axis=-1,
    )


def _compute_departure_coefficients(lte, nlte, eps=1e-30):
    """
    lte/nlte: [nx, ny, nz, nlev] or [nx, ny, nz, Cout]
    return: log10(nlte/lte)
    """
    return np.log10((nlte + eps) / (lte + eps))


def invert_log_departure(pred_log):
    return torch.pow(10.0, pred_log)


def _make_inputs_ch_first(
    rho, z_scale,
    temp, vx, vy, vz, ne,
    *,
    ndep
):
    """
    returns inputs: [Cin, ndep, nx, ny]  (channel-first with depth first)
    """
    cmass_grid = np.logspace(-6, 2, ndep)

    features = _prepare_input_features(temp, vx, vy, vz, ne, rho)  # [nx,ny,nz,Cin]

    features = interpolate_everything(
        rho, z_scale, features, cmass_grid
    )  # expected output: [nx,ny,ndep,Cin]

    features = np.transpose(features, (3, 2, 0, 1)).astype(np.float32, copy=False)

    return features, cmass_grid


def _make_targets_ch_first(
    rho, z_scale,
    lte, nlte,
    *,
    ndep
):
    """
    returns dep: [Cout, ndep, nx, ny]
    """
    cmass_grid = np.logspace(-6, 2, ndep)

    dep = _compute_departure_coefficients(lte, nlte)  # [nx,ny,nz,Cout] (or nlev)

    dep = interpolate_everything(
        rho, z_scale, dep, cmass_grid
    )  # [nx,ny,ndep,Cout]

    dep = np.transpose(dep, (3, 2, 0, 1)).astype(np.float32, copy=False)

    return dep


# ============================================================
# ---------------- PATCH EXTRACTION ---------------------------
# ============================================================

def _extract_patches_xy(X, Y, patch, stride):
    """
    X: [Cin, D, nx, ny]
    Y: [Cout, D, nx, ny]

    returns:
        Xp [N, Cin, D, patch, patch]
        Yp [N, Cout, D, patch, patch]
    """

    Cin, D, nx, ny = X.shape
    Cout = Y.shape[0]

    xs, ys = [], []

    for i in range(0, nx - patch + 1, stride):
        for j in range(0, ny - patch + 1, stride):

            xs.append(X[:, :, i:i+patch, j:j+patch])
            ys.append(Y[:, :, i:i+patch, j:j+patch])

    if len(xs) == 0:
        raise ValueError(
            f"Patch too large: patch={patch} for nx,ny={nx},{ny}"
        )

    Xp = np.stack(xs, axis=0).astype(np.float32, copy=False)
    Yp = np.stack(ys, axis=0).astype(np.float32, copy=False)

    return Xp, Yp


# ------------------------------------------------------------
# Downsampling with anti-alias filter
# ------------------------------------------------------------
def _downsample_xy(X, Y, scale):
    """
    Downsample spatial dimensions by factor `scale`
    with anti-alias filtering.

    X: [Cin, D, nx, ny]
    Y: [Cout, D, nx, ny]
    """

    if scale == 1:
        return X, Y

    sigma = scale / 2

    Xf = gaussian_filter(X, sigma=(0, 0, sigma, sigma))
    Yf = gaussian_filter(Y, sigma=(0, 0, sigma, sigma))

    Xs = Xf[:, :, ::scale, ::scale]
    Ys = Yf[:, :, ::scale, ::scale]

    return Xs, Ys


# ============================================================
# ---------------- HDF5 WRITERS -------------------------------
# ============================================================

def _save_hdf5_patches(
    path,
    Xp,
    Yp,
    dxs,
    dys,
    cmass_grid,
    *,
    scales=None,
    weights=None,
    mean_X=None,
    std_X=None,
    mean_Y=None,
    std_Y=None,
    attrs=None,
):
    """
    Save training patches + normalization stats.

    Parameters
    ----------
    Xp : [N, Cin, D, P, P]
    Yp : [N, Cout, D, P, P]

    dxs : [N]
    dys : [N]

    scales : [N] optional
    weights : [N] optional

    mean_X, std_X : [Cin]
    mean_Y, std_Y : [Cout]

    cmass_grid : [D]
    """

    if os.path.isfile(path):
        raise IOError(f"Output exists: {path}")

    attrs = attrs or {}

    N = Xp.shape[0]
    Cin = Xp.shape[1]
    Cout = Yp.shape[1]

    # ------------------------------------------------------------
    # sanity checks (VERY IMPORTANT)
    # ------------------------------------------------------------
    if mean_X is not None:
        assert mean_X.shape[0] == Cin, "mean_X shape mismatch"
        assert std_X.shape[0] == Cin, "std_X shape mismatch"

    if mean_Y is not None:
        assert mean_Y.shape[0] == Cout, "mean_Y shape mismatch"
        assert std_Y.shape[0] == Cout, "std_Y shape mismatch"

    with h5py.File(path, "w") as f:

        # ========================================================
        # main datasets
        # ========================================================

        f.create_dataset(
            "inputs",
            data=Xp,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        f.create_dataset(
            "targets",
            data=Yp,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        f.create_dataset("dx", data=dxs.astype(np.float32))
        f.create_dataset("dy", data=dys.astype(np.float32))

        # ========================================================
        # optional metadata datasets
        # ========================================================

        if scales is not None:
            f.create_dataset("scale", data=scales.astype(np.int32))

        if weights is not None:
            f.create_dataset("weights", data=weights.astype(np.float32))

        f.create_dataset(
            "cmass_grid",
            data=cmass_grid.astype(np.float64),
        )

        # ========================================================
        # NORMALIZATION STATS (NEW)
        # ========================================================

        if mean_X is not None:
            f.create_dataset("mean_X", data=mean_X.astype(np.float32))
            f.create_dataset("std_X", data=std_X.astype(np.float32))

        if mean_Y is not None:
            f.create_dataset("mean_Y", data=mean_Y.astype(np.float32))
            f.create_dataset("std_Y", data=std_Y.astype(np.float32))

        # ========================================================
        # attributes
        # ========================================================

        for k, v in attrs.items():
            f.attrs[k] = v

        f.attrs["N"] = N
        f.attrs["Cin"] = Cin
        f.attrs["Cout"] = Cout
        f.attrs["D"] = Xp.shape[2]
        f.attrs["P"] = Xp.shape[3]

        if scales is not None:
            f.attrs["n_scales"] = int(len(np.unique(scales)))

        # flag for downstream safety
        f.attrs["normalized"] = int(mean_X is not None)


def _save_hdf5_cube(path, X, cmass_grid, dx, dy, attrs=None):
    """
    Save inference cube.

    X : [Cin, D, nx, ny]
    """

    if os.path.isfile(path):
        raise IOError(f"Output exists: {path}")

    attrs = attrs or {}

    X = X[None, ...].astype(np.float32, copy=False)

    with h5py.File(path, "w") as f:

        f.create_dataset(
            "inputs",
            data=X,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        f.create_dataset("dx", data=np.array([dx], dtype=np.float32))
        f.create_dataset("dy", data=np.array([dy], dtype=np.float32))

        f.create_dataset(
            "cmass_grid",
            data=cmass_grid.astype(np.float64),
        )

        for k, v in attrs.items():
            f.attrs[k] = v

        f.attrs["N"] = 1
        f.attrs["Cin"] = X.shape[1]
        f.attrs["D"] = X.shape[2]
        f.attrs["nx"] = X.shape[3]
        f.attrs["ny"] = X.shape[4]


def _read_io_channels(h5_path):
    """
    Reads Cin and Cout from training HDF5 file.
    """
    with h5py.File(h5_path, "r") as f:
        Cin = int(f.attrs["Cin"])
        Cout = int(f.attrs["Cout"])
    return Cin, Cout


# ============================================================
# ---------------- BUILD TRAINING SET -------------------------
# ============================================================

# ------------------------------------------------------------
# Main builder
# ------------------------------------------------------------
def build_dataset_ffno(
    temp_list,
    vx_list,
    vy_list,
    vz_list,
    ne_list,
    lte_list,
    nlte_list,
    rho_list,
    z_list,
    dx_list,
    dy_list,
    *,
    save_path,
    ndep=400,
    patch=96,
    stride=48,
    scales=(1,2,3,4),
    stat_file=None
):

    if os.path.isfile(save_path):
        raise IOError(f"Output exists: {save_path}")

    cmass_grid_ref = None

    if stat_file is None:
        # ============================================================
        # -------- PASS 1: compute global normalization stats ---------
        # ============================================================

        X_stats_list = []
        Y_stats_list = []

        for temp, vx, vy, vz, ne, lte, nlte, rho, z in zip(
            temp_list,
            vx_list,
            vy_list,
            vz_list,
            ne_list,
            lte_list,
            nlte_list,
            rho_list,
            z_list,
        ):

            X, cmass_grid = _make_inputs_ch_first(
                rho, z, temp, vx, vy, vz, ne,
                ndep=ndep
            )

            Y = _make_targets_ch_first(
                rho, z, lte, nlte,
                ndep=ndep
            )

            mx, sx = compute_channel_stats(X)
            my, sy = compute_channel_stats(Y)

            X_stats_list.append((mx, sx))
            Y_stats_list.append((my, sy))

        # ------------------------------------------------------------
        # GLOBAL stats (across simulations)
        # ------------------------------------------------------------

        means_X = np.stack([m for m, s in X_stats_list])
        stds_X  = np.stack([s for m, s in X_stats_list])

        means_Y = np.stack([m for m, s in Y_stats_list])
        stds_Y  = np.stack([s for m, s in Y_stats_list])

        mean_X = means_X.mean(axis=0)
        std_X  = stds_X.mean(axis=0)

        mean_Y = means_Y.mean(axis=0)
        std_Y  = stds_Y.mean(axis=0)

    else:
        mean_X, std_X, mean_Y, std_Y = read_normalization(stat_file)

    print("==== NORMALIZATION STATS ====")
    print("mean_X:", mean_X)
    print("std_X :", std_X)
    print("mean_Y:", mean_Y)
    print("std_Y :", std_Y)

    # ============================================================
    # -------- PASS 2: build dataset (with normalization) ---------
    # ============================================================

    X_all = []
    Y_all = []

    dx_all = []
    dy_all = []
    scale_all = []

    for temp, vx, vy, vz, ne, lte, nlte, rho, z, dx, dy in zip(
        temp_list,
        vx_list,
        vy_list,
        vz_list,
        ne_list,
        lte_list,
        nlte_list,
        rho_list,
        z_list,
        dx_list,
        dy_list
    ):

        X, cmass_grid = _make_inputs_ch_first(
            rho, z, temp, vx, vy, vz, ne,
            ndep=ndep
        )

        Y = _make_targets_ch_first(
            rho, z, lte, nlte,
            ndep=ndep
        )

        if cmass_grid_ref is None:
            cmass_grid_ref = cmass_grid
        elif not np.allclose(cmass_grid_ref, cmass_grid):
            raise ValueError("cmass_grid mismatch")

        # ---------------- NORMALIZE HERE ----------------
        X = normalize_channels(X, mean_X, std_X)
        Y = normalize_channels(Y, mean_Y, std_Y)

        for s in scales:

            Xs, Ys = _downsample_xy(X, Y, s)

            nx, ny = Xs.shape[2:]

            if nx < patch or ny < patch:
                continue

            Xp, Yp = _extract_patches_xy(
                Xs,
                Ys,
                patch=patch,
                stride=stride,
            )

            n = Xp.shape[0]

            X_all.append(Xp)
            Y_all.append(Yp)

            dx_all.append(np.full(n, dx * s))
            dy_all.append(np.full(n, dy * s))
            scale_all.append(np.full(n, s))

    # ------------------------------------------------------------
    # concatenate
    # ------------------------------------------------------------

    if len(X_all) == 0:
        raise RuntimeError("No patches generated. Check patch size and scales.")

    X_all = np.concatenate(X_all)
    Y_all = np.concatenate(Y_all)

    dx_all = np.concatenate(dx_all)
    dy_all = np.concatenate(dy_all)
    scale_all = np.concatenate(scale_all)

    # ------------------------------------------------------------
    # weights (per scale balancing)
    # ------------------------------------------------------------

    weights = np.zeros_like(scale_all, dtype=np.float32)

    for s in np.unique(scale_all):
        mask = scale_all == s
        weights[mask] = 1.0 / mask.sum()

    weights *= len(weights)
    weights /= weights.mean()

    # ------------------------------------------------------------
    # sanity check
    # ------------------------------------------------------------

    print("==== DATA CHECK ====")
    print("X mean:", X_all.mean(), "std:", X_all.std())
    print("Y mean:", Y_all.mean(), "std:", Y_all.std())

    # ============================================================
    # save dataset
    # ============================================================

    _save_hdf5_patches(
        save_path,
        X_all,
        Y_all,
        dx_all,
        dy_all,
        cmass_grid_ref,
        scales=scale_all,
        weights=weights,
        attrs=dict(
            ndep=int(ndep),
            patch=int(patch),
            stride=int(stride),
            scales=np.array(scales),
        ),
        mean_X=mean_X,
        std_X=std_X,
        mean_Y=mean_Y,
        std_Y=std_Y,
    )


# ============================================================
# ---------------- BUILD SOLVING SET --------------------------
# ============================================================

def build_solving_set_ffno(
    rho,
    z_scale,
    temp,
    vx,
    vy,
    vz,
    ne,
    *,
    dx,
    dy,
    save_path,
    ndep=400
):
    """
    Build dataset for prediction (no targets).

    Only atmospheric inputs are stored.
    """

    if os.path.isfile(save_path):
        raise IOError(f"Output exists: {save_path}")

    X, cmass_grid = _make_inputs_ch_first(
        rho,
        z_scale,
        temp,
        vx,
        vy,
        vz,
        ne,
        ndep=ndep
    )  # [Cin, D, nx, ny]

    _save_hdf5_cube(
        save_path,
        X,
        cmass_grid,
        dx,
        dy,
        attrs=dict(ndep=int(ndep)),
    )


# ============================================================
# ---------------- TRAINING (FFNO) ----------------------------
# ============================================================

def ffno_train_model(
    *,
    model,
    train_h5,
    val_h5,
    save_path,
    lines,
    wave,
    chi,
    levels,
    atom_names,
    model_config,
    dataset_type,
    num_epochs=50,
    batch_size=1,
    lr=2e-4,
    resume_last_epoch=None,
    resume_last_lr=None,
    weight_decay=1e-4,
    num_workers=8,
    pin_memory=True,
    grad_clip=1.0,
    device="cuda",
    multi_gpu=False,
    debug_loss=False,
    patience=10,
    min_delta=1e-5,
    use_cosine=False,
    min_learning_rate=1e-6,
    resume=False,
    bestpath=False,
    load_earlier_val=False,
    expand_from_checkpoint=None,
    zero_init_new_blocks=True,
):
    resume_path = get_resume_checkpoint_path(save_path)
    load_path = None
    load_optimizer_state = False

    if expand_from_checkpoint is not None and resume:
        raise ValueError("expand_from_checkpoint cannot be combined with resume")

    if os.path.isfile(save_path) and not resume:
        if expand_from_checkpoint is not None:
            raise IOError(
                f"Output exists: {save_path}. "
                "For --expand, set MODEL_FILE / MODEL_DIR to a new checkpoint path "
                "for the expanded model."
            )

        raise IOError(
            f"Output exists: {save_path}. Use --resume with --train to continue training."
        )

    if resume:
        if os.path.isfile(resume_path):
            load_path = resume_path
            load_optimizer_state = True
        elif bestpath and os.path.isfile(save_path):
            load_path = save_path
        else:
            raise IOError(f"Resume checkpoint not found: {resume_path}")

    Cin, Cout = _read_io_channels(train_h5)

    mean_X, std_X, mean_Y, std_Y = read_normalization(train_h5)

    model_config = dict(model_config)
    model_config["in_channels"] = Cin
    model_config["out_channels"] = Cout

    resume_state = {}
    best_val_init = None

    if load_path is not None:
        resume_state = torch.load(load_path, map_location="cpu")
        if not isinstance(resume_state, dict):
            raise RuntimeError(f"Invalid checkpoint: {load_path}")

        resumed_lr = resume_state.get("current_lr", resume_last_lr)
        if resumed_lr is not None:
            lr = resumed_lr

        if load_path == resume_path:
            print(f"Resuming training from {resume_path}")
        else:
            print(f"Resuming training from best checkpoint {save_path}")
            if resume_last_epoch is not None:
                resume_state["epoch"] = resume_last_epoch
                resume_state["completed_epochs"] = resume_last_epoch
            if resume_last_lr is not None:
                resume_state["current_lr"] = resume_last_lr

        if os.path.isfile(save_path):
            best_state = torch.load(save_path, map_location="cpu")
            if not isinstance(best_state, dict):
                raise RuntimeError(f"Invalid best checkpoint: {save_path}")
            best_val_init = best_state.get("val_loss")

        if best_val_init is None:
            best_val_init = resume_state.get("val_loss")

    if load_earlier_val is False:
        best_val_init = None

    builder = ModelBuilder(
        model=model,
        model_config=model_config,
        chi=chi,
        lines=lines,
        wave=wave,
        levels=levels,
        atom_names=atom_names,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        multi_gpu=multi_gpu,
        debug_loss=debug_loss,
        mean_X=mean_X,
        std_X=std_X,
        mean_Y=mean_Y,
        std_Y=std_Y,
        num_epochs=num_epochs,
        use_cosine=use_cosine,
        lr_min=min_learning_rate
    )

    if expand_from_checkpoint is not None:
        model = builder.build_model(wrap_fsdp=False)

        expand_info = expand_model_from_checkpoint(
            expand_from_checkpoint,
            model,
            map_location="cpu",
            zero_init_new_blocks=zero_init_new_blocks,
        )

        model = builder.wrap_model(model)
        optimizer, scheduler, loss_fn = builder.build_training_components(model)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if (
            not multi_gpu
            or not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        ):
            print(
                f"Expanded model from {expand_from_checkpoint}: "
                f"copied {len(expand_info['copied_keys'])} tensors, "
                f"zeroed {len(expand_info['zeroed_keys'])} residual tensors, "
                f"old_n_blocks={expand_info['old_n_blocks']}"
            )

    else:
        model, scheduler, optimizer, loss_fn = builder.build()

    if expand_from_checkpoint is None and load_path is not None:
        load_training_state(
            load_path,
            model,
            optimizer=optimizer if load_optimizer_state else None,
            scheduler=scheduler if load_optimizer_state else None,
            map_location="cpu",
        )

        if not load_optimizer_state:
            if resume_last_lr is not None:
                for group in optimizer.param_groups:
                    group["lr"] = resume_last_lr
        elif resume_state.get("opt_state") is None:
            current_lr = resume_state.get("current_lr")
            if current_lr is not None:
                for group in optimizer.param_groups:
                    group["lr"] = current_lr

        if str(builder.device).startswith("cuda"):
            torch.cuda.synchronize(builder.device)

    data_builder = DataLoaderBuilder(
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    train_loader, val_loader, _, _ = data_builder.build(
        train_h5,
        val_h5,
    )

    # run training
    train(
        model,
        train_loader,
        val_loader,
        scheduler,
        optimizer,
        loss_fn,
        save_path,
        num_epochs=num_epochs,
        device=builder.device,
        grad_clip=grad_clip,
        resume_last_epoch=resume_last_epoch,
        resume_state=resume_state,
        resume_path=resume_path,
        best_val_init=best_val_init,
        early_stopping=dict(
            enabled=True,
            patience=patience,
            min_delta=min_delta,
        )
    )


# ============================================================
# ---------------- VALIDATION TEST MODE -----------------------
# ============================================================

def _to_serializable(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]

    return value


def ffno_test_model(
    *,
    model,
    checkpoint_path,
    train_h5,
    val_h5,
    diagnostic_path,
    lines,
    wave,
    chi,
    levels,
    atom_names,
    model_config,
    dataset_type,
    batch_size=1,
    num_workers=8,
    pin_memory=True,
    device="cuda",
):
    if diagnostic_path is None:
        raise ValueError("diagnostic_path is required for test mode")

    os.makedirs(os.path.dirname(diagnostic_path) or ".", exist_ok=True)

    Cin, Cout = _read_io_channels(train_h5)
    mean_X, std_X, mean_Y, std_Y = read_normalization(train_h5)

    model_config = dict(model_config)
    model_config["in_channels"] = Cin
    model_config["out_channels"] = Cout

    builder = ModelBuilder(
        model=model,
        model_config=model_config,
        chi=chi,
        lines=lines,
        wave=wave,
        levels=levels,
        atom_names=atom_names,
        device=device,
        lr=0.0,
        weight_decay=0.0,
        multi_gpu=False,
        debug_loss=False,
        mean_X=mean_X,
        std_X=std_X,
        mean_Y=mean_Y,
        std_Y=std_Y,
    )

    model, _, optimizer, loss_fn = builder.build()
    del optimizer

    load_training_state(
        checkpoint_path,
        model,
        optimizer=None,
        scheduler=None,
        map_location="cpu",
    )
    model.eval()

    data_builder = DataLoaderBuilder(
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_dataset = data_builder.build_dataset(val_h5)
    val_loader, _ = data_builder.build_dataloader(val_dataset, shuffle=False)

    baseline_loss, baseline_comp, baseline_stats = validate(
        model=model,
        loader=val_loader,
        loss_fn=loss_fn,
        device=builder.device,
        collect_model_stats=True,
    )

    baseline_stats = dict(baseline_stats)
    baseline_stats.update(compute_mean_stats(baseline_stats))

    branch_masks = {
        "full": {"spec": 1.0, "vertical": 1.0, "pw": 1.0, "mlp": 1.0},
        "no_spec": {"spec": 0.0, "vertical": 1.0, "pw": 1.0, "mlp": 1.0},
        "no_vertical": {"spec": 1.0, "vertical": 0.0, "pw": 1.0, "mlp": 1.0},
        "no_pw": {"spec": 1.0, "vertical": 1.0, "pw": 0.0, "mlp": 1.0},
        "no_mlp": {"spec": 1.0, "vertical": 1.0, "pw": 1.0, "mlp": 0.0},
    }

    ablations = {}
    for name, mask in branch_masks.items():
        if name == "full":
            loss = baseline_loss
            comp = baseline_comp
            stats = baseline_stats
        else:
            loss, comp, stats = validate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=builder.device,
                collect_model_stats=True,
                forward_kwargs={"branch_mask": mask},
            )
            stats = dict(stats)
            stats.update(compute_mean_stats(stats))

        ablations[name] = {
            "branch_mask": mask,
            "loss": float(loss),
            "loss_delta_vs_full": float(loss - baseline_loss),
            "components": _to_serializable(comp),
            "stats": _to_serializable(stats),
        }

    branch_importance = {
        name: vals["loss_delta_vs_full"]
        for name, vals in ablations.items()
        if name != "full"
    }

    summary = {
        "checkpoint_path": checkpoint_path,
        "val_h5": val_h5,
        "baseline_loss": float(baseline_loss),
        "baseline_components": _to_serializable(baseline_comp),
        "baseline_stats": _to_serializable(baseline_stats),
        "branch_importance": branch_importance,
        "ablations": ablations,
    }

    with open(diagnostic_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved validation diagnostics to {diagnostic_path}")

    return summary


# ============================================================
# ---------------- PREDICTION (FFNO) --------------------------
# ============================================================

def _hann_2d(patch, device, dtype, eps=1e-3):
    w = torch.hann_window(patch, device=device, dtype=dtype)
    w = torch.clamp(w, min=eps)
    ww = torch.outer(w, w)
    ww = ww / (ww.max() + 1e-12)
    return ww


def _tile_positions(n, patch, stride):
    pos = list(range(0, n - patch + 1, stride))

    if pos[-1] != n - patch:
        pos.append(n - patch)

    return pos


@torch.no_grad()
def _predict_tiled(model, X, dx, dy, patch, stride, device="cuda"):

    model.eval()

    X = X.to(device)
    dx = dx.to(device)
    dy = dy.to(device)

    B, Cin, D, nx, ny = X.shape

    assert patch <= nx and patch <= ny

    print("=== TILE PREDICTION DEBUG ===")
    print("Input shape:", X.shape)
    print("Input range:", X.min().item(), X.max().item())
    print("dx:", dx)
    print("dy:", dy)

    if torch.isnan(X).any():
        print("WARNING: NaN detected in input")

    if torch.isinf(X).any():
        print("WARNING: Inf detected in input")

    # ------------------------------------------------
    # probe first tile
    # ------------------------------------------------

    x0 = X[:, :, :, 0:patch, 0:patch]

    y0 = model(x0, dx, dy)

    if torch.isnan(y0).any():
        print("WARNING: NaN detected in FIRST tile output")

    if torch.isinf(y0).any():
        print("WARNING: Inf detected in FIRST tile output")

    Cout = y0.shape[1]

    # ------------------------------------------------
    # accumulators
    # ------------------------------------------------

    Y_acc = torch.zeros((1, Cout, D, nx, ny), device=device, dtype=y0.dtype)
    W_acc = torch.zeros((1, 1, 1, nx, ny), device=device, dtype=y0.dtype)

    w2 = _hann_2d(patch, device=device, dtype=y0.dtype)[None, None, None, :, :]

    # ------------------------------------------------
    # tile positions
    # ------------------------------------------------

    xs = _tile_positions(nx, patch, stride)
    ys = _tile_positions(ny, patch, stride)

    assert all(x + patch <= nx for x in xs)
    assert all(y + patch <= ny for y in ys)

    print("Tile grid:", len(xs), "x", len(ys))

    print(f"patch={patch}, stride={stride}, nx={nx}, ny={ny}")
    print("xs =", xs)
    print("ys =", ys)

    # ------------------------------------------------
    # tiled inference
    # ------------------------------------------------

    # all_stats = []

    for i0, j0 in tqdm(itertools.product(xs, ys), total=len(xs)*len(ys), desc="Tiles"):

        xt = X[:, :, :, i0:i0+patch, j0:j0+patch]

        yt = model(xt, dx, dy)

        tile_weight = float(w2.sum().item())  # scalar weight

        # for s in stats:
        #     s = s.copy()
        #     s["_weight"] = tile_weight
        #     all_stats.append(s)

        if torch.isnan(yt).any() or torch.isinf(yt).any():

            print(f"NaN/Inf detected in tile ({i0},{j0})")

            finite = yt[torch.isfinite(yt)]

            if finite.numel() > 0:
                print(
                    "Tile output range:",
                    finite.min().item(),
                    finite.max().item()
                )

            # prevent contamination
            yt = torch.nan_to_num(
                yt,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

        Y_acc[:, :, :, i0:i0+patch, j0:j0+patch] += yt * w2
        W_acc[:, :, :, i0:i0+patch, j0:j0+patch] += w2

    # ------------------------------------------------
    # check weight coverage
    # ------------------------------------------------

    zero_pixels = (W_acc == 0).sum().item()

    print("W_acc min:", W_acc.min().item())
    print("Uncovered pixels:", zero_pixels)

    if zero_pixels > 0:
        print("WARNING: some pixels were not covered by tiles")

    # ------------------------------------------------
    # normalize safely
    # ------------------------------------------------

    W_acc = torch.clamp(W_acc, min=1e-12)

    out = Y_acc / W_acc

    if torch.isnan(out).any():
        print("WARNING: NaN present after stitching")

    if torch.isinf(out).any():
        print("WARNING: Inf present after stitching")

    out = torch.nan_to_num(out)

    print("Output range:", out.min().item(), out.max().item())
    print("=== TILE PREDICTION END ===")

    return out


def ffno_predict_populations(
    *,
    model,
    checkpoint_path,
    solve_h5,
    train_h5,          # <-- add this
    save_path,
    model_config,
    lines,
    wave,
    chi,
    levels,
    atom_names,
    diagnostic_path=None,
    cuda=True,
    tiled=True,
    patch=128,
    stride=64
):

    def _aggregate_stats(all_stats):

        layer_dict = {}

        for s in all_stats:
            layer = s["layer"]
            w = s.get("_weight", 1.0)

            if layer not in layer_dict:
                layer_dict[layer] = {}

            for k, v in s.items():
                if k in ("layer", "_weight"):
                    continue

                if k not in layer_dict[layer]:
                    layer_dict[layer][k] = {"sum": 0.0, "w": 0.0}

                layer_dict[layer][k]["sum"] += v * w
                layer_dict[layer][k]["w"] += w

        # weighted mean
        agg = {}
        for layer, vals in layer_dict.items():
            agg[layer] = {
                k: vals[k]["sum"] / (vals[k]["w"] + 1e-12)
                for k in vals
            }

        return agg

    """
    Predict log-departure coeffs -> convert to departure coeffs -> (optional) to populations downstream.

    This function writes:
      save_path: dataset "departure_coefficients" (linear, not log)
                + attrs "cmass_scale"
    """
    if os.path.isfile(save_path):
        raise IOError(f"Output exists: {save_path}")

    device = "cuda" if (cuda and torch.cuda.is_available()) else "cpu"

    Cin, Cout = _read_io_channels(train_h5)

    mean_X, std_X, mean_Y, std_Y = read_normalization(train_h5)

    model_config = dict(model_config)
    model_config["in_channels"] = Cin
    model_config["out_channels"] = Cout

    builder = ModelBuilder(
        model=model,
        model_config=model_config,
        chi=chi,
        lines=lines,
        wave=wave,
        levels=levels,
        atom_names=atom_names,
        device=device,
        lr=0.0,
        weight_decay=0.0,
        multi_gpu=False,
        debug_loss=False,
        mean_X=mean_X,
        std_X=std_X,
        mean_Y=mean_Y,
        std_Y=std_Y
    )

    model, scheduler, optimizer, loss_fn = builder.build()

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            print("NaN weights in", name)


    for name, buf in model.named_buffers():
        if torch.isnan(buf).any():
            print("NaN in buffer:", name, buf.shape)
        if torch.isinf(buf).any():
            print("Inf in buffer:", name, buf.shape)

    model.eval()

    with h5py.File(solve_h5, "r") as f:
        X = f["inputs"][...]   # [1,Cin,D,nx,ny]
        cmass_grid = f["cmass_grid"][...]
        # note: X stored float32 already
        dx = f["dx"][...]
        dy = f["dy"][...]
    
    X = torch.from_numpy(X).to(device)
    dx = torch.from_numpy(dx).to(device)
    dy = torch.from_numpy(dy).to(device)

    mean_X_t = torch.from_numpy(mean_X).float()[None, :, None, None, None].to(device)
    std_X_t  = torch.from_numpy(std_X).float()[None, :, None, None, None].to(device)

    X = (X - mean_X_t) / std_X_t

    print("dx =", dx, dx.dtype, dx.shape)
    print("dy =", dy, dy.dtype, dy.shape)
    print("X nan:", torch.isnan(X).sum().item())
    print("X inf:", torch.isinf(X).sum().item())

    print("X mean:", X.mean().item(), "std:", X.std().item())

    if not tiled:
        X = X.to(device)
        dx = dx.to(device)
        dy = dy.to(device)

        pred_log = model(X, dx, dy)
    else:
        pred_log = _predict_tiled(
            model,
            X,
            dx,
            dy,
            patch=patch,
            stride=stride,
            device=device,
        )

    if torch.isnan(pred_log).any():
        print("NaN in prediction")

    if torch.isinf(pred_log).any():
        print("Inf in prediction")

    print(
        "pred_log (normalized) range:",
        pred_log.min().item(),
        pred_log.max().item()
    )

    # ------------------------------------------------
    # DENORMALIZE OUTPUT (log space)
    # ------------------------------------------------
    mean_Y_t = torch.from_numpy(mean_Y).float()[None, :, None, None, None].to(pred_log.device)
    std_Y_t  = torch.from_numpy(std_Y).float()[None, :, None, None, None].to(pred_log.device)

    pred_log = pred_log * std_Y_t + mean_Y_t

    print(
        "pred_log (denorm) range:",
        pred_log.min().item(),
        pred_log.max().item()
    )

    # pred_log is log10(dep). Convert to linear dep:
    dep = invert_log_departure(pred_log).float().cpu().numpy()  # [1,Cout,D,nx,ny]

    print(
        "dep range:",
        dep.min().item(),
        dep.max().item()
    )

    dep = np.transpose(dep, (0, 3, 4, 2, 1)).astype(np.float32, copy=False)

    dep = dep[0]

    # Save
    with h5py.File(save_path, "w") as f:
        d = f.create_dataset("departure_coefficients", data=dep, compression="gzip", compression_opts=4, shuffle=True)
        d.attrs["cmass_scale"] = cmass_grid
        if "val_loss" in ckpt:
            f.attrs["val_loss"] = float(ckpt["val_loss"])
        f.attrs["epoch"] = int(ckpt.get("epoch", -1))

    return save_path
