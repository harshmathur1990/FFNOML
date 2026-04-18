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
from train_utils import (
    train,
    validate,
    compute_mean_stats,
    expand_model_from_checkpoint,
    get_checkpoint_io_metadata,
    get_resume_checkpoint_path,
    get_checkpoint_normalization,
    load_training_state,
)
from scipy.ndimage import gaussian_filter
from model_builder import ModelBuilder
from data_builder import DataLoaderBuilder
from normalize_utils import *
from tqdm import tqdm
import itertools
import math
from loss.nlte_composite_loss import (
    compute_Sv_all_lines_T_batched,
    extract_temperature,
    c_AHz,
    h,
    c,
)


# ============================================================
# ---------------- PREPROCESSING ------------------------------
# ============================================================

def _expand_z_to_shape(z_scale, target_shape):
    z_scale = np.asarray(z_scale, dtype=np.float32)

    if z_scale.shape == target_shape:
        return z_scale

    if z_scale.ndim == 1 and z_scale.shape[0] == target_shape[-1]:
        return np.broadcast_to(z_scale.reshape(1, 1, -1), target_shape).astype(
            np.float32,
            copy=False,
        )

    raise ValueError(
        f"z_scale must have shape {target_shape} or ({target_shape[-1]},), got {z_scale.shape}"
    )


def _expand_z_to_match_rho(z_scale, rho):
    return _expand_z_to_shape(z_scale, rho.shape)


def _normalize_z_scale(z_scale):
    return np.asarray(z_scale, dtype=np.float32) / 1e6


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


def _infer_patch_group_prefix(save_path):
    base = os.path.basename(save_path).lower()
    if "train" in base:
        return "train"
    if "test" in base or "val" in base:
        return "test"
    return "dataset"


def _normalization_stats_dict(mean_X, std_X, mean_Y, std_Y):
    return {
        "mean_X": np.asarray(mean_X, dtype=np.float32),
        "std_X": np.asarray(std_X, dtype=np.float32),
        "mean_Y": np.asarray(mean_Y, dtype=np.float32),
        "std_Y": np.asarray(std_Y, dtype=np.float32),
    }


def _io_metadata_dict(Cin, Cout):
    return {
        "Cin": int(Cin),
        "Cout": int(Cout),
    }


def _load_normalization_from_checkpoint_or_h5(checkpoint_path, h5_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    stats = get_checkpoint_normalization(ckpt)
    if stats is not None:
        return stats["mean_X"], stats["std_X"], stats["mean_Y"], stats["std_Y"]
    return read_normalization(h5_path)


def _load_inference_metadata_from_checkpoint(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    stats = get_checkpoint_normalization(ckpt)
    io_meta = get_checkpoint_io_metadata(ckpt)

    if stats is None:
        raise RuntimeError(
            f"Checkpoint missing normalization_stats: {checkpoint_path}"
        )
    if io_meta is None:
        raise RuntimeError(
            f"Checkpoint missing io_metadata (Cin/Cout): {checkpoint_path}"
        )

    return (
        io_meta["Cin"],
        io_meta["Cout"],
        stats["mean_X"],
        stats["std_X"],
        stats["mean_Y"],
        stats["std_Y"],
    )


def _make_inputs_ch_first(rho, temp, vx, vy, vz, ne):
    """
    returns inputs: [Cin, nz, nx, ny]  (channel-first with depth first)
    """
    features = _prepare_input_features(
        temp,
        vx,
        vy,
        vz,
        ne,
        rho,
    )  # [nx,ny,nz,Cin]
    features = np.transpose(features, (3, 2, 0, 1)).astype(np.float32, copy=False)
    return features


def _make_targets_ch_first(lte, nlte):
    """
    returns dep: [Cout, nz, nx, ny]
    """
    dep = _compute_departure_coefficients(lte, nlte)  # [nx,ny,nz,Cout] (or nlev)
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


def _extract_z_patches_xy(Z, patch, stride):
    """
    Z: [D, nx, ny]

    returns:
        Zp [N, D, patch, patch]
    """
    zs = []

    for i in range(0, Z.shape[1] - patch + 1, stride):
        for j in range(0, Z.shape[2] - patch + 1, stride):
            zs.append(Z[:, i:i+patch, j:j+patch])

    if len(zs) == 0:
        raise ValueError(
            f"Patch too large: patch={patch} for nx,ny={Z.shape[1]},{Z.shape[2]}"
        )

    return np.stack(zs, axis=0).astype(np.float32, copy=False)


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


def _flatten_columns_ch_first(x):
    x_t = torch.from_numpy(x[None, ...])
    return x_t.permute(0, 3, 4, 1, 2).reshape(-1, x.shape[0], x.shape[1])


def _restore_columns_ch_first(cols, channels, depth, height, width):
    return cols.reshape(1, height, width, channels, depth).permute(0, 3, 4, 1, 2)[0]


# ============================================================
# ---------------- HDF5 WRITERS -------------------------------
# ============================================================

def _save_hdf5_patches(
    path,
    patch_groups,
    *,
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
    patch_groups : list of dicts with per-group patch tensors/metadata

    mean_X, std_X : [Cin]
    mean_Y, std_Y : [Cout]

    """

    if os.path.isfile(path):
        raise IOError(f"Output exists: {path}")

    attrs = attrs or {}

    if len(patch_groups) == 0:
        raise ValueError("patch_groups must not be empty")

    sample_group = patch_groups[0]
    Cin = sample_group["inputs"].shape[1]
    Cout = sample_group["targets"].shape[1]
    total_patches = int(sum(group["inputs"].shape[0] for group in patch_groups))
    group_names = [group["name"] for group in patch_groups]

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
        for group in patch_groups:
            g = f.create_group(group["name"])

            g.create_dataset(
                "inputs",
                data=group["inputs"],
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            g.create_dataset(
                "targets",
                data=group["targets"],
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            g.create_dataset(
                "z_scale",
                data=group["z_scale"].astype(np.float32, copy=False),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            g.create_dataset("dx", data=group["dx"].astype(np.float32))
            g.create_dataset("dy", data=group["dy"].astype(np.float32))
            g.create_dataset("scale", data=group["scale"].astype(np.int32))
            g.create_dataset("weights", data=group["weights"].astype(np.float32))

            for k, v in group.get("attrs", {}).items():
                g.attrs[k] = v

            g.attrs["N"] = int(group["inputs"].shape[0])
            g.attrs["Cin"] = int(group["inputs"].shape[1])
            g.attrs["Cout"] = int(group["targets"].shape[1])
            g.attrs["D"] = int(group["inputs"].shape[2])
            g.attrs["P"] = int(group["inputs"].shape[3])

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

        f.attrs["N"] = total_patches
        f.attrs["Cin"] = Cin
        f.attrs["Cout"] = Cout
        f.attrs["n_patch_datasets"] = len(patch_groups)

        # flag for downstream safety
        f.attrs["normalized"] = int(mean_X is not None)
        f.create_dataset(
            "patch_dataset_names",
            data=np.asarray(group_names, dtype=h5py.string_dtype(encoding="utf-8")),
        )


def _save_hdf5_cube(path, X, z_scale, dx, dy, attrs=None):
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

        z_native = _expand_z_to_shape(
            z_scale,
            (X.shape[3], X.shape[4], X.shape[2]),
        )
        z_native = np.transpose(z_native, (2, 0, 1)).astype(np.float32, copy=False)
        f.create_dataset(
            "z_scale",
            data=z_native[None, ...],
            compression="gzip",
            compression_opts=4,
            shuffle=True,
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
    chi=None,
    lines=None,
    wave=None,
    levels=None,
    patch=96,
    stride=48,
    scales=(1,2,3,4),
    stat_file=None
):

    if os.path.isfile(save_path):
        raise IOError(f"Output exists: {save_path}")

    if stat_file is None:
        # ============================================================
        # -------- PASS 1: compute global normalization stats ---------
        # ============================================================

        x_sum = None
        x_sq_sum = None
        y_sum = None
        y_sq_sum = None
        x_count = 0
        y_count = 0

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

            X = _make_inputs_ch_first(rho, temp, vx, vy, vz, ne)

            Y = _make_targets_ch_first(lte, nlte)

            X_flat = X.reshape(X.shape[0], -1).astype(np.float64, copy=False)
            Y_flat = Y.reshape(Y.shape[0], -1).astype(np.float64, copy=False)

            x_sum_i = X_flat.sum(axis=1)
            x_sq_sum_i = np.square(X_flat).sum(axis=1)
            y_sum_i = Y_flat.sum(axis=1)
            y_sq_sum_i = np.square(Y_flat).sum(axis=1)

            if x_sum is None:
                x_sum = x_sum_i
                x_sq_sum = x_sq_sum_i
                y_sum = y_sum_i
                y_sq_sum = y_sq_sum_i
            else:
                x_sum += x_sum_i
                x_sq_sum += x_sq_sum_i
                y_sum += y_sum_i
                y_sq_sum += y_sq_sum_i

            x_count += X_flat.shape[1]
            y_count += Y_flat.shape[1]

        mean_X = (x_sum / max(1, x_count)).astype(np.float32)
        var_X = np.maximum(x_sq_sum / max(1, x_count) - np.square(mean_X, dtype=np.float64), 1e-12)
        std_X = np.sqrt(var_X).astype(np.float32)

        mean_Y = (y_sum / max(1, y_count)).astype(np.float32)
        var_Y = np.maximum(y_sq_sum / max(1, y_count) - np.square(mean_Y, dtype=np.float64), 1e-12)
        std_Y = np.sqrt(var_Y).astype(np.float32)

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

    patch_groups = []
    all_scale_arrays = []
    group_prefix = _infer_patch_group_prefix(save_path)

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

        X = _make_inputs_ch_first(rho, temp, vx, vy, vz, ne)

        Y = _make_targets_ch_first(lte, nlte)

        # ---------------- NORMALIZE HERE ----------------
        X = normalize_channels(X, mean_X, std_X)
        Y = normalize_channels(Y, mean_Y, std_Y)

        group_inputs = []
        group_targets = []
        group_z = []
        group_dx = []
        group_dy = []
        group_scale = []

        for s in scales:
            Xs, Ys = _downsample_xy(X, Y, s)

            z_native = _normalize_z_scale(_expand_z_to_match_rho(z, rho))
            z_native = np.transpose(z_native, (2, 0, 1)).astype(np.float32, copy=False)
            if s != 1:
                z_native = z_native[:, ::s, ::s]

            nx, ny = Xs.shape[2:]

            if nx < patch or ny < patch:
                continue

            Xp, Yp = _extract_patches_xy(
                Xs,
                Ys,
                patch=patch,
                stride=stride,
            )

            Zp = _extract_z_patches_xy(
                z_native,
                patch=patch,
                stride=stride,
            )

            n = Xp.shape[0]

            group_inputs.append(Xp)
            group_targets.append(Yp)
            group_z.append(Zp)

            group_dx.append(np.full(n, dx * s, dtype=np.float32))
            group_dy.append(np.full(n, dy * s, dtype=np.float32))
            group_scale.append(np.full(n, s, dtype=np.int32))

        if len(group_inputs) == 0:
            continue

        group_inputs = np.concatenate(group_inputs)
        group_targets = np.concatenate(group_targets)
        group_z = np.concatenate(group_z)
        group_dx = np.concatenate(group_dx)
        group_dy = np.concatenate(group_dy)
        group_scale = np.concatenate(group_scale)

        patch_groups.append(
            dict(
                name=f"{group_prefix}_{len(patch_groups) + 1}",
                inputs=group_inputs,
                targets=group_targets,
                z_scale=group_z,
                dx=group_dx,
                dy=group_dy,
                scale=group_scale,
                attrs=dict(native_depth=int(group_inputs.shape[2])),
            )
        )
        all_scale_arrays.append(group_scale)

    if len(patch_groups) == 0:
        raise RuntimeError("No patches generated. Check patch size and scales.")

    scale_all = np.concatenate(all_scale_arrays)
    scale_weights = np.zeros_like(scale_all, dtype=np.float32)

    for s in np.unique(scale_all):
        mask = scale_all == s
        scale_weights[mask] = 1.0 / mask.sum()

    scale_weights *= len(scale_weights)
    scale_weights /= scale_weights.mean()

    offset = 0
    x_sum = 0.0
    x_sq_sum = 0.0
    y_sum = 0.0
    y_sq_sum = 0.0
    x_count = 0
    y_count = 0

    for group in patch_groups:
        n = group["inputs"].shape[0]
        group["weights"] = scale_weights[offset:offset + n]
        offset += n

        x_sum += float(group["inputs"].sum(dtype=np.float64))
        x_sq_sum += float(np.square(group["inputs"], dtype=np.float64).sum(dtype=np.float64))
        y_sum += float(group["targets"].sum(dtype=np.float64))
        y_sq_sum += float(np.square(group["targets"], dtype=np.float64).sum(dtype=np.float64))
        x_count += int(group["inputs"].size)
        y_count += int(group["targets"].size)

    x_mean = x_sum / max(1, x_count)
    y_mean = y_sum / max(1, y_count)
    x_std = math.sqrt(max(0.0, x_sq_sum / max(1, x_count) - x_mean ** 2))
    y_std = math.sqrt(max(0.0, y_sq_sum / max(1, y_count) - y_mean ** 2))

    print("==== DATA CHECK ====")
    print("X mean:", x_mean, "std:", x_std)
    print("Y mean:", y_mean, "std:", y_std)

    # ============================================================
    # save dataset
    # ============================================================

    _save_hdf5_patches(
        save_path,
        patch_groups,
        attrs=dict(
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
):
    """
    Build dataset for prediction (no targets).

    Only atmospheric inputs are stored.
    """

    if os.path.isfile(save_path):
        raise IOError(f"Output exists: {save_path}")

    X = _make_inputs_ch_first(rho, temp, vx, vy, vz, ne)  # [Cin, D, nx, ny]

    _save_hdf5_cube(
        save_path,
        X,
        _normalize_z_scale(z_scale),
        dx,
        dy,
        attrs=dict(native_depth=int(X.shape[1])),
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
    normalization_stats = _normalization_stats_dict(mean_X, std_X, mean_Y, std_Y)
    io_metadata = _io_metadata_dict(Cin, Cout)

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

    effective_resume_last_epoch = None if expand_from_checkpoint is not None else resume_last_epoch
    effective_resume_state = {} if expand_from_checkpoint is not None else resume_state
    effective_resume_path = None if expand_from_checkpoint is not None else resume_path
    effective_best_val_init = None if expand_from_checkpoint is not None else best_val_init

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
        resume_last_epoch=effective_resume_last_epoch,
        resume_state=effective_resume_state,
        resume_path=effective_resume_path,
        best_val_init=effective_best_val_init,
        normalization_stats=normalization_stats,
        io_metadata=io_metadata,
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
    mean_X, std_X, mean_Y, std_Y = _load_normalization_from_checkpoint_or_h5(
        checkpoint_path,
        train_h5,
    )

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
def _predict_tiled(model, X, z_scale, dx, dy, patch, stride, device="cuda"):

    model.eval()

    X = X.to(device)
    z_scale = z_scale.to(device)
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
    z0 = z_scale[:, :, 0:patch, 0:patch]

    y0 = model(x0, z0, dx, dy)

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
        zt = z_scale[:, :, i0:i0+patch, j0:j0+patch]

        yt = model(xt, zt, dx, dy)

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
                + attrs "z_scale"
    """
    if os.path.isfile(save_path):
        raise IOError(f"Output exists: {save_path}")

    device = "cuda" if (cuda and torch.cuda.is_available()) else "cpu"

    Cin, Cout, mean_X, std_X, mean_Y, std_Y = _load_inference_metadata_from_checkpoint(
        checkpoint_path,
    )

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
        z_scale = f["z_scale"][...]
        # note: X stored float32 already
        dx = f["dx"][...]
        dy = f["dy"][...]
    
    X = torch.from_numpy(X).to(device)
    z_scale = torch.from_numpy(np.transpose(z_scale, (0, 3, 1, 2))).to(device)
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

        pred_log = model(X, z_scale, dx, dy)
    else:
        pred_log = _predict_tiled(
            model,
            X,
            z_scale,
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
        d.attrs["z_scale"] = z_scale[0]
        d.attrs["depth_scale_type"] = "z"
        if "val_loss" in ckpt:
            f.attrs["val_loss"] = float(ckpt["val_loss"])
        f.attrs["epoch"] = int(ckpt.get("epoch", -1))

    return save_path
