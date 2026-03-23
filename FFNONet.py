# ============================================================
# FFNO Utilities (full-volume / patch-to-patch operator training)
#
# Public API (new):
#   build_dataset_ffno
#   build_solving_set_ffno
#   ffno_train_model
#   ffno_predict_populations
#
# ============================================================

import sys
import os
import numpy as np
import h5py
import torch
from interp_utils import interpolate_everything
from train_utils import train
from scipy.ndimage import gaussian_filter
from model_builder import ModelBuilder
from data_builder import DataLoaderBuilder
from normalize_utils import *
from tqdm import tqdm
import itertools
import math


def compute_k_cutoff(dx, dy=None, k_scale=1e5, radial=True):
    """
    Compute training-band cutoff in the SAME scaled k-space
    used by your layer.

    Parameters
    ----------
    dx : float
        Training grid spacing in x.
    dy : float or None
        Training grid spacing in y. If None, dy=dx.
    k_scale : float
        The same scaling factor used in the model.
    radial : bool
        If True, return radial cutoff based on k_mag.
        If False, return per-axis cutoffs.

    Returns
    -------
    radial=True:
        k_cutoff : float

    radial=False:
        kx_cutoff, ky_cutoff : float, float
    """
    if dy is None:
        dy = dx

    kx_cutoff = math.pi / dx * k_scale
    ky_cutoff = math.pi / dy * k_scale

    if radial:
        return math.sqrt(kx_cutoff**2 + ky_cutoff**2)
    else:
        return kx_cutoff, ky_cutoff


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
    weight_decay=1e-4,
    num_workers=8,
    pin_memory=True,
    amp=True,
    grad_clip=1.0,
    device="cuda",
    multi_gpu=False,
    debug_loss=False,
    patience=10,
    min_delta=1e-5,
    use_cosine=False,
    min_learning_rate=1e-6
):

    if os.path.isfile(save_path):
        raise IOError(f"Output exists: {save_path}")

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
        lr=lr,
        weight_decay=weight_decay,
        amp=amp,
        multi_gpu=multi_gpu,
        debug_loss=debug_loss,
        mean_X=mean_X,
        std_X=std_X,
        mean_Y=mean_Y,
        std_Y=std_Y,
        num_epochs=NUM_EPOCHS,
        use_cosine=use_cosine,
        lr_min=min_learning_rate
    )

    model, scheduler, optimizer, loss_fn, scaler = builder.build()

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
        scaler,
        save_path,
        num_epochs=num_epochs,
        device=builder.device,
        amp=amp,
        grad_clip=grad_clip,
        early_stopping=dict(
            enabled=True,
            patience=patience,
            min_delta=min_delta,
        )
    )


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
def _predict_tiled(model, X, dx, dy, patch, stride, device="cuda", amp=True):

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

    with torch.amp.autocast("cuda", enabled=(amp and str(device).startswith("cuda"))):
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

    all_stats = []

    for i0, j0 in tqdm(itertools.product(xs, ys), total=len(xs)*len(ys), desc="Tiles"):

        xt = X[:, :, :, i0:i0+patch, j0:j0+patch]

        with torch.amp.autocast("cuda", enabled=(amp and str(device).startswith("cuda"))):

            yt, stats = model(xt, dx, dy, collect_stats=True)

            tile_weight = float(w2.sum().item())  # scalar weight

            for s in stats:
                s = s.copy()
                s["_weight"] = tile_weight
                all_stats.append(s)

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

    return out, all_stats


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
    stride=64,
    dx_cutoff=None,
    dy_cutoff=None
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

    amp = False

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
    kx_cutoff, ky_cutoff = compute_k_cutoff(dx=dx_cutoff, dy=dy_cutoff, k_scale=1e5, radial=False)

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
        amp=amp,
        multi_gpu=False,
        debug_loss=False,
        mean_X=mean_X,
        std_X=std_X,
        mean_Y=mean_Y,
        std_Y=std_Y,
        kx_cutoff=kx_cutoff,
        ky_cutoff=ky_cutoff
    )

    model, optimizer, loss_fn, scaler = builder.build()

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

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(amp and str(device).startswith("cuda"))):
            pred_log, all_stats = model(X, dx, dy, collect_stats=True)
    else:
        pred_log, all_stats = _predict_tiled(
            model,
            X,
            dx,
            dy,
            patch=patch,
            stride=stride,
            device=device,
            amp=amp
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

    # Diagnostics
    if diagnostic_path is not None:

        agg_stats = _aggregate_stats(all_stats)

        if len(agg_stats) == 0:
            print("WARNING: No stats collected")
            return save_path

        summary = {}

        first_layer = next(iter(agg_stats.values()))
        for k in first_layer.keys():
            vals = [agg_stats[layer][k] for layer in agg_stats]
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"]  = float(np.std(vals))

        summary["spec_to_pw_ratio"] = float(
            summary["spec_norm_mean"] / (summary["pw_norm_mean"] + 1e-12)
        )

        summary["vertical_to_mlp_ratio"] = float(
            summary["vertical_norm_mean"] / (summary["mlp_norm_mean"] + 1e-12)
        )

        summary["alpha_spec_mean"] = float(np.mean([
            agg_stats[l]["alpha_spec"] for l in agg_stats
        ]))
        summary["alpha_pw_mean"] = float(np.mean([
            agg_stats[l]["alpha_pw"] for l in agg_stats
        ]))
        summary["alpha_vert_mean"] = float(np.mean([
            agg_stats[l]["alpha_vert"] for l in agg_stats
        ]))
        summary["alpha_mlp_mean"] = float(np.mean([
            agg_stats[l]["alpha_mlp"] for l in agg_stats
        ]))

        summary["pred_log_mean"] = float(pred_log.mean().item())
        summary["pred_log_std"]  = float(pred_log.std().item())

        summary["dep_mean"] = float(dep.mean())
        summary["dep_std"]  = float(dep.std())
        summary["dep_min"]  = float(dep.min())
        summary["dep_max"]  = float(dep.max())
        stats_flat = {}

        for layer, vals in agg_stats.items():
            for k, v in vals.items():
                stats_flat[f"layer{layer}_{k}"] = v

        stats_flat.update(summary)

        np.savez(diagnostic_path, **stats_flat)

    return save_path
