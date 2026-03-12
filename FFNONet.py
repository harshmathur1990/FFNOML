# ============================================================
# FFNO Utilities (full-volume / patch-to-patch operator training)
#
# Drop-in replacement for your window-based SunnyNet pipeline.
#
# What changes:
#   - NO 7x7 -> center-column extraction
#   - We train an operator: [Cin, D, Px, Py] -> [Cout, D, Px, Py]
#   - You can generate many TRAINING patches via (patch, stride)
#   - At inference, you can predict either:
#       (A) full cube in one go (if GPU allows), or
#       (B) tiled prediction with overlap + blending
#
# Public API (new):
#   build_dataset_ffno
#   build_solving_set_ffno
#   ffno_train_model
#   ffno_predict_populations
#
# ============================================================

import os
import numpy as np
import h5py
import torch
from interp_utils import interpolate_everything
from train_utils import train
from scipy.ndimage import gaussian_filter
from model_builder import ModelBuilder
from data_builder import DataLoaderBuilder


# ============================================================
# ---------------- PREPROCESSING ------------------------------
# ============================================================

LOG_SCALE = 10.0

def _prepare_input_features(temp, vx, vy, vz, ne, rho):
    """
    temp: [nx, ny, nz]
    returns features: [nx, ny, nz, Cin]
    """
    return np.stack(
        [
            np.log10(temp) / 10,
            vx / 100.0,
            vy / 100.0,
            vz / 100.0,
            np.log10(ne) / 10,
            np.log10(rho) / 10,
        ],
        axis=-1,
    )


def _compute_departure_coefficients(lte, nlte, eps=1e-30):
    """
    lte/nlte: [nx, ny, nz, nlev] or [nx, ny, nz, Cout]
    return: log10(nlte/lte)
    """
    return np.log10((nlte + eps) / (lte + eps)) / LOG_SCALE


def invert_log_departure(pred_log):
    return torch.pow(10.0, pred_log * LOG_SCALE)


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
    ndep,
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
    attrs=None,
):
    """
    Save training patches.

    Parameters
    ----------
    Xp : [N, Cin, D, P, P]
    Yp : [N, Cout, D, P, P]

    dxs : [N]
    dys : [N]

    scales : [N] optional
        resolution scale factor

    weights : [N] optional
        sample weight for loss

    cmass_grid : [D]
    """

    if os.path.isfile(path):
        raise IOError(f"Output exists: {path}")

    attrs = attrs or {}

    N = Xp.shape[0]

    with h5py.File(path, "w") as f:

        # ------------------------------------------------
        # main datasets
        # ------------------------------------------------

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

        # ------------------------------------------------
        # optional metadata datasets
        # ------------------------------------------------

        if scales is not None:
            f.create_dataset("scale", data=scales.astype(np.int32))

        if weights is not None:
            f.create_dataset("weights", data=weights.astype(np.float32))

        f.create_dataset(
            "cmass_grid",
            data=cmass_grid.astype(np.float64),
        )

        # ------------------------------------------------
        # attributes
        # ------------------------------------------------

        for k, v in attrs.items():
            f.attrs[k] = v

        f.attrs["N"] = N
        f.attrs["Cin"] = Xp.shape[1]
        f.attrs["Cout"] = Yp.shape[1]
        f.attrs["D"] = Xp.shape[2]
        f.attrs["P"] = Xp.shape[3]

        if scales is not None:
            f.attrs["n_scales"] = int(len(np.unique(scales)))


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
    scales=(1,2,3,4)
):

    if os.path.isfile(save_path):
        raise IOError(f"Output exists: {save_path}")

    X_all = []
    Y_all = []

    dx_all = []
    dy_all = []
    scale_all = []

    cmass_grid_ref = None

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
            ndep=ndep,
        )

        Y = _make_targets_ch_first(
            rho, z, lte, nlte,
            ndep=ndep,
        )

        if cmass_grid_ref is None:
            cmass_grid_ref = cmass_grid
        elif not np.allclose(cmass_grid_ref, cmass_grid):
            raise ValueError("cmass_grid mismatch")

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

    # --------------------------------------------
    # concatenate
    # --------------------------------------------

    if len(X_all) == 0:
        raise RuntimeError("No patches generated. Check patch size and scales.")

    X_all = np.concatenate(X_all)
    Y_all = np.concatenate(Y_all)

    dx_all = np.concatenate(dx_all)
    dy_all = np.concatenate(dy_all)
    scale_all = np.concatenate(scale_all)

    # --------------------------------------------
    # compute weights
    # --------------------------------------------

    weights = np.zeros_like(scale_all, dtype=np.float32)

    for s in np.unique(scale_all):
        mask = scale_all == s
        weights[mask] = 1.0 / mask.sum()

    weights *= len(weights)

    weights /= weights.mean()

    # --------------------------------------------
    # save dataset
    # --------------------------------------------

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
    ndep=400,
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
        ndep=ndep,
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
    min_delta=1e-5
):

    if os.path.isfile(save_path):
        raise IOError(f"Output exists: {save_path}")

    Cin, Cout = _read_io_channels(train_h5)
    
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
        debug_loss=debug_loss
    )

    model, optimizer, loss_fn, scaler = builder.build()

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
        ),
    )


# ============================================================
# ---------------- PREDICTION (FFNO) --------------------------
# ============================================================

def _hann_2d(patch, device, dtype):
    """
    Smooth blending window to reduce seams in tiled prediction.
    """
    w = torch.hann_window(patch, device=device, dtype=dtype)
    ww = torch.outer(w, w)  # [P,P]
    ww = ww / (ww.max() + 1e-12)
    return ww


@torch.no_grad()
def _predict_tiled(model, X, dx, dy, patch, stride, device="cuda", amp=True):

    model.eval()

    X = X.to(device)
    dx = dx.to(device)
    dy = dy.to(device)

    B, Cin, D, nx, ny = X.shape

    assert patch <= nx and patch <= ny

    i0, j0 = 0, 0
    x0 = X[:, :, :, i0:i0+patch, j0:j0+patch]

    with torch.amp.autocast("cuda", enabled=(amp and str(device).startswith("cuda"))):
        y0 = model(x0, dx, dy)

    Cout = y0.shape[1]

    Y_acc = torch.zeros((1, Cout, D, nx, ny), device=device, dtype=y0.dtype)
    W_acc = torch.zeros((1, 1, 1, nx, ny), device=device, dtype=y0.dtype)

    w2 = _hann_2d(patch, device=device, dtype=y0.dtype)[None, None, None, :, :]

    for i in range(0, nx - patch + 1, stride):
        for j in range(0, ny - patch + 1, stride):

            i0 = min(i, nx - patch)
            j0 = min(j, ny - patch)

            xt = X[:, :, :, i0:i0+patch, j0:j0+patch]

            with torch.amp.autocast("cuda", enabled=(amp and str(device).startswith("cuda"))):
                yt = model(xt, dx, dy)

            if torch.isnan(yt).any():
                print("NaN detected in tile", i0, j0)

            Y_acc[:, :, :, i0:i0+patch, j0:j0+patch] += yt * w2
            W_acc[:, :, :, i0:i0+patch, j0:j0+patch] += w2

    return Y_acc / (W_acc + 1e-12)


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
    amp=True,
    tiled=True,
    patch=128,
    stride=64,
):
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
        amp=False,
        multi_gpu=False,
        debug_loss=False
    )

    model, optimizer, loss_fn, scaler = builder.build()

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            print("NaN weights in", name)

    model.eval()

    with h5py.File(solve_h5, "r") as f:
        X = f["inputs"][...]   # [1,Cin,D,nx,ny]
        cmass_grid = f["cmass_grid"][...]
        # note: X stored float32 already
        dx = f["dx"][...]
        dy = f["dy"][...]
    
    X = torch.from_numpy(X)
    dx = torch.from_numpy(dx)
    dy = torch.from_numpy(dy)

    print("X nan:", np.isnan(X).sum())
    print("X inf:", np.isinf(X).sum())

    if not tiled:
        X = X.to(device)
        dx = dx.to(device)
        dy = dy.to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(amp and str(device).startswith("cuda"))):
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
            amp=amp
        )

    if torch.isnan(pred_log).any():
        print("NaN in prediction")

    if torch.isinf(pred_log).any():
        print("Inf in prediction")

    print(
        "pred_log range:",
        pred_log.min().item(),
        pred_log.max().item()
    )

    # pred_log is log10(dep). Convert to linear dep:
    dep = invert_log_departure(pred_log).float().cpu().numpy()  # [1,Cout,D,nx,ny]

    # Save
    with h5py.File(save_path, "w") as f:
        d = f.create_dataset("departure_coefficients", data=dep, compression="gzip", compression_opts=4, shuffle=True)
        d.attrs["cmass_scale"] = cmass_grid
        if "val_loss" in ckpt:
            f.attrs["val_loss"] = float(ckpt["val_loss"])
        f.attrs["epoch"] = int(ckpt.get("epoch", -1))

    # Diagnostics
    if diagnostic_path is not None:
        # simple summary stats
        stats = dict(
            dep_mean=float(np.mean(dep)),
            dep_std=float(np.std(dep)),
            dep_min=float(np.min(dep)),
            dep_max=float(np.max(dep)),
        )
        np.savez(diagnostic_path, **stats)

    return save_path
