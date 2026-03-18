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


# ============================================================
# ---------------- PREPROCESSING ------------------------------
# ============================================================

def _prepare_input_features(temp, vx, vy, vz, ne, rho, tscale=1):
    """
    temp: [nx, ny, nz]
    returns features: [nx, ny, nz, Cin]
    """
    return np.stack(
        [
            np.log10(temp),
            vx / 100,
            vy / 100,
            vz / 100,
            np.log10(ne),
            np.log10(rho),
        ],
        axis=-1,
    )


def _compute_departure_coefficients(lte, nlte, eps=1e-30, tscale=1):
    """
    lte/nlte: [nx, ny, nz, nlev] or [nx, ny, nz, Cout]
    return: log10(nlte/lte)
    """
    return np.log10((nlte + eps) / (lte + eps))


def invert_log_departure(pred_log, tscale=1):
    return torch.pow(10.0, pred_log)


def _make_inputs_ch_first(
    rho, z_scale,
    temp, vx, vy, vz, ne,
    *,
    ndep,
    tscale
):
    """
    returns inputs: [Cin, ndep, nx, ny]  (channel-first with depth first)
    """
    cmass_grid = np.logspace(-6, 2, ndep)

    features = _prepare_input_features(temp, vx, vy, vz, ne, rho, tscale=tscale)  # [nx,ny,nz,Cin]

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
    tscale
):
    """
    returns dep: [Cout, ndep, nx, ny]
    """
    cmass_grid = np.logspace(-6, 2, ndep)

    dep = _compute_departure_coefficients(lte, nlte, tscale=tscale)  # [nx,ny,nz,Cout] (or nlev)

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
    tscale=1,
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

        f.create_dataset("tscale", data=tscale)

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


def _save_hdf5_cube(path, X, cmass_grid, dx, dy, tscale, attrs=None):
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

        f.create_dataset("tscale", data=tscale)

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


def read_tscale(h5_path):
    """
    Reads tscale from training HDF5 file.
    """
    with h5py.File(h5_path, "r") as f:
        tscale = f['tscale'][()] if 'tscale' in f.keys() else None

    return tscale


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
    tscale=1
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
            tscale=tscale
        )

        Y = _make_targets_ch_first(
            rho, z, lte, nlte,
            ndep=ndep,
            tscale=tscale
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
        tscale=tscale,
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
    tscale=1
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
        tscale=tscale
    )  # [Cin, D, nx, ny]

    _save_hdf5_cube(
        save_path,
        X,
        cmass_grid,
        dx,
        dy,
        tscale,
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
    tscale=1
):

    if os.path.isfile(save_path):
        raise IOError(f"Output exists: {save_path}")

    Cin, Cout = _read_io_channels(train_h5)
    
    tscale_read = read_tscale(train_h5)

    if tscale_read is None or tscale_read != tscale:
        sys.stderr.write(f"\n tscale_read is {tscale_read} and tscale is {tscale} \n")
        sys.exit(-1)

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
        tscale=tscale
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

    for i0 in xs:
        for j0 in ys:

            xt = X[:, :, :, i0:i0+patch, j0:j0+patch]

            with torch.amp.autocast("cuda", enabled=(amp and str(device).startswith("cuda"))):
                yt = model(xt, dx, dy)

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
    stride=64,
    tscale=1
):
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

    tscale_read = read_tscale(train_h5)

    if tscale_read is None or tscale_read != tscale:
        sys.stderr.write(f"\n tscale_read is {tscale_read} and tscale is {tscale} \n")
        sys.exit(-1)

    tscale_solve = read_tscale(solve_h5)

    if tscale_solve is None or tscale_solve != tscale:
        sys.stderr.write(f"\n tscale_solve is {tscale_solve} and tscale is {tscale} \n")
        sys.exit(-1)

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
        debug_loss=False
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
    
    X = torch.from_numpy(X)
    dx = torch.from_numpy(dx)
    dy = torch.from_numpy(dy)

    print("dx =", dx, dx.dtype, dx.shape)
    print("dy =", dy, dy.dtype, dy.shape)
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
    dep = invert_log_departure(pred_log, tscale=tscale).float().cpu().numpy()  # [1,Cout,D,nx,ny]

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
        # simple summary stats
        stats = dict(
            dep_mean=float(np.mean(dep)),
            dep_std=float(np.std(dep)),
            dep_min=float(np.min(dep)),
            dep_max=float(np.max(dep)),
        )
        np.savez(diagnostic_path, **stats)

    return save_path
