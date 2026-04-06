import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from helita.sim.multi3d import Multi3dAtmos, Multi3dOut

from config import ACTIVE_ATOMS, MODEL, MODEL_DIR, MULTI3D_PRED_DATA
from interp_utils import interpolate_everything
from pipeline import compute_dx_dy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=MODEL_DIR,
        help="Directory where plots are written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser.parse_args()


def plot_population_error_envelopes(
    pred,
    true,
    cmass,
    level_names=None,
    figsize=(10, 10),
    ncols=2,
):
    """
    Plot median + 68% + 95% relative-error envelopes vs log10(cmass).
    """

    assert pred.shape == true.shape
    nlevels, ndepth, nx, ny = pred.shape

    eps = 1e-30
    rel_err = (pred - true) / (true + eps)
    rel_err[true <= 0] = np.nan
    rel_err = rel_err.reshape(nlevels, ndepth, -1)

    p50 = np.nanmedian(rel_err, axis=-1)
    p16 = np.nanpercentile(rel_err, 16, axis=-1)
    p84 = np.nanpercentile(rel_err, 84, axis=-1)
    p025 = np.nanpercentile(rel_err, 2.5, axis=-1)
    p975 = np.nanpercentile(rel_err, 97.5, axis=-1)

    log_cmass = np.log10(cmass)

    nrows = int(np.ceil(nlevels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i in range(nlevels):
        ax = axes[i // ncols, i % ncols]

        ax.fill_between(log_cmass, p025[i], p975[i], color="#e6b98c", alpha=0.8)
        ax.fill_between(log_cmass, p16[i], p84[i], color="#6f8fa6", alpha=0.9)
        ax.plot(log_cmass, p50[i], color="#1f77b4", lw=1.5)

        if level_names is not None:
            ax.set_title(level_names[i], fontsize=12)

        ax.axhline(0.0, color="k", lw=0.5, alpha=0.5)

    for ax in axes[-1]:
        ax.set_xlabel(r"log cmass")

    for ax in axes[:, 0]:
        ax.set_ylabel(r"(b$_{NN}$ - b) / b")

    for j in range(nlevels, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    plt.tight_layout()
    return fig, axes


def plot_log_population_error_envelopes(
    pred,
    true,
    cmass,
    level_names=None,
    figsize=(10, 10),
    ncols=2,
):
    """
    Plot median + 68% + 95% envelopes for (log10(b_NN) - log10(b)) / log10(b)
    vs log10(cmass).
    """

    assert pred.shape == true.shape
    nlevels, ndepth, nx, ny = pred.shape

    eps = 1e-30
    log_pred = np.log10(np.maximum(pred, eps))
    log_true = np.log10(np.maximum(true, eps))
    log_rel_err = (log_pred - log_true) / (log_true + eps)
    log_rel_err[true <= 0] = np.nan
    log_rel_err = log_rel_err.reshape(nlevels, ndepth, -1)

    p50 = np.nanmedian(log_rel_err, axis=-1)
    p16 = np.nanpercentile(log_rel_err, 16, axis=-1)
    p84 = np.nanpercentile(log_rel_err, 84, axis=-1)
    p025 = np.nanpercentile(log_rel_err, 2.5, axis=-1)
    p975 = np.nanpercentile(log_rel_err, 97.5, axis=-1)

    log_cmass = np.log10(cmass)

    nrows = int(np.ceil(nlevels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i in range(nlevels):
        ax = axes[i // ncols, i % ncols]

        ax.fill_between(log_cmass, p025[i], p975[i], color="#d9c2f0", alpha=0.8)
        ax.fill_between(log_cmass, p16[i], p84[i], color="#7b8fc7", alpha=0.9)
        ax.plot(log_cmass, p50[i], color="#243b6b", lw=1.5)

        if level_names is not None:
            ax.set_title(level_names[i], fontsize=12)

        ax.axhline(0.0, color="k", lw=0.5, alpha=0.5)

    for ax in axes[-1]:
        ax.set_xlabel(r"log cmass")

    for ax in axes[:, 0]:
        ax.set_ylabel(r"($\log b_{NN}$ - $\log b$) / $\log b$")

    for j in range(nlevels, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    plt.tight_layout()
    return fig, axes


def load_true_multi3d_departures(dataset, active_atoms=None):
    """
    Load Multi3D truth for one snapshot and concatenate atoms in ACTIVE_ATOMS order.
    """

    active_atoms = active_atoms or ACTIVE_ATOMS

    lte_blocks = []
    nlte_blocks = []
    level_names = []

    atmos = None
    rho = None
    z_scale = None

    for atom in active_atoms:
        mpath = dataset["MULTI3D_PATHS"][atom]
        m3d = Multi3dOut(directory=mpath)
        m3d.readall()

        lte = m3d.atom.nstar[:] * 1e6
        nlte = m3d.atom.n[:] * 1e6

        lte_blocks.append(lte)
        nlte_blocks.append(nlte)
        level_names.extend([f"{atom} {ilevel + 1}" for ilevel in range(lte.shape[-1])])

        if atmos is None:
            nx, ny, nz, _ = lte.shape
            atmos = Multi3dAtmos(dataset["MULTI3D_ATMOS"], nx, ny, nz)

            if hasattr(atmos, "readall"):
                atmos.readall()

            rho = atmos.rho[:] * 1e3
            z_scale = m3d.geometry.z[:] * 1e-2

            dx, dy = compute_dx_dy(dataset["MESH"])
            print(f"{dataset['NAME']}: dx={abs(dx):.4e} m, dy={abs(dy):.4e} m")

    lte_block = np.concatenate(lte_blocks, axis=-1)
    nlte_block = np.concatenate(nlte_blocks, axis=-1)

    return rho, z_scale, lte_block, nlte_block, level_names


def build_truth_paths(dataset):
    base_dir = os.path.dirname(dataset["MULTI3D_ATMOS"])
    return {atom: os.path.join(base_dir, atom) for atom in ACTIVE_ATOMS}


def build_prediction_file(dataset):
    return os.path.join(MODEL_DIR, f"output_3D_sim_s5_{dataset['NAME']}_{MODEL}.hdf5")


def build_plot_jobs():
    jobs = []

    for dataset in MULTI3D_PRED_DATA:
        truth_paths = build_truth_paths(dataset)
        pred_file = build_prediction_file(dataset)

        if not os.path.exists(pred_file):
            print(f"Skipping {dataset['NAME']}: missing prediction file {pred_file}")
            continue

        missing_truth = [path for path in truth_paths.values() if not os.path.exists(path)]
        if missing_truth:
            print(f"Skipping {dataset['NAME']}: missing Multi3D truth paths {missing_truth}")
            continue

        job = dict(dataset)
        job["MULTI3D_PATHS"] = truth_paths
        job["PRED_FILE"] = pred_file
        jobs.append(job)

    return jobs


def load_prediction_file(pred_file):
    with h5py.File(pred_file, "r") as f:
        dep = f["departure_coefficients"][...]
        cmass_grid = f["departure_coefficients"].attrs["cmass_scale"]

    return dep, np.asarray(cmass_grid)


def interpolate_true_departures(rho, z_scale, lte, nlte, cmass_grid):
    """
    Recreate the training/prediction target:
    interpolate log10(departure coefficient) to cmass grid, then invert to linear.
    """

    eps = 1e-30
    log_dep = np.log10((nlte + eps) / (lte + eps))
    true_log_dep = interpolate_everything(rho, z_scale, log_dep, cmass_grid)
    true_dep = np.power(10.0, true_log_dep, dtype=np.float64)
    return true_dep.astype(np.float32, copy=False)


def prepare_plot_arrays(pred_dep, true_dep):
    """
    Convert [nx, ny, ndep, nlevels] -> [nlevels, ndep, nx, ny].
    """

    if pred_dep.shape != true_dep.shape:
        raise ValueError(
            f"Prediction/true shape mismatch: {pred_dep.shape} vs {true_dep.shape}"
        )

    pred_plot = np.transpose(pred_dep, (3, 2, 0, 1))
    true_plot = np.transpose(true_dep, (3, 2, 0, 1))
    return pred_plot, true_plot


def make_snapshot_plot(dataset, output_dir, show=False):
    pred_dep, cmass_grid = load_prediction_file(dataset["PRED_FILE"])
    rho, z_scale, lte, nlte, level_names = load_true_multi3d_departures(dataset)
    true_dep = interpolate_true_departures(rho, z_scale, lte, nlte, cmass_grid)
    pred_plot, true_plot = prepare_plot_arrays(pred_dep, true_dep)

    fig, _ = plot_population_error_envelopes(
        pred_plot,
        true_plot,
        cmass_grid,
        level_names=level_names,
        figsize=(12, 10),
        ncols=2,
    )
    fig.suptitle(dataset["NAME"], fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"population_error_envelopes_{dataset['NAME']}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved {outpath}")

    fig_log, _ = plot_log_population_error_envelopes(
        pred_plot,
        true_plot,
        cmass_grid,
        level_names=level_names,
        figsize=(12, 10),
        ncols=2,
    )
    fig_log.suptitle(f"{dataset['NAME']} (log space)", fontsize=14)
    fig_log.tight_layout(rect=[0, 0, 1, 0.97])

    outpath_log = os.path.join(
        output_dir,
        f"log_population_error_envelopes_{dataset['NAME']}.png",
    )
    fig_log.savefig(outpath_log, dpi=200, bbox_inches="tight")
    print(f"Saved {outpath_log}")

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_log)


def main():
    args = parse_args()

    for dataset in build_plot_jobs():
        make_snapshot_plot(dataset, output_dir=args.output_dir, show=args.show)


if __name__ == "__main__":
    main()
