import argparse
import os
import subprocess
import tempfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from helita.sim.multi3d import Multi3dAtmos, Multi3dOut

from config import ACTIVE_ATOMS, MODEL, MODEL_DIR, MULTI3D_PRED_DATA
from pipeline import compute_dx_dy
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


def active_atom_names_tag():
    return "_".join(ACTIVE_ATOMS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser.parse_args()


def plot_population_error_envelopes(
    pred,
    true,
    z_scale,
    level_names=None,
    figsize=(10, 10),
    ncols=2,
    ylabel=r"(b$_{NN}$ - b) / b",
):
    """
    Plot median + 68% + 95% relative-error envelopes vs z_scale.
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

    z_axis = np.asarray(z_scale)
    if z_axis.ndim != 1:
        raise ValueError(f"Expected 1D z_scale for plotting, got shape {z_axis.shape}")

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

        ax.fill_between(z_axis, p025[i], p975[i], color="#e6b98c", alpha=0.8)
        ax.fill_between(z_axis, p16[i], p84[i], color="#6f8fa6", alpha=0.9)
        ax.plot(z_axis, p50[i], color="#1f77b4", lw=1.5)

        if level_names is not None:
            ax.set_title(level_names[i], fontsize=12)

        ax.axhline(0.0, color="k", lw=0.5, alpha=0.5)

    for ax in axes[-1]:
        ax.set_xlabel(r"z [Mm]")

    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)

    for j in range(nlevels, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    finite_rel_err = rel_err[np.isfinite(rel_err)]
    if finite_rel_err.size:
        ylim = max(np.max(np.abs(finite_rel_err)), 1e-12)
        for ax in axes.flat:
            if ax in fig.axes:
                ax.set_ylim(-1, 1)
                # ax.yaxis.set_minor_locator(MultipleLocator(0.1))
                ax.set_xlim(-1, 5)
                ax.set_yticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1], [-1, "", -0.6, "", -0.2, "", 0.2, "", 0.6, "", 1])
    plt.tight_layout()
    return fig, axes


def plot_log_population_error_envelopes(
    pred,
    true,
    z_scale,
    level_names=None,
    figsize=(10, 10),
    ncols=2,
):
    """
    Plot median + 68% + 95% envelopes for log10(b_NN / b) vs z_scale.
    """

    assert pred.shape == true.shape
    nlevels, ndepth, nx, ny = pred.shape

    eps = 1e-30
    log_ratio = np.log10((pred + eps) / (true + eps))
    log_ratio[true <= 0] = np.nan
    log_ratio = log_ratio.reshape(nlevels, ndepth, -1)

    p50 = np.nanmedian(log_ratio, axis=-1)
    p16 = np.nanpercentile(log_ratio, 16, axis=-1)
    p84 = np.nanpercentile(log_ratio, 84, axis=-1)
    p025 = np.nanpercentile(log_ratio, 2.5, axis=-1)
    p975 = np.nanpercentile(log_ratio, 97.5, axis=-1)

    z_axis = np.asarray(z_scale)
    if z_axis.ndim != 1:
        raise ValueError(f"Expected 1D z_scale for plotting, got shape {z_axis.shape}")

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

        ax.fill_between(z_axis, p025[i], p975[i], color="#d9c2f0", alpha=0.8)
        ax.fill_between(z_axis, p16[i], p84[i], color="#7b8fc7", alpha=0.9)
        ax.plot(z_axis, p50[i], color="#243b6b", lw=1.5)

        if level_names is not None:
            ax.set_title(level_names[i], fontsize=12)

        ax.axhline(0.0, color="k", lw=0.5, alpha=0.5)

    for ax in axes[-1]:
        ax.set_xlabel(r"z [Mm]")

    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10}(b_{NN} / b)$")

    for j in range(nlevels, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    plt.tight_layout()
    return fig, axes


def plot_departure_coefficient_scatter_by_height(
    pred,
    true,
    z_scale,
    level_names=None,
    figsize=(16, 10),
    ncols=3,
    max_points_per_level=250_000,
):
    """Compare predicted and true departure coefficients at every level.

    Both axes are logarithmic and points are colored by physical height,
    matching the height-coded comparison in the supplied reference plot.
    Large cubes are sampled uniformly in flattened-array order to keep the
    figure responsive; pass ``None`` for ``max_points_per_level`` to plot all
    valid cells.
    """

    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape or pred.ndim != 4:
        raise ValueError(
            "Expected matching [nlevels, ndepth, nx, ny] arrays, got "
            f"{pred.shape} and {true.shape}"
        )

    nlevels, ndepth, nx, ny = pred.shape
    z_axis = np.asarray(z_scale, dtype=np.float64)
    if z_axis.shape != (ndepth,):
        raise ValueError(
            f"Expected z_scale with shape ({ndepth},), got {z_axis.shape}"
        )
    if not np.all(np.isfinite(z_axis)):
        raise ValueError("z_scale contains non-finite values")
    if level_names is not None and len(level_names) != nlevels:
        raise ValueError(
            f"Expected {nlevels} level names, got {len(level_names)}"
        )
    if max_points_per_level is not None and max_points_per_level <= 0:
        raise ValueError("max_points_per_level must be positive or None")

    nrows = int(np.ceil(nlevels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True
    )
    height_km = np.repeat(z_axis * 1e3, nx * ny)

    for ilevel in range(nlevels):
        ax = axes[ilevel // ncols, ilevel % ncols]
        actual = true[ilevel].ravel()
        predicted = pred[ilevel].ravel()
        valid = np.flatnonzero(
            np.isfinite(actual)
            & np.isfinite(predicted)
            & (actual > 0)
            & (predicted > 0)
        )

        if max_points_per_level is not None and valid.size > max_points_per_level:
            sample = np.linspace(
                0, valid.size - 1, max_points_per_level, dtype=np.int64
            )
            valid = valid[sample]

        if valid.size:
            points = ax.scatter(
                actual[valid],
                predicted[valid],
                c=height_km[valid],
                cmap="viridis",
                vmin=np.min(height_km),
                vmax=np.max(height_km),
                s=2,
                alpha=0.18,
                linewidths=0,
                rasterized=True,
            )
            lower = min(actual[valid].min(), predicted[valid].min())
            upper = max(actual[valid].max(), predicted[valid].max())
            ax.plot(
                [lower, upper],
                [lower, upper],
                "--",
                color="#b07a3c",
                lw=1.2,
                label=r"$y=x$ (perfect fit)",
            )
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            colorbar = fig.colorbar(points, ax=ax, pad=0.02)
            colorbar.set_label(r"z [km]")
            ax.legend(loc="upper left", fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "No positive finite values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        title = level_names[ilevel] if level_names is not None else f"Level {ilevel + 1}"
        ax.set_title(f"{title} (color-coded by height)")
        ax.set_xlabel("Actual departure coefficient")
        ax.set_ylabel("Predicted departure coefficient")
        ax.grid(True, which="both", alpha=0.2)

    for j in range(nlevels, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    return fig, axes


def plot_departure_coefficient_error_assessment(
    pred,
    true,
    z_scale,
    level_names=None,
    figsize=None,
    residual_range=(-50, 50),
    residual_bins=100,
    coefficient_bins=80,
):
    """Plot height-binned residual and coefficient distributions per level.

    The top row shows relative errors in percent and the bottom row shows the
    corresponding true departure-coefficient distributions on a logarithmic
    x axis. Cells are grouped into the physical-height ranges requested for
    the comparison: below 750 km, 750--1200 km, 1200--2500 km, 2500--5000 km,
    and 5000 km and above.
    """

    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape or pred.ndim != 4:
        raise ValueError(
            "Expected matching [nlevels, ndepth, nx, ny] arrays, got "
            f"{pred.shape} and {true.shape}"
        )

    nlevels, ndepth, _, _ = pred.shape
    z_axis = np.asarray(z_scale, dtype=np.float64)
    if z_axis.shape != (ndepth,):
        raise ValueError(
            f"Expected z_scale with shape ({ndepth},), got {z_axis.shape}"
        )
    if not np.all(np.isfinite(z_axis)):
        raise ValueError("z_scale contains non-finite values")
    if level_names is not None and len(level_names) != nlevels:
        raise ValueError(
            f"Expected {nlevels} level names, got {len(level_names)}"
        )
    z_km = z_axis * 1e3
    regions = (
        (z_km < 750, r"$z < 750$ km", "#482878"),
        ((z_km >= 750) & (z_km < 1200), r"$750 \leq z < 1200$ km", "#3e7c98"),
        ((z_km >= 1200) & (z_km < 2500), r"$1200 \leq z < 2500$ km", "#55a868"),
        ((z_km >= 2500) & (z_km < 5000), r"$2500 \leq z < 5000$ km", "#dd9c3c"),
        (z_km >= 5000, r"$z \geq 5000$ km", "#c44e52"),
    )
    if figsize is None:
        figsize = (4.0 * nlevels, 7.5)

    fig, axes = plt.subplots(
        2,
        nlevels,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )

    for ilevel in range(nlevels):
        residual_ax = axes[0, ilevel]
        coefficient_ax = axes[1, ilevel]
        title = level_names[ilevel] if level_names is not None else f"Level {ilevel + 1}"

        positive_actual = true[ilevel][
            np.isfinite(true[ilevel]) & (true[ilevel] > 0)
        ]
        log_bins = None
        if positive_actual.size:
            log_min = np.log10(positive_actual.min())
            log_max = np.log10(positive_actual.max())
            if np.isclose(log_min, log_max):
                log_min -= 0.5
                log_max += 0.5
            log_bins = np.logspace(log_min, log_max, coefficient_bins + 1)

        for height_mask, label, color in regions:
            actual = true[ilevel, height_mask, :, :].ravel()
            predicted = pred[ilevel, height_mask, :, :].ravel()
            valid = (
                np.isfinite(actual)
                & np.isfinite(predicted)
                & (actual > 0)
            )
            residual = 100.0 * (predicted[valid] - actual[valid]) / actual[valid]
            residual = residual[np.isfinite(residual)]
            if residual.size:
                residual_ax.hist(
                    residual,
                    bins=residual_bins,
                    range=residual_range,
                    density=True,
                    histtype="stepfilled",
                    color=color,
                    alpha=0.45,
                    label=label,
                )

            actual = actual[np.isfinite(actual) & (actual > 0)]
            if actual.size and log_bins is not None:
                coefficient_ax.hist(
                    actual,
                    bins=log_bins,
                    histtype="step",
                    color=color,
                    linewidth=1.4,
                    hatch="//",
                    label=f"Actual: {label}",
                )

        residual_ax.axvline(0, color="0.25", linestyle="--", lw=1)
        residual_ax.set_xlim(*residual_range)
        residual_ax.set_title(f"{title} residuals")
        residual_ax.set_xlabel("Relative error (%)")
        residual_ax.grid(True, alpha=0.2)

        coefficient_ax.set_xscale("log")
        coefficient_ax.set_title(f"{title} coefficient distribution")
        coefficient_ax.set_xlabel("Departure coefficient")
        coefficient_ax.grid(True, which="both", alpha=0.2)

    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Count")
    axes[0, -1].legend(loc="upper right", fontsize=8)
    axes[1, -1].legend(loc="upper right", fontsize=8)

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
    return os.path.join(
        MODEL_DIR,
        f"output_3D_sim_s5_{dataset['NAME']}_{MODEL}_{active_atom_names_tag()}.hdf5",
    )


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
        populations = f["nlte_populations"][...]
        z_scale = f["z_scale"][...]

    return populations, np.asarray(z_scale)


def compute_departure_coefficients(lte, nlte):
    eps = 1e-30
    true_dep = (nlte + eps) / (lte + eps)
    return true_dep.astype(np.float32, copy=False)


def extract_plot_z_axis(z_scale, *, scale_to_mm=False):
    z_axis = np.asarray(z_scale, dtype=np.float32)

    if z_axis.ndim == 3:
        ref = z_axis[:, 0, 0][:, None, None]
        if not np.allclose(z_axis, ref):
            raise ValueError(
                "Expected z_scale to be identical across x/y columns for plotting."
            )
        z_axis = z_axis[:, 0, 0]
    elif z_axis.ndim != 1:
        raise ValueError(f"Expected 1D or 3D z_scale, got shape {z_axis.shape}")

    if scale_to_mm:
        z_axis = z_axis / 1e6

    return z_axis


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


def prepare_forward_lte_populations(muspel_lte, simulation_lte_shape):
    """Put Muspel LTE populations from ``Forward.jl`` in Multi3D layout.

    ``lte_pops_saha`` in ``Forward.jl`` returns ``[nz, nx, ny, nlevels]``,
    whereas helita reads the simulation populations as
    ``[nx, ny, nz, nlevels]``.
    """

    muspel_lte = np.asarray(muspel_lte)
    if muspel_lte.ndim != 4:
        raise ValueError(
            "Expected 4D Muspel LTE populations [nz, nx, ny, nlevels], "
            f"got shape {muspel_lte.shape}"
        )

    forward_shape = (
        simulation_lte_shape[2],
        simulation_lte_shape[0],
        simulation_lte_shape[1],
        simulation_lte_shape[3],
    )
    if muspel_lte.shape == forward_shape:
        muspel_lte = np.transpose(muspel_lte, (1, 2, 0, 3))
    elif muspel_lte.shape != simulation_lte_shape:
        raise ValueError(
            "Muspel/simulation LTE shape mismatch: expected Forward.jl layout "
            f"{forward_shape} (or Multi3D layout {simulation_lte_shape}), got "
            f"{muspel_lte.shape}"
        )

    return muspel_lte


def compute_muspel_lte(dataset, active_atoms=None):
    """Run the Julia/Muspel LTE calculation and return one combined array."""

    active_atoms = active_atoms or ACTIVE_ATOMS
    julia_script = os.path.join(os.path.dirname(__file__), "compute_muspel_lte.jl")
    output_file = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            output_file = tmp.name

        command = [
            os.environ.get("JULIA", "julia"),
            "--startup-file=no",
            julia_script,
            dataset["MESH"],
            dataset["MULTI3D_ATMOS"],
            output_file,
            *active_atoms,
        ]
        print(f"Computing Muspel LTE populations for {dataset['NAME']}...")
        subprocess.run(command, check=True)

        blocks = []
        with h5py.File(output_file, "r") as file:
            for atom in active_atoms:
                if atom not in file:
                    raise KeyError(f"Julia LTE output is missing atom {atom}")
                group = file[atom]
                shape = tuple(int(value) for value in group["shape"][...])
                values = group["values"][...]
                blocks.append(values.reshape(shape, order="F"))

        return np.concatenate(blocks, axis=-1)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not launch Julia. Install Julia or set JULIA to its executable path."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Muspel LTE calculation failed for {dataset['NAME']}"
        ) from exc
    finally:
        if output_file is not None and os.path.exists(output_file):
            os.remove(output_file)


def make_lte_population_comparison_plot(
    dataset,
    output_dir,
    show=False,
):
    """Compare simulation LTE populations with those computed by Muspel.

    Parameters
    ----------
    dataset : dict
        A plot job produced by :func:`build_plot_jobs`.
    output_dir : path-like
        Directory in which to save the PDF.

    The plotted quantity is ``(n_LTE,Muspel - n_LTE,simulation) /
    n_LTE,simulation``. Percentile envelopes and panel layout match
    :func:`make_snapshot_plot`.
    """

    muspel_lte = compute_muspel_lte(dataset)
    _, true_z_scale, simulation_lte, _, level_names = (
        load_true_multi3d_departures(dataset)
    )
    muspel_lte = prepare_forward_lte_populations(
        muspel_lte, simulation_lte.shape
    )
    z_axis = extract_plot_z_axis(true_z_scale, scale_to_mm=True)
    muspel_plot, simulation_plot = prepare_plot_arrays(
        muspel_lte, simulation_lte
    )

    fig, axes = plot_population_error_envelopes(
        muspel_plot,
        simulation_plot,
        z_axis,
        level_names=level_names,
        figsize=(12, 10),
        ncols=2,
        ylabel=r"$(n_{\mathrm{LTE,Muspel}}-n_{\mathrm{LTE,sim}}) / "
        r"n_{\mathrm{LTE,sim}}$",
    )
    fig.tight_layout(rect=[0, 0, 1, 1])

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(
        output_dir,
        f"lte_population_comparison_{dataset['NAME']}_{active_atom_names_tag()}.pdf",
    )
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved {outpath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return outpath, axes


def make_snapshot_plot(dataset, output_dir, show=False):
    pred_nlte, pred_z_scale = load_prediction_file(dataset["PRED_FILE"])
    _, true_z_scale, lte, nlte, level_names = load_true_multi3d_departures(dataset)
    pred_dep = compute_departure_coefficients(lte, pred_nlte)
    true_dep = compute_departure_coefficients(lte, nlte)

    pred_z_axis = extract_plot_z_axis(pred_z_scale)
    true_z_axis = extract_plot_z_axis(true_z_scale, scale_to_mm=True)

    if pred_dep.shape != true_dep.shape:
        raise ValueError(
            f"Prediction/true shape mismatch: {pred_dep.shape} vs {true_dep.shape}"
        )
    if pred_z_axis.shape != true_z_axis.shape or not np.allclose(pred_z_axis, true_z_axis):
        raise ValueError(
            f"Prediction/true z_scale mismatch: {pred_z_axis.shape} vs {true_z_axis.shape}"
        )

    pred_plot, true_plot = prepare_plot_arrays(pred_dep, true_dep)

    fig, _ = plot_population_error_envelopes(
        pred_plot,
        true_plot,
        pred_z_axis,
        level_names=level_names,
        figsize=(12, 10),
        ncols=2,
    )
    # fig.suptitle(dataset["NAME"], fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 1])

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(
        output_dir,
        f"population_error_envelopes_{dataset['NAME']}_{active_atom_names_tag()}.pdf",
    )
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved {outpath}")

    scatter_fig, _ = plot_departure_coefficient_scatter_by_height(
        pred_plot,
        true_plot,
        pred_z_axis,
        level_names=level_names,
    )
    scatter_outpath = os.path.join(
        output_dir,
        f"departure_coefficient_scatter_{dataset['NAME']}_{active_atom_names_tag()}.pdf",
    )
    scatter_fig.savefig(scatter_outpath, dpi=200, bbox_inches="tight")
    print(f"Saved {scatter_outpath}")

    assessment_fig, _ = plot_departure_coefficient_error_assessment(
        pred_plot,
        true_plot,
        pred_z_axis,
        level_names=level_names,
    )
    assessment_outpath = os.path.join(
        output_dir,
        f"departure_coefficient_error_assessment_{dataset['NAME']}_{active_atom_names_tag()}.pdf",
    )
    assessment_fig.savefig(assessment_outpath, dpi=200, bbox_inches="tight")
    print(f"Saved {assessment_outpath}")

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(scatter_fig)
        plt.close(assessment_fig)


def main():
    args = parse_args()

    for dataset in build_plot_jobs():
        make_snapshot_plot(dataset, output_dir=MODEL_DIR, show=args.show)
        make_lte_population_comparison_plot(
            dataset, output_dir=MODEL_DIR, show=args.show
        )


if __name__ == "__main__":
    main()
