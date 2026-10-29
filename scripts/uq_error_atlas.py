"""Actual-error atlas for FFNO3D NLTE populations against MULTI3D truth, optionally scoring an
uncertainty estimate from scripts/uq_predict.py.

Error convention: e = log10(n_pred) - log10(n_truth) in dex, per level, on the native grid.
Prediction and truth are both [nx, ny, nz, nlev] in the out_pop orientation (no flip).

Writes to --out-dir:
    <case>_<atom>_summary.csv    per-level bias, MAE, quantiles of |e|, exceedance fractions,
                                 plus uncertainty-vs-error scores when --uq is given
    <case>_<atom>_height.csv     the same statistics in height bins
    <case>_<atom>_atlas.h5       median |e| and counts on (logT, logne) and (z, logT) grids,
                                 and a random voxel sample of descriptors, errors and uncertainties

Usage:
    python scripts/uq_error_atlas.py --case en024048_hion_385 --atom H \
        --truth ../bifrost_data/en024048_hion/385/H --uq uq/en024048_hion_385_H.h5 --out-dir uq/atlas
"""
import argparse
import csv
import os
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from config import MODEL, MODEL_DIR  # noqa: E402

FEATURES = ("logT", "vx", "vy", "vz", "logne", "logrho")
THRESHOLDS = (0.05, 0.1, 0.2, 0.3)
LOGT_EDGES = np.arange(3.0, 7.01, 0.1)
LOGNE_EDGES = np.arange(14.0, 22.01, 0.2)
Z_EDGES_MM = np.arange(-1.0, 14.51, 0.25)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True)
    p.add_argument("--atom", required=True, choices=["H", "CA"])
    p.add_argument("--truth", required=True, help="MULTI3D output directory (contains out_pop)")
    p.add_argument("--pred", help="output_3D_sim population file (linear m^-3); default: production file")
    p.add_argument("--uq", help="uq_predict.py file; its det_log replaces --pred, and its spreads are scored")
    p.add_argument("--inputs", help="solving set with log features; default: MODEL_DIR/3D_sim_predict_<case>.hdf5")
    p.add_argument("--atmos", help="atm3d to build features from when there is no solving set")
    p.add_argument("--mesh", help="mesh for --atmos")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample", type=int, default=2_000_000, help="voxels in the random sample table")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_truth(directory):
    from helita.sim.multi3d import Multi3dOut
    m3d = Multi3dOut(directory=directory)
    m3d.readall()
    log_truth = np.log10(np.maximum(np.asarray(m3d.atom.n, dtype=np.float32) * 1e6, 1e-30))
    z_mm = np.asarray(m3d.geometry.z, dtype=np.float64) * 1e-8  # cm -> Mm
    return log_truth, z_mm


def load_prediction(args):
    """Return (log_pred, spreads dict, source path)."""
    if args.uq:
        with h5py.File(args.uq, "r") as f:
            spreads = {k: f[k][...] for k in ("mc_std_log", "tile_std_log") if k in f}
            if "mc_mean_log" in f:
                spreads["mc_shift_log"] = np.abs(f["mc_mean_log"][...] - f["det_log"][...])
            return f["det_log"][...], spreads, args.uq
    path = args.pred or os.path.join(MODEL_DIR, f"output_3D_sim_s5_{args.case}_{MODEL}_{args.atom}.hdf5")
    with h5py.File(path, "r") as f:
        pops = f["nlte_populations"][...]
    return np.log10(np.maximum(pops, 1e-30)), {}, path


def load_features(args, shape):
    """Return dict of [nx, ny, nz] float32 descriptors, or {} if the solving set is missing."""
    path = args.inputs or os.path.join(MODEL_DIR, f"3D_sim_predict_{args.case}.hdf5")
    if os.path.exists(path):
        with h5py.File(path, "r") as f:
            X = f["inputs"][0]  # [Cin, D, nx, ny]
        feats = {name: np.ascontiguousarray(np.transpose(X[i], (1, 2, 0))) for i, name in enumerate(FEATURES)}
    elif args.atmos:
        from pipeline import load_pred_data
        from FFNONet import _prepare_input_features
        rho, _, temp, vx, vy, vz, ne, _, _ = load_pred_data(mesh_file=args.mesh, atmos_file=args.atmos)
        X = _prepare_input_features(temp, vx, vy, vz, ne, rho).astype(np.float32)  # [nx, ny, nz, Cin]
        feats = {name: np.ascontiguousarray(X[..., i]) for i, name in enumerate(FEATURES)}
    else:
        print(f"no solving set at {path} and no --atmos; regime maps skipped")
        return {}
    if feats["logT"].shape != shape[:3]:
        raise ValueError(f"solving set {feats['logT'].shape} does not match populations {shape[:3]}")
    return feats


def level_stats(e):
    a = np.abs(e)
    row = dict(n=a.size, bias=float(e.mean()), mae=float(a.mean()), rmse=float(np.sqrt((e * e).mean())))
    for q in (50, 90, 95, 99):
        row[f"p{q}"] = float(np.percentile(a, q))
    for t in THRESHOLDS:
        row[f"frac_gt_{t}"] = float((a > t).mean())
    return row


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def uq_scores(err, u, rng, n_rank=2_000_000):
    """How well spread u ranks |e|: Spearman, capture of the top-5% errors by the top-5% u,
    error of the most-confident 80%, and the ratio RMS(e)/RMS(u) (1 = calibrated Gaussian scale)."""
    a, u = np.abs(err).ravel(), u.ravel()
    idx = rng.choice(a.size, size=min(n_rank, a.size), replace=False)
    top_e = a > np.percentile(a, 95)
    top_u = u > np.percentile(u, 95)
    keep = u <= np.percentile(u, 80)
    return dict(
        spearman=spearman(a[idx], u[idx]),
        top5_capture=float((top_e & top_u).sum() / max(top_e.sum(), 1)),
        mae_conf80=float(a[keep].mean()),
        mae_all=float(a.mean()),
        rms_ratio=float(np.sqrt((a * a).mean() / max((u * u).mean(), 1e-30))),
        median_u=float(np.median(u)),
    )


def binned_median(x, y, vals, xedges, yedges):
    """Median of vals in (x, y) bins; also counts. x, y, vals are 1-D."""
    ix = np.digitize(x, xedges) - 1
    iy = np.digitize(y, yedges) - 1
    ok = (ix >= 0) & (ix < len(xedges) - 1) & (iy >= 0) & (iy < len(yedges) - 1)
    flat = ix[ok] * (len(yedges) - 1) + iy[ok]
    v = vals[ok]
    order = np.argsort(flat, kind="stable")
    flat, v = flat[order], v[order]
    med = np.full((len(xedges) - 1) * (len(yedges) - 1), np.nan, np.float32)
    cnt = np.bincount(flat, minlength=med.size)
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    for b in np.nonzero(cnt)[0]:
        med[b] = np.median(v[starts[b]:starts[b] + cnt[b]])
    shape = (len(xedges) - 1, len(yedges) - 1)
    return med.reshape(shape), cnt.reshape(shape)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tag = f"{args.case}_{args.atom}"

    log_truth, z_mm = load_truth(args.truth)
    log_pred, spreads, src = load_prediction(args)
    if log_pred.shape != log_truth.shape:
        raise ValueError(f"prediction {log_pred.shape} vs truth {log_truth.shape}")
    err = log_pred - log_truth
    nx, ny, nz, nlev = err.shape
    print(f"{tag}: {err.shape}, prediction {src}")
    if not np.isfinite(err).all():
        print(f"warning: {np.count_nonzero(~np.isfinite(err))} non-finite errors")

    base = dict(case=args.case, atom=args.atom)
    rows = []
    for lev in range(nlev):
        row = dict(base, level=lev + 1, **level_stats(err[..., lev]))
        for name, u in spreads.items():
            row.update({f"{name}:{k}": v for k, v in uq_scores(err[..., lev], u[..., lev], rng).items()})
        rows.append(row)
    write_csv(os.path.join(args.out_dir, f"{tag}_summary.csv"), rows)

    hrows = []
    zb = np.digitize(z_mm, Z_EDGES_MM) - 1
    for b in np.unique(zb):
        if b < 0 or b >= len(Z_EDGES_MM) - 1:
            continue
        sel = zb == b
        for lev in range(nlev):
            row = dict(base, z_lo_Mm=Z_EDGES_MM[b], z_hi_Mm=Z_EDGES_MM[b + 1], level=lev + 1,
                       **level_stats(err[:, :, sel, lev]))
            for name, u in spreads.items():
                row[f"{name}:median"] = float(np.median(u[:, :, sel, lev]))
            hrows.append(row)
    write_csv(os.path.join(args.out_dir, f"{tag}_height.csv"), hrows)

    feats = load_features(args, err.shape)
    with h5py.File(os.path.join(args.out_dir, f"{tag}_atlas.h5"), "w") as f:
        f.attrs.update(dict(case=args.case, atom=args.atom, prediction=os.path.abspath(src),
                            truth=os.path.abspath(args.truth), error="log10(pred) - log10(truth) [dex]"))
        f["z_Mm"] = z_mm
        f["logT_edges"], f["logne_edges"], f["z_edges_Mm"] = LOGT_EDGES, LOGNE_EDGES, Z_EDGES_MM
        if feats:
            logT, logne = feats["logT"].ravel(), feats["logne"].ravel()
            zfull = np.broadcast_to(z_mm[None, None, :], (nx, ny, nz)).ravel()
            for lev in range(nlev):
                a = np.abs(err[..., lev]).ravel()
                g = f.create_group(f"level{lev + 1}")
                g["mae_logT_logne"], g["count_logT_logne"] = binned_median(logT, logne, a, LOGT_EDGES, LOGNE_EDGES)
                g["mae_z_logT"], g["count_z_logT"] = binned_median(zfull, logT, a, Z_EDGES_MM, LOGT_EDGES)
                for name, u in spreads.items():
                    g[f"{name}_logT_logne"], _ = binned_median(logT, logne, u[..., lev].ravel(),
                                                               LOGT_EDGES, LOGNE_EDGES)

            idx = np.sort(rng.choice(nx * ny * nz, size=min(args.sample, nx * ny * nz), replace=False))
            ix, iy, iz = np.unravel_index(idx, (nx, ny, nz))
            s = f.create_group("sample")
            s["ix"], s["iy"], s["iz"], s["z_Mm"] = ix, iy, iz, z_mm[iz]
            for name, v in feats.items():
                s[name] = v.ravel()[idx]
            # vertical and horizontal log T gradients (per Mm and per pixel) as context descriptors
            s["dlogT_dz"] = np.gradient(feats["logT"], z_mm, axis=2).ravel()[idx]
            gx, gy = np.gradient(feats["logT"], axis=(0, 1))
            s["gradh_logT_px"] = np.hypot(gx, gy).ravel()[idx]
            s["err"] = err.reshape(-1, nlev)[idx]
            for name, u in spreads.items():
                s[name] = u.reshape(-1, nlev)[idx]
    print(f"wrote {os.path.join(args.out_dir, tag + '_atlas.h5')}")


if __name__ == "__main__":
    main()
