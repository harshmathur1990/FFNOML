"""Line-profile errors from scripts/uq_synth.jl outputs: a prediction against MULTI3D truth for the
same atmosphere and synthesis settings.

Per column, with I the profile and the core the minimum within --core-window nm of the line centre:
    core_rel     (I_core,pred - I_core,true) / I_core,true
    core_shift   Doppler shift of the core position (parabolic sub-grid vertex), km/s
    nrms         RMS over wavelength of (I_pred - I_true) / I_cont,true, I_cont = mean of the two end points
Columns where either profile is non-finite (the formal solver occasionally fails, also on MULTI3D truth
populations) are excluded and counted in n_excluded. Writes one CSV row per (case, atom, label) with
median, p95 and fractions above thresholds.

Usage:
    python scripts/uq_spectral_errors.py --truth T.h5 --pred P.h5 --atom H --case X --label prod --out uq/spectral.csv
"""
import argparse
import csv
import os

import h5py
import numpy as np

C_KMS = 299792.458


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--atom", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True, help="CSV; one row is appended per call")
    p.add_argument("--core-window", type=float, default=0.08, help="nm either side of the centre")
    return p.parse_args()


def read(path, atom):
    with h5py.File(path, "r") as f:
        return f[atom]["intensity"][...].astype(np.float64), f[atom]["wave"][...].astype(np.float64)


def core(I, w, window):
    """Core intensity and wavelength per column; I is [..., nλ]."""
    c = w[len(w) // 2]
    sel = np.flatnonzero(np.abs(w - c) <= window)
    k = sel[I[..., sel].argmin(-1)]
    # sub-grid core position: vertex of the parabola through the minimum and its neighbours
    km = np.clip(k, 1, len(w) - 2)
    x0, x1, x2 = w[km - 1], w[km], w[km + 1]
    y0, y1, y2 = (np.take_along_axis(I, (km + d)[..., None], -1)[..., 0] for d in (-1, 0, 1))
    num = (x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0)
    den = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)
    lam = np.where(np.abs(den) > 0, x1 - 0.5 * num / np.where(den == 0, 1, den), x1)
    lam = np.clip(lam, x0, x2)
    return np.take_along_axis(I, k[..., None], -1)[..., 0], lam, c


def main():
    args = parse_args()
    T, w = read(args.truth, args.atom)
    P, wp = read(args.pred, args.atom)
    if T.shape != P.shape or np.abs(w - wp).max() > 1e-9:
        raise ValueError("truth and prediction differ in shape or wavelength grid")
    good = np.isfinite(T).all(-1) & np.isfinite(P).all(-1)
    T, P = T[good], P[good]  # [ncolumns, nλ]
    cT, lT, c = core(T, w, args.core_window)
    cP, lP, _ = core(P, w, args.core_window)
    core_rel = (cP - cT) / cT
    shift = (lP - lT) / c * C_KMS
    cont = 0.5 * (T[..., 0] + T[..., -1])
    nrms = np.sqrt((((P - T) / cont[..., None]) ** 2).mean(-1))

    row = dict(case=args.case, atom=args.atom, label=args.label, columns=core_rel.size,
               n_excluded=int((~good).sum()),
               core_bias=float(np.median(core_rel)), core_mae=float(np.abs(core_rel).mean()),
               core_p95=float(np.percentile(np.abs(core_rel), 95)),
               core_frac_gt_5pct=float((np.abs(core_rel) > 0.05).mean()),
               core_frac_gt_10pct=float((np.abs(core_rel) > 0.10).mean()),
               shift_mae_kms=float(np.abs(shift).mean()), shift_p95_kms=float(np.percentile(np.abs(shift), 95)),
               nrms_median=float(np.median(nrms)), nrms_p95=float(np.percentile(nrms, 95)))
    print(row)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(row))
        if new:
            wr.writeheader()
        wr.writerow(row)


if __name__ == "__main__":
    main()
