"""Check that prediction, input features and MULTI3D truth are aligned beyond matching shapes.

Per case and atom:
  orientation  max |log10 T_truth - logT_features| for the identity mapping and for x/y flips and a
               transpose; only the identity should be ~0
  height       max |z_truth - z_mesh| in Mm; horizontal spacing of truth geometry vs mesh dx
  units        median log10(sum_i n_truth,i / ntot) and the same for the prediction (both ~0 when
               populations are m^-3 and complete); log10(sum n / n_H) - (abnd - 12) (~0 for the
               element abundance); median log10 ne_truth[m^-3] - logne_features (~0)
  levels       of all permutations of predicted vs truth levels, the identity must minimize the mean
               |pred - truth|; reports the margin (dex per level) over the closest alternative and the
               level energies (eV)

Usage:
    python scripts/uq_verify_alignment.py --case nw012023_915 --atom H --truth .../915/H \
        --uq uq/mc/nw012023_915_H.h5 --atmos .../915/atm3d --mesh .../915/mesh --out uq/alignment.csv
"""
import argparse
import csv
import itertools
import os
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "ffno"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True)
    p.add_argument("--atom", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--uq", required=True)
    p.add_argument("--atmos", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", required=True, help="CSV; one row is appended per call")
    p.add_argument("--sample", type=int, default=500_000)
    return p.parse_args()


def main():
    args = parse_args()
    from helita.sim.multi3d import Multi3dOut
    from pipeline import load_pred_data
    from FFNONet import _prepare_input_features

    m3d = Multi3dOut(directory=args.truth)
    m3d.readall()
    n_truth = np.asarray(m3d.atom.n, np.float64) * 1e6  # cm^-3 -> m^-3
    ntot = np.asarray(m3d.atom.ntot, np.float64) * 1e6
    nh_tot = np.asarray(m3d.atmos.nh, np.float64).sum(-1) * 1e6
    T_truth = np.asarray(m3d.atmos.tg, np.float64)
    ne_truth = np.asarray(m3d.atmos.ne, np.float64) * 1e6  # cm^-3 -> m^-3
    zt = np.asarray(m3d.geometry.z, np.float64) * 1e-8  # cm -> Mm
    xt = np.asarray(m3d.geometry.x, np.float64) * 1e-8

    rho, zm, temp, vx, vy, vz, ne, dx, _ = load_pred_data(mesh_file=args.mesh, atmos_file=args.atmos)
    feats = _prepare_input_features(temp, vx, vy, vz, ne, rho)  # [nx, ny, nz, 6]
    logT, logne = feats[..., 0].astype(np.float64), feats[..., 4].astype(np.float64)
    zm = np.asarray(zm, np.float64) * 1e-6  # m -> Mm

    with h5py.File(args.uq, "r") as f:
        pred = f["det_log"][...].astype(np.float64)

    row = dict(case=args.case, atom=args.atom)
    lt = np.log10(T_truth)
    variants = {"identity": logT, "flip_x": logT[::-1], "flip_y": logT[:, ::-1]}
    if logT.shape[0] == logT.shape[1]:
        variants["transpose"] = logT.transpose(1, 0, 2)
    for name, v in variants.items():
        row[f"dlogT_{name}"] = float(np.abs(lt - v).max())
    row["dz_max_Mm"] = float(np.abs(zt - zm).max())
    row["dx_truth_Mm"] = float(np.diff(xt).mean())
    row["dx_mesh_Mm"] = float(np.ravel(dx)[0]) * 1e-6
    row["logne_offset"] = float(np.median(np.log10(ne_truth) - logne))

    rng = np.random.default_rng(0)
    nlev = n_truth.shape[-1]
    idx = rng.choice(lt.size, size=min(args.sample, lt.size), replace=False)
    nt, npd = n_truth.reshape(-1, nlev)[idx], pred.reshape(-1, nlev)[idx]
    tn = ntot.reshape(-1)[idx]
    nhs = nh_tot.reshape(-1)[idx]
    row["log_sum_truth_over_ntot"] = float(np.median(np.log10(nt.sum(1) / tn)))
    row["log_sum_pred_over_ntot"] = float(np.median(np.log10((10.0 ** npd).sum(1) / tn)))
    row["abundance_check"] = float(np.median(np.log10(nt.sum(1) / nhs)) - (float(m3d.atom.abnd) - 12.0))
    # level ordering: of all permutations of the predicted levels, the identity must give the lowest
    # mean |pred - truth| (per-level nearest matching is ambiguous: photospheric n4/n5 differ by ~0.06 dex
    # and coronal n3/n4 are close)
    ltr = np.log10(np.maximum(nt, 1e-30))
    cost = np.array([[np.abs(npd[:, i] - ltr[:, j]).mean() for j in range(nlev)] for i in range(nlev)])
    perms = list(itertools.permutations(range(nlev)))
    total = np.array([cost[np.arange(nlev), list(q)].sum() for q in perms])
    best = int(total.argmin())
    alt = min((t, q) for t, q in zip(total, perms) if q != tuple(range(nlev)))
    row["level_perm_best"] = " ".join(str(j + 1) for j in perms[best])
    row["levels_ok"] = perms[best] == tuple(range(nlev))
    row["level_margin_dex"] = float((alt[0] - total[perms.index(tuple(range(nlev)))]) / nlev)
    row["level_closest_alt"] = " ".join(str(j + 1) for j in alt[1])
    row["level_ev"] = " ".join(f"{e:.3f}" for e in np.asarray(m3d.atom.ev))
    row["aligned"] = bool(row["dlogT_identity"] < 1e-3 and row["dz_max_Mm"] < 1e-3 and row["levels_ok"]
                          and abs(row["dx_truth_Mm"] - row["dx_mesh_Mm"]) < 1e-4
                          and abs(row["log_sum_truth_over_ntot"]) < 0.01
                          and abs(row["abundance_check"]) < 0.01 and abs(row["logne_offset"]) < 0.01)
    print(row)

    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
