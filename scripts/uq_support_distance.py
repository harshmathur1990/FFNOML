"""Training-support distance as an error indicator for FFNO3D populations.

For each voxel in the atlas sample tables (scripts/uq_error_atlas.py), compute the distance to the
k nearest training voxels in a standardized descriptor space, and score how well that distance
ranks the actual |error|. Descriptors are the six model inputs at the voxel plus the same inputs
at fixed height offsets above and below (column context), and z.

Training references are drawn from the TRAIN_SPLIT atmospheres (atm3d), uniformly per snapshot.
Scaling is fitted on the training references only. When the evaluated case is itself a training
snapshot, references from that snapshot within --exclude-mm (a box in x, y and z, horizontally
periodic) of the query voxel are excluded, so self- and adjacent-voxel matches do not count.

Usage:
    python scripts/uq_support_distance.py --atlas uq/atlas_prod/*_atlas.h5 uq/atlas_mc/*_atlas.h5 \
        --out uq/support_distance.csv
"""
import argparse
import csv
import os
import sys

import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from config import SIMULATIONS, TRAIN_SPLIT  # noqa: E402
from uq_error_atlas import REGIMES as ATLAS_REGIMES, spearman  # noqa: E402

FEATURES = ("logT", "vx", "vy", "vz", "logne", "logrho")
OFFSETS_MM = (-0.5, 0.5, 1.5)  # column context relative to the voxel height
REGIMES = {"all": lambda T, z: np.ones_like(T, bool), **ATLAS_REGIMES}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atlas", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-train", type=int, default=400_000, help="reference voxels per training snapshot")
    p.add_argument("--n-eval", type=int, default=300_000, help="sample voxels scored per case")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--exclude-mm", type=float, default=1.0,
                   help="exclusion box half-width for same-snapshot references on training cases")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_cube(atmos, mesh):
    """[nx, ny, nz, 6] log features, z in Mm and horizontal spacing dx in Mm for one atmosphere."""
    from pipeline import load_pred_data
    from FFNONet import _prepare_input_features
    rho, z, temp, vx, vy, vz, ne, dx, _ = load_pred_data(mesh_file=mesh, atmos_file=atmos)
    cube = _prepare_input_features(temp, vx, vy, vz, ne, rho).astype(np.float32)
    return cube, np.asarray(z) * 1e-6, float(np.ravel(dx)[0]) * 1e-6


def descriptors(cube, z_mm, ix, iy, iz):
    """Voxel features, features at OFFSETS_MM along the column (clamped to the box), and z."""
    cols = [cube[ix, iy, iz]]
    for dz in OFFSETS_MM:
        jz = np.abs(z_mm[None, :] - (z_mm[iz] + dz)[:, None]).argmin(1)
        cols.append(cube[ix, iy, jz])
    cols.append(z_mm[iz][:, None])
    return np.concatenate(cols, axis=1).astype(np.float32)


def own_k_distances(nn, q, k, ref_pos, q_pos, n_box, dx_mm, half_mm, chunk=20_000):
    """Sorted distances [N, k] to the k nearest references, skipping references within half_mm (box, periodic in
    x and y) of the query voxel. ref_pos/q_pos are [N, 3] (ix, iy, z_Mm); n_box is (nx, ny)."""
    out = np.empty((len(q), k), np.float64)
    for c0 in range(0, len(q), chunk):
        qs, qp = q[c0:c0 + chunk], q_pos[c0:c0 + chunk]
        kq = 4 * k
        while True:
            d, i = nn.kneighbors(qs, n_neighbors=min(kq, nn.n_samples_fit_))
            rp = ref_pos[i]  # [n, kq, 3]
            near = np.ones(i.shape, bool)
            for ax in (0, 1):
                dpx = np.abs(rp[..., ax] - qp[:, None, ax])
                near &= np.minimum(dpx, n_box[ax] - dpx) * dx_mm < half_mm
            near &= np.abs(rp[..., 2] - qp[:, None, 2]) < half_mm
            ok = ~near
            if ok.sum(1).min() >= k or kq >= nn.n_samples_fit_:
                break
            kq *= 4
        d = np.where(ok, d, np.inf)
        out[c0:c0 + chunk] = np.sort(d, axis=1)[:, :k]
    return out


def case_paths(case):
    """Map an atlas case name back to its atm3d/mesh via SIMULATIONS, else bifrost_data/<sim>/<snap>."""
    sim, snap = case.rsplit("_", 1)
    base = SIMULATIONS[sim]["base_path"] if sim in SIMULATIONS else os.path.join(
        os.path.dirname(SIMULATIONS["nw012023"]["base_path"]), sim)
    return os.path.join(base, snap, "atm3d"), os.path.join(base, snap, "mesh")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ref, ref_pos = [], {}
    for sim, snaps in TRAIN_SPLIT.items():
        for snap in snaps:
            base = SIMULATIONS[sim]["base_path"]
            cube, z, _ = load_cube(f"{base}/{snap}/atm3d", f"{base}/{snap}/mesh")
            nx, ny, nz, _ = cube.shape
            idx = rng.choice(nx * ny * nz, size=args.n_train, replace=False)
            ix, iy, iz = np.unravel_index(idx, (nx, ny, nz))
            ref.append(descriptors(cube, z, ix, iy, iz))
            ref_pos[f"{sim}_{snap}"] = np.stack([ix, iy, z[iz]], 1)
            print(f"reference {sim}/{snap}: {cube.shape}", flush=True)
            del cube
    ref = np.concatenate(ref)
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    ref = (ref - mu) / sd
    nn = NearestNeighbors(n_neighbors=args.k, n_jobs=32).fit(ref)
    # per training snapshot: an index over the other snapshots and one over its own references
    starts = np.cumsum([0] + [len(v) for v in ref_pos.values()])
    own = {}
    for j, name in enumerate(ref_pos):
        sel = np.zeros(len(ref), bool)
        sel[starts[j]:starts[j + 1]] = True
        own[name] = (NearestNeighbors(n_jobs=32).fit(ref[~sel]), NearestNeighbors(n_jobs=32).fit(ref[sel]))

    rows = []
    for path in args.atlas:
        with h5py.File(path, "r") as f:
            case, atom = f.attrs["case"], f.attrs["atom"]
            s = f["sample"]
            sel = np.sort(rng.choice(s["ix"].shape[0], size=min(args.n_eval, s["ix"].shape[0]), replace=False))
            ix, iy, iz = (s[k][...][sel] for k in ("ix", "iy", "iz"))
            err = np.abs(s["err"][...][sel])
            logT, zs = s["logT"][...][sel], s["z_Mm"][...][sel]
            mc = s["mc_std_log"][...][sel] if "mc_std_log" in s else None
        cube, z, dx_mm = load_cube(*case_paths(case))
        q = (descriptors(cube, z, ix, iy, iz) - mu) / sd
        if case in own:
            # merge the k nearest from other snapshots with the k nearest far-enough own references
            nn_other, nn_own = own[case]
            d_other = np.sort(nn_other.kneighbors(q, n_neighbors=args.k)[0], axis=1)
            q_pos = np.stack([ix, iy, z[iz]], 1)
            d_own = own_k_distances(nn_own, q, args.k, ref_pos[case], q_pos, cube.shape[:2], dx_mm, args.exclude_mm)
            d = np.sort(np.concatenate([d_other, d_own], 1), axis=1)[:, :args.k].mean(1)
        else:
            d = nn.kneighbors(q)[0].mean(1)
        del cube
        for reg, fn in REGIMES.items():
            m = fn(logT, zs)
            if not m.any():
                continue
            for lev in range(err.shape[1]):
                row = dict(case=case, atom=atom, regime=reg, level=lev + 1, n=int(m.sum()),
                           mae=float(err[m, lev].mean()), median_dist=float(np.median(d[m])),
                           spearman_dist=spearman(err[m, lev], d[m]))
                if mc is not None:
                    row["spearman_mc"] = spearman(err[m, lev], mc[m, lev])
                rows.append(row)
        print(f"{case}/{atom}: median distance {np.median(d):.2f}, chromosphere "
              f"{np.median(d[REGIMES['chromosphere'](logT, zs)]):.2f}", flush=True)

    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
