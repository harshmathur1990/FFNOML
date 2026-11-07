"""Population errors of FFNO checkpoints against MULTI3D, snapshot by snapshot, grouped by what the checkpoint saw.

Answers whether the next gains come from the model or from the data: if the training snapshots are fitted well and the
validation, same-simulation and unseen-simulation snapshots are not, the network generalizes poorly (data); if the
training snapshots themselves are fitted poorly where the lines form, the model or the loss limits it.

For every snapshot and atom it builds the solving set if needed (pipeline.py --buildforpredict, raw inputs, shared by all
checkpoints), predicts the whole cube with the batched backend on the visible GPUs, and compares with MULTI3D's out_pop
in bins of log10 tau500: |delta log10 n| per level and |delta log10 (n_u / n_l)| of the line's two levels (its source
function). Predictions are cached, so a rerun only scores.

    source ~/loadnvidia.sh
    python scripts/population_error_diagnostic.py --run-dir RUN [--model-dir DIR] [--snapshot role=sim/snap ...]

RUN gets solving sets under training_FFNO3D_zscale_expand_lognlte/ (those already in --model-dir are linked, not
rebuilt), predictions under predictions/<label>/ and population_errors_<label>.json with a printed summary.
With FFNOML_PACKED_DATA set (packed_data.py), the snapshots are read from that file instead of bifrost_data/.
"""
import argparse
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ffno_batched_backend import BatchedFFNOBackend, linear_populations  # noqa: E402
from ffno_model_factory import create_ffno3d  # noqa: E402
import packed_data  # noqa: E402

# a packed_data.py file holding the snapshots, read instead of bifrost_data/ (as by pipeline.py --buildforpredict)
PACKED_DATA = os.environ.get("FFNOML_PACKED_DATA") or None


def packed_key(snapshot_dir):
    """.../bifrost_data/<sim>/<snap> -> (sim, snap)"""
    return os.path.basename(os.path.dirname(snapshot_dir)), os.path.basename(snapshot_dir)


SOLVE_SUBDIR = "training_FFNO3D_zscale_expand_lognlte"

# what the production checkpoints saw (config.TRAIN_SPLIT, VAL_SPLIT): one training and one validation snapshot of five
# simulations, two more snapshots of two of them, and one simulation they never saw
DEFAULT_SNAPSHOTS = [
    "train=en024048_hion/465", "train=nw012023/915", "train=ch012012_hion/910", "train=ch012006/836",
    "train=qs006003_sap/689",
    "val=en024048_hion/700", "val=nw012023/940", "val=ch012012_hion/984", "val=ch012006/849", "val=qs006003_sap/900",
    "same_sim=en024048_hion/385", "same_sim=nw012023/1050",
    "unseen_sim=ch024031_by200bz005/450",
]
ROLE_ORDER = ["train", "val", "same_sim", "unseen_sim"]

# the line each atom is scored on, as 0-based (lower, upper) levels: H alpha, Ca II 8542
LINE_LEVELS = {"H": (1, 2), "CA": (2, 4)}
LOGTAU_BINS = [(-np.inf, -6.0), (-6.0, -4.0), (-4.0, -2.0), (-2.0, 0.0), (0.0, np.inf)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-dir", default=os.path.join(REPO, SOLVE_SUBDIR),
                        help="directory with 3D_sim_train_<atom>.pt (default: the repository's production checkpoints)")
    parser.add_argument("--solve-run-dir", default=None,
                        help="run directory whose solving sets are used and built, shared between checkpoints (default: --run-dir)")
    parser.add_argument("--label", default=None, help="name of this checkpoint set in the outputs (default: model dir name)")
    parser.add_argument("--data-dir", default=None, help="directory holding bifrost_data/ (default: config.DATA_DIR)")
    parser.add_argument("--atoms", default="H,CA")
    parser.add_argument("--snapshot", action="append", default=None,
                        help="role=sim/snap, repeatable; roles: train, val, same_sim, unseen_sim")
    parser.add_argument("--snapshots-file", default=None, help="file of role=sim/snap lines, added to --snapshot")
    parser.add_argument("--score-only", action="store_true", help="score the cached predictions, predict nothing")
    return parser.parse_args()


def parse_snapshot(text):
    role, _, path = text.partition("=")
    sim, _, snap = path.partition("/")
    if not (role and sim and snap):
        raise ValueError(f"--snapshot {text!r}: expected role=sim/snap")
    return role, sim, snap


def solving_set(run_dir, model_dir, name):
    """The solving set of a snapshot: linked from the model directory when it is there, otherwise built."""
    target_dir = os.path.join(run_dir, SOLVE_SUBDIR)
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"3D_sim_predict_{name}.hdf5")
    existing = os.path.join(model_dir, f"3D_sim_predict_{name}.hdf5")
    if not os.path.exists(path) and os.path.exists(existing) and os.path.abspath(existing) != os.path.abspath(path):
        os.symlink(existing, path)
    if not os.path.exists(path):
        env = dict(os.environ, FFNOML_RUN_DIR=os.path.abspath(run_dir))
        subprocess.run([sys.executable, "pipeline.py", "--buildforpredict", "--predname", name], cwd=REPO, env=env, check=True)
    return path


def predict(backend, solve_path, out_path):
    """log10 populations (x, y, z, level) of a whole cube, cached as float32 log10 in out_path."""
    if os.path.exists(out_path):
        with h5py.File(out_path, "r") as f:
            return f["log10_populations"][...]
    with h5py.File(solve_path, "r") as f:
        features, z_scale, dx, dy = f["inputs"][0], f["z_scale"][0], float(f["dx"][0]), float(f["dy"][0])
    log10 = backend.predict_log10(features, z_scale, dx, dy).float()          # (level, z, x, y)
    linear = np.asarray(linear_populations(log10))                             # the float32 limit of the service
    log10 = np.log10(np.maximum(linear, np.finfo(np.float32).tiny)).transpose(2, 3, 1, 0).astype(np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path + ".tmp", "w") as f:
        f["log10_populations"] = log10
        f["z_scale"] = np.asarray(z_scale, dtype=np.float32)
    os.replace(out_path + ".tmp", out_path)
    return log10


def multi3d_log10(snapshot_dir, atom, shape):
    """MULTI3D's NLTE populations as log10 m^-3, (x, y, z, level): out_pop is Fortran (x, y, z, level), cm^-3, top first."""
    nx, ny, nz, levels = shape
    if PACKED_DATA:
        pops = packed_data.populations(PACKED_DATA, *packed_key(snapshot_dir), atom, lte=False)[0][..., :levels]
        if pops.shape != tuple(shape):
            raise ValueError(f"{PACKED_DATA} {snapshot_dir} {atom}: populations {pops.shape}, predicted {shape}")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(pops.astype(np.float64)) + 6.0
    raw = np.fromfile(os.path.join(snapshot_dir, atom, "out_pop"), dtype=np.float32, count=nx * ny * nz * levels)
    if raw.size != nx * ny * nz * levels:
        raise ValueError(f"{snapshot_dir}/{atom}/out_pop is shorter than {shape}")
    pops = raw.reshape((levels, nz, ny, nx)).transpose(3, 2, 1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(pops.astype(np.float64)) + 6.0


def log10_tau500(snapshot_dir, atom, shape):
    nx, ny, nz = shape
    if PACKED_DATA:
        with np.errstate(divide="ignore"):
            return np.log10(packed_data.tau500(PACKED_DATA, *packed_key(snapshot_dir), atom).astype(np.float64))
    for path in (os.path.join(snapshot_dir, atom, "tau500"), os.path.join(snapshot_dir, "H", "tau500")):
        if os.path.exists(path):
            count = nx * ny * nz
            dtype = np.float32 if os.path.getsize(path) == 4 * count else np.float64
            tau = np.fromfile(path, dtype=dtype, count=count).reshape((nz, ny, nx)).transpose(2, 1, 0)
            with np.errstate(divide="ignore"):
                return np.log10(tau.astype(np.float64))
    raise FileNotFoundError(f"no tau500 for {snapshot_dir} ({atom})")


def summary(values):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    p50, p90 = np.percentile(values, [50, 90])
    return {"median": float(p50), "p90": float(p90), "mean": float(values.mean()), "cells": int(values.size)}


def score(pred, true, logtau, atom):
    lower, upper = LINE_LEVELS[atom]
    bins = [("all", np.ones(logtau.shape, dtype=bool))]
    bins += [(f"logtau[{lo:g},{hi:g})", (logtau >= lo) & (logtau < hi)) for lo, hi in LOGTAU_BINS]
    result = {}
    for name, mask in bins:
        entry = {f"level{l + 1}": summary(np.abs(pred[..., l] - true[..., l])[mask]) for l in range(pred.shape[-1])}
        ratio_error = (pred[..., upper] - pred[..., lower]) - (true[..., upper] - true[..., lower])
        entry["line_ratio"] = summary(np.abs(ratio_error)[mask])
        entry["line_lower"] = entry[f"level{lower + 1}"]
        result[name] = entry
    return result


def print_summary(results, atoms):
    """Median over each role's snapshots of the per-snapshot medians, for the line's lower level and level ratio."""
    for atom in atoms:
        print(f"\n{atom}: median |delta log10| (dex), lower level of the line / upper-lower ratio, median over snapshots")
        names = ["all"] + [f"logtau[{lo:g},{hi:g})" for lo, hi in LOGTAU_BINS]
        print("role        n  " + "  ".join(f"{n:>22s}" for n in names))
        for role in ROLE_ORDER:
            rows = [r for r in results if r["role"] == role and atom in r["atoms"]]
            if not rows:
                continue
            cells = []
            for n in names:
                low = [r["atoms"][atom][n]["line_lower"]["median"] for r in rows if r["atoms"][atom][n]["line_lower"]]
                rat = [r["atoms"][atom][n]["line_ratio"]["median"] for r in rows if r["atoms"][atom][n]["line_ratio"]]
                cells.append(f"{np.median(low):.3f} / {np.median(rat):.3f}" if low else "-")
            print(f"{role:10s} {len(rows):2d}  " + "  ".join(f"{c:>22s}" for c in cells))


def main():
    args = parse_args()
    import config
    data_dir = args.data_dir or config.DATA_DIR
    atoms = args.atoms.split(",")
    label = args.label or os.path.basename(os.path.normpath(args.model_dir))
    listed = list(args.snapshot or [])
    if args.snapshots_file:
        with open(args.snapshots_file) as f:
            listed += [line.strip() for line in f if line.strip() and not line.startswith("#")]
    snapshots = [parse_snapshot(s) for s in (listed or DEFAULT_SNAPSHOTS)]
    os.makedirs(args.run_dir, exist_ok=True)

    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if not devices and not args.score_only:
        raise RuntimeError("no GPU visible; use --score-only to score cached predictions")
    results = []
    for atom in atoms:
        checkpoint = os.path.join(args.model_dir, f"3D_sim_train_{atom}.pt")
        backend = None
        for role, sim, snap in snapshots:
            name = f"{sim}_{snap}"
            snapshot_dir = os.path.join(data_dir, "bifrost_data", sim, snap)
            out_path = os.path.join(args.run_dir, "predictions", label, f"{name}_{atom}.hdf5")
            if not (packed_data.holds(PACKED_DATA, sim, snap, atom) if PACKED_DATA
                    else os.path.exists(os.path.join(snapshot_dir, atom, "out_pop"))):
                print(f"skip {name} {atom}: no MULTI3D out_pop", flush=True)
                continue
            if not os.path.exists(out_path):
                if args.score_only:
                    print(f"skip {name} {atom}: no cached prediction", flush=True)
                    continue
                if backend is None:
                    backend = BatchedFFNOBackend(checkpoint, lambda Cin, Cout, device: create_ffno3d(Cin, Cout, device),
                                                 devices=devices, log=lambda message: None, vertical_norm_slabs=1)
                solve_path = solving_set(args.solve_run_dir or args.run_dir, args.model_dir, name)
            else:
                solve_path = None
            pred = predict(backend, solve_path, out_path).astype(np.float64)
            true = multi3d_log10(snapshot_dir, atom, pred.shape)
            logtau = log10_tau500(snapshot_dir, atom, pred.shape[:3])
            entry = next((r for r in results if r["name"] == name), None)
            if entry is None:
                entry = {"role": role, "name": name, "atoms": {}}
                results.append(entry)
            entry["atoms"][atom] = score(pred, true, logtau, atom)
            line = entry["atoms"][atom]["all"]
            print(f"{role:10s} {name:28s} {atom:2s} lower level {line['line_lower']['median']:.3f} dex, "
                  f"ratio {line['line_ratio']['median']:.3f} dex (median, all cells)", flush=True)
        del backend
        torch.cuda.empty_cache()

    out = os.path.join(args.run_dir, f"population_errors_{label}.json")
    with open(out, "w") as f:
        json.dump({"model_dir": os.path.abspath(args.model_dir), "line_levels": LINE_LEVELS, "results": results}, f, indent=1)
    print_summary(results, atoms)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
