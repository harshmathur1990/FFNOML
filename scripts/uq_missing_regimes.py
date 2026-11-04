"""Which atmospheric configurations carry the population errors? (claude-next-steps-uq step 2)

Pools the uq_error_atlas.py sample tables of all cases (an equal number of voxels per case, no case
identity among the descriptors) and fits a shallow regression tree of |e| (mean over --levels) on
physical descriptors. Each leaf is a box of parameter ranges. Per leaf:

    n, n_cases, n_sims          voxels and independent cases/simulations represented (>= --min-case)
    mae, mae_train, mae_heldout mean |e| over all, training-snapshot and non-training voxels
    train_coverage              (leaf share of training voxels) / (leaf share of all voxels); < 1 means
                                the training snapshots populate this configuration less than the pool
    bias, mae_local             mean signed e, and mean |e - b| where b is each case's mean e in the
                                voxel's regime (photosphere/chromosphere/TR/corona)
    snapshot_wide_frac          1 - mae_local / mae: the part of the error explained by per-case offsets
    kind                        "snapshot-wide" if snapshot_wide_frac > 0.5 else "localized"
    cases_mae_gt                cases whose leaf MAE exceeds --bad, worst first
    example                     a representative voxel (worst case; error nearest that case's leaf median)

Associations are diagnostic; they do not establish causal parameter effects.

Usage:
    python scripts/uq_missing_regimes.py --atlas uq/atlas_v2/*_H_atlas.h5 --levels 2-5 --out uq/missing_regimes_H.csv
"""
import argparse
import csv
import os
import sys

import h5py
import numpy as np
from sklearn.tree import DecisionTreeRegressor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "ffno"))

from config import TRAIN_SPLIT  # noqa: E402
from uq_error_atlas import REGIMES  # noqa: E402

DESCRIPTORS = ("logT", "logne", "logrho", "vz", "vh", "z_Mm", "log_cmass", "dlogT_dz", "dlogrho_dz",
               "dvz_dz", "gradh_logT", "divh_v", "logT_contrast_1Mm")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atlas", nargs="+", required=True)
    p.add_argument("--levels", default="2-5", help="1-based level range averaged into the target")
    p.add_argument("--out", required=True)
    p.add_argument("--per-case", type=int, default=100_000)
    p.add_argument("--leaves", type=int, default=32)
    p.add_argument("--min-leaf-frac", type=float, default=0.005)
    p.add_argument("--min-case", type=int, default=200, help="voxels for a case to count in a leaf")
    p.add_argument("--bad", type=float, default=0.2, help="dex; leaf MAE threshold for cases_mae_gt")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def is_train(case):
    sim, snap = case.rsplit("_", 1)
    return snap in map(str, TRAIN_SPLIT.get(sim, ()))


def load(paths, levels, per_case, rng):
    cols = {k: [] for k in DESCRIPTORS + ("e", "e_local", "case", "ix", "iy")}
    for path in sorted(paths):
        with h5py.File(path, "r") as f:
            case = str(f.attrs["case"])
            s = f["sample"]
            n = s["ix"].shape[0]
            sel = np.sort(rng.choice(n, size=min(per_case, n), replace=False))
            d = {k: s[k][...][sel] for k in ("logT", "logne", "logrho", "vx", "vy", "vz", "z_Mm", "ix", "iy",
                                              "log_cmass", "dlogT_dz", "dlogrho_dz", "dvz_dz", "gradh_logT",
                                              "divh_v", "logT_contrast_1Mm")}
            e = s["err"][...][sel][:, levels].mean(1)
        d["vh"] = np.hypot(d.pop("vx"), d.pop("vy"))
        e_local = np.empty_like(e)
        for fn in REGIMES.values():
            m = fn(d["logT"], d["z_Mm"])
            if m.any():
                e_local[m] = e[m] - e[m].mean()
        for k in DESCRIPTORS + ("ix", "iy"):
            cols[k].append(d[k])
        cols["e"].append(e)
        cols["e_local"].append(e_local)
        cols["case"].append(np.full(e.size, case, dtype=object))
        print(f"{case}: {e.size} voxels, mean |e| {np.abs(e).mean():.3f}", flush=True)
    return {k: np.concatenate(v) for k, v in cols.items()}


def leaf_boxes(tree, names):
    """Per leaf: {feature: [lo, hi]} from the split thresholds along its path."""
    t = tree.tree_
    boxes = {}

    def walk(node, box):
        if t.children_left[node] == -1:
            boxes[node] = box
            return
        f, thr = names[t.feature[node]], t.threshold[node]
        lo, hi = box.get(f, (-np.inf, np.inf))
        walk(t.children_left[node], {**box, f: (lo, min(hi, thr))})
        walk(t.children_right[node], {**box, f: (max(lo, thr), hi)})

    walk(0, {})
    return boxes


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    a, b = (int(x) for x in args.levels.split("-"))
    levels = list(range(a - 1, b))
    d = load(args.atlas, levels, args.per_case, rng)
    X = np.stack([d[k] for k in DESCRIPTORS], 1).astype(np.float32)
    y = np.abs(d["e"])
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    d = {k: v[ok] for k, v in d.items()}

    tree = DecisionTreeRegressor(max_leaf_nodes=args.leaves, min_samples_leaf=int(args.min_leaf_frac * len(y)),
                                 random_state=args.seed).fit(X, y)
    leaf = tree.apply(X)
    boxes = leaf_boxes(tree, DESCRIPTORS)
    train = np.array([is_train(c) for c in d["case"]])
    sims = np.array([c.rsplit("_", 1)[0] for c in d["case"]])
    share_train = train.mean()

    rows = []
    for node, box in boxes.items():
        m = leaf == node
        cases, counts = np.unique(d["case"][m], return_counts=True)
        cases = cases[counts >= args.min_case]
        per_case = {c: float(np.abs(d["e"][m & (d["case"] == c)]).mean()) for c in cases}
        worst = sorted(per_case, key=per_case.get, reverse=True)
        bad = [f"{c}:{per_case[c]:.2f}" for c in worst if per_case[c] > args.bad]
        mae, mae_local = float(y[m].mean()), float(np.abs(d["e_local"][m]).mean())
        mw = m & (d["case"] == worst[0]) if worst else m
        k = np.flatnonzero(mw)[np.abs(np.abs(d["e"][mw]) - np.median(np.abs(d["e"][mw]))).argmin()]
        row = dict(
            leaf=int(node), n=int(m.sum()), share=float(m.mean()), n_cases=len(cases),
            n_sims=len({c.rsplit("_", 1)[0] for c in cases}), mae=mae,
            mae_train=float(y[m & train].mean()) if (m & train).any() else float("nan"),
            mae_heldout=float(y[m & ~train].mean()) if (m & ~train).any() else float("nan"),
            train_coverage=float((m & train).sum() / max(m.sum(), 1) / share_train),
            bias=float(d["e"][m].mean()), mae_local=mae_local,
            snapshot_wide_frac=float(1 - mae_local / mae) if mae > 0 else float("nan"),
        )
        row["kind"] = "snapshot-wide" if row["snapshot_wide_frac"] > 0.5 else "localized"
        row["cases_mae_gt"] = " ".join(bad)
        row["example"] = f"{d['case'][k]} ix={d['ix'][k]} iy={d['iy'][k]} z={d['z_Mm'][k]:.2f}Mm |e|={y[k]:.2f}"
        for f in DESCRIPTORS:
            lo, hi = box.get(f, (np.nan, np.nan))
            row[f"{f}_lo"], row[f"{f}_hi"] = lo, hi
        rows.append(row)
    rows.sort(key=lambda r: -(np.nan_to_num(r["mae_heldout"]) * r["share"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} leaves, tree R^2 {tree.score(X, y):.3f}; simulations {sorted(set(sims))}")


if __name__ == "__main__":
    main()
