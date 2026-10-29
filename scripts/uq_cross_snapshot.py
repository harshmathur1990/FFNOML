"""Cross-snapshot error table from uq_error_atlas.py sample tables: per case, atom, regime and level,
the MAE, p95 and bias of e = log10(pred) - log10(truth), with each case's role in the training split.

Usage:
    python scripts/uq_cross_snapshot.py --atlas uq/atlas_mc/*_atlas.h5 --out uq/cross_snapshot.csv
"""
import argparse
import csv
import os
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from config import TRAIN_SPLIT, VAL_SPLIT  # noqa: E402
from uq_error_atlas import REGIMES  # noqa: E402


def role(case):
    sim, snap = case.rsplit("_", 1)
    for name, split in (("train", TRAIN_SPLIT), ("val", VAL_SPLIT)):
        if any(str(s) == snap for s in split.get(sim, ())):
            return name
    return "held-out"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atlas", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = []
    for path in sorted(args.atlas):
        with h5py.File(path, "r") as f:
            case, atom = f.attrs["case"], f.attrs["atom"]
            s = f["sample"]
            err, logT, z = s["err"][...], s["logT"][...], s["z_Mm"][...]
        for reg, fn in REGIMES.items():
            m = fn(logT, z)
            if not m.any():
                continue
            for lev in range(err.shape[1]):
                e = err[m, lev]
                rows.append(dict(case=case, atom=atom, role=role(case), regime=reg, level=lev + 1, n=int(m.sum()),
                                 mae=float(np.abs(e).mean()), p95=float(np.percentile(np.abs(e), 95)),
                                 bias=float(e.mean())))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
