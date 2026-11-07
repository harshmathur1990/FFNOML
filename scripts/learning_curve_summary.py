"""Compare the population errors of several checkpoints (population_error_diagnostic.py outputs) side by side.

    python scripts/learning_curve_summary.py ROOT_OR_JSON [...] [--bin "logtau[-6,-4)"] [--metric line_ratio]

A directory is searched for */eval/population_errors_*.json (the learning-curve layout) and population_errors_*.json.
Each row is one checkpoint; each column the median, over that role's snapshots, of the per-snapshot median |delta log10|
in dex, for the line's lower level (or --metric line_ratio, the upper/lower ratio) in the chosen tau500 bin.
"""
import argparse
import glob
import json
import os

import numpy as np

ROLES = ["train", "val", "same_sim", "unseen_sim"]
ORDER = ["sims1", "sims2", "sims3", "sims4", "same3", "all5", "sims1_CA", "sims2_CA", "sims3_CA", "sims4_CA", "same3_CA", "all5_CA"]


def files(paths):
    found = []
    for path in paths:
        if os.path.isdir(path):
            found += sorted(glob.glob(os.path.join(path, "*", "eval", "population_errors_*.json")))
            found += sorted(glob.glob(os.path.join(path, "population_errors_*.json")))
        else:
            found.append(path)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--bin", action="append", default=None,
                        help='tau500 bin, repeatable (default: all, logtau[-6,-4), logtau[-4,-2))')
    parser.add_argument("--metric", default="line_lower", choices=["line_lower", "line_ratio"])
    args = parser.parse_args()
    bins = args.bin or ["all", "logtau[-6,-4)", "logtau[-4,-2)"]

    table = {}
    for path in files(args.paths):
        label = os.path.basename(path)[len("population_errors_"):-len(".json")]
        with open(path) as f:
            table[label] = json.load(f)["results"]
    labels = sorted(table, key=lambda l: (ORDER.index(l) if l in ORDER else len(ORDER), l))
    atoms = sorted({atom for rows in table.values() for r in rows for atom in r["atoms"]})

    for atom in atoms:
        for b in bins:
            print(f"\n{atom}  {args.metric}  {b}: median over snapshots of the median |delta log10| [dex] (snapshots)")
            print(f"{'checkpoint':14s}" + "".join(f"{role:>18s}" for role in ROLES))
            for label in labels:
                cells = []
                for role in ROLES:
                    values = [r["atoms"][atom][b][args.metric]["median"] for r in table[label]
                              if r["role"] == role and atom in r["atoms"] and r["atoms"][atom][b][args.metric]]
                    cells.append(f"{np.median(values):.3f} ({len(values)})" if values else "-")
                print(f"{label:14s}" + "".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
