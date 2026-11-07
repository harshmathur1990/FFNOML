"""Submit the learning-curve experiments of one input file on Olivia: does the FFNO gain more from more simulations or
from more of the same one?

    python3 scripts/submit_learning_curve.py scripts/learning_curve.ini [EXPERIMENT ...] [--dry-run] [--resume]

The input file (INI) has a [DEFAULT] section of settings shared by every experiment, and one section per experiment
naming its training and validation snapshots (sim/snap, separated by spaces or newlines). The snapshots are read
from packed_data, one file made on eagle7 with packed_data.py, or else from data_dir/bifrost_data. Any setting can be overridden
in an experiment's section. Every experiment trains from scratch with the production model (config.py) and is then
scored against MULTI3D on its own training and validation snapshots and on the held_out ones: a held-out snapshot of a
simulation the experiment trains on is scored as same_sim, any other as unseen_sim.

Each experiment gets ROOT/<section>/ with experiment.env, eval_snapshots.txt, the datasets, the checkpoint and
eval/population_errors_<section>.json; the solving sets (raw inputs, the same for every atom) are shared in
ROOT/solving_sets, so the evaluations run one after another. Before anything is submitted, every snapshot is checked for
the files its jobs read, or for its entry in the packed file's manifest. With no EXPERIMENT names, every section is submitted. --resume continues training jobs stopped
by their time limit (their datasets are kept and the build is skipped).
Afterwards, on eagle7 or anywhere: python scripts/learning_curve_summary.py ROOT

Only the standard library is used, so it runs with the login node's python3.
"""
import argparse
import configparser
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATOMS = ("H", "CA")
# helita's Multi3dOut.readall opens multi3d.input and memory-maps these; the ie_* and zt1_* spectra are never read
BUILD_FILES = ("multi3d.input", "out_par", "out_nu", "out_pop", "out_atm", "out_rtq")
SETTINGS = ("root", "data_dir", "packed_data", "atoms", "epochs", "nodes", "time", "train_flags", "held_out", "train", "val")


def snapshots(text):
    """'sim/snap sim/snap ...' -> [(sim, snap), ...]"""
    out = []
    for item in text.split():
        sim, _, snap = item.partition("/")
        if not (sim and snap) or "/" in snap:
            raise ValueError(f"{item!r}: expected sim/snap")
        out.append((sim, snap))
    return out


def split_json(pairs):
    split = {}
    for sim, snap in pairs:
        split.setdefault(sim, []).append(snap)
    return json.dumps(split)


def experiment(parser, name):
    section = parser[name]
    unknown = sorted(set(section) - set(SETTINGS))
    if unknown:
        raise ValueError(f"[{name}]: unknown settings {', '.join(unknown)} (known: {', '.join(SETTINGS)})")
    e = {key: section.get(key, "").strip() for key in SETTINGS}
    if e["atoms"] not in ATOMS:
        raise ValueError(f"[{name}] atoms = {e['atoms']!r}: one of {', '.join(ATOMS)} (production trains H and CA apart; "
                         "make one section per atom)")
    e["train"], e["val"], e["held_out"] = snapshots(e["train"]), snapshots(e["val"]), snapshots(e["held_out"])
    if not e["train"] or not e["val"]:
        raise ValueError(f"[{name}]: train and val must each name at least one sim/snap")
    for key in ("root", "epochs", "nodes", "time"):
        if not e[key]:
            raise ValueError(f"[{name}]: {key} is not set")
    if not (e["data_dir"] or e["packed_data"]):
        raise ValueError(f"[{name}]: set packed_data (a packed_data.py file) or data_dir (holding bifrost_data/)")
    int(e["epochs"]), int(e["nodes"])
    e["root"] = os.path.abspath(e["root"])
    e["data_dir"] = os.path.join(os.path.abspath(e["data_dir"]), "") if e["data_dir"] else ""
    e["packed_data"] = os.path.abspath(e["packed_data"]) if e["packed_data"] else ""
    seen_sims = {sim for sim, _ in e["train"] + e["val"]}
    seen = set(e["train"] + e["val"])
    e["eval"] = ([("train", s) for s in e["train"]] + [("val", s) for s in e["val"]] +
                 [("same_sim" if s[0] in seen_sims else "unseen_sim", s) for s in e["held_out"] if s not in seen])
    return e


def missing_files(e):
    """What the jobs read: training and validation snapshots go through readall; held-out ones need the atmosphere for
    the solving set, and out_pop and tau500 (in the atom's directory or H/) for scoring. A packed file holds all of these
    for every snapshot and atom it packed, as its manifest (<file>.json) lists."""
    if e["packed_data"]:
        if not os.path.exists(e["packed_data"]):
            return [e["packed_data"]]
        try:
            with open(e["packed_data"] + ".json") as f:
                held = json.load(f)["snapshots"]
        except OSError:
            return [e["packed_data"] + ".json (the manifest packed_data.py writes beside it)"]
        return [f"{e['packed_data']}: {sim}/{snap} {e['atoms']}" for _, (sim, snap) in e["eval"]
                if e["atoms"] not in held.get(f"{sim}/{snap}", {}).get("atoms", {})]
    missing = []
    for role, (sim, snap) in e["eval"]:
        d = os.path.join(e["data_dir"], "bifrost_data", sim, snap)
        a = os.path.join(d, e["atoms"])
        needed = [os.path.join(d, f) for f in ("atm3d", "mesh")]
        needed += [os.path.join(a, f) for f in (BUILD_FILES if role in ("train", "val") else ("out_pop",))]
        missing += [p for p in needed if not os.path.exists(p)]
        if not any(os.path.exists(os.path.join(d, sub, "tau500")) for sub in (e["atoms"], "H")):
            missing.append(os.path.join(a, "tau500") + " (or H/tau500)")
    return missing


def sbatch(args):
    return subprocess.run(["sbatch", "--parsable"] + args, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("input")
    cli.add_argument("experiments", nargs="*", help="sections to submit (default: all)")
    cli.add_argument("--dry-run", action="store_true", help="check the data and write the run directories, submit nothing")
    cli.add_argument("--resume", action="store_true", help="continue the training of experiments stopped by their time limit")
    args = cli.parse_args()

    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
    if not parser.read(args.input):
        sys.exit(f"cannot read {args.input}")
    names = args.experiments or parser.sections()
    absent = [n for n in names if not parser.has_section(n)]
    if absent:
        sys.exit(f"no section {', '.join(absent)} in {args.input} (sections: {', '.join(parser.sections())})")
    try:
        experiments = {name: experiment(parser, name) for name in names}
    except ValueError as err:
        sys.exit(f"{args.input}: {err}")

    # everything is checked before anything is submitted, so a missing file cannot stop the chain halfway
    problems = {name: missing_files(e) for name, e in experiments.items()}
    if any(problems.values()):
        sys.exit("missing:\n" + "\n".join(f"  [{n}] {p}" for n, ps in problems.items() for p in ps))

    previous_eval = {}   # root -> last evaluation job: evaluations sharing ROOT/solving_sets run one after another
    for name, e in experiments.items():
        run = os.path.join(e["root"], name)
        os.makedirs(os.path.join(e["root"], "solving_sets"), exist_ok=True)
        os.makedirs(run, exist_ok=True)
        flags = " ".join(f for f in (e["train_flags"], "--resume" if args.resume else "") if f)
        env = {
            "FFNOML_EXPERIMENT": name,
            "FFNOML_REPO_DIR": REPO,
            "FFNOML_DATA_DIR": e["data_dir"],
            "FFNOML_PACKED_DATA": e["packed_data"],
            "FFNOML_TRAIN_SPLIT": split_json(e["train"]),
            "FFNOML_VAL_SPLIT": split_json(e["val"]),
            "FFNOML_ACTIVE_ATOMS": e["atoms"],
            "FFNOML_NUM_EPOCHS": e["epochs"],
            "FFNOML_TRAIN_FLAGS": flags,
        }
        with open(os.path.join(run, "experiment.env"), "w") as f:
            f.writelines(f"{k}='{v}'\n" for k, v in env.items())
        with open(os.path.join(run, "eval_snapshots.txt"), "w") as f:
            f.writelines(f"{role}={sim}/{snap}\n" for role, (sim, snap) in e["eval"])
        counts = {}
        for role, _ in e["eval"]:
            counts[role] = counts.get(role, 0) + 1
        print(f"{name}: {e['atoms']}, {len(e['train'])} training snapshots, {e['epochs']} epochs on {e['nodes']} nodes; "
              f"scored on {', '.join(f'{n} {r}' for r, n in counts.items())}; {run}", flush=True)
        if args.dry_run:
            continue
        common = ["--chdir", run, "--export", f"ALL,FFNOML_RUN_DIR={run}"]
        train_id = sbatch(common + ["--nodes", e["nodes"], "--time", e["time"], "--job-name", f"lc_{name}",
                                    "--output", f"{run}/slurm-%j.out", "--error", f"{run}/slurm-%j.err",
                                    os.path.join(REPO, "scripts", "learning_curve_train.sbatch")])
        dependency = f"afterok:{train_id}" + (f",afterany:{previous_eval[e['root']]}" if e["root"] in previous_eval else "")
        previous_eval[e["root"]] = sbatch(common + ["--dependency", dependency, "--job-name", f"lc_{name}_eval",
                                                    "--output", f"{run}/slurm-eval-%j.out", "--error", f"{run}/slurm-eval-%j.err",
                                                    os.path.join(REPO, "scripts", "learning_curve_eval.sbatch")])
        print(f"  training job {train_id}, evaluation job {previous_eval[e['root']]}", flush=True)


if __name__ == "__main__":
    main()
