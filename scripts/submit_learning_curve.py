"""Submit the learning-curve experiments of one input file on Olivia: does the FFNO gain more from more simulations or
from more of the same one?

    python3 scripts/submit_learning_curve.py scripts/learning_curve.ini [EXPERIMENT ...] [--dry-run] [--resume]

The input file (INI) has a [DEFAULT] section of settings shared by every experiment, and one section per experiment
naming its training and validation snapshots (sim/snap, separated by spaces or newlines). Any setting can be overridden
in an experiment's section. Every experiment trains from scratch with the production model (config.py) and is then
scored against MULTI3D on its own training and validation snapshots and on the held_out ones: a held-out snapshot of a
simulation the experiment trains on is scored as same_sim, any other as unseen_sim.

Only GPU work runs on Olivia. On eagle7, scripts/learning_curve_eagle7.py prepare builds every experiment's datasets
(pipeline.py --build) and the scoring snapshots' solving sets, which are copied to ROOT/<section>/IO and
ROOT/solving_sets. One job per experiment then trains (pipeline.py --train) and predicts every scoring snapshot
(pipeline.py --fsdppredict); the checkpoint and predictions are copied back and scored on eagle7
(learning_curve_eagle7.py score). Before anything is submitted, every experiment is checked for its datasets and solving
sets. With no EXPERIMENT names, every section is submitted. --resume continues jobs stopped by their time limit:
training carries on from its last epoch and predictions already written are kept.

Only the standard library of Python 3.6 is used, so it runs with the login node's python3 (3.6.15 on Olivia).
"""
import argparse
import configparser
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATOMS = ("H", "CA")
# root: Olivia; stage, data_dir, solving_sets_from: eagle7 (learning_curve_eagle7.py)
SETTINGS = ("root", "stage", "data_dir", "solving_sets_from", "atoms", "epochs", "nodes", "time", "train_flags",
            "held_out", "train", "val")
MODEL_SUBDIR = "training_FFNO3D_zscale_expand_lognlte"   # config.MODEL_DIR below the run directory


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
    for key in ("root", "stage", "epochs", "nodes", "time"):
        if not e[key]:
            raise ValueError(f"[{name}]: {key} is not set")
    int(e["epochs"]), int(e["nodes"])
    e["name"] = name
    seen_sims = {sim for sim, _ in e["train"] + e["val"]}
    seen = set(e["train"] + e["val"])
    e["eval"] = ([("train", s) for s in e["train"]] + [("val", s) for s in e["val"]] +
                 [("same_sim" if s[0] in seen_sims else "unseen_sim", s) for s in e["held_out"] if s not in seen])
    return e


def read_experiments(path, names=None):
    """The experiments of an input file, by section name; every one of names, or all sections."""
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
    if not parser.read(path):
        sys.exit(f"cannot read {path}")
    names = names or parser.sections()
    absent = [n for n in names if not parser.has_section(n)]
    if absent:
        sys.exit(f"no section {', '.join(absent)} in {path} (sections: {', '.join(parser.sections())})")
    try:
        return {name: experiment(parser, name) for name in names}
    except ValueError as err:
        sys.exit(f"{path}: {err}")


def prediction_name(sim, snap):
    return f"{sim}_{snap}"


def eval_lines(e):
    return [f"{role}={sim}/{snap}\n" for role, (sim, snap) in e["eval"]]


def missing_inputs(e, root):
    """What the job reads: the datasets learning_curve_eagle7.py prepare listed in prepared.json, with their sizes, and
    the solving set of every scoring snapshot."""
    run = os.path.join(root, e["name"])
    try:
        with open(os.path.join(run, "prepared.json")) as f:
            prepared = json.load(f)
    except OSError:
        return [os.path.join(run, "prepared.json") + " (prepare and copy the experiment from eagle7)"]
    missing = []
    for rel, size in prepared["datasets"].items():
        path = os.path.join(run, rel)
        if not os.path.exists(path):
            missing.append(path)
        elif os.path.getsize(path) != size:
            missing.append(f"{path} ({os.path.getsize(path)} bytes, prepared {size}: incomplete copy?)")
    for _, (sim, snap) in e["eval"]:
        path = os.path.join(root, "solving_sets", f"3D_sim_predict_{prediction_name(sim, snap)}.hdf5")
        if not os.path.exists(path):
            missing.append(path)
    return missing


def sbatch(args):
    return subprocess.run(["sbatch", "--parsable"] + args, check=True, stdout=subprocess.PIPE,
                          universal_newlines=True).stdout.strip().split(";")[0]


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("input")
    cli.add_argument("experiments", nargs="*", help="sections to submit (default: all)")
    cli.add_argument("--dry-run", action="store_true", help="check the inputs and write the run directories, submit nothing")
    cli.add_argument("--resume", action="store_true", help="continue the experiments stopped by their time limit")
    args = cli.parse_args()
    experiments = read_experiments(args.input, args.experiments)

    # everything is checked before anything is submitted
    problems = {name: missing_inputs(e, os.path.abspath(e["root"])) for name, e in experiments.items()}
    if any(problems.values()):
        sys.exit("missing:\n" + "\n".join(f"  [{n}] {p}" for n, ps in problems.items() for p in ps))

    for name, e in experiments.items():
        root = os.path.abspath(e["root"])
        run = os.path.join(root, name)
        flags = " ".join(f for f in (e["train_flags"], "--resume" if args.resume else "") if f)
        env = {
            "FFNOML_EXPERIMENT": name,
            "FFNOML_REPO_DIR": REPO,
            "FFNOML_SOLVING_SETS": os.path.join(root, "solving_sets"),
            "FFNOML_TRAIN_SPLIT": split_json(e["train"]),
            "FFNOML_VAL_SPLIT": split_json(e["val"]),
            "FFNOML_ACTIVE_ATOMS": e["atoms"],
            "FFNOML_NUM_EPOCHS": e["epochs"],
            "FFNOML_TRAIN_FLAGS": flags,
        }
        with open(os.path.join(run, "experiment.env"), "w") as f:
            f.writelines(f"{k}='{v}'\n" for k, v in env.items())
        with open(os.path.join(run, "eval_snapshots.txt"), "w") as f:
            f.writelines(eval_lines(e))
        print(f"{name}: {e['atoms']}, {len(e['train'])} training snapshots, {e['epochs']} epochs on {e['nodes']} nodes, "
              f"then {len(e['eval'])} predictions; {run}", flush=True)
        if args.dry_run:
            continue
        job = sbatch(["--chdir", run, "--export", f"ALL,FFNOML_RUN_DIR={run}", "--nodes", e["nodes"], "--time", e["time"],
                      "--job-name", f"lc_{name}", "--output", f"{run}/slurm-%j.out", "--error", f"{run}/slurm-%j.err",
                      os.path.join(REPO, "scripts", "learning_curve_train.sbatch")])
        print(f"  job {job}", flush=True)


if __name__ == "__main__":
    main()
