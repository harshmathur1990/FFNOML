"""Submit the learning-curve experiments of one input file on Olivia: does the FFNO gain more from more simulations or
from more of the same one?

    python3 scripts/submit_learning_curve.py scripts/learning_curve.ini [EXPERIMENT ...] [--dry-run] [--resume]
    python3 scripts/submit_learning_curve.py scripts/learning_curve.ini EXPERIMENT --predict-only SIM/SNAP ... [--nodes N]

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
training carries on from its last epoch and predictions already written are kept. --predict-only runs no training: the
job predicts the snapshots named (sim/snap; their solving sets in ROOT/solving_sets) with the experiment's trained
checkpoint, e.g. snapshots outside its scoring set; --nodes and --time override the section's for that job.

Only the standard library of Python 3.6 is used, so it runs with the login node's python3 (3.6.15 on Olivia).
"""
import argparse
import configparser
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATOMS = ("H", "CA")
# root: Olivia; stage, data_dir, solving_sets_from: eagle7 (learning_curve_eagle7.py)
SETTINGS = ("root", "stage", "data_dir", "solving_sets_from", "atoms", "epochs", "nodes", "time", "train_flags",
            "held_out", "train", "val", "output_mode", "lam_total", "model_config", "trainselect", "predict")
OUTPUT_MODES = ("log_population", "level_fraction")   # config.MODEL_CONFIG output_mode (models/ffno_model.py)
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
    e["output_mode"] = e["output_mode"] or OUTPUT_MODES[0]
    e["lam_total"] = e["lam_total"] or "0"
    e["trainselect"] = e["trainselect"] or "1"
    e["predict"] = e["predict"] or "yes"
    if not 0 < float(e["trainselect"]) <= 1:
        raise ValueError(f"[{name}] trainselect = {e['trainselect']!r}: the fraction of training patches per epoch, in (0, 1]")
    if e["predict"] not in ("yes", "no"):
        raise ValueError(f"[{name}] predict = {e['predict']!r}: yes or no")
    if e["output_mode"] not in OUTPUT_MODES:
        raise ValueError(f"[{name}] output_mode = {e['output_mode']!r}: one of {', '.join(OUTPUT_MODES)}")
    if float(e["lam_total"]) < 0:
        raise ValueError(f"[{name}] lam_total = {e['lam_total']!r}: must be >= 0")
    if e["model_config"]:
        try:
            overrides = json.loads(e["model_config"])
        except ValueError as err:
            raise ValueError(f"[{name}] model_config is not JSON: {err}")
        if not isinstance(overrides, dict):
            raise ValueError(f"[{name}] model_config = {e['model_config']!r}: a JSON object, e.g. {{\"width\": 96}}")
        e["model_config"] = json.dumps(overrides, sort_keys=True)
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


def model_env(e):
    """The environment that selects the experiment's head, loss and model_config overrides of MODEL_CONFIG (config.py);
    training and every prediction need it, since a checkpoint loads only into the model it was trained with."""
    return {"FFNOML_OUTPUT_MODE": e["output_mode"], "FFNOML_LAM_TOTAL": e["lam_total"],
            "FFNOML_MODEL_CONFIG": e["model_config"]}


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


def missing_for_prediction(e, root, names):
    """What a --predict-only job reads: the experiment's checkpoint and the named snapshots' solving sets."""
    run = os.path.join(root, e["name"])
    paths = [os.path.join(run, MODEL_SUBDIR, "3D_sim_train_{}.pt".format(e["atoms"]))]
    paths += [os.path.join(root, "solving_sets", "3D_sim_predict_{}.hdf5".format(prediction_name(*s))) for s in names]
    return [p for p in paths if not os.path.exists(p)]


def job_env(e, root, resume=False):
    """experiment.env: what learning_curve_train.sbatch sources"""
    return {
        "FFNOML_EXPERIMENT": e["name"],
        "FFNOML_REPO_DIR": REPO,
        "FFNOML_SOLVING_SETS": os.path.join(root, "solving_sets"),
        "FFNOML_TRAIN_SPLIT": split_json(e["train"]),
        "FFNOML_VAL_SPLIT": split_json(e["val"]),
        "FFNOML_ACTIVE_ATOMS": e["atoms"],
        "FFNOML_NUM_EPOCHS": e["epochs"],
        "FFNOML_TRAINSELECT": e["trainselect"],
        "FFNOML_PREDICT": e["predict"],
        "FFNOML_TRAIN_FLAGS": " ".join(f for f in (e["train_flags"], "--resume" if resume else "") if f),
        **model_env(e),
    }


def write_job_files(e, run, root, resume=False):
    with open(os.path.join(run, "experiment.env"), "w") as f:
        f.writelines(f"{k}='{v}'\n" for k, v in job_env(e, root, resume).items())
    with open(os.path.join(run, "eval_snapshots.txt"), "w") as f:
        f.writelines(eval_lines(e))


def sbatch(args):
    return subprocess.run(["sbatch", "--parsable"] + args, check=True, stdout=subprocess.PIPE,
                          universal_newlines=True).stdout.strip().split(";")[0]


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("input")
    cli.add_argument("experiments", nargs="*", help="sections to submit (default: all)")
    cli.add_argument("--dry-run", action="store_true", help="check the inputs and write the run directories, submit nothing")
    cli.add_argument("--resume", action="store_true", help="continue the experiments stopped by their time limit")
    cli.add_argument("--predict-only", nargs="+", metavar="SIM/SNAP",
                     help="no training: predict these snapshots with the experiment's trained checkpoint")
    cli.add_argument("--nodes", help="with --predict-only: nodes for the job (default: the section's)")
    cli.add_argument("--time", help="with --predict-only: time limit for the job (default: the section's)")
    args = cli.parse_args()
    if (args.nodes or args.time) and not args.predict_only:
        cli.error("--nodes and --time go with --predict-only")
    experiments = read_experiments(args.input, args.experiments)
    to_predict = snapshots(" ".join(args.predict_only)) if args.predict_only else None
    if to_predict is not None and len(experiments) != 1:
        cli.error("--predict-only takes one experiment")

    # everything is checked before anything is submitted
    problems = {name: (missing_for_prediction(e, os.path.abspath(e["root"]), to_predict) if to_predict is not None
                       else missing_inputs(e, os.path.abspath(e["root"]))) for name, e in experiments.items()}
    if any(problems.values()):
        sys.exit("missing:\n" + "\n".join(f"  [{n}] {p}" for n, ps in problems.items() for p in ps))

    for name, e in experiments.items():
        root = os.path.abspath(e["root"])
        run = os.path.join(root, name)
        export = f"ALL,FFNOML_RUN_DIR={run}"
        nodes, time = args.nodes or e["nodes"], args.time or e["time"]
        if to_predict is not None:
            # the list in a file of its own, so that jobs predicting different snapshots can be queued together
            lines = [f"predict={sim}/{snap}\n" for sim, snap in to_predict]
            listing = "predict_only_{}.txt".format(hashlib.md5("".join(lines).encode()).hexdigest()[:8])
            with open(os.path.join(run, listing), "w") as f:
                f.writelines(lines)
            export += f",FFNOML_PREDICT_ONLY={listing}"
            print(f"{name}: predict only, {len(to_predict)} snapshots ({listing}) on {nodes} nodes; {run}", flush=True)
        else:
            write_job_files(e, run, root, args.resume)
            print(f"{name}: {e['atoms']}, {e['output_mode']} (lam_total {e['lam_total']}), "
                  f"{len(e['train'])} training snapshots, {e['epochs']} epochs on {e['nodes']} nodes, "
                  f"then {len(e['eval'])} predictions; {run}", flush=True)
        if args.dry_run:
            continue
        job = sbatch(["--chdir", run, "--export", export, "--nodes", nodes, "--time", time,
                      "--job-name", f"lc_{name}", "--output", f"{run}/slurm-%j.out", "--error", f"{run}/slurm-%j.err",
                      os.path.join(REPO, "scripts", "learning_curve_train.sbatch")])
        print(f"  job {job}", flush=True)


if __name__ == "__main__":
    main()
