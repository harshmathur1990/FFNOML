"""The CPU side of the learning curve, off Olivia (whose hours go to GPU work only): build every experiment's inputs on
eagle7, copy them to Olivia, copy the checkpoints and predictions back, and score them.

    python scripts/learning_curve_eagle7.py prepare scripts/learning_curve.ini [EXPERIMENT ...]   # on eagle7
    python3 scripts/learning_curve_eagle7.py upload scripts/learning_curve.ini [EXPERIMENT ...]   # where ssh reaches Olivia
    (on Olivia: python3 scripts/submit_learning_curve.py scripts/learning_curve.ini [EXPERIMENT ...])
    python3 scripts/learning_curve_eagle7.py fetch scripts/learning_curve.ini [EXPERIMENT ...]
    python scripts/learning_curve_eagle7.py predict scripts/learning_curve.ini [EXPERIMENT ...]   # on eagle7, if the job timed out
    python scripts/learning_curve_eagle7.py score scripts/learning_curve.ini [EXPERIMENT ...]     # on eagle7

prepare builds, for each experiment, its training and validation datasets in STAGE/<section>/IO (pipeline.py --build
with the experiment's splits, from data_dir/bifrost_data) and records them with their sizes in prepared.json; every
scoring snapshot's solving set goes to STAGE/solving_sets, linked from solving_sets_from when it is there and built
(pipeline.py --buildforpredict) otherwise. upload copies those to ROOT/<section> and ROOT/solving_sets on host, and
fetch copies back each experiment's checkpoint, --fsdppredict outputs and logs, skipping files already there with the
same size. predict makes, on eagle7's GPUs, the --fsdppredict outputs a job stopped by its time limit did not reach,
from the fetched best checkpoint. score runs scripts/population_error_diagnostic.py --score-only on them, writing
STAGE/<section>/eval/population_errors_<section>.json; scripts/learning_curve_summary.py STAGE compares them.

ssh and scp use LC_SSH_OPTS (default: the ControlMaster socket ~/.ssh/olivia.sock when it exists, so one login with
password and TOTP serves every copy). prepare and score need the repository's Python environment (nvidiaenv on
eagle7); upload and fetch only the standard library.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from submit_learning_curve import (MODEL_SUBDIR, REPO, eval_lines, model_env, prediction_name, read_experiments,  # noqa: E402
                                   split_json)


def env_for(e, run_dir):
    env = dict(os.environ, FFNOML_RUN_DIR=run_dir, FFNOML_TRAIN_SPLIT=split_json(e["train"]),
               FFNOML_VAL_SPLIT=split_json(e["val"]), FFNOML_ACTIVE_ATOMS=e["atoms"], FFNOML_NUM_EPOCHS=e["epochs"],
               **model_env(e))
    env.pop("FFNOML_PACKED_DATA", None)
    if e["data_dir"]:
        env["FFNOML_DATA_DIR"] = e["data_dir"]
    return env


def run_logged(cmd, env, log):
    print(f"  {' '.join(cmd)}  (log: {log})", flush=True)
    with open(log, "a") as f:
        subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def prepare(experiments):
    for name, e in experiments.items():
        stage = os.path.abspath(e["stage"])
        run = os.path.join(stage, name)
        os.makedirs(os.path.join(run, "IO"), exist_ok=True)
        env = env_for(e, run)
        files = subprocess.run([sys.executable, "-c", "import config; print(config.TRAIN_FILE); print(config.TEST_FILE)"],
                               cwd=REPO, env=env, check=True, stdout=subprocess.PIPE, universal_newlines=True).stdout.split()
        print(f"{name}: datasets", flush=True)
        if not all(os.path.exists(p) for p in files):
            run_logged([sys.executable, "pipeline.py", "--build"], env, os.path.join(run, "build.txt"))
        prepared = {"datasets": {os.path.relpath(p, run): os.path.getsize(p) for p in files},
                    "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, stdout=subprocess.PIPE,
                                             universal_newlines=True).stdout.strip()}
        with open(os.path.join(run, "prepared.json"), "w") as f:
            json.dump(prepared, f, indent=1)
        with open(os.path.join(run, "eval_snapshots.txt"), "w") as f:
            f.writelines(eval_lines(e))

        solving = os.path.join(stage, "solving_sets")
        os.makedirs(solving, exist_ok=True)
        for _, (sim, snap) in e["eval"]:
            pname = prediction_name(sim, snap)
            target = os.path.join(solving, f"3D_sim_predict_{pname}.hdf5")
            if os.path.exists(target):
                continue
            existing = os.path.join(e["solving_sets_from"], f"3D_sim_predict_{pname}.hdf5") if e["solving_sets_from"] else ""
            if existing and os.path.exists(existing):
                os.symlink(os.path.realpath(existing), target)
                print(f"  solving set {pname}: linked from {os.path.realpath(existing)}", flush=True)
                continue
            build = os.path.join(stage, "solving_sets_build")
            os.makedirs(os.path.join(build, MODEL_SUBDIR), exist_ok=True)   # pipeline.py writes there, does not create it
            run_logged([sys.executable, "pipeline.py", "--buildforpredict", "--predname", pname], env_for(e, build),
                       os.path.join(stage, "solving_sets_build.txt"))
            os.replace(os.path.join(build, MODEL_SUBDIR, f"3D_sim_predict_{pname}.hdf5"), target)
        print(f"  {run}: {', '.join(f'{k} ({v / 2**30:.1f} GiB)' for k, v in prepared['datasets'].items())}", flush=True)


def ssh_opts():
    if "LC_SSH_OPTS" in os.environ:
        return shlex.split(os.environ["LC_SSH_OPTS"])
    sock = os.path.expanduser("~/.ssh/olivia.sock")
    return ["-o", f"ControlPath={sock}", "-o", "BatchMode=yes"] if os.path.exists(sock) else []


def remote_sizes(host, paths):
    """{path: size} of the files under paths on host (directories are walked, links followed; absent paths are left
    out)."""
    script = "find -L " + " ".join(shlex.quote(p) for p in paths) + " -type f -printf '%s %p\\n' 2>/dev/null; true"
    out = subprocess.run(["ssh"] + ssh_opts() + [host, script], check=True, stdout=subprocess.PIPE,
                         universal_newlines=True).stdout
    return {line.split(" ", 1)[1]: int(line.split(" ", 1)[0]) for line in out.splitlines() if " " in line}


def scp(sources, dest, host, *, to_remote):
    if not sources:
        return
    print(f"  scp {len(sources)} file(s) -> {dest}", flush=True)
    if to_remote:
        cmd = ["scp", "-q"] + ssh_opts() + list(sources) + [f"{host}:{dest}/"]
    else:
        cmd = ["scp", "-q"] + ssh_opts() + [f"{host}:{s}" for s in sources] + [f"{dest}/"]
    subprocess.run(cmd, check=True)


def upload(experiments, host):
    for name, e in experiments.items():
        stage, root = os.path.abspath(e["stage"]), e["root"]
        run, remote_run = os.path.join(stage, name), f"{root}/{name}"
        with open(os.path.join(run, "prepared.json")) as f:
            prepared = json.load(f)
        solving = [os.path.join(stage, "solving_sets", f"3D_sim_predict_{prediction_name(*s)}.hdf5") for _, s in e["eval"]]
        subprocess.run(["ssh"] + ssh_opts() + [host, f"mkdir -p {shlex.quote(remote_run)}/IO {shlex.quote(root)}/solving_sets"],
                       check=True)
        have = remote_sizes(host, [f"{remote_run}/IO", f"{root}/solving_sets"])
        print(f"{name}: upload to {host}:{remote_run}", flush=True)
        scp([os.path.join(run, rel) for rel, size in prepared["datasets"].items() if have.get(f"{remote_run}/{rel}") != size],
            f"{remote_run}/IO", host, to_remote=True)
        scp([p for p in solving if have.get(f"{root}/solving_sets/{os.path.basename(p)}") != os.path.getsize(p)],
            f"{root}/solving_sets", host, to_remote=True)
        scp([os.path.join(run, "prepared.json")], remote_run, host, to_remote=True)   # last: the job's check of a complete copy


def fetch(experiments, host):
    for name, e in experiments.items():
        stage, remote_run = os.path.abspath(e["stage"]), f"{e['root']}/{name}"
        run = os.path.join(stage, name)
        model = os.path.join(run, MODEL_SUBDIR)
        os.makedirs(model, exist_ok=True)
        there = remote_sizes(host, [remote_run])
        print(f"{name}: fetch from {host}:{remote_run}", flush=True)
        wanted = [p for p in there if os.path.dirname(p) == f"{remote_run}/{MODEL_SUBDIR}" and
                  (os.path.basename(p) in (f"3D_sim_train_{e['atoms']}.pt",) or
                   (os.path.basename(p).startswith("output_3D_sim_s5_") and p.endswith(f"_{e['atoms']}.hdf5")))]
        logs = [p for p in there if os.path.dirname(p) == remote_run and os.path.basename(p).startswith(("slurm-", "output-"))]
        local = lambda p, d: os.path.join(d, os.path.basename(p))   # noqa: E731
        scp([p for p in wanted if not os.path.exists(local(p, model)) or os.path.getsize(local(p, model)) != there[p]],
            model, host, to_remote=False)
        scp([p for p in logs if not os.path.exists(local(p, run)) or os.path.getsize(local(p, run)) != there[p]],
            run, host, to_remote=False)
        checkpoint = os.path.exists(os.path.join(model, f"3D_sim_train_{e['atoms']}.pt"))
        print(f"  {len([p for p in wanted if 'output_3D' in p])} of {len(e['eval'])} predictions, "
              f"checkpoint {'there' if checkpoint else 'missing'}", flush=True)


def predict(experiments, batched_slabs=None):
    """--fsdppredict of every scoring snapshot from the experiment's best checkpoint, as the job's last step does, for a
    job stopped by its time limit before it got there. One torchrun on this host's visible GPUs (CUDA_VISIBLE_DEVICES),
    or, with batched_slabs, scripts/predict_batched.py for cubes whose activations do not fit in them."""
    gpus = len([g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if g.strip()])
    torchrun = os.path.join(os.path.dirname(sys.executable), "torchrun")
    for name, e in experiments.items():
        stage = os.path.abspath(e["stage"])
        run = os.path.join(stage, name)
        model = os.path.join(run, MODEL_SUBDIR)
        if not os.path.exists(os.path.join(model, f"3D_sim_train_{e['atoms']}.pt")):
            raise FileNotFoundError(f"{name}: no checkpoint in {model}; fetch it first")
        for role, (sim, snap) in e["eval"]:
            pred = prediction_name(sim, snap)
            if os.path.exists(os.path.join(model, f"output_3D_sim_s5_{pred}_FFNO3D_{e['atoms']}.hdf5")):
                print(f"{name}: {pred} ({role}) already predicted", flush=True)
                continue
            link = os.path.join(model, f"3D_sim_predict_{pred}.hdf5")
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(os.path.join(stage, "solving_sets", f"3D_sim_predict_{pred}.hdf5"), link)
            print(f"{name}: predicting {pred} ({role}) on {gpus} GPU(s)", flush=True)
            if batched_slabs:
                cmd = [sys.executable, "scripts/predict_batched.py", os.path.join(model, f"3D_sim_train_{e['atoms']}.pt"), link,
                       os.path.join(model, f"output_3D_sim_s5_{pred}_FFNO3D_{e['atoms']}.hdf5"),
                       "--vertical-norm-slabs", str(batched_slabs)]
            else:
                cmd = [torchrun, "--standalone", f"--nproc_per_node={gpus}", "pipeline.py", "--fsdppredict", "--predname", pred]
            run_logged(cmd, env_for(e, run), os.path.join(run, "predict_eagle7.log"))


def score(experiments):
    for name, e in experiments.items():
        run = os.path.join(os.path.abspath(e["stage"]), name)
        with open(os.path.join(run, "eval_snapshots.txt"), "w") as f:
            f.writelines(eval_lines(e))
        cmd = [sys.executable, "scripts/population_error_diagnostic.py", "--score-only", "--run-dir", os.path.join(run, "eval"),
               "--model-dir", os.path.join(run, MODEL_SUBDIR), "--label", name, "--atoms", e["atoms"],
               "--snapshots-file", os.path.join(run, "eval_snapshots.txt")]
        if e["data_dir"]:
            cmd += ["--data-dir", e["data_dir"]]
        print(f"{name}: scoring", flush=True)
        subprocess.run(cmd, cwd=REPO, env=env_for(e, run), check=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("step", choices=["prepare", "upload", "fetch", "predict", "score"])
    cli.add_argument("input")
    cli.add_argument("experiments", nargs="*", help="sections (default: all)")
    cli.add_argument("--host", default="harshm@olivia.sigma2.no", help="where upload and fetch copy to and from")
    cli.add_argument("--batched", type=int, metavar="SLABS", help="predict: with scripts/predict_batched.py, its vertical "
                     "norm statistics over SLABS slabs as an FSDP run on SLABS ranks (32 on Olivia's 8 nodes)")
    args = cli.parse_args()
    experiments = read_experiments(args.input, args.experiments)
    if args.step == "prepare":
        prepare(experiments)
    elif args.step == "upload":
        upload(experiments, args.host)
    elif args.step == "fetch":
        fetch(experiments, args.host)
    elif args.step == "predict":
        predict(experiments, args.batched)
    else:
        score(experiments)


if __name__ == "__main__":
    main()
