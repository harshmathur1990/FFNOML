"""Checks of learning-curve experiments on eagle7 before anything is uploaded or submitted, so no Olivia hours go to a
job that would fail or train the wrong thing:

    python scripts/preflight_learning_curve.py scripts/learning_curve.ini [EXPERIMENT ...]   # on eagle7, nvidiaenv

1. job environment: experiment.env as submit_learning_curve.py writes it, sourced and exported as
   learning_curve_train.sbatch does, gives config.py the experiment's atoms, splits, epochs, head and loss weight, and
   the datasets prepare recorded in prepared.json;
2. datasets: present with the prepared sizes, targets log10 n in m^-3; for the level-fraction head, the targets' level
   sum against rho / mass from the inputs, the total the head starts from (it must sit near 0 dex);
3. model and loss at production size, on one training patch on the CPU: the head's channels, the total at
   initialisation equal to rho / mass, a finite loss with its species-total component, finite gradients everywhere.

Prints each check and exits non-zero if any fails.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from submit_learning_curve import MODEL_SUBDIR, REPO, read_experiments, write_job_files  # noqa: E402

FAILED = []


def check(ok, what):
    print(f"  [{'ok' if ok else 'FAIL'}] {what}", flush=True)
    if not ok:
        FAILED.append(what)
    return ok


def job_config(e, run):
    """config.py's view under the job's environment: experiment.env, then the sbatch's own export lines"""
    sbatch = open(os.path.join(REPO, "scripts", "learning_curve_train.sbatch")).read().splitlines()
    exports = [line for line in sbatch if line.startswith("export FFNOML_")]
    probe = ("import config, json; print(json.dumps(dict(atoms=config.ACTIVE_ATOMS, train=config.TRAIN_SPLIT, "
             "val=config.VAL_SPLIT, epochs=config.NUM_EPOCHS, output_mode=config.MODEL_CONFIG['output_mode'], "
             "species=config.MODEL_CONFIG.get('species'), lam_total=config.LAM_TOTAL, train_file=config.TRAIN_FILE, "
             "test_file=config.TEST_FILE, model_dir=config.MODEL_DIR)))")
    with tempfile.TemporaryDirectory() as tmp:
        write_job_files(e, tmp, os.path.abspath(e["root"]))
        script = "\n".join(["set -o errexit -o nounset", f"source {tmp}/experiment.env", f"FFNOML_RUN_DIR={run}",
                            "export FFNOML_RUN_DIR", *exports, f"cd {REPO}", f"{sys.executable} -c \"{probe}\""])
        env = {k: v for k, v in os.environ.items() if not k.startswith("FFNOML_")}
        if e["data_dir"]:
            env["FFNOML_DATA_DIR"] = e["data_dir"]
        out = subprocess.run(["bash", "-c", script], env=env, check=True, stdout=subprocess.PIPE,
                             universal_newlines=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def split_of(pairs):
    split = {}
    for sim, snap in pairs:
        split.setdefault(sim, []).append(snap)
    return split


def preflight(name, e):
    run = os.path.join(os.path.abspath(e["stage"]), name)
    print(f"{name}: {run}", flush=True)
    try:
        with open(os.path.join(run, "prepared.json")) as f:
            prepared = json.load(f)
    except OSError:
        check(False, "prepared.json (run learning_curve_eagle7.py prepare first)")
        return

    # 1. job environment
    c = job_config(e, run)
    check(c["atoms"] == [e["atoms"]], f"atoms {c['atoms']}")
    check(c["train"] == split_of(e["train"]) and c["val"] == split_of(e["val"]), f"splits train {c['train']} val {c['val']}")
    check(c["epochs"] == int(e["epochs"]), f"epochs {c['epochs']}")
    check(c["output_mode"] == e["output_mode"], f"head {c['output_mode']}")
    check(math.isclose(c["lam_total"], float(e["lam_total"])), f"lam_total {c['lam_total']}")
    if e["output_mode"] == "level_fraction":
        check(c["species"] is not None and len(c["species"]) == 1,
              f"species {c['species']} (mass {10 ** c['species'][0]['log10_particle_mass']:.4e} kg)" if c["species"] else "species")
    datasets = sorted(os.path.join(run, rel) for rel in prepared["datasets"])
    check(sorted([c["train_file"], c["test_file"]]) == datasets, "the job reads the prepared datasets")
    check(os.path.normpath(c["model_dir"]) == os.path.join(run, MODEL_SUBDIR), f"model dir {c['model_dir']}")

    # 2. datasets
    import h5py
    import numpy as np
    for rel, size in prepared["datasets"].items():
        path = os.path.join(run, rel)
        check(os.path.exists(path) and os.path.getsize(path) == size, f"{rel}: {size / 2**30:.2f} GiB as prepared")
    with h5py.File(c["train_file"], "r") as f:
        check(f.attrs["target_representation"] == "log10_nlte_population_m-3", f"targets {f.attrs['target_representation']}")
        stats = {k: np.asarray(f[k], np.float32) for k in ("mean_X", "std_X", "mean_Y", "std_Y")}
        group = f[sorted(k for k in f.keys() if isinstance(f[k], h5py.Group))[0]]
        i = int(np.argmax(np.asarray(group["scale"]) == 1))                      # a full-resolution patch
        patch = {k: np.asarray(group[k][i:i + 1]) for k in ("inputs", "targets", "z_scale", "dx", "dy", "log_lte_populations")}
    if e["output_mode"] == "level_fraction":
        log10_rho = patch["inputs"][0, 5] * stats["std_X"][5] + stats["mean_X"][5]
        log10_n = patch["targets"][0].astype(np.float64) * stats["std_Y"][:, None, None, None] + stats["mean_Y"][:, None, None, None]
        total = np.log10(np.sum(10.0 ** log10_n, 0))
        s = total - (log10_rho - c["species"][0]["log10_particle_mass"])
        check(abs(np.median(s)) < 0.1 and np.abs(s).max() < 6.0,
              f"targets' total against rho / mass on a training patch: median {np.median(s):+.3f} dex, "
              f"range {s.min():+.2f} to {s.max():+.2f} (the head reaches 6 dex)")

    # 3. model and loss at production size
    import torch
    env_keys = {"FFNOML_RUN_DIR": run, "FFNOML_TRAIN_SPLIT": json.dumps(c["train"]), "FFNOML_VAL_SPLIT": json.dumps(c["val"]),
                "FFNOML_ACTIVE_ATOMS": e["atoms"], "FFNOML_OUTPUT_MODE": e["output_mode"], "FFNOML_LAM_TOTAL": e["lam_total"]}
    os.environ.update(env_keys)
    sys.path.insert(0, REPO)
    for module in [m for m in sys.modules if m == "config"]:
        del sys.modules[module]
    import config
    from model_builder import ModelBuilder
    from train_utils import compute_loss, flatten_columns_logb
    builder = ModelBuilder(config.MODEL, model_config=config.MODEL_CONFIG, chi=config.chi, lines=config.lines,
                           wave=config.wave, levels=config.levels, atom_names=config.atom_names, device="cpu",
                           lam_total=config.LAM_TOTAL, **stats)
    torch.manual_seed(0)
    model = builder.build_model(wrap_fsdp=False)
    _, _, loss_fn = builder.build_training_components(model)
    head = sum(config.levels) + (len(config.levels) if e["output_mode"] == "level_fraction" else 0)
    check(model.proj2.out_channels == head, f"head channels {model.proj2.out_channels}")
    check(math.isclose(loss_fn.lam_total, float(e["lam_total"])), f"loss weight on the total {loss_fn.lam_total}")
    print(f"  model: {sum(p.numel() for p in model.parameters()):,} parameters ({config.MODEL_CONFIG})", flush=True)
    model.train()
    x = torch.from_numpy(patch["inputs"])
    y = torch.from_numpy(patch["targets"])
    t0 = time.time()
    pred = model(x, torch.from_numpy(patch["z_scale"]), torch.from_numpy(patch["dx"]), torch.from_numpy(patch["dy"]))
    loss, comp = compute_loss(pred=flatten_columns_logb(pred), target=flatten_columns_logb(y), weight=None, loss_fn=loss_fn,
                              x=flatten_columns_logb(x), pred_full=pred, target_full=y,
                              log_lte=flatten_columns_logb(torch.from_numpy(patch["log_lte_populations"])))
    loss.backward()
    seconds = time.time() - t0
    check(bool(torch.isfinite(loss)), f"loss {loss.item():.4f} (levels {float(comp['data']):.4f}, total {float(comp['total']):.4f})")
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    check(all(g is not None and bool(torch.isfinite(g).all()) for g in grads), f"finite gradients on all {len(grads)} tensors "
          f"({seconds:.0f} s for one patch forward and backward on the CPU)")
    if e["output_mode"] == "level_fraction":
        with torch.no_grad():
            log10_n = pred.detach().double() * model.std_Y.double()[None, :, None, None, None] + model.mean_Y.double()[None, :, None, None, None]
            total = torch.logsumexp(log10_n * math.log(10.0), 1) / math.log(10.0)
            ref = (x[:, 5].double() * stats["std_X"][5] + stats["mean_X"][5]) - c["species"][0]["log10_particle_mass"]
        check(float((total - ref).abs().max()) < 1e-3, f"total at initialisation is rho / mass to {float((total - ref).abs().max()):.1e} dex")


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("input")
    cli.add_argument("experiments", nargs="*", help="sections (default: all)")
    args = cli.parse_args()
    for name, e in read_experiments(args.input, args.experiments).items():
        preflight(name, e)
    print("preflight: " + ("all checks passed" if not FAILED else f"{len(FAILED)} failed:\n  " + "\n  ".join(FAILED)))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
