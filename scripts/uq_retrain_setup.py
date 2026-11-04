"""Set up one branch of the controlled fine-tuning experiment (claude-next-steps-uq step 4).

A branch is a run directory for `FFNOML_RUN_DIR=<dir> torchrun pipeline.py --train --resume --bestpath`:

    <dir>/IO/<TRAIN_FILE name>   HDF5 of external links: the production training groups plus one
                                 group per --add snapshot, with the production normalization copied
                                 in (the added patches are built with stat_file = production train
                                 file, so normalization is fixed across branches)
    <dir>/IO/<TEST_FILE name>    symlink to the production validation set (checkpoint selection)
    <dir>/training_.../3D_sim_train_<atom>.pt   copy of the production checkpoint (the common start)
    <dir>/manifest.json          branch definition, sources and checksums

Added-snapshot patch files are cached in <root>/data/<sim>_<snap>_<atom>.hdf5.

Usage:
    FFNOML_ACTIVE_ATOMS=H python scripts/uq_retrain_setup.py --branch B_nw1050 --add nw012023/1050
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import h5py

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--branch", required=True)
    p.add_argument("--add", nargs="*", default=[], help="sim/snap to add to the training set")
    p.add_argument("--root", default=os.path.join(REPO, "uq", "retrain"))
    p.add_argument("--notes", default="")
    return p.parse_args()


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_extra(sim, snap, out):
    """Patch file for one snapshot with the production normalization."""
    if os.path.exists(out):
        return
    from pipeline import load_multi3d_blocks
    from FFNONet import build_dataset_ffno
    blocks = load_multi3d_blocks(config.build_multi3d_entries({sim: [snap]}))
    tmp = out + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    build_dataset_ffno(
        blocks["temp_list"], blocks["vx_list"], blocks["vy_list"], blocks["vz_list"], blocks["ne_list"],
        blocks["lte_list"], blocks["nlte_list"], blocks["rho_list"], blocks["z_list"], blocks["dx_list"],
        blocks["dy_list"], save_path=tmp, chi=config.chi, lines=config.lines, wave=config.wave,
        levels=config.levels, patch=config.PATCH, stride=config.STRIDE, scales=config.SCALES,
        stat_file=config.TRAIN_FILE,
    )
    os.replace(tmp, out)


def main():
    args = parse_args()
    atoms = config.atoms_tag
    run = os.path.join(args.root, f"{atoms}_{args.branch}")
    io_dir = os.path.join(run, "IO")
    model_dir = os.path.join(run, os.path.basename(config.MODEL_DIR.rstrip(os.sep)))
    os.makedirs(io_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(args.root, "data"), exist_ok=True)

    extras = []
    for cs in args.add:
        sim, snap = cs.split("/")
        path = os.path.join(args.root, "data", f"{sim}_{snap}_{atoms}.hdf5")
        build_extra(sim, snap, path)
        extras.append((cs, path))

    train_link = os.path.join(io_dir, os.path.basename(config.TRAIN_FILE))
    if os.path.exists(train_link):
        os.remove(train_link)
    with h5py.File(config.TRAIN_FILE, "r") as src, h5py.File(train_link, "w") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in src["patch_dataset_names"][...]]
        for k in ("mean_X", "std_X", "mean_Y", "std_Y"):
            f[k] = src[k][...]
        f.attrs.update(dict(src.attrs))
        n_total = int(src.attrs["N"])
        for name in names:
            f[name] = h5py.ExternalLink(os.path.abspath(config.TRAIN_FILE), name)
        for i, (cs, path) in enumerate(extras):
            with h5py.File(path, "r") as e:
                enames = [n.decode() if isinstance(n, bytes) else str(n) for n in e["patch_dataset_names"][...]]
                for k in ("mean_X", "std_X", "mean_Y", "std_Y"):
                    if not (e[k][...] == f[k][...]).all():
                        raise RuntimeError(f"{path}: normalization differs from the production training set")
                for en in enames:
                    new = f"added_{i + 1}_{en}"
                    f[new] = h5py.ExternalLink(os.path.abspath(path), en)
                    names.append(new)
                    n_total += int(e[en]["inputs"].shape[0])
        f["patch_dataset_names"] = [n.encode() for n in names]
        f.attrs["N"] = n_total
        f.attrs["n_patch_datasets"] = len(names)
        f.attrs["added_snapshots"] = json.dumps([cs for cs, _ in extras])

    test_link = os.path.join(io_dir, os.path.basename(config.TEST_FILE))
    if not os.path.lexists(test_link):
        os.symlink(os.path.abspath(config.TEST_FILE), test_link)

    ckpt = os.path.join(model_dir, os.path.basename(config.MODEL_FILE))
    if not os.path.exists(ckpt):
        shutil.copy2(config.MODEL_FILE, ckpt)

    manifest = dict(
        branch=args.branch, atoms=atoms, added=[cs for cs, _ in extras], train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT, n_train_patches=n_total, production_train_file=config.TRAIN_FILE,
        production_checkpoint=config.MODEL_FILE, checkpoint_sha256=sha256(config.MODEL_FILE),
        extra_files={cs: dict(path=p, sha256=sha256(p)) for cs, p in extras}, notes=args.notes,
    )
    with open(os.path.join(run, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"{run}: {n_total} training patches, added {manifest['added']}")


if __name__ == "__main__":
    main()
