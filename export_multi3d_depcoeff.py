#!/usr/bin/env python3

import argparse
import os

import h5py
import numpy as np

from config import ATOM_CONFIG, SIMULATIONS
from pipeline import load_multi3d_blocks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export real Multi3D departure coefficients in the same HDF5 layout used by FFNONet predictions."
    )
    parser.add_argument("--sim", default="en024048_hion")
    parser.add_argument("--snap", default="385")
    parser.add_argument("--atoms", nargs="+", default=["H"])
    parser.add_argument(
        "--save-path",
        default="training/original_multi3d_dep_en024048_hion_385.h5",
    )
    return parser.parse_args()


def build_entry(sim_name, snap, atoms):
    base_path = SIMULATIONS[sim_name]["base_path"]
    return {
        "MULTI3D_PATHS": [
            os.path.join(base_path, str(snap), ATOM_CONFIG[a]["subdir"]) for a in atoms
        ],
        "MULTI3D_ATMOS": os.path.join(base_path, str(snap), "atm3d"),
        "MESH": os.path.join(base_path, str(snap), "mesh"),
    }


def compute_dep(nlte, lte, eps=1e-30):
    return (nlte + eps) / (lte + eps)


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    entry = build_entry(args.sim, args.snap, args.atoms)
    data = load_multi3d_blocks([entry])

    lte = np.asarray(data["lte_list"][0], dtype=np.float32)
    nlte = np.asarray(data["nlte_list"][0], dtype=np.float32)
    z_scale = np.asarray(data["z_list"][0], dtype=np.float32)

    dep = compute_dep(nlte, lte).astype(np.float32, copy=False)

    print("Writing reference departure coefficients...")
    print("sim      =", args.sim)
    print("snap     =", args.snap)
    print("atoms    =", ",".join(args.atoms))
    print("lte shape =", lte.shape)
    print("nlte shape =", nlte.shape)
    print("dep shape =", dep.shape)
    print("save_path =", args.save_path)

    with h5py.File(args.save_path, "w") as f:
        d = f.create_dataset(
            "departure_coefficients",
            data=np.asfortranarray(dep),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        d.attrs["depth_scale_type"] = "z"
        f.create_dataset(
            "z_scale",
            data=z_scale,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        f.attrs["epoch"] = -1
        f.attrs["sim_name"] = args.sim
        f.attrs["snap"] = str(args.snap)
        f.attrs["atoms"] = ",".join(args.atoms)


if __name__ == "__main__":
    main()
