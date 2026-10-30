#!/usr/bin/env python3
"""Compare the inversion's FFNO populations with a direct FFNONet.py prediction.

The forward run (FFNOInversion.jl) reconstructs the column-HSE atmosphere from
the log10tau500 input with HSEMHS and predicts populations through the FSDP
service. It writes both to ``forward_atmosphere.h5``. This script takes that
same HSE atmosphere and predicts populations with FFNONet.py's own route
(build_solving_set_ffno + ffno_predict_populations_distributed_full), then
compares the two. Both use the same checkpoints, so the populations should
agree to float32 round-off.

    # 1. solving set (training conventions: log10 T, km/s, z in Mm)
    python scripts/compare_service_with_ffnonet.py build \\
        --atmosphere outputs/forward_atmosphere.h5 --out compare/solve.h5
    # 2. direct prediction, once per species
    torchrun --nproc_per_node=4 scripts/compare_service_with_ffnonet.py predict \\
        --solve compare/solve.h5 --species H --checkpoint .../3D_sim_train_H.pt \\
        --out compare/direct_H.h5
    # 3. comparison
    python scripts/compare_service_with_ffnonet.py compare \\
        --atmosphere outputs/forward_atmosphere.h5 --direct H=compare/direct_H.h5 CA=compare/direct_CA.h5
"""

import argparse
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

# MULTI3D atmospheres (the training data) store velocities in km/s.
VELOCITY_M_S_PER_KM_S = 1.0e3


def read_hse_atmosphere(path):
    """forward_atmosphere.h5 fields as FFNONet arrays [nx, ny, nz] (top first), SI units."""
    with h5py.File(path, "r") as f:
        order = f.attrs.get("axis_order", b"")
        order = order.decode() if isinstance(order, bytes) else str(order)
        if not order.startswith("time,z,y,x"):
            raise ValueError(f"{path}: unexpected axis_order {order!r}")
        # Julia writes (time, z, y, x); h5py reads the reverse, (x, y, z, time).
        field = {name: np.asarray(f[name][..., 0], dtype=np.float64)
                 for name in ("temperature", "vx", "vy", "vz", "rho", "ne", "z", "pgas")}
        x = np.asarray(f["x_m"][...], dtype=np.float64)
        y = np.asarray(f["y_m"][...], dtype=np.float64)
        populations = {name: np.asarray(f["populations"][name][...], dtype=np.float64)
                       for name in f["populations"]} if "populations" in f else {}
    for name in ("temperature", "rho", "ne"):
        if not (field[name] > 0).all():
            raise ValueError(f"{path}: {name} must be positive")
    if not (np.diff(field["z"], axis=2) < 0).all():
        raise ValueError(f"{path}: z must strictly decrease with depth index (top first)")
    return field, uniform_spacing(x, "x"), uniform_spacing(y, "y"), populations


def uniform_spacing(values, name):
    steps = np.diff(values)
    if steps.size == 0 or np.max(np.abs(steps - steps.mean())) > 1e-6 * abs(steps.mean()):
        raise ValueError(f"{name} spacing must be uniform")
    return float(abs(steps.mean()))


def build(arguments):
    from FFNONet import build_solving_set_ffno

    field, dx, dy, _ = read_hse_atmosphere(arguments.atmosphere)
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_solving_set_ffno(
        rho=field["rho"],
        z_scale=field["z"],                       # metres; FFNONet converts to Mm
        temp=field["temperature"],                # FFNONet takes log10
        vx=field["vx"] / VELOCITY_M_S_PER_KM_S,
        vy=field["vy"] / VELOCITY_M_S_PER_KM_S,
        vz=field["vz"] / VELOCITY_M_S_PER_KM_S,
        ne=field["ne"],
        dx=dx,
        dy=dy,
        save_path=str(out),
    )
    nx, ny, nz = field["temperature"].shape
    print(f"solving set {out}: nx={nx} ny={ny} nz={nz} dx={dx} dy={dy}")


def predict(arguments):
    import torch
    import torch.distributed as dist
    import config
    from FFNONet import ffno_predict_populations_distributed_full

    atom = config.ATOM_CONFIG[arguments.species]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    try:
        ffno_predict_populations_distributed_full(
            model=config.MODEL,
            checkpoint_path=arguments.checkpoint,
            solve_h5=arguments.solve,
            save_path=arguments.out,
            model_config=config.MODEL_CONFIG,
            lines=[atom["lines"]],
            wave=[atom["wave"]],
            chi=[atom["chi"]],
            levels=[atom["levels"]],
            atom_names=[arguments.species],
            cuda=True,
        )
    finally:
        dist.destroy_process_group()


def compare(arguments):
    _, _, _, service = read_hse_atmosphere(arguments.atmosphere)
    report = {}
    worst = 0.0
    for entry in arguments.direct:
        species, path = entry.split("=", 1)
        if species not in service:
            raise KeyError(f"{arguments.atmosphere} has no populations/{species}")
        with h5py.File(path, "r") as f:
            direct = np.asarray(f["nlte_populations"][...], dtype=np.float64)   # [nx, ny, nz, level]
        mine = service[species]                                                  # [nx, ny, nz, level]
        if direct.shape != mine.shape:
            raise ValueError(f"{species}: direct {direct.shape} vs service {mine.shape}")
        relative = np.abs(mine - direct) / np.abs(direct)
        dex = np.abs(np.log10(mine) - np.log10(direct))
        levels = []
        for level in range(direct.shape[-1]):
            r = relative[..., level]
            levels.append(dict(level=level + 1,
                               max_relative=float(r.max()),
                               median_relative=float(np.median(r)),
                               p99_relative=float(np.quantile(r, 0.99)),
                               max_dex=float(dex[..., level].max())))
        report[species] = levels
        worst = max(worst, float(relative.max()))
        print(f"{species}: max relative difference {relative.max():.3e}, "
              f"median {np.median(relative):.3e}, max |dlog10| {dex.max():.3e}")
        for item in levels:
            print("   level {level}: max {max_relative:.3e}  p99 {p99_relative:.3e}  "
                  "median {median_relative:.3e}".format(**item))
    if arguments.report:
        Path(arguments.report).write_text(json.dumps(report, indent=2))
    verdict = "AGREE" if worst <= arguments.tolerance else "DIFFER"
    print(f"SERVICE_VS_FFNONET_{verdict} max_relative={worst:.3e} tolerance={arguments.tolerance:.1e}")
    return 0 if verdict == "AGREE" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("build", help="write an FFNONet solving set from forward_atmosphere.h5")
    p.add_argument("--atmosphere", required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser("predict", help="FFNONet distributed full-volume prediction (run with torchrun)")
    p.add_argument("--solve", required=True)
    p.add_argument("--species", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser("compare", help="compare service and direct populations")
    p.add_argument("--atmosphere", required=True)
    p.add_argument("--direct", nargs="+", required=True, metavar="SPECIES=FILE")
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.add_argument("--report")
    arguments = parser.parse_args()
    if arguments.command == "build":
        build(arguments)
    elif arguments.command == "predict":
        predict(arguments)
    else:
        sys.exit(compare(arguments))


if __name__ == "__main__":
    main()
