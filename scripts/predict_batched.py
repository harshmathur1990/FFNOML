"""--fsdppredict for GPUs too small to hold a cube's activations: the same checkpoint, solving set and output file
(output_3D_sim_s5_<name>_FFNO3D_<atoms>.hdf5), computed by ffno_batched_backend.BatchedFFNOBackend from host RAM on the
visible GPUs instead of by FSDP.

    python scripts/predict_batched.py CHECKPOINT SOLVING_SET OUTPUT [--vertical-norm-slabs N]

The FSDP route computes the vertical branch's out_gn statistics per rank slab, so its result depends on the number of
ranks; --vertical-norm-slabs N reproduces an FSDP run on N ranks (32 on Olivia's 8 nodes), 1 the single volume statistic.
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffno_batched_backend import BatchedFFNOBackend  # noqa: E402
from ffno_model_factory import create_ffno3d  # noqa: E402


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("checkpoint")
    cli.add_argument("solving_set")
    cli.add_argument("output")
    cli.add_argument("--vertical-norm-slabs", type=int, default=32)
    args = cli.parse_args()

    backend = BatchedFFNOBackend(args.checkpoint, create_ffno3d, vertical_norm_slabs=args.vertical_norm_slabs,
                                 log=lambda m: print(m, flush=True))
    with h5py.File(args.solving_set, "r") as f:
        features = f["inputs"][0].astype(np.float32)      # (6, D, nx, ny), the training's units
        z_scale = f["z_scale"][0].astype(np.float32)      # (D, nx, ny)
        dx, dy = float(np.ravel(f["dx"][...])[0]), float(np.ravel(f["dy"][...])[0])
    log10_populations = backend.predict_log10(features, z_scale, dx, dy).numpy()   # (levels, D, nx, ny)

    # as ffno_predict_populations_distributed_full writes it: 10**log, (nx, ny, D, levels) in Fortran order
    dep = np.transpose(np.power(np.float32(10.0), log10_populations), (2, 3, 1, 0)).astype(np.float32, copy=False)
    tmp = args.output + ".tmp"
    with h5py.File(tmp, "w") as f:
        d = f.create_dataset("nlte_populations", data=np.asfortranarray(dep), compression="gzip", compression_opts=4,
                             shuffle=True)
        d.attrs["depth_scale_type"] = "z"
        f.create_dataset("z_scale", data=z_scale, compression="gzip", compression_opts=4, shuffle=True)
        f.attrs["epoch"] = int(backend.epoch if backend.epoch is not None else -1)
        f.attrs["vertical_norm_slabs"] = args.vertical_norm_slabs
    os.replace(tmp, args.output)
    print(f"wrote {args.output}: populations {dep.shape}, range {dep.min():.3e} to {dep.max():.3e}", flush=True)


if __name__ == "__main__":
    main()
