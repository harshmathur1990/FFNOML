"""Compare ffno_batched_backend.py with populations already computed by the full-domain FSDP route, on the same input.

    python scripts/compare_batched_with_reference.py SOLVE.hdf5 REFERENCE.hdf5 CHECKPOINT.pt [--slabs N] [--out OUT.json]

SOLVE is a 3D_sim_predict_* file (the exact network inputs) and REFERENCE the output_3D_sim_s5_* made from it. Differences
are in dex of the populations, per level, with the outer columns reported apart from the interior."""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ffno"))
from ffno_batched_backend import BatchedFFNOBackend
from ffno_model_factory import create_ffno3d

parser = argparse.ArgumentParser()
parser.add_argument("solve")
parser.add_argument("reference")
parser.add_argument("checkpoint")
parser.add_argument("--slabs", type=int, nargs="+", default=[1])
parser.add_argument("--chunk-depth", type=int, default=16)
parser.add_argument("--chunk-rows", type=int, default=16)
parser.add_argument("--vertical-chunk", type=int, default=1024)
parser.add_argument("--out", default=None)
arguments = parser.parse_args()

with h5py.File(arguments.solve) as f:
    features = f["inputs"][0]
    z = f["z_scale"][0]
    dx = float(f["dx"][0])
    dy = float(f["dy"][0])
with h5py.File(arguments.reference) as f:
    reference = np.log10(f["nlte_populations"][...])            # (nx, ny, nz, levels)
    print("reference attributes", dict(f.attrs), flush=True)

results = []
for slabs in arguments.slabs:
    backend = BatchedFFNOBackend(arguments.checkpoint, create_ffno3d, chunk_depth=arguments.chunk_depth,
                                 chunk_rows=arguments.chunk_rows, vertical_chunk=arguments.vertical_chunk,
                                 vertical_norm_slabs=slabs)
    print("checkpoint epoch", backend.epoch, flush=True)
    start = time.time()
    predicted = backend.predict_log10(features, z, dx, dy).permute(2, 3, 1, 0).numpy()  # (nx, ny, nz, levels)
    seconds = time.time() - start
    difference = np.abs(predicted.astype(np.float64) - reference)
    nx, ny = difference.shape[:2]
    interior = difference[10:nx - 10, 10:ny - 10]
    row = dict(slabs=slabs, seconds=seconds, shape=list(difference.shape),
               max_dex=float(difference.max()), p99_dex=float(np.percentile(difference, 99)),
               median_dex=float(np.median(difference)), interior_max_dex=float(interior.max()),
               max_dex_by_level=[float(difference[..., l].max()) for l in range(difference.shape[-1])],
               median_dex_by_level=[float(np.median(difference[..., l])) for l in range(difference.shape[-1])],
               gpu_peak_gb=[torch.cuda.max_memory_allocated(i) / 1e9 for i in range(torch.cuda.device_count())])
    results.append(row)
    print(json.dumps(row), flush=True)
    del backend, predicted, difference, interior
    torch.cuda.empty_cache()
if arguments.out:
    with open(arguments.out, "w") as f:
        json.dump(results, f, indent=1)
