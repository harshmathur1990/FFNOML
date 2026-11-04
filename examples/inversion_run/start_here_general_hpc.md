# Quick start on any HPC with GPUs: forward synthesis with the batched service

This guide runs `FFNOInversion.jl/scripts/invert.jl` as a batch job on one GPU node of any HPC system. The batched service predicts a whole cube on GPUs too small to hold it: the hidden states stay in host RAM and the GPUs process batches. This is not tiling. See the [package README](../../FFNOInversion.jl/README.md#batched-forward-service-small-gpus) for how it works.

**Forward runs on one node only.** The service is a single process, so it can't span nodes, and it refuses the VJP and JVP requests an inversion needs. An inversion needs the FSDP service, which has submission scripts only for [Olivia](START_HERE.md).

Other guides: [eagle7](start_here_eagle7.md), and [a GPU host without a scheduler](start_here_local_cluster_gpu.md).

This route has been run on eagle7, which has no scheduler. **The Slurm script below hasn't been run on any system**, so try it on a small cube first.

## 1. What the system needs

| Need | Notes |
| --- | --- |
| A GPU node | NVIDIA with CUDA. T4s (16 GB) have run 504 × 504 × 467. |
| Host RAM | About 760 bytes per cell for the hidden states: about 90 GB at 504 × 504 × 467. Julia's own memory comes on top and hasn't been measured for a large cube, so request well above this. |
| Julia 1.12 | MPI is off in this route, so no system MPI is needed. |
| Python 3.11 or newer | With CUDA `torch` and `numpy`. |
| A C++ compiler | For the Wittmann EOS library. |

Depth is the population mesh's 452 levels by default, whatever the input grid.

## 2. Install, once

Run this on a node of the same CPU type as the GPU nodes. Login nodes can have a different architecture (Olivia's do), and then the Julia depot and the Wittmann library must be built on the GPU nodes.

```bash
repo=/path/to/FFNOML
git clone https://github.com/harshmathur1990/FFNOML.git "$repo"   # or your own checkout
export JULIA_DEPOT_PATH=/path/with/storage/julia-depot            # if the home directory has little storage
export JULIA_PKG_USE_CLI_GIT=true                                 # avoids LibGit2 errors on some systems
julia --project="$repo/FFNOInversion.jl" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Check that Python finds the GPUs and the service's modules. Run this on a GPU node:

```bash
cd "$repo" && /path/to/python -c "import torch, ffno_batched_service; print(torch.cuda.device_count(), 'GPUs')"
```

## 3. Make a run folder and supply the files

Use a new folder for each experiment on a file system the GPU node can read, because output files with the same names are overwritten.

```bash
run=/path/to/scratch/ffnoml_forward_001
mkdir -p "$run"
cp -R "$repo/examples/inversion_run/." "$run/"
ln -s /path/to/initial_atmosphere.h5 "$run/inputs/initial_atmosphere.h5"
ln -s /path/to/kurucz_6302.list "$run/inputs/kurucz_6302.list"
ln -s /path/to/atom.h6_tiago2.yaml "$run/inputs/atoms/atom.h6_tiago2.yaml"
ln -s /path/to/atom.ca2.yaml "$run/inputs/atoms/atom.ca2.yaml"
ln -s "$repo/scripts/pf_Kurucz.input" "$run/inputs/pf_Kurucz.input"
ln -s /path/to/eos_table_fullz.h5 "$run/inputs/eos_table_fullz.h5"
ln -s /path/to/3D_sim_train_H.pt "$run/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_H.pt"
ln -s /path/to/3D_sim_train_CA.pt "$run/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_CA.pt"
```

The factory needs both checkpoints and both atoms, even for a spectrum with only one of the species. `eos_table_fullz.h5` is the HSEMHS EOS table, `HSEMHS/cache/eos_table_fullz.h5` in the HSEMHS checkout. The population mesh is always on and needs it.

## 4. Edit the configuration

Start from `forward.toml` (log tau), `forward_logm.toml` (log column mass) or `forward_z.toml` (height). The dataset table and every setting are described in [START_HERE.md, section 3](START_HERE.md#3-check-the-atmosphere-and-edit-forwardtoml).

The batched service runs in one Julia process, so MPI must be off:

```bash
sed -i '/^\[parallel\]/,/^\[/ s/^enabled = true/enabled = false/' "$run/forward.toml"
```

Then check these settings:

- **`storage_order`:** use `"python"` for files written by NumPy or h5py, and `"julia"` for files written by HDF5.jl.
- **`[atmosphere.datasets]`:** add `ne`, `rho`, `pgas` or `hydrogen_populations` if you have them. HSEMHS is then not used. For a simulation, `ne`, `rho` and `pgas` together are the most faithful.
- **IRIS/MURaM FITS:** set `atmos_model_format = "iris"` and `atmos_directory` under `[atmosphere]`.

To check that the configuration loads and to see what it selected:

```bash
julia --startup-file=no --project="$repo/FFNOInversion.jl" "$repo/FFNOInversion.jl/scripts/dry_run.jl" "$run/forward.toml"
```

## 5. Submit

For Slurm, save this as `$run/run_batched.sbatch` and replace the placeholders. Keep `--nodes=1`, and use your site's own way of loading modules and requesting GPUs. On another scheduler, keep everything from `export` onward and translate the header.

```bash
#!/bin/bash
#SBATCH --job-name=ffno-forward
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
repo=/path/to/FFNOML
run=/path/to/scratch/ffnoml_forward_001
# load your site's Julia, Python and compiler modules here
# export JULIA_DEPOT_PATH=/path/with/storage/julia-depot

export FFNOML_RUN_DIR=$run
export FFNO_FSDP_LAUNCHER=$repo/FFNOInversion.jl/scripts/ffno_batched_service_local.sh
export FFNO_FSDP_PYTHON=/path/to/python     # with CUDA torch
export FFNO_GPU_CONTROL_HOST=127.0.0.1
export FFNO_BATCHED_VERTICAL_NORM_SLABS=1   # changes the result: see below
export OMP_NUM_THREADS=1
export FFNO_THREADS_PER_RANK=$SLURM_CPUS_PER_TASK

cd "$run"
# built on the GPU node, because the login node's CPU may differ
[ -f inputs/wittmann/libwitt_ffno.so ] || CXX=${CXX:-g++} julia --startup-file=no \
  "$repo/FFNOInversion.jl/scripts/build_wittmann_backend.jl" "$run/inputs/wittmann/libwitt_ffno.so"
julia --startup-file=no --project="$repo/FFNOInversion.jl" --threads="$SLURM_CPUS_PER_TASK" \
  "$repo/FFNOInversion.jl/scripts/invert.jl" "$run/forward.toml" "$run/model_factory.jl"
```

```bash
cd "$run" && sbatch run_batched.sbatch
```

Julia and the service run on the same node and talk over `127.0.0.1`. Without `CUDA_VISIBLE_DEVICES`, the service uses every GPU it can see, and schedulers normally expose only the allocated ones.

`FFNO_BATCHED_VERTICAL_NORM_SLABS` is the only batched setting that changes the result. It doesn't need to match the number of GPUs you use here:

| Value | Result |
| --- | --- |
| `1` (default) | the volume statistic of `FFNO3D.forward` |
| `N` | reproduces an FSDP run on N GPUs |

The FSDP routes normalize one layer per GPU slab, so their results depend on the GPU count. Pick the value you want before comparing with another run.

`FFNO_BATCHED_CHUNK_DEPTH`, `FFNO_BATCHED_CHUNK_ROWS` and `FFNO_BATCHED_VERTICAL_CHUNK` (16, 16, 1024) set the batch sizes. They change memory use and speed but not the result. A batch that runs out of GPU memory is split and retried, and more GPUs only make the run faster.

Measured on 3 T4s, for the H model:

| Cube | Prediction time per species | Peak per GPU |
| --- | --- | --- |
| 64 × 64 × 452 | 16–22 s | not recorded |
| 252 × 252 × 467 | 123 s | 2.8 GB |
| 504 × 504 × 467 | 428 s | 11 GB |

## 6. Check the result

The service log, with one `BATCHED` line per stage, is `diagnostics/ffno-fsdp-service-*.log`. A successful run writes:

- **`outputs/synthesis.h5`:** `intensity`, `wavelength_m` and `normalization`.
- **`outputs/forward_atmosphere.h5`:** the atmosphere, with `populations/H` and `populations/CA`.

[START_HERE.md, section 5](START_HERE.md#5-check-the-result-later-switch-to-inversion) describes both files.

## If something goes wrong

| Symptom | Cause and fix |
| --- | --- |
| `FSDP launcher exited before the service became ready` | Read `diagnostics/ffno-fsdp-service-*.log`. A missing library usually means the Python environment's modules weren't loaded in the job. |
| `failed to map segment` from Julia | Julia on a network file system can fail this way. Copy it to node-local storage, such as `/tmp`, at the start of the job. |
| `Illegal instruction`, or the Wittmann library doesn't load | The library or depot was built on a different CPU. Delete `inputs/wittmann/libwitt_ffno.so` and let the job rebuild it. |
| The service refuses VJP or JVP | The configuration is an inversion. This service is forward only. |

`diagnostics/ffno-fsdp-service-*.toml` contains the service's auth token. Don't commit it.

## What has been validated

| Check | Result |
| --- | --- |
| H prediction, 256 × 256 × 350, against FSDP on the same machine with 3 slabs | at most 7.6e-6 dex (float32 rounding) |
| H prediction, 504 × 504 × 467, against the Olivia output with 32 slabs | at most 0.0087 dex, median 0.00065 |
| Complete run on eagle7, 64 × 64 crop with given `ne`, `rho`, `pgas`, H and Ca, on 3 T4s | completed; spectra finite, H alpha and Ca II 8542 cores at the expected wavelengths |

Not yet done: any run under a scheduler, a complete run on a 504 × 504 cube, a comparison of the final spectra with a reference, and any comparison of the Ca predictions.
