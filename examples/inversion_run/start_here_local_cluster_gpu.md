# Quick start on a local GPU machine: forward synthesis with the batched service

This guide runs `FFNOInversion.jl/scripts/invert.jl` on a GPU host you log in to directly: a group cluster node or workstation with no scheduler. The batched service predicts a whole cube on GPUs too small to hold it: the hidden states stay in host RAM and the GPUs process batches. This is not tiling. See the [package README](../../FFNOInversion.jl/README.md#batched-forward-service-small-gpus) for how it works.

**Forward runs on one host only.** The service is a single process, so it can't span machines, and it refuses the VJP and JVP requests an inversion needs. Julia and the service must run on the machine that has the GPUs.

Other guides: [eagle7](start_here_eagle7.md), which has this machine's paths filled in, and [any HPC with GPUs](start_here_general_hpc.md), for a batch system.

## 1. What the machine needs

| Need | Notes |
| --- | --- |
| NVIDIA GPUs with CUDA | T4s (16 GB) have run 504 × 504 × 467. |
| Host RAM | About 760 bytes per cell for the hidden states: about 90 GB at 504 × 504 × 467. Julia's own memory comes on top and hasn't been measured for a large cube. |
| Julia 1.12 | MPI is off in this route, so no system MPI is needed. If your Julia is on a network file system and fails with `failed to map segment`, copy it to `/tmp`. |
| Python 3.11 or newer | With CUDA `torch` and `numpy`. |
| A C++ compiler | For the Wittmann EOS library. |

Depth is the population mesh's 452 levels by default, whatever the input grid.

## 2. Install, once

Load your machine's environment first, if it has one. Then:

```bash
repo=/path/to/FFNOML
python=/path/to/python    # with CUDA torch
git clone https://github.com/harshmathur1990/FFNOML.git "$repo"   # or your own checkout
export JULIA_PKG_USE_CLI_GIT=true                                 # avoids LibGit2 errors on some systems
julia --project="$repo/FFNOInversion.jl" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
cd "$repo" && "$python" -c "import torch, ffno_batched_service; print(torch.cuda.device_count(), 'GPUs')"
```

## 3. Make a run folder and supply the files

Use a new folder for each experiment, because output files with the same names are overwritten.

```bash
run=/path/to/runs/forward_001
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
CXX=${CXX:-g++} julia --startup-file=no "$repo/FFNOInversion.jl/scripts/build_wittmann_backend.jl" "$run/inputs/wittmann/libwitt_ffno.so"
```

The factory needs both checkpoints and both atoms, even for a spectrum with only one of the species. `eos_table_fullz.h5` is the HSEMHS EOS table, `HSEMHS/cache/eos_table_fullz.h5` in the HSEMHS checkout. The population mesh is always on and needs it.

If your group's machines have different CPUs and share this run folder, build the library on the machine that runs the job.

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

## 5. Choose the GPUs and run

With no scheduler, nothing reserves the GPUs for you, so see what is free first:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
free -g
```

Write the run script, then edit the GPU list and set the threads to the cores you can use:

```bash
cat > "$run/run_local.sh" <<EOF
#!/bin/bash
# load your machine's environment here, if it has one
export FFNOML_RUN_DIR=$run
export FFNO_FSDP_LAUNCHER=$repo/FFNOInversion.jl/scripts/ffno_batched_service_local.sh
export FFNO_FSDP_PYTHON=$python
export FFNO_GPU_CONTROL_HOST=127.0.0.1
export CUDA_VISIBLE_DEVICES=0,1,2           # the free GPUs
export FFNO_BATCHED_VERTICAL_NORM_SLABS=1   # changes the result: see below
export OMP_NUM_THREADS=1
export FFNO_THREADS_PER_RANK=16             # the same as --threads below
cd $run
julia --startup-file=no --project=$repo/FFNOInversion.jl --threads=16 \\
  $repo/FFNOInversion.jl/scripts/invert.jl $run/forward.toml $run/model_factory.jl
EOF
```

`nohup` keeps the run going after you log out:

```bash
nohup bash "$run/run_local.sh" > "$run/run.log" 2>&1 &
```

`FFNO_BATCHED_VERTICAL_NORM_SLABS` is the only batched setting that changes the result. It doesn't need to match the number of GPUs you use here:

| Value | Result |
| --- | --- |
| `1` (default) | the volume statistic of `FFNO3D.forward` |
| `N` | reproduces an FSDP run on N GPUs |

The FSDP routes normalize one layer per GPU slab, so their results depend on the GPU count. Pick the value you want before comparing with another run.

`FFNO_BATCHED_CHUNK_DEPTH`, `FFNO_BATCHED_CHUNK_ROWS` and `FFNO_BATCHED_VERTICAL_CHUNK` (16, 16, 1024) set the batch sizes. They change memory use and speed but not the result. A batch that runs out of GPU memory is split and retried, and more GPUs only make the run faster. Lower the batch sizes if someone else's job leaves too little GPU memory.

Measured on 3 T4s, for the H model:

| Cube | Prediction time per species | Peak per GPU |
| --- | --- | --- |
| 64 × 64 × 452 | 16–22 s | not recorded |
| 252 × 252 × 467 | 123 s | 2.8 GB |
| 504 × 504 × 467 | 428 s | 11 GB |

## 6. Check the result

Follow the run in `run.log`. The service writes its own log, with one `BATCHED` line per stage:

```bash
tail -f "$run"/diagnostics/ffno-fsdp-service-*.log
```

A successful run writes:

- **`outputs/synthesis.h5`:** `intensity`, `wavelength_m` and `normalization`.
- **`outputs/forward_atmosphere.h5`:** the atmosphere, with `populations/H` and `populations/CA`.

[START_HERE.md, section 5](START_HERE.md#5-check-the-result-later-switch-to-inversion) describes both files.

## If something goes wrong

| Symptom | Cause and fix |
| --- | --- |
| `FSDP launcher exited before the service became ready` | Read `diagnostics/ffno-fsdp-service-*.log`. A missing library usually means the Python environment's modules weren't loaded in the run script. |
| `failed to map segment` from Julia | Julia is on a network file system. Copy it to `/tmp` and put that copy first on `PATH`. |
| The service refuses VJP or JVP | The configuration is an inversion. This service is forward only. |
| GPUs stay busy after you killed a run | Check `nvidia-smi` and stop your own leftover process: `pkill -u $USER -f ffno_batched_service.py`. |

`diagnostics/ffno-fsdp-service-*.toml` contains the service's auth token. Don't commit it.

## What has been validated

| Check | Result |
| --- | --- |
| H prediction, 256 × 256 × 350, against FSDP on the same machine with 3 slabs | at most 7.6e-6 dex (float32 rounding) |
| H prediction, 504 × 504 × 467, against the Olivia output with 32 slabs | at most 0.0087 dex, median 0.00065 |
| Complete run on eagle7, 64 × 64 crop with given `ne`, `rho`, `pgas`, H and Ca, on 3 T4s | completed; spectra finite, H alpha and Ca II 8542 cores at the expected wavelengths |

Not yet done: a complete run on a 504 × 504 cube, a comparison of the final spectra with a reference, and any comparison of the Ca predictions.
