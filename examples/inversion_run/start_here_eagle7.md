# Quick start on eagle7: forward synthesis with the batched service

This guide runs `FFNOInversion.jl/scripts/invert.jl` on **eagle7** (6 × Tesla T4, 16 GB each, 64 cores, about 500 GB RAM, no Slurm). The batched service predicts a whole cube on GPUs too small to hold it: the hidden states stay in host RAM and the GPUs process batches. This is not tiling. See the [package README](../../FFNOInversion.jl/README.md#batched-forward-service-small-gpus) for how it works.

**Forward runs only.** The service refuses the VJP and JVP requests an inversion needs. For inversions, use [Olivia](START_HERE.md). For other machines, see [any HPC with GPUs](start_here_general_hpc.md) or [a GPU host without a scheduler](start_here_local_cluster_gpu.md).

## 1. Make a run folder

Use a new folder for each experiment, because output files with the same names are overwritten. Run this on eagle7, in bash:

```bash
repo=/mn/stornext/d9/data/harshm/FFNOML
run=/mn/stornext/d9/data/harshm/ffnoml_runs/forward_001
mkdir -p "$run"
cp -R "$repo/examples/inversion_run/." "$run/"
```

## 2. Link the files

Replace `/path/to/initial_atmosphere.h5` with your atmosphere. An IRIS/MURaM FITS atmosphere needs no file here (step 3).

```bash
assets=$repo/response_run/inputs
models=$repo/training_FFNO3D_zscale_expand_lognlte
ln -s /path/to/initial_atmosphere.h5 "$run/inputs/initial_atmosphere.h5"
ln -s "$assets/kurucz_6302.list" "$run/inputs/kurucz_6302.list"
ln -s "$assets/atoms/atom.h6_tiago2.yaml" "$run/inputs/atoms/atom.h6_tiago2.yaml"
ln -s "$assets/atoms/atom.ca2.yaml" "$run/inputs/atoms/atom.ca2.yaml"
ln -s "$repo/scripts/pf_Kurucz.input" "$run/inputs/pf_Kurucz.input"
ln -s /mn/stornext/d9/data/harshm/MHS/HSEMHS/cache/eos_table_fullz.h5 "$run/inputs/eos_table_fullz.h5"
ln -s "$models/3D_sim_train_H.pt" "$run/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_H.pt"
ln -s "$models/3D_sim_train_CA.pt" "$run/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_CA.pt"
```

The factory needs both checkpoints and both atoms, even for a spectrum with only one of the species.

Build the Wittmann EOS library once for the run:

```bash
source ~/loadnvidia.sh >/dev/null 2>&1
export PATH=/tmp/julia-1.12.3/bin:$PATH
CXX=g++ julia --startup-file=no "$repo/FFNOInversion.jl/scripts/build_wittmann_backend.jl" "$run/inputs/wittmann/libwitt_ffno.so"
```

## 3. Edit the configuration

Start from `forward.toml` (log tau), `forward_logm.toml` (log column mass) or `forward_z.toml` (height). The dataset table and every setting are described in [START_HERE.md, section 3](START_HERE.md#3-check-the-atmosphere-and-edit-forwardtoml).

The batched service runs in one Julia process, so MPI must be off. This is the one change the templates need on eagle7:

```bash
sed -i '/^\[parallel\]/,/^\[/ s/^enabled = true/enabled = false/' "$run/forward.toml"
```

Then check these settings:

- **`storage_order`:** use `"python"` for files written by NumPy or h5py, and `"julia"` for files written by HDF5.jl.
- **`[atmosphere.datasets]`:** add `ne`, `rho`, `pgas` or `hydrogen_populations` if you have them. HSEMHS is then not used. For a simulation, `ne`, `rho` and `pgas` together are the most faithful.
- **IRIS/MURaM FITS:** set `atmos_model_format = "iris"` and `atmos_directory` under `[atmosphere]`.
- **`eos_table_file`:** the template's `inputs/eos_table_fullz.h5` is the link made above. The population mesh is always on and needs it.

To check that the configuration loads and to see what it selected:

```bash
julia --startup-file=no --project="$repo/FFNOInversion.jl" "$repo/FFNOInversion.jl/scripts/dry_run.jl" "$run/forward.toml"
```

## 4. Choose the GPUs and check the memory

eagle7 is shared, so see what is free first:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
free -g
```

Use `CUDA_VISIBLE_DEVICES` in step 5 to list only free GPUs. More GPUs only make a run faster; a batch that runs out of GPU memory is split and retried.

Measured on 3 T4s, for the H model:

| Cube | Prediction time per species | Peak per GPU |
| --- | --- | --- |
| 64 × 64 × 452 | 16–22 s | not recorded |
| 252 × 252 × 467 | 123 s | 2.8 GB |
| 504 × 504 × 467 | 428 s | 11 GB |

The hidden states took about 90 GB of host RAM at 504 × 504 × 467, which is roughly 760 bytes per cell. The depth is the population mesh's 452 levels by default, whatever the input grid. Julia's own memory comes on top of this and hasn't been measured for a large cube.

## 5. Run

Write the run script:

```bash
cat > "$run/run_eagle7.sh" <<EOF
#!/bin/bash
source ~/loadnvidia.sh >/dev/null 2>&1
export PATH=/tmp/julia-1.12.3/bin:\$PATH
export FFNOML_RUN_DIR=$run
export FFNO_FSDP_LAUNCHER=$repo/FFNOInversion.jl/scripts/ffno_batched_service_local.sh
export FFNO_FSDP_PYTHON=\$HOME/nvidiaenv/bin/python
export FFNO_GPU_CONTROL_HOST=127.0.0.1
export CUDA_VISIBLE_DEVICES=3,4,5           # the free GPUs from step 4
export FFNO_BATCHED_VERTICAL_NORM_SLABS=1   # changes the result: see below
export OMP_NUM_THREADS=1
export FFNO_THREADS_PER_RANK=64             # the same as --threads below
cd $run
julia --startup-file=no --project=$repo/FFNOInversion.jl --threads=64 \\
  $repo/FFNOInversion.jl/scripts/invert.jl $run/forward.toml $run/model_factory.jl
EOF
```

Edit the GPU list if needed, and lower the threads if others are using the CPUs, then start it. `nohup` keeps it running after you log out:

```bash
nohup bash "$run/run_eagle7.sh" > "$run/run.log" 2>&1 &
```

`FFNO_BATCHED_VERTICAL_NORM_SLABS` is the only batched setting that changes the result. It doesn't need to match the number of GPUs you use here:

| Value | Result |
| --- | --- |
| `1` (default) | the volume statistic of `FFNO3D.forward` |
| `N` | reproduces an FSDP run on N GPUs. Use `32` to match Olivia's default 8 nodes × 4 GPUs. |

The FSDP routes normalize one layer per GPU slab, so their results depend on the GPU count. Pick the value you want before comparing with another run.

`FFNO_BATCHED_CHUNK_DEPTH`, `FFNO_BATCHED_CHUNK_ROWS` and `FFNO_BATCHED_VERTICAL_CHUNK` set the batch sizes. They change memory use and speed but not the result.

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
| `libpython3.12.so.1.0: cannot open shared object file` | `~/nvidiaenv/bin/python` was started without `source ~/loadnvidia.sh`. |
| `failed to map segment` from Julia | Julia was started from stornext. Put `/tmp/julia-1.12.3/bin` first on `PATH`, as above. |
| `FSDP launcher exited before the service became ready` | Read `diagnostics/ffno-fsdp-service-*.log`. |
| The service refuses VJP or JVP | The configuration is an inversion. This service is forward only. |
| GPUs stay busy after you killed a run | Check `nvidia-smi` and stop your own leftover process: `pkill -u $USER -f ffno_batched_service.py`. |

`diagnostics/ffno-fsdp-service-*.toml` contains the service's auth token. Don't commit it.

## What has been validated

| Check | Result |
| --- | --- |
| H prediction, 256 × 256 × 350, against FSDP on the same machine with 3 slabs | at most 7.6e-6 dex (float32 rounding) |
| H prediction, 504 × 504 × 467, against the Olivia output with 32 slabs | at most 0.0087 dex, median 0.00065 |
| Complete run following this guide, 64 × 64 crop with given `ne`, `rho`, `pgas`, H and Ca, on 3 T4s | completed; spectra finite, H alpha and Ca II 8542 cores at the expected wavelengths |

Not yet done: a complete run on a 504 × 504 cube, a comparison of the final spectra with a reference, and any comparison of the Ca predictions.
