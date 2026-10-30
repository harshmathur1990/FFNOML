# FFNOML

FFNOML predicts NLTE level populations in 3D solar atmospheres with Fourier Neural
Operators, and uses those predictions to synthesize and invert chromospheric spectra.

The repository has two halves:

1. **Python: learning NLTE populations.** `pipeline.py` builds training sets from
   MULTI3D/Bifrost simulations, trains the FFNO3D network, validates it, and predicts
   populations for new atmospheres (single GPU, tiled, or distributed over many GPUs).
2. **Julia: forward synthesis and 3D inversion.**
   [`FFNOInversion.jl`](FFNOInversion.jl/README.md) takes an atmosphere, reconstructs
   its hydrostatic structure with HSEMHS, gets populations from the trained networks
   through a persistent multi-GPU service, synthesizes spectra (H alpha, Ca II 8542 and
   Kurucz LTE lines), and fits the whole 3D volume to observations with bounded L-BFGS.

```text
MULTI3D / Bifrost ──► pipeline.py --build ──► --train ──► checkpoint (.pt)
                                                             │
          atmosphere ──► --buildforpredict ──► --predict / --fsdppredict ──► populations (HDF5) ──► Forward.jl
                                                             │
log-tau or z atmosphere ──► FFNOInversion.jl ── HSEMHS ── FFNO service ── synthesis ──► spectra, atmosphere
                                  ▲                                             │
                            observations ◄──────── L-BFGS (inversion mode) ◄────┘
```

## Contents

- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [What the model learns](#what-the-model-learns)
- [Data and configuration](#data-and-configuration)
- [Installation](#installation)
- [Pipeline modes](#pipeline-modes)
- [Forward synthesis with Forward.jl](#forward-synthesis-with-forwardjl)
- [Persistent FFNO service](#persistent-ffno-service)
- [3D forward synthesis and inversion](#3d-forward-synthesis-and-inversion)
- [Preparing inversion inputs](#preparing-inversion-inputs)
- [Other scripts](#other-scripts)
- [Tests](#tests)
- [Documentation](#documentation)
- [Caveats](#caveats)

## Repository layout

| Path | Role |
| --- | --- |
| `pipeline.py` | CLI entry point for dataset building, training, validation, prediction |
| `config.py` | dataset splits, active atoms, model hyperparameters, paths, runtime flags |
| `FFNONet.py` | dataset builders, HDF5 writers, training loop, validation diagnostics, inference |
| `models/ffno_model.py` | the FFNO3D architecture |
| `model_builder.py`, `ffno_model_factory.py` | model/loss/optimizer construction, FSDP wrapping, factory for the service |
| `data_builder.py`, `data/dataset.py` | patch or cube datasets and dataloaders (lazy HDF5) |
| `loss/` | composite NLTE loss, weighted MSE, gradient loss |
| `train_utils.py`, `normalize_utils.py` | checkpoints, resume, model expansion, normalization |
| `distributed_inference.py` | H-slab distributed FFTs and collectives for full-volume inference |
| `ffno_fsdp_service.py`, `ffno_fsdp_runtime.py`, `ffno_fsdp_diagnostics.py` | persistent multi-GPU prediction/VJP service used by the inversion |
| `Forward.jl` | standalone Julia synthesis from predicted (or MULTI3D) populations |
| `FFNOInversion.jl/` | Julia forward-synthesis and inversion package ([README](FFNOInversion.jl/README.md)) |
| `examples/` | run-directory templates for Python jobs, inversions and Olivia runtime tests |
| `scripts/` | input builders, converters, EOS sources, comparisons, Slurm helpers |
| `errorplots.py`, `make_paper_plots.py`, `compare_population_models.py` | analysis and figures |
| `train_gpu.sh`, `predict_gpu.sh` | Olivia Slurm scripts for training and prediction |
| `tests/` | Python unit tests |
| `docs/` | architecture diagrams, paper draft, media |

## Quick start

Train and predict (Python):

```bash
python pipeline.py --build
python pipeline.py --train
python pipeline.py --test
python pipeline.py --buildforpredict
python pipeline.py --predict
```

Synthesize spectra from a 3D atmosphere on Olivia (Julia), after preparing a run
directory as described in the
[starter guide](examples/inversion_run/START_HERE.md):

```bash
FFNOML_RUN_DIR="$run" FFNO_INVERSION_CONFIG="$run/forward.toml" \
  bash "$repo/FFNOInversion.jl/scripts/submit_olivia_inversion.sh"
```

Switch to `inversion.toml` and add `inputs/observations.h5` to invert.

## What The Model Learns

The pipeline works on the native simulation height scale (`z_scale`), not on column
mass. The model maps atmospheric state variables to NLTE populations:

`[B, Cin, D, H, W] -> [B, Cout, D, H, W]`

Input channels are constructed in `FFNONet.py` from:

- `log10(temp)`
- `vx`
- `vy`
- `vz`
- `log10(ne)`
- `log10(rho)`

Targets are:

- `log10(n_NLTE [m^-3])`

Prediction outputs are written back in linear space as:

- `nlte_populations = n_NLTE [m^-3]`

## Data And Configuration

All project-specific configuration lives in `config.py`.

Important sections:

- `SIMULATIONS`: available simulation roots and snapshots
- `TRAIN_SPLIT` / `VAL_SPLIT`: snapshots used for dataset building
- `ACTIVE_ATOMS`: atoms to concatenate into one target tensor
- `MULTI3D_PRED_DATA`: atmospheres used by `--predict` and `--fsdppredict`
- `MODEL`, `MODEL_CONFIG`: architecture choice and hyperparameters
- `PATCH`, `STRIDE`, `SCALES`: patch extraction and multiscale dataset settings
- `TRAIN_FILE`, `TEST_FILE`: HDF5 dataset paths derived from the active split and patch config
- `MODEL_DIR`, `MODEL_FILE`: checkpoint and diagnostics output location
- `MULTI_GPU`, `DEVICE`, `CUDA`, `TILED`: runtime behavior
- `EXPAND_FROM_CHECKPOINT`, `ZERO_INIT_NEW_BLOCKS`: model expansion settings for `--train --expand`

### Separate run directory (Olivia)

The repository and run files do not need to live in the same directory. `DATA_DIR`
continues to locate permanent Bifrost/MULTI3D source data, while the environment
variable `FFNOML_RUN_DIR` anchors both:

- `IO/` for generated or symlinked input datasets
- `training_FFNO3D_zscale_expand_lognlte/` for checkpoints and other model outputs

The Slurm scripts set `FFNOML_RUN_DIR` to the submission directory by default and
locate the Python code from the batch script itself. For example:

```bash
repo=/path/to/project/FFNOML
run=/path/to/hot/ffnoml_run1

mkdir -p "$run/IO" "$run/training_FFNO3D_zscale_expand_lognlte"
# Add the required symlinks under "$run/IO" here.
cd "$run"
sbatch "$repo/train_gpu.sh"
```

This places Slurm logs, the combined `output-<job-id>.txt` log, and model outputs
under the disposable run directory. To submit from elsewhere, pass an explicit
run root, for example `sbatch --export=ALL,FFNOML_RUN_DIR="$run" "$repo/train_gpu.sh"`.
Without `FFNOML_RUN_DIR`, non-Slurm Python runs retain the old behavior and use
the repository directory.

Ready-to-copy layouts for Python runs, production Julia inversions, and Olivia
runtime tests are available under [`examples/`](examples/README.md).

The currently implemented model names are:

- `FFNO3D`

`FFNO3D` uses a cheap coordinate-conditioned vertical branch: it keeps absolute
`z_scale` as an input feature and also derives local normalized depth,
local `dz`, and total z-span before the vertical projection. This is lighter
than the deleted z-operator variant, but gives the model more information about
nonuniform z grids than depth-index convolutions alone.

## Installation

At minimum you need:

- Python 3.10+
- PyTorch
- NumPy
- SciPy
- h5py
- matplotlib
- `helita`

Example:

```bash
pip install torch numpy scipy h5py matplotlib helita
```

If you use multi-GPU training, your PyTorch install must support distributed CUDA/FSDP execution.

## Input Data Assumptions

The Python pipeline expects MULTI3D/Bifrost-style files:

- atmosphere file: `atm3d`
- mesh file: `mesh`
- one MULTI3D output directory per active atom

For training/build mode, each dataset entry must provide:

- `MULTI3D_ATMOS`
- `MESH`
- `MULTI3D_PATHS`

For prediction mode, each entry in `MULTI3D_PRED_DATA` must provide:

- `MULTI3D_ATMOS`
- `MESH`
- `NAME`

Unit conversions currently applied by the pipeline:

- `rho`: `g/cm^3 -> kg/m^3`
- `ne`: `cm^-3 -> m^-3`
- `z` from mesh or geometry: `cm -> m`

`dx` and `dy` are computed from the mesh and converted to meters.

## Pipeline Modes

`pipeline.py` takes exactly one mode flag:

| Flag | Does |
| --- | --- |
| `--build` | build the HDF5 training and validation datasets from MULTI3D outputs |
| `--train` | train a checkpoint (`--resume`, `--bestpath`, `--expand` modify it) |
| `--val` | run one validation pass, print the loss terms, exit |
| `--test` | validation diagnostics and branch ablations, written as JSON |
| `--buildforpredict` | build the prediction solving sets (`--predname` for one atmosphere) |
| `--predict` | predict NLTE populations, full cube or tiled |
| `--fsdppredict` | the same through `torchrun`, distributed over GPUs by H slabs |
| `--inspectfreqgate` | print and save the FFNO frequency-gate diagnostics (`--inspect-h`, `--inspect-w`, `--inspect-dx`, `--inspect-dy`, `--inspect-out`) |

## Dataset Build Mode

Run:

```bash
python pipeline.py --build
```

What it does:

1. Reads all configured training snapshots from `MULTI3D_TRAIN_DATA`
2. Reads all configured validation snapshots from `MULTI3D_VAL_DATA`
3. Loads LTE and NLTE populations from each atom directory
4. Concatenates active atoms along the output-channel dimension
5. Builds input features and `log10(n_NLTE)` targets
6. Normalizes channels using global statistics from the training set
7. Extracts multiscale XY patches using `PATCH`, `STRIDE`, and `SCALES`
8. Writes grouped HDF5 patch datasets to `TRAIN_FILE` and `TEST_FILE`

Important behavior:

- The train file computes normalization statistics itself
- The validation file reuses normalization statistics from `TRAIN_FILE`
- Existing output files are not overwritten
- Grouped patch datasets preserve native depth per simulation

### Training/Validation HDF5 Layout

Patch datasets are stored as grouped HDF5 files. Each group contains:

- `inputs`: `[N, Cin, D, P, P]`
- `targets`: `[N, Cout, D, P, P]`
- `z_scale`: `[N, D, P, P]`
- `dx`: `[N]`
- `dy`: `[N]`
- `scale`: `[N]`
- `weights`: `[N]`

Root-level metadata includes:

- `mean_X`, `std_X`
- `mean_Y`, `std_Y`
- `patch_dataset_names`
- attributes such as `Cin`, `Cout`, `N`, `n_patch_datasets`, `normalized`

## Train Mode

Run:

```bash
python pipeline.py --train
```

Train mode requires that `TRAIN_FILE` and `TEST_FILE` already exist. If they do not, run `--build` first.

What it does:

1. Loads input/output channel counts from `TRAIN_FILE`
2. Loads normalization stats from `TRAIN_FILE`
3. Builds the configured model from `MODEL` and `MODEL_CONFIG`
4. Builds patch or cube dataloaders according to `DATASET_TYPE`
5. Trains with the composite NLTE loss
6. Saves the best checkpoint to `MODEL_FILE`
7. Saves a resumable checkpoint to `MODEL_FILE` with `.resume` inserted before the extension

Current training details from the code:

- optimizer: `AdamW`
- optional scheduler: cosine annealing when `USE_COSINE=True`
- early stopping controlled by `PATIENCE` and `MIN_DELTA`
- gradient clipping controlled by `GRAD_CLIP`
- grouped native-depth datasets require `BATCH_SIZE = 1`

### Resume Options

`pipeline.py` supports these train-only flags:

- `--resume`: resume from the `.resume` checkpoint
- `--bestpath`: when used with `--resume`, load the best checkpoint at `MODEL_FILE` instead of the `.resume` file
- `--expand`: initialize a larger model from `EXPAND_FROM_CHECKPOINT`

Rules enforced by the CLI:

- `--resume` only works with `--train`
- `--bestpath` only works with `--train`
- `--expand` only works with `--train`
- `--expand` cannot be combined with `--resume` or `--bestpath`
- `EXPAND_FROM_CHECKPOINT` must be set in `config.py` before using `--expand`

Behavior notes:

- `--train` refuses to overwrite an existing `MODEL_FILE` unless `--resume` is used
- `--train --expand` should point `MODEL_FILE` at a new output checkpoint path
- `ZERO_INIT_NEW_BLOCKS` controls whether newly added residual tensors are zero-initialized during expansion

## Test Mode

Run:

```bash
python pipeline.py --test
```

With `MULTI_GPU=True`, run test mode through `torchrun` so validation uses the
same distributed full H-slab inference strategy as `--fsdppredict`:

```bash
torchrun --nproc_per_node=4 pipeline.py --test
```

Test mode requires:

- `TEST_FILE`
- `MODEL_FILE`

What it does:

1. Builds the validation dataset loader from `TEST_FILE`
2. Restores channel metadata and normalization stats directly from `MODEL_FILE`
3. Loads the trained checkpoint from `MODEL_FILE`
4. Evaluates validation loss/components with distributed full H-slab inference
5. Runs branch ablations for spectral, vertical, pointwise, and MLP branches
6. Writes a JSON diagnostic summary

Current output path:

- `MODEL_DIR + "val_diagnostics_<MODEL>_<ACTIVE_ATOMS>.json"`

The JSON summary includes:

- baseline loss
- baseline component breakdown
- baseline model statistics
- validation coverage metadata, including inference strategy, visited dataset items, and columns
- ablation results
- branch importance scores

## Predict Mode

Build the solving set for every entry in `MULTI3D_PRED_DATA`:

```bash
python pipeline.py --buildforpredict
```

To build one atmosphere/snapshot without declaring it in `MULTI3D_PRED_DATA`,
pass a name in `<atmosphere>_<snap>` form:

```bash
python pipeline.py --buildforpredict --predname en024048_hion_386
```

This reads `PRED_DIR/bifrost_data/en024048_hion/386/atm3d` and `mesh`, and
writes `MODEL_DIR/3D_sim_predict_en024048_hion_386.hdf5`. Missing source files
are reported before loading begins.

Run:

```bash
python pipeline.py --predict
```

Predict mode uses `MULTI3D_PRED_DATA` from `config.py`.

What it does for each configured atmosphere:

1. Loads `atm3d` and `mesh`
2. Computes `dx`, `dy`, and `z_scale`
3. Requires the solving-set HDF5 from `--buildforpredict` (it stops with a message if the file is missing)
4. Loads normalization stats and channel metadata from `MODEL_FILE`
5. Runs full-cube or tiled inference
6. Writes predicted NLTE populations in linear space

Important current behavior:

- `--predict` does not require `TRAIN_FILE` or `TEST_FILE`
- it does require a trained checkpoint at `MODEL_FILE`
- it skips rebuilding outputs if the final prediction file already exists
- tiling is controlled by `TILED`, `PATCH`, and `STRIDE`

## Distributed Full Prediction Mode

Run with `torchrun`:

```bash
torchrun --nproc_per_node=4 pipeline.py --fsdppredict
```

For multi-node jobs, pass the usual `torchrun` rendezvous arguments, for example:

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=<rank> \
  --rdzv_id=<job_id> \
  --rdzv_backend=c10d \
  --rdzv_endpoint=<master_addr>:<master_port> \
  pipeline.py --fsdppredict
```

`--fsdppredict` uses the same configured atmospheres, checkpoint, solving-set path, and final HDF5 output path as `--predict`. The difference is the inference path:

- it initializes `torch.distributed` with NCCL
- each rank owns a slab along the `nx`/H dimension
- the spectral branch uses distributed inference helpers instead of local full-cube FFTs
- rank 0 gathers the slabs and writes the final prediction HDF5

Use this mode when the full prediction volume is too large for the single-process or tiled `--predict` path. Existing final prediction files are still not overwritten.

### Solving-Set HDF5 Layout

The intermediate solving input written by `build_solving_set_ffno` contains:

- `inputs`: `[1, Cin, D, nx, ny]`
- `z_scale`: `[1, D, nx, ny]`
- `dx`: `[1]`
- `dy`: `[1]`

### MURaM FITS Prediction Inputs

For MURaM atmospheres stored as FITS cubes, generate the same solving-set HDF5
directly with:

```bash
cp scripts/convert_iris_sim_fits_to_ffno_hdf5.example.toml muram-convert.toml
# Edit muram-convert.toml, then run:
python scripts/convert_iris_sim_fits_to_ffno_hdf5.py muram-convert.toml
```

Every converter setting is read from the TOML file: inputs, x/y/z crop,
height range, grid spacing, Witt EOS options, outputs, compression,
overwrite behavior, upsampling, interpolation, and validation. Relative paths
are resolved from the directory containing the TOML file. The complete,
commented schema is in
`scripts/convert_iris_sim_fits_to_ffno_hdf5.example.toml`.

The converter reads `lgtg`, `lgr`, `ux`, `uy`, and `uz` FITS files, rotates each
horizontal plane with `[::-1, :].T`, reverses the selected height range so the
first depth index is the top of the atmosphere, and writes `inputs`, `z_scale`,
`dx`, and `dy` in the layout used by
`--fsdppredict`. Electron density is selected automatically in priority order:
`lgne` is read directly, otherwise `lgp` is passed with temperature to the Witt
EOS, otherwise the Witt EOS derives it from `lgr`. Conversion fails if none of
these three FITS files exists. The converter finds the repo-local
`scripts/witt.py` and `scripts/pf_Kurucz.input`
automatically; set `witt_path` in `[eos]` only when those files live somewhere else. By
default, this uses the C++ full-atmosphere EOS backend and all visible CPU
threads; `show_progress = true` prints a C++-side progress line without Python
callbacks. Use `backend = "python"` only for debugging or if no C++ compiler
is available. Spatial resampling is also compiled at first use and requires a
C++17 compiler plus Boost headers (set `BOOST_INCLUDEDIR` when they are not in
a standard include location). The optional `multi3d_atmos` and `multi3d_mesh` outputs write a
Multi3D atmosphere for reference calculations. If the complete `lgn1` through
`lgn6` FITS set is present, the converter also reads and writes all six hydrogen
populations; incomplete sets are ignored. The output contains no magnetic
field.

Each source spatial axis can be cropped or subsampled in `[selection]` with a
slice string or `start`/`stop`/`step` table. The x/y strides are included in the
output spacing, and the z selection is applied before the height filter. After
coordinate rotation, every final axis can be upsampled to a requested physical
spacing in metres with `[resampling.target_spacing_m]`; its x/y values become
the written `dx`/`dy`, and z becomes the absolute increment in `z_scale`.
Factors and target grid sizes remain available as alternatives. Before
resampling starts, the converter prints the complete source selection, source
and final grids, fields, methods, units, derived quantities, and output shapes.
During resampling it prints both per-field and whole-job completion estimates
with an ETA. Interpolation runs across all CPUs available to the process by
default (respecting Linux/Slurm CPU affinity); set `[resampling].workers` to a
positive integer to override it. The native C++ backend supports
`nearest`, `linear`, and Boost.Math shape-preserving `cubic` (PCHIP) methods
independently per x/y/z direction, with per-channel overrides
for temperature, density, velocity components, electron density, and hydrogen
populations. Positive channels are interpolated in log space.

After interpolation, temperatures below `temperature_floor_k` (3250 K by
default) are raised to that floor. When electron density is derived from `lgp`
or `lgr`, the Witt EOS runs only after this step, using the interpolated gas
pressure or density together with the floored interpolated temperature. A
direct `lgne` electron-density cube is interpolated as its own channel.

Before writing, all channels are checked for finite values; temperature,
`rho`, and `ne` must be positive. Adjacent-cell continuity checks for `rho` and
`ne` default to a maximum 3-dex jump in each direction and can be tightened or
disabled in `[validation.continuity_max_log10_step]`.

To run distributed prediction on only this generated file:

```bash
torchrun --nproc_per_node=4 pipeline.py --fsdppredict --predname ar098192_270000
```

The prediction path will look for
`MODEL_DIR + "3D_sim_predict_ar098192_270000.hdf5"`.

### Prediction Output Layout

Prediction outputs are written to:

- `MODEL_DIR + "output_3D_sim_s5_<NAME>_<MODEL>_<ACTIVE_ATOMS>.hdf5"`

The file contains:

- `nlte_populations`: `[nx, ny, D, Cout]`
- `z_scale`: `[D, nx, ny]`

Current attributes include:

- `nlte_populations.attrs["depth_scale_type"] = "z"`
- file attribute `epoch`
- file attribute `val_loss` when available in the checkpoint

Prediction also writes a diagnostics path per atmosphere:

- `MODEL_DIR + "diagnostics_3D_sim_s5_<NAME>_<MODEL>_<ACTIVE_ATOMS>.npz"`

Note: the diagnostics path is passed into inference, but the current `ffno_predict_populations()` implementation only writes the HDF5 prediction output.

## Runtime Device Behavior

Before `--train`, `--test`, `--predict`, or `--fsdppredict`, `pipeline.py` calls `validate_runtime_device()`.

If `DEVICE` or `CUDA` requests GPU execution:

- PyTorch CUDA availability is checked
- visible GPU names are printed

If CUDA is requested but unavailable, the script raises an error and suggests setting `DEVICE="cpu"`.

## Multi-GPU Training

The code supports FSDP-wrapped multi-GPU training when `MULTI_GPU=True`.

Implementation notes:

- distributed initialization is done in `ModelBuilder._init_distributed()`
- it uses `torch.distributed.init_process_group("nccl")`
- FSDP auto-wrap targets the FFNO block class
- dataloaders switch to `DistributedSampler` automatically when distributed is initialized

In practice, multi-GPU training should be launched with `torchrun`, for example:

```bash
torchrun --nproc_per_node=4 pipeline.py --train
```

Single-GPU or CPU execution can still use plain `python`.

## Checkpoints

Best-checkpoint save path:

- `MODEL_FILE`

Resume-checkpoint save path:

- same path as `MODEL_FILE`, but with `.resume` inserted before the file extension

Checkpoints may store:

- `model_state`
- `opt_state`
- `scheduler_state`
- `epoch`
- `completed_epochs`
- `current_lr`
- `train_loss`, `val_loss`
- train/validation loss components
- `normalization_stats`
- `io_metadata`

Inference and test mode both rely on checkpoint-carried normalization and I/O metadata.

## Dataset Types

`DATASET_TYPE` in `config.py` can be:

- `patch`
- `cube`

The current build pipeline writes grouped patch datasets, and the default config uses:

- `DATASET_TYPE = "patch"`

For grouped native-depth patch datasets, `batch_size` must remain `1`.

`TRAINSELECT` can be set below `1.0` to randomly thin each grouped training
dataset at load time without rebuilding HDF5 files. Selection is done
independently within each stored weight value in each `train_*` group, and the
remaining sample weights are rescaled so every weight class keeps the same total
weight. Single-sample weight classes are kept unchanged. The selected rows are
resampled every epoch using `TRAINSELECT_SEED + epoch`, so the run is
reproducible while each epoch can see a different subset.

## Current End-To-End Workflow

Typical usage is:

```bash
python pipeline.py --build
python pipeline.py --train
python pipeline.py --test
python pipeline.py --buildforpredict
python pipeline.py --predict
```

For distributed full-volume prediction, replace the last command with:

```bash
torchrun --nproc_per_node=4 pipeline.py --fsdppredict
```

For continuing an interrupted run:

```bash
python pipeline.py --train --resume
```

For resuming from the best checkpoint while resetting optimizer/scheduler state using the configured fallback epoch/LR:

```bash
python pipeline.py --train --resume --bestpath
```

For initializing a larger model from an earlier checkpoint:

```bash
python pipeline.py --train --expand
```

## Plotting And Analysis

`errorplots.py` loads prediction outputs and MULTI3D truth, computes:

- relative population error envelopes
- log-ratio error envelopes

It converts predicted and true NLTE populations to departure coefficients using
the same MULTI3D LTE populations, then compares them on shared `z_scale`
directly without interpolating to column mass.

## Forward Synthesis With Forward.jl

`Forward.jl` is the Julia-side consumer for predicted NLTE populations and related atmospheric inputs. Use it for downstream synthesis after Python-side prediction has produced the HDF5 outputs.

For ML mode, `Forward.jl` reads `output_3D_sim_s5_<NAME>_<MODEL>_<FORWARD_ATOMS>.hdf5` and writes intensity files with the same Forward atom tag.

Set `FORWARD_ATOMS` near the top of `Forward.jl` to choose which atoms to synthesize, for example `["H"]`, `["CA"]`, or `["H", "CA"]`. The generated intensity filename uses that selection as its atom tag.

`Forward.jl` skips atom outputs already present in the intensity HDF5. In Bifrost mode, if a required active-atom population folder or `out_pop` file is missing, it skips that atmosphere and prints the dataset/snapshot that could not run.

Run the synthesis directly with `julia Forward.jl`. To process only one
atmosphere and snapshot, including one not declared in `MULTI3D_PRED_DATA`, use:

```bash
julia Forward.jl --predname en024048_hion_386
```

An undeclared name is interpreted as `<atmosphere>_<snap>` and uses the same
`PRED_DIR/bifrost_data/<atmosphere>/<snap>` layout as `pipeline.py`. Before
synthesis, `Forward.jl` checks the selected `mesh`, `atm3d`, and ML prediction
files and reports any that are missing.

When no Julia thread count is configured, the script restarts itself with
`--threads=auto` and preserves `--predname`. To choose a specific count, use
`julia --threads=N Forward.jl --predname en024048_hion_386` or set
`JULIA_NUM_THREADS=N` before starting Julia. The `Threads` line printed at
startup is the number of CPU threads the synthesis can use.

## Persistent FFNO Service

The inversion does not call `pipeline.py`. It uses a persistent prediction service
that keeps the trained models on the GPUs for the whole run:

- `ffno_fsdp_service.py`: a socket service (protocol version 2) that answers
  prediction and VJP requests for one or more species (H and Ca together in the
  example) until it is told to shut down.
- `ffno_fsdp_runtime.py`: FSDP `FULL_SHARD` parameters plus H-slab-distributed
  activations and FFTs, with eval-mode activation checkpointing. FSDP alone would
  replicate the full-volume activations. Only torchrun rank 0 reads each checkpoint.
- `ffno_fsdp_diagnostics.py`: opt-in per-rank logs (`FFNO_FSDP_DIAGNOSTICS=1` or `2`)
  and core dumps (`FFNO_FSDP_CORE_DUMPS=1`) for debugging native crashes.
- `ffno_model_factory.py`: builds FFNO3D from `MODEL_CONFIG` for the service.

Julia MPI rank 0 launches it with `FFNOInversion.jl/scripts/ffno_fsdp_service.sh`
(one `srun --overlap` torchrun job across all nodes of the allocation). The service
feeds log10 T and velocities in km/s, the same as training.
`scripts/compare_service_with_ffnonet.py` (with
`scripts/run_olivia_service_vs_ffnonet.sbatch`) checks that the service and a direct
`FFNONet.py` prediction agree to float32 round-off on the same atmosphere.

## 3D Forward Synthesis And Inversion

[`FFNOInversion.jl`](FFNOInversion.jl/README.md) is the Julia package for
atmosphere-to-spectrum synthesis and for fitting 3D atmospheres to observations.
In brief:

- **Inputs and outputs.** One atmosphere HDF5 (plus one observation HDF5 when
  inverting) in, one synthesis HDF5 and one atmosphere HDF5 out. A TOML file
  configures the run, and a model-factory Julia file names the checkpoints and atom
  data.
- **Vertical grids.** Native log10 tau500 (`forward.toml`, `inversion.toml`) or native
  height z (`forward_z.toml`, `inversion_z.toml`).
- **Equilibrium.** HSEMHS column hydrostatic equilibrium reconstructs gas pressure,
  density, electron density, heights (or log tau) and the 500 nm opacity. It sets the
  pressure boundary.
- **Populations.** FFNO3D through the persistent service above. For log-tau
  atmospheres each column is first mapped onto a training-like height mesh
  (`[population.mesh]`), because the networks were trained on simulation height grids.
- **Synthesis.** Any number of spectral regions, each combining FFNO lines (H alpha,
  Ca II 8542 through Muspel) and Kurucz LTE lines in one formal solution. On log-tau
  grids the optical depth is integrated on the native tau grid, with sub-stepping
  across opacity jumps.
- **Inversion.** Temperature, velocity, microturbulence and magnetic-field
  corrections on coarse node maps, weighted chi-square plus vertical and horizontal
  regularization, and bounded L-BFGS with matrix-free adjoint gradients.
  Checkpoints restart on a different MPI/thread layout.
- **Runtime.** Hybrid MPI ranks × Julia threads on CPU, with rank 0 controlling the
  multi-GPU FFNO service. Production runs go on Olivia through
  `FFNOInversion.jl/scripts/submit_olivia_inversion.sh`, which runs a login-node
  preflight before `sbatch`.

Start with [`examples/inversion_run/START_HERE.md`](examples/inversion_run/START_HERE.md).
Run-directory templates are in [`examples/`](examples/README.md).

## Preparing Inversion Inputs

`scripts/build_input_ltau.py` builds a native log-tau input atmosphere from a Bifrost
snapshot:

```bash
python scripts/build_input_ltau.py ATMOSPHERE.nc SUPPLEMENTARY.nc OUTPUT.nc
```

Each column's temperature, velocities and B (Bifrost height indices 0–466) are
interpolated linearly in log tau500 onto the 95-point grid log tau500 = −8.0 … 1.4
(step 0.1). The MULTI3D supplementary file stores `ltau500` with x and y swapped
relative to the Bifrost netCDF, and the script corrects that. `--as-stored`
reproduces the older, misregistered `input-mhs-ltau.nc` exactly. The output holds
`log10tau500`, `temperature`, `velocity_{x,y,z}` and `B_{x,y,z}` in NumPy order.
Map those names in the run's `[atmosphere.datasets]` with
`storage_order = "python"`, and check the units the inversion reader expects
(K, m/s, T).

Native-z inputs need `z_m`, `x_m` and `y_m` axes in metres in addition to the fields.

## Other Scripts

| Script | Purpose |
| --- | --- |
| `scripts/convert_iris_sim_fits_to_ffno_hdf5.py` | MURaM FITS to prediction solving set (see above) |
| `scripts/witt.py`, `scripts/witt_eos_cpp.cpp`, `scripts/pf_Kurucz.input` | Wittmann EOS (Python and C++) and partition functions. The C++ source is also compiled into the inversion's `libwitt_ffno` |
| `scripts/atmosphere_resampler.cpp` | C++/Boost spatial resampler used by the MURaM converter |
| `export_multi3d_depcoeff.py` | write MULTI3D departure coefficients in the FFNONet prediction layout |
| `compute_muspel_lte.jl`, `validate_multi3d_dep_loading.jl` | Muspel LTE populations, check of MULTI3D departure-coefficient loading |
| `compare_population_models.py` | per-level NRMSE of two prediction directories against MULTI3D |
| `make_paper_plots.py`, `generate_ffno_schematic_matplotlib.py` | paper figures and the architecture schematic |
| `scripts/generate_patch_training_animation.py` | animation of multiscale patch extraction (needs Pillow and ffmpeg) |
| `run_script_regression.sh` | example invocation of the Olivia runtime-test bootstrap |

## Tests

Python:

```bash
python -m pytest tests
```

The tests cover the composite NLTE loss, spacing-grouped batches, checkpoint
consistency, the FSDP service launcher, the Olivia case outcomes and the C++ Witt
EOS.

Julia (from `FFNOInversion.jl/`):

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The MPI parity checks and the Olivia regression chain are described in the
[package README](FFNOInversion.jl/README.md#tests-and-validation).

## Documentation

- `docs/ffno_model_flowchart.md` and the `docs/ffno_*.svg` diagrams: FFNO3D
  architecture (full forward pass, balanced block, spectral convolution, vertical
  stack).
- `docs/paper_draft_ffno_nlte.md`: paper draft.
- `output/pdf/`: inversion development plan, implementation report and HSE bootstrap
  review.
- `FFNOInversion.jl/docs/`: architecture decisions, the HSEMHS integration plan and
  validation reports.

## Caveats

- The code does not overwrite existing dataset, solving-set, prediction, or checkpoint outputs by default
- `--predict` and `--fsdppredict` assume `MODEL_FILE` already contains normalization and I/O metadata
- grouped patch datasets can have variable native depth across samples
- some configured output names still contain historical strings such as `s5`; these are just file naming conventions
- config values are the source of truth for splits, active atoms, paths, and model dimensions
- `FFNOInversion.jl` overwrites its configured output files; use a new run directory per experiment
- the inversion supports Stokes I, non-PRD synthesis and column HSE only; see the [package limitations](FFNOInversion.jl/README.md#known-limitations)

## Usage And Citation

If you want to use this code, please contact the author.

Please cite:

- H. Mathur and T. Pereira 2026, in preparation

## Author

Harsh Mathur
