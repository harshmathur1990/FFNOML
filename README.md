# FFNOML

FFNOML is a training and inference pipeline for predicting NLTE departure coefficients from 3D MULTI3D/Bifrost atmospheres with Fourier Neural Operators.

The codebase currently supports five main workflows through `pipeline.py`:

- `--build`: build HDF5 training and validation datasets from MULTI3D outputs
- `--train`: train a model checkpoint from the built HDF5 datasets
- `--test`: run validation diagnostics and branch-ablation analysis on the validation set
- `--predict`: build inference inputs for configured atmospheres and write predicted departure coefficients
- `--fsdppredict`: run distributed full-volume prediction through `torchrun`

This README describes the code as it exists now. The current pipeline is `z_scale`-based, not column-mass-based.

## What The Model Learns

The model maps atmospheric state variables to NLTE departure coefficients:

`[B, Cin, D, H, W] -> [B, Cout, D, H, W]`

Input channels are constructed in `FFNONet.py` from:

- `log10(temp)`
- `vx`
- `vy`
- `vz`
- `log10(ne)`
- `log10(rho)`

Targets are:

- `log10(n_NLTE / n_LTE)`

Prediction outputs are written back in linear space as:

- `departure_coefficients = n_NLTE / n_LTE`

## Main Files

- `pipeline.py`: CLI entrypoint for dataset building, training, validation testing, and prediction
- `config.py`: dataset splits, active atoms, model hyperparameters, file paths, and runtime flags
- `FFNONet.py`: dataset builders, HDF5 writers, training loop entrypoints, validation diagnostics, and inference
- `model_builder.py`: model/loss/optimizer construction and optional FSDP wrapping
- `data_builder.py`: selects patch or cube datasets and builds PyTorch dataloaders
- `data/dataset.py`: lazy HDF5 dataset implementations
- `train_utils.py`: checkpoint save/load, resume handling, validation helpers, and model expansion utilities
- `Forward.jl`: downstream Julia-side forward synthesis utilities
- `errorplots.py`: compares saved predictions against MULTI3D truth and plots error envelopes on `z_scale`

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
5. Builds input features and target departure coefficients
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

- `MODEL_DIR + "val_diagnostics_<MODEL>.json"`

The JSON summary includes:

- baseline loss
- baseline component breakdown
- baseline model statistics
- validation coverage metadata, including inference strategy, visited dataset items, and columns
- ablation results
- branch importance scores

## Predict Mode

Run:

```bash
python pipeline.py --predict
```

Predict mode uses `MULTI3D_PRED_DATA` from `config.py`.

What it does for each configured atmosphere:

1. Loads `atm3d` and `mesh`
2. Computes `dx`, `dy`, and `z_scale`
3. Builds a solving-set HDF5 if it does not already exist
4. Loads normalization stats and channel metadata from `MODEL_FILE`
5. Runs full-cube or tiled inference
6. Writes predicted departure coefficients in linear space

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
python scripts/convert_muram_fits_to_ffno_hdf5.py \
  --folder /mn/stornext/d9/data/harshm/bifrost_data/ar098192/atmos \
  --simulation-code MURaM \
  --simulation-name ar098192 \
  --snap 270000 \
  --output training_FFNO3D_zscale_expand/3D_sim_predict_ar098192_270000.hdf5 \
  --electron-density-mode witt-rho \
  --show-eos-progress \
  --multi3d-atmos-out /mn/stornext/d9/data/harshm/bifrost_data/ar098192/270000/atm3d \
  --multi3d-mesh-out /mn/stornext/d9/data/harshm/bifrost_data/ar098192/270000/mesh
```

The converter reads `lgtg`, `lgr`, `ux`, `uy`, and `uz` FITS files, reverses the
selected height range so the first depth index is the top of the atmosphere, and
writes `inputs`, `z_scale`, `dx`, and `dy` in the layout used by
`--fsdppredict`. With `--electron-density-mode witt-rho`, electron density is
computed from temperature and gas density using the Witt EOS. The converter
finds the repo-local `scripts/witt.py` and `scripts/pf_Kurucz.input`
automatically; use `--witt-path` only when those files live somewhere else. By
default, this uses the C++ full-atmosphere EOS backend and all visible CPU
threads; `--show-eos-progress` prints a C++-side progress line without Python
callbacks. Use `--eos-backend python` only for debugging or if no C++ compiler
is available.
The optional `--multi3d-atmos-out` and `--multi3d-mesh-out` outputs write a
plain Multi3D atmosphere for reference calculations. They contain temperature,
electron density, gas density, and velocity only, with no magnetic field or
hydrogen populations.

To run distributed prediction on only this generated file:

```bash
torchrun --nproc_per_node=4 pipeline.py --fsdppredict --predname ar098192_270000
```

The prediction path will look for
`MODEL_DIR + "3D_sim_predict_ar098192_270000.hdf5"`.

### Prediction Output Layout

Prediction outputs are written to:

- `MODEL_DIR + "output_3D_sim_s5_<NAME>_<MODEL>.hdf5"`

The file contains:

- `departure_coefficients`: `[nx, ny, D, Cout]`
- `z_scale`: `[D, nx, ny]`

Current attributes include:

- `departure_coefficients.attrs["depth_scale_type"] = "z"`
- file attribute `epoch`
- file attribute `val_loss` when available in the checkpoint

Prediction also writes a diagnostics path per atmosphere:

- `MODEL_DIR + "diagnostics_3D_sim_s5_<NAME>_<MODEL>.npz"`

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

It now compares on shared `z_scale` directly and does not interpolate to column mass.

## Forward Synthesis

`Forward.jl` is the Julia-side consumer for predicted departure coefficients and related atmospheric inputs. Use it for downstream synthesis after Python-side prediction has produced the HDF5 outputs.

## Caveats

- The code does not overwrite existing dataset, solving-set, prediction, or checkpoint outputs by default
- `--predict` and `--fsdppredict` assume `MODEL_FILE` already contains normalization and I/O metadata
- grouped patch datasets can have variable native depth across samples
- some configured output names still contain historical strings such as `s5`; these are just file naming conventions
- config values are the source of truth for splits, active atoms, paths, and model dimensions

## Usage And Citation

If you want to use this code, please contact the author.

Please cite:

- H. Mathur and T. Pereira 2026, in preparation

## Author

Harsh Mathur
