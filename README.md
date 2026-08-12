# FFNOML

FFNOML is a training and inference pipeline for predicting NLTE populations from 3D MULTI3D/Bifrost atmospheres with Fourier Neural Operators.

The codebase currently supports five main workflows through `pipeline.py`:

- `--build`: build HDF5 training and validation datasets from MULTI3D outputs
- `--train`: train a model checkpoint from the built HDF5 datasets
- `--test`: run validation diagnostics and branch-ablation analysis on the validation set
- `--predict`: build inference inputs for configured atmospheres and write predicted NLTE populations
- `--fsdppredict`: run distributed full-volume prediction through `torchrun`

This README describes the code as it exists now. The current pipeline is `z_scale`-based, not column-mass-based.

## What The Model Learns

The model maps atmospheric state variables to NLTE populations:

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

For a named prediction, both paths can be overridden without changing
`config.py`. This is used by `Forward.jl` so electron-density consistency runs
operate on a copy instead of changing the original solving set:

```bash
torchrun --nproc_per_node=4 pipeline.py --fsdppredict \
  --predname ar098192_270000 \
  --solve-h5 /path/to/working-solving-set.hdf5 \
  --prediction-output /path/to/working-populations.hdf5
```

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
is available. The optional `multi3d_atmos` and `multi3d_mesh` outputs write a
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
Factors and target grid sizes remain available as alternatives. Interpolation
supports `nearest`, `linear`, and natural `cubic`
spline methods independently per x/y/z direction, with per-channel overrides
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
- `electron_density`: `[nx, ny, D]` in m^-3 when the solving HDF5 contains a
  `gpu_charge_model` request from Forward.jl charge-only mode

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

## Forward Synthesis

`Forward.jl` is the Julia-side consumer for predicted NLTE populations and related atmospheric inputs. Use it for downstream synthesis after Python-side prediction has produced the HDF5 outputs.

For ML mode, `Forward.jl` reads `output_3D_sim_s5_<NAME>_<MODEL>_<FORWARD_ATOMS>.hdf5` and writes intensity files with the same Forward atom tag.

Set `FORWARD_ATOMS` near the top of `Forward.jl` to choose which atoms to synthesize, for example `["H"]`, `["CA"]`, or `["H", "CA"]`. The generated intensity filename uses that selection as its atom tag.

`Forward.jl` skips atom outputs already present in the intensity HDF5. In Bifrost mode, if a required active-atom population folder or `out_pop` file is missing, it skips that atmosphere and prints the dataset/snapshot that could not run.

Each `MULTI3D_PRED_DATA` entry in `config.py` can also control Forward synthesis:

```python
{
    "NAME": "en024048_hion_385",
    "MESH": ".../mesh",
    "MULTI3D_ATMOS": ".../atm3d",
    "FORWARD": True,
    "NONLTE_NE": True,
    "CHARGE_CONSERVATION_MAX_ITERATIONS": 10,
    "POPULATION_CONSISTENCY_MODE": "charge-only",
    "HYDROGEN_SE_WAVELENGTH_STRIDE": 2,
}
```

`FORWARD=False` excludes that atmosphere from `Forward.jl` without affecting
Python prediction commands. `NONLTE_NE=True` requires non-LTE electron-density
consistency and therefore also requires
`--charge-conservation-max-iterations N` with `N > 0`. `NONLTE_NE=False` forces
one-shot synthesis for that atmosphere even when the global iteration option is
present. Omitting `NONLTE_NE` (or setting it to `None`) preserves the previous
behavior and follows the global command-line option. Both keys are optional.

`CHARGE_CONSERVATION_MAX_ITERATIONS`, `POPULATION_CONSISTENCY_MODE`, and
`HYDROGEN_SE_WAVELENGTH_STRIDE` provide per-atmosphere versions of the matching
Forward command-line options. A value set in the atmosphere entry takes
precedence over the command line. Set a key to `None` or omit it to use the
command-line value (and ultimately its normal default).

In ML mode, charge-conserving inference can be enabled with a maximum number of
fixed-point iterations:

```bash
julia Forward.jl --charge-conservation-max-iterations 10
```

Each iteration launches `pipeline.py --fsdppredict`, reads the predicted
non-LTE hydrogen populations, recomputes electron density from charge neutrality
(hydrogen in non-LTE and all background elements in LTE), writes `log10(ne)` to
channel 5 of a method-specific working/result HDF5, and repeats. The original
solving-set and population-prediction HDF5 files are never opened for writing by
the consistency loop. The loop stops when the
maximum relative electron-density residual is at most `1e-4`, or at the supplied
iteration limit. Hydrogen must be included in `FORWARD_ATOMS`. Use
`--charge-conservation-tolerance VALUE` to change the tolerance and
`--fsdp-nproc-per-node N` to change the `torchrun` process count. Omitting the
maximum-iteration option keeps the existing one-shot behavior.

In `charge-only` mode, every distributed torch worker evaluates the same
Saha-Boltzmann equations directly on its GPU slab immediately after FFNoML
prediction. Julia exports the fixed hydrogen density and compact background
atomic data (energies, statistical weights, stages, and abundances) into the
working solving HDF5. GPU arithmetic is Float64 for the Saha calculation and
the final electron density is stored as Float32, matching the CPU reference
path. This is a direct calculation, not a lookup-table approximation. The
prediction HDF5 carries the resulting `electron_density` back to the Julia MPI
ranks. On every iteration, every Julia rank recomputes 64 deterministic sample
cells with the original Muspel CPU routine; the run aborts if the global maximum
sampled relative error exceeds `5e-5`. `hydrogen-se-1p5d` retains the CPU charge calculation because its
hydrogen populations are corrected after FFNoML has returned.

Two population-consistency modes are available. The existing default is the
inexpensive charge-only update:

```bash
julia Forward.jl \
  --charge-conservation-max-iterations 10 \
  --population-consistency-mode charge-only
```

The preferred statistical-equilibrium correction is selected with:

```bash
julia Forward.jl \
  --charge-conservation-max-iterations 10 \
  --population-consistency-mode hydrogen-se-1p5d
```

`hydrogen-se-1p5d` uses FFNoML populations as the initial state, solves the
radiative transfer equation independently in every vertical column, computes
bound-bound and bound-free radiative rates, adds the hydrogen atom file's
CE/CI/Omega electron-collision rates, solves the stationary hydrogen rate
equations with hydrogen-particle conservation, and then imposes charge
conservation. The corrected electron density is fed to the next FFNoML call.
Convergence requires the electron-density residual, SE residual, and relative
FFNoML-to-SE population correction all to meet the requested tolerance.

Every method writes a separate result next to the original solving set:

```text
nonlte_electron_density_<NAME>_<MODEL>_<ATOMS>_<METHOD>.hdf5
```

The file begins as a copy of the original solving set and contains:

- `initial_electron_density` and the latest `electron_density`, in m^-3 with
  dimensions `(z,y,x)`;
- `nlte_populations`, including the SE-corrected hydrogen populations in
  `hydrogen-se-1p5d` mode;
- `last_ffno_hydrogen_populations`, preserving the uncorrected FFNoML state for
  an accurate population delta after restart;
- `iteration_history/` datasets for the iteration number, electron-density and
  population deltas, SE residual/correction, background-scattering iteration
  count/residual, density/population ranges,
  per-hydrogen-level deltas/ranges, and all phase timings;
- file attributes including `method`, `converged`, `iterations_completed`,
  `convergence_iteration`, tolerance, HSE settings, final residuals, timestamps,
  and fingerprints of the source solving set, hydrogen atom, and model
  checkpoint.

When iteration is requested again, `Forward.jl` checks this file first. A
converged result that meets the newly requested tolerance is loaded directly
and FFNoML is skipped. An unconverged result, or a result that met only a looser
tolerance, resumes from its latest electron density and appends up to the newly
requested number of additional iterations. If the source solving set, hydrogen
atom, model checkpoint, method, or HSE numerical settings changed, the old file
is moved to a `.stale-TIMESTAMP` name and a fresh result is started.

This mode is substantially more expensive than `charge-only`. It is a stationary
3D CRD correction with periodic horizontal boundaries and inclined
characteristics through the complete x/y planes. All velocity components are
projected onto the azimuthal ray quadrature. Atom transitions marked PRD are
treated in CRD. Background H-minus, hydrogen free-free, H2-plus,
Thomson, and neutral-hydrogen Rayleigh processes are included. Following RH,
true continuum absorption has thermal emissivity while continuum scattering is
iterated as `eta_scat = chi_scat J`; it is not folded into a Planck source.
LTE background-metal bound-free opacity is also included as true absorption.
As in Multi3D, transport reuses precomputed periodic interpolation stencils
within each height sweep instead of searching the horizontal axes for every
cell and field. Continuum coefficients use Muspel's 100-point log-temperature
and log-electron-density interpolation tables, cached per wavelength and
invalidated automatically if a later consistency iteration leaves their
thermodynamic range.
The compact corrector still omits background bound-bound line haze and the
H2/H2-minus and helium Rayleigh terms for which its atmosphere currently has no
molecular/helium populations. `--hydrogen-se-relaxation VALUE`
damps population updates. `--hydrogen-se-wavelength-stride N` reduces cost by
sampling every Nth transition wavelength; the default `1` uses every wavelength.

### MPI Forward Synthesis

`Forward.jl` distributes the horizontal x dimension over MPI ranks for input,
the original synthesis path, and `charge-only`:

```bash
srun --ntasks=32 \
  julia --threads=8 Forward.jl --mpi
```

The atmosphere is read once on MPI rank 0 and distributed with `MPI.Scatterv!`.
Each rank retains only its local x slab, reads only that slab from the FFNoML
prediction HDF5, and uses Julia threads over the flattened set of local `(x,y)`
columns. This lets a rank use all of its threads even when its local x count is
smaller than `--threads`; synthesis keeps one reusable radiative-transfer buffer
per Julia worker thread. Intensities are gathered in rank/x order for the final
serial HDF5 write. In the consistency modes,
electron-density and population residuals use a global MPI maximum reduction.
The 3D hydrogen-SE stage cannot treat x/y slab boundaries as physical transfer
boundaries. It therefore gathers complete horizontal planes onto one Julia rank
per node, distributes wavelengths over those node ranks, and threads cell-local
work in height slabs. Height planes in each characteristic remain causally
ordered, while the independent destination pixels of each complete x/y plane
are evaluated by all node threads. Radiative rates are summed across the
wavelength ranks;
the final rate-matrix solves are divided by height and reduced before corrected
populations are scattered back to the synthesis x slabs. Rank 0 gathers
electron-density slabs for durable checkpoints. The full population dataset is
copied directly between HDF5 files rather than gathered through rank-0 memory.

MPI mode requires MPI.jl. On a Cray/Slurm system, configure MPI.jl against the
loaded system MPI before launching the job, then restart Julia:

```bash
julia -e 'import Pkg; Pkg.add(["MPI", "MPIPreferences"])'
julia -e 'using MPIPreferences; MPIPreferences.use_system_binary(mpiexec="srun", vendor="cray")'
```

The supplied [`forward_mpi_gpu.sh`](forward_mpi_gpu.sh) is an eight-node Slurm
template using one wavelength MPI rank per node with 256 Julia threads. This
gives the full-volume SE worker access to all 256 CPU cores on its node while
the separate torchrun step continues to launch four GPU workers per node.
Adjust its resource directives to the atmosphere size and cluster policy.
It defaults prediction data and Multi3D atom files to Olivia project storage.
Set `FNOML_PRED_DIR` (the directory containing `bifrost_data`) and
`FNOML_ATOM_DIR` (the directory containing the atom YAML files) to use another
filesystem. `FNOML_PROJECT_STORAGE_ROOT` changes both Olivia-derived defaults
at once.

Distributed FFNoML inference is deliberately not launched by every MPI rank.
Pass a launcher executable with `--fsdp-launcher`; only MPI rank 0 invokes it,
and every rank waits at an MPI collective until it succeeds:

```bash
sbatch forward_mpi_gpu.sh \
  --max-iterations 10 \
  --population-consistency-mode hydrogen-se-1p5d
```

[`forward_fsdppredict.sh`](forward_fsdppredict.sh) implements the launcher
contract `PATH PREDICTION_NAME SOLVE_H5 PREDICTION_OUTPUT`. It creates one
overlapping Slurm step with one `torchrun` launcher per
node and one worker per GPU, matching `predict_gpu.sh`. Set `FORWARD_TORCHRUN`,
`FORWARD_REPO_DIR`, or `FORWARD_MASTER_PORT` when the cluster defaults differ.
The Julia ranks never launch independent `torchrun` jobs and never read the
prediction file until the coordinated GPU step has completed. While that step
runs, each non-root Julia rank blocks on a small TCP control connection to rank
0 instead of holding an outer MPI collective open alongside NCCL. Rank 0 sends
one success or failure message when the launcher returns, so there is no polling
loop or shared-filesystem marker. The socket wait has no timeout by default;
Slurm's job wall-time remains the overall limit because valid prediction time
depends on atmosphere size. Set `FORWARD_FSDP_STATUS_TIMEOUT` to a positive
number of seconds to add a per-launch timeout. MPI resumes only after every rank
has received the launcher status.

### Progress, resource monitoring, and crash diagnostics

`Forward.jl` keeps the Slurm stdout readable: only MPI rank 0 prints the main
progress. A normal consistency iteration reports the FFNoML population change
(starting with `n/a` on iteration 1), the electron-density residual and range,
the hydrogen SE residual and FFNoML-to-SE correction in preferred mode, and a
compact timing split for FFNoML, prediction reading, SE, charge conservation,
HDF5 I/O, and the whole iteration. When the CPU charge path is used after a
hydrogen-SE correction, rank 0 reports percentage complete, elapsed time,
estimated remaining time, and processing rate every 30 seconds. The estimate
follows rank 0's local x slab, which is one of the largest rank partitions and
therefore approximates the completion time of the full distributed phase.
Progress accounting uses small 100-cell work chunks so updates remain timely
even when each cell is expensive. Charge-only mode instead reports the direct
GPU Saha stage's total duration. Before that GPU charge calculation, the six
predicted hydrogen levels are rescaled independently in every cell so that
their sum equals the fixed total hydrogen density from the atmosphere. The
relative distribution predicted by FFNoML is preserved. Stdout reports the
raw predicted-sum/fixed-density range and the applied scale-factor range, and
the same values are stored as attributes on the prediction HDF5 file.
Atmosphere distribution, each atom's
synthesis progress/time, output gathering, and completion are also reported.
The Slurm template writes these streams to `forward-JOBID.out` and
`forward-JOBID.err`.

Detailed information goes to a diagnostics directory instead of stdout. Its
default name is `forward-diagnostics-slurm-JOBID`; the Slurm template sets the
absolute path in `FORWARD_DIAGNOSTICS_DIR` and prints it at startup. It can be
overridden through the environment so Julia and the Slurm termination handler
write to the same place:

```bash
FORWARD_DIAGNOSTICS_DIR=/cluster/work/projects/nn2834k/harshm/my-forward-debug \
sbatch forward_mpi_gpu.sh \
  --resource-monitor-interval 15 \
  --max-iterations 10 \
  --population-consistency-mode hydrogen-se-1p5d
```

The directory contains one set of files per MPI rank:

- `events-rank-NNNN.log` records timestamped phase transitions and timings. For
  every consistency iteration it includes global overall and per-hydrogen-level
  population changes/minima/maxima, SE metrics, electron-density metrics, the
  local x slab, and phase timings.
- `resources-rank-NNNN.csv` is sampled every 30 seconds by default and at major
  phase boundaries. It contains process RSS and peak RSS, Julia live GC bytes,
  aggregate process CPU percentage, one-minute node load, available/total node
  memory, Julia thread count, Slurm CPUs per task, and Linux CPU affinity. A
  fully busy 64-thread rank can read near 6400% CPU because this is aggregate
  process CPU, not a value normalized to one allocation.
- `failure-rank-NNNN.log` is created on a caught failure and records the current
  dataset, iteration, phase, resource snapshot, exception, and full Julia stack
  trace. In MPI mode the message is flushed before the communicator is aborted,
  avoiding ranks hanging indefinitely after a collective failure.

Set `--resource-monitor-interval 0` or `--no-resource-monitor` to disable timed
sampling; phase-boundary resource snapshots remain enabled. Diagnostics files
are flushed after every record so useful data normally survives a Julia error.
An operating-system OOM kill, node loss, or `SIGKILL` cannot be caught by Julia.
For that case the Slurm template requests a two-minute termination warning and
writes `slurm-termination.log` when Slurm delivers `TERM`/`INT`. The scheduler is
still authoritative; while a job runs and after it finishes, useful checks are:

```bash
sstat -j "${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,AveRSS
sacct -j "${SLURM_JOB_ID}" --format=JobID,State,ExitCode,Elapsed,AllocCPUS,MaxRSS
```

Before expensive work begins, the program checks required atmosphere, mesh,
atom, solving-set/prediction, and launcher files, launcher executability, and
the output directory. Prediction populations are checked for non-finite or
negative values on every iteration; electron-density writes reject non-finite
or non-positive values. The active phase in the failure log identifies which
preflight, FFNoML, SE, charge, synthesis, collective, or HDF5 operation failed.

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
