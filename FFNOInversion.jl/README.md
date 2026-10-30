# FFNOInversion.jl

`FFNOInversion.jl` is the Julia forward-synthesis and inversion layer of FFNOML.
It takes a 3D solar atmosphere (temperature, velocities, optional magnetic field),
reconstructs its hydrostatic structure with HSEMHS, predicts NLTE level populations
with the trained FFNO3D networks, synthesizes Stokes I spectra for several spectral
regions at once, and, in inversion mode, fits the atmosphere to observed spectra
with a spatially coupled, bounded L-BFGS solver.

The whole volume is treated as one problem: columns are coupled through the FFNO
Fourier layers, the spatial PSF and horizontal regularization, so the inversion is
not a set of independent 1D fits.

**First run?** Follow the [starter guide](../examples/inversion_run/START_HERE.md).
It walks through the run directory, input files, Olivia submission and outputs.
This README is the reference for everything the guide leaves out.

## Contents

- [What one evaluation does](#what-one-evaluation-does)
- [Status](#status)
- [Run contract: two inputs, two outputs](#run-contract-two-inputs-two-outputs)
- [Native vertical coordinates](#native-vertical-coordinates)
- [Configuration reference](#configuration-reference)
- [The model factory](#the-model-factory)
- [Running](#running)
- [Physics and numerics](#physics-and-numerics)
- [Parallel runtime](#parallel-runtime)
- [Package layout](#package-layout)
- [Tests and validation](#tests-and-validation)
- [Design documents and reports](#design-documents-and-reports)
- [Known limitations](#known-limitations)

## What one evaluation does

Every forward evaluation, and every objective evaluation inside the inversion,
runs the same phases in the same order on every MPI rank:

```text
control maps (inversion only)
   │  A = A_start + interpolate(p − p_initial) on each rank's tile
   ▼
HSEMHS equilibrium          Pgas, rho, ne, z (or log10 tau500) and chi500 from T (and B)
   ▼
population mesh (log tau)   columns mapped onto a training-like height mesh
   ▼
FFNO3D populations          persistent multi-GPU FSDP service, one model per species
   ▼
mixed synthesis             FFNO NLTE lines + Kurucz LTE lines + continuum, per region
   ▼
observation model           Gaussian spectral/spatial PSF on the full grid
   ▼
objective (inversion only)  weighted chi-square + vertical/horizontal regularization
```

One-rank development runs and multi-rank production runs use the same
`HybridForwardModel` and `forward!` code path.

## Status

| Area | State |
| --- | --- |
| Forward synthesis (`mode = "forward"`) | Production. Native log10 tau500 or native z atmospheres. |
| Inversion (`mode = "inversion"`) | Implemented: bounded L-BFGS with matrix-free gradients, checkpoints, restart across topologies. |
| Equilibrium | HSEMHS column HSE (`column-hse-ltau`, `column-hse-z`). MHS Poisson modes (`model = "mhs"`) are rejected until HSEMHS-integration Phase 4. |
| Populations | FFNO3D through the persistent FSDP GPU service (H and Ca II, 6 levels each in the example factory). At least two GPU ranks are required. |
| Spectral lines | FFNO: H alpha (`halpha`), Ca II 8542 (`ca_ii_8542`). LTE: any Kurucz line list. |
| Stokes / redistribution | Stokes I, non-PRD (Release 1). The type boundaries for PRD and full Stokes exist (ADR-004) but are not implemented. |
| Clusters | Olivia (Slurm `accel` partition, NVIDIA GPUs) for production; any MPI machine for CPU tests. |

The legacy native force-balance (HE3D/MHS) backend has been removed. HSEMHS is the
only equilibrium backend (ADR-012).

## Run contract: two inputs, two outputs

A run has exactly two scientific inputs and two scientific outputs (ADR-006). The
TOML file only configures paths and options; it is not a scientific input. The model
factory supplies checkpoints and atom data; it is not a scientific input either.

| | File | Used in |
| --- | --- | --- |
| Input 1 | initial atmosphere (`[inputs] initial_atmosphere_file`) | forward and inversion |
| Input 2 | observations (`[inputs] observation_file`) | inversion only |
| Output 1 | synthesis (`[outputs] synthesis_file`) | both |
| Output 2 | atmosphere (`[outputs] atmosphere_file`) | both |

Output files are overwritten. Use a new run directory or new output names per
experiment.

### Initial atmosphere (HDF5)

| Dataset | Shape | Units | Notes |
| --- | --- | --- | --- |
| `logtau_500` | `(z)` | log10 tau at 500 nm | log-tau runs only; strictly monotonic |
| `z_m` | `(z)` | m, positive upward | z runs only; either order |
| `x_m`, `y_m` | `(x)`, `(y)` | m | required for z runs, optional for log tau; uniform and increasing |
| `temperature` | `(time,z,y,x)` or `(z,y,x)` | K | required |
| `vx`, `vy`, `vz` | as temperature | m/s | all three required |
| `vturb` | as temperature | m/s | optional; zero when not mapped |
| `Bx`, `By`, `Bz` | as temperature | T | optional, all three or none; absent means B = 0 |

Dataset names are configurable in `[atmosphere.datasets]`. Set
`[atmosphere] storage_order = "python"` for files written by NumPy/h5py or C, so
row-major dimensions are reversed on read; use `"julia"` (the default) for
HDF5.jl files. `[inputs] time_index` (1-based) selects the snapshot.

Inputs may be stored bottom-first or top-first. They are reversed once on read so
that inside the package index `k = 1` is always the top of the atmosphere.

Pressure, density and electron density are not inputs. They are reconstructed by
HSEMHS. A starting atmosphere must already satisfy the physical bounds configured
for the controls.

[`../scripts/build_input_ltau.py`](../scripts/build_input_ltau.py) builds a native
log-tau input from a Bifrost snapshot. It writes `log10tau500`, `temperature`,
`velocity_{x,y,z}` and `B_{x,y,z}` in Python order, so map those names in
`[atmosphere.datasets]` and check that the units match the table above.

### Observations (HDF5, inversion only)

| Dataset | Shape | Notes |
| --- | --- | --- |
| `intensity` | `(time,Stokes,wavelength,y,x)` or without time | same wavelength grid as the configured regions |
| `sigma` | as `intensity` | strictly positive |
| `wavelength_weights` | `(wavelength,Stokes)`, optionally with time | zero excludes a wavelength or Stokes component |
| `spatial_weights` | `(y,x)`, optionally with time | zero excludes a pixel |

Names are configurable in `[observation.datasets]`. The observation grid must match
the atmosphere's spatial grid. Nothing is interpolated.

### Synthesis output

| Dataset / attribute | Content |
| --- | --- |
| `intensity` | `(time,Stokes,wavelength,y,x)` after the configured PSF |
| `wavelength_m` | concatenated region wavelengths (air, m) |
| `objective_total`, `objective_data`, `objective_regularization` | inversion only |
| `execution_mode`, `axis_order`, `provenance_toml` | attributes |

### Atmosphere output

| Dataset / attribute | Content |
| --- | --- |
| `logtau_500` or `z_m`, `x_m`, `y_m` | native axes |
| `temperature`, `vx`, `vy`, `vz`, `vturb`, `Bx`, `By`, `Bz` | recovered (or input) fields, `(time,z,y,x)` |
| `pgas`, `rho`, `ne`, `z`, `logtau500`, `chi500` | HSEMHS equilibrium, `(time,z,y,x)` |
| `populations/<species>` | FFNO populations, `(level,z,y,x)`, m⁻³ |
| `vertical_coordinate`, `vertical_unit`, `vertical_order = "top_first"` | attributes |
| `execution_mode`, `solver_termination`, `provenance_toml` | attributes |

`provenance_toml` records the configuration hash, parallel topology, equilibrium
mode and final status, EOS provenance, and the control mapping.

## Native vertical coordinates

`[atmosphere] vertical_coordinate` chooses the native grid. Everything else follows
from it:

| | `"log10tau500"` | `"z"` |
| --- | --- | --- |
| Example configs | `forward.toml`, `inversion.toml` | `forward_z.toml`, `inversion_z.toml` |
| Vertical dataset | `logtau500` | `z_m`, plus `x_m` and `y_m` |
| HSEMHS mode | `column-hse-ltau`: heights derived per column | `column-hse-z`: log10 tau500 derived per column |
| Pressure boundary | HSEMHS tau-hydrostatic top closure (`[equilibrium.column_ltau]`) | `[equilibrium.column_z] anchor`, `P_ref_pa` |
| FFNO populations | on the `[population.mesh]` height mesh, then interpolated back | directly on the native grid |
| Formal solution | optical depth integrated on the native log tau500 grid using chi500 | optical depth integrated in geometric height |
| Control nodes | `log_tau_nodes` | `z_nodes_m` |
| Regularization length | `length_scale_dex` (default 1) | `length_scale_m` (required for derivative types 1 and 4) |

A native height axis is never treated as log tau, and a column-dependent derived log
tau is never collapsed into one shared vector.

## Configuration reference

The checked-in configs in [`../examples/inversion_run`](../examples/inversion_run)
are commented and complete. `scripts/dry_run.jl` prints the parsed configuration.
Unknown keys in `[equilibrium]` and its subsections are errors.

### Top level, `[inputs]`, `[outputs]`

| Key | Meaning |
| --- | --- |
| `mode` | `"inversion"` (default) or `"forward"`. `[solver] max_iterations = 0` also selects forward. |
| `[inputs] initial_atmosphere_file`, `observation_file`, `time_index` | input paths (relative to the run directory) and 1-based snapshot |
| `[outputs] synthesis_file`, `atmosphere_file` | output paths |

### `[atmosphere]`

`vertical_coordinate`, `storage_order` and the `[atmosphere.datasets]` name map
(`logtau500` or `z_m`, `x_m`, `y_m`, `temperature`, `vx`, `vy`, `vz`, optional
`vturb`, `Bx`, `By`, `Bz`). There is no top-pressure or top-density option: configs
that still contain `pressure_top_pa` or `pressure_top_dataset` are rejected.

### `[equilibrium]`

| Key | Meaning |
| --- | --- |
| `backend` | must be `"HSEMHS"` |
| `model` | `"column_hse"`; `"mhs"` is reserved for Phase 4 |
| `gravity_m_s2` | surface gravity (default 274.0) |
| `eos_table_file` | HSEMHS EOS table, e.g. a link to `HSEMHS/cache/eos_table_fullz.h5` |
| `on_nonconvergence` | `"reject_trial"` (an inversion trial is rejected) or `"abort"` |
| `[equilibrium.column_ltau] align` | `"ptot"` (P + B²/2μ₀ equal at one height) or `"none"` |
| `[equilibrium.column_z]` | `anchor` (`bottom`, `top`, `height`), `P_ref_pa` (required), `z_ref_m`, `tau_top` (`exp_tail`, `half_cell`, `zero`), and the HSEMHS integrator options `n_sub`, `adapt`, `rk_tol`, `n_sub_max`, `H_max_m`, `mu` |

These options use HSEMHS's names, defaults and semantics. HSEMHS is the authority
on the pressure boundary, and the boundary is not an inversion control.
`prepare_equilibrium` checks that the EOS table was built with the same partition
functions and abundances as the Wittmann library used by synthesis.

### `[grid]`

`dx_m`, `dy_m`: horizontal spacing in metres. With explicit `x_m`/`y_m` datasets
they are optional, and must agree with the axes to 1e-9 when given.

### `[[regions]]` and `[[regions.sources]]`

Each `[[regions]]` table is one spectral window, synthesized simultaneously with the
others:

| Key | Meaning |
| --- | --- |
| `start_angstrom`, `step_angstrom`, `count` | uniform air-wavelength grid (Å) |
| `normalization` | positive number, validated (see [Known limitations](#known-limitations)) |
| `psf_type`, `psf_file` | `"none"`, `"fpi"` or `"file"`; a file is required unless `none` (see [Known limitations](#known-limitations)) |

Each region has one or more sources:

| `mode` | Keys | Meaning |
| --- | --- | --- |
| `"ffno"` | `species` (`"H"`, `"CA"`), `line` (`"halpha"`, `"ca_ii_8542"`) | NLTE line from FFNO populations through Muspel |
| `"kurucz_lte"` | `linelist_file` | LTE lines from a Kurucz list, Wittmann EOS |

Sources in one region share a single formal solution, so blends are synthesized
together. The first source owns the continuum, so it is not counted twice. A region
with only LTE sources does not call FFNO.

### `[population.mesh]` (log-tau runs)

`enabled` (default true for log tau, ignored for z), `bottom_m`, `top_m`,
`spacing_m`, `uniform_top_m`, `stretch`, `max_spacing_m`, `nabla`. The defaults
(−1 Mm to 14 Mm, 20 km spacing up to 4.5 Mm, then growing by 1 % per point to 100 km,
∇ = 0.13) lie inside the range of the training grids. See
[Population mesh](#population-mesh-for-log-tau-atmospheres).

### `[physics]`, `[observation]`

| Key | Meaning |
| --- | --- |
| `[physics] redistribution` | `"non_prd"` |
| `[observation] stokes` | `["I"]` |
| `[observation.datasets]` | names of `intensity`, `sigma`, `wavelength_weights`, `spatial_weights` |
| `[observation.gaussian_psf]` | `spatial_fwhm_x_m`, `spatial_fwhm_y_m`, `spectral_fwhm_nm` (0 disables a direction) |

The PSF applies to forward runs too: set the FWHM values to 0 for unblurred spectra.

### `[regularization]`

| Key | Meaning |
| --- | --- |
| `horizontal_order` | 1 or 2 (derivative order of the horizontal penalty) |
| `[regularization.vertical] types` | seven types for the slots `(Temp, Vlos, vturb, B, inc, azi, pgas_boundary)`: 0 none, 1 first derivative, 2 deviation from the depth mean, 3 deviation from zero, 4 second derivative |
| `[regularization.vertical] regularize`, `weights` | global multiplier and seven relative weights |
| `[regularization.vertical] length_scale_dex` / `length_scale_m` | vertical length unit for derivatives |
| `[regularization.horizontal]` | per-variable strengths (`temperature`, `vx`, `vy`, `vz`, ...) |
| `[regularization.scales]` | per-variable scales the fields are divided by before penalizing |

Temperature types 2 and 3 are normalized to 0. Regularization is evaluated on the
full recovered atmosphere, independent of the chi-square weights (ADR-007).

### `[inversion]` controls

Repeat `[[inversion.controls]]` for each inverted variable:

| Key | Meaning |
| --- | --- |
| `variable` | `temperature`, `vx`, `vy`, `vz`, `vlos` (alias of `vz`), `vturb`, `Bx`, `By`, `Bz` |
| `log_tau_nodes` or `z_nodes_m` | vertical node positions (must match the native coordinate) |
| `control_nx`, `control_ny` | coarse horizontal control-map size |
| `lower`, `upper` | physical bounds of the full field |
| `scale` | unit scaling of this variable's parameters |

### `[solver]`

`method = "lbfgs"` (default) is the production solver. Options: `max_iterations`,
`history_length`, `gradient_tolerance`, `step_tolerance`, `objective_tolerance`,
`initial_step`, `backtracking_factor`, `armijo_coefficient`,
`maximum_line_search_trials`, `maximum_forward_evaluations`, `curvature_tolerance`,
`checkpoint_every`, `checkpoint_path`.

`method = "prototype_pattern_search"` is the derivative-free Phase 5 reference
solver (`maximum_evaluations`, `minimum_step`, `step_shrink`,
`improvement_tolerance`, ...). It is for validation, not production.

### `[parallel]`

| Key | Meaning |
| --- | --- |
| `enabled` | use MPI |
| `decomposition` | `"x_slabs"` (required by HSEMHS when MPI is enabled) or `"cartesian_2d"` |
| `threads_per_rank` | Julia threads per rank; the Olivia batch script sets it from `cpus-per-task`, so leave it out of copied configs |
| `min_columns_per_rank` | smallest tile width |
| `gpu_launcher_rank` | rank that launches and talks to the GPU service (0) |
| `gpu_connect_timeout_seconds`, `gpu_status_timeout_seconds` | GPU-service connect and status timeouts |
| `gpu_diagnostic_interval_seconds`, `gpu_diagnostics_directory` | periodic per-rank diagnostics |

The `FFNO_GPU_*` environment variables override these (ADR-009).

## The model factory

`scripts/invert.jl CONFIG.toml MODEL_FACTORY.jl` includes the factory file, which
must define `FFNO_INVERSION_FACTORY::InversionModelFactory`. The factory keeps
deployment details (checkpoint paths, atom files, EOS library) out of the scientific
inputs. [`../examples/inversion_run/model_factory.jl`](../examples/inversion_run/model_factory.jl)
is the production example. Its `build_production_model`:

1. builds the Wittmann EOS used by synthesis from `inputs/wittmann/libwitt_ffno.so`
   and `inputs/pf_Kurucz.input`;
2. prepares the HSEMHS equilibrium (`prepare_equilibrium`);
3. launches one persistent FSDP service for H and Ca
   (`launch_fsdp_population_models`, with `population_mesh_options(config)`);
4. reconstructs the initial equilibrium, predicts populations once and builds the
   cached synthesis setup (`build_synthesis_setup`);
5. returns a `HybridForwardModel`.

On an error it aborts all MPI ranks (`abort_on_rank_failure`) rather than leaving
them blocked in a collective. The file also returns the asset lists
(`assets`, `generated_assets`) that the Olivia preflight checks before submission.
The example factory loads both H and Ca even when only one is used by the lines.

## Running

### Install

Use Julia 1.12 (the manifest is resolved with 1.12.3). From this directory:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/build_wittmann_backend.jl
```

`Project.toml` pins HSEMHS (from `harshmathur1990/1DHSE3dMHS`, subdirectory
`HSEMHS`) and Muspel (`harshmathur1990/Muspel.jl`, commit `01ec68d`) by Git revision.
`build_wittmann_backend.jl` compiles `../scripts/witt_eos_cpp.cpp` with `$CXX`
(default `c++`) into `deps/libwitt_ffno.<ext>`, or into the path given as its
argument. Build it on the architecture that will load it.

For MPI runs, point MPI.jl at the system MPI with `MPIPreferences.use_system_binary()`.
The Olivia scripts do this for you.

### Locally

```bash
# unit, integration and contract tests (mock populations, no GPU)
julia --project=. -e 'using Pkg; Pkg.test()'

# parse a configuration and print a one-line summary
julia --project=. scripts/dry_run.jl configs/example_intensity_nonprd.toml

# forward or inversion run
julia --project=. --threads=auto scripts/invert.jl CONFIG.toml MODEL_FACTORY.jl
```

Production runs need the FSDP GPU service, so they run on Olivia.

### On Olivia

Keep the code in a permanent checkout and each run in its own directory:

```bash
repo=/cluster/projects/nn2834k/harshm/FFNOML
run=/cluster/work/projects/nn2834k/harshm/ffnoml_forward_001
mkdir -p "$run"
cp -R "$repo/examples/inversion_run/." "$run/"
# link inputs, atoms, partition functions, EOS table and checkpoints (see START_HERE.md)
```

```bash
FFNOML_RUN_DIR="$run" FFNO_INVERSION_CONFIG="$run/forward.toml" \
  bash "$repo/FFNOInversion.jl/scripts/submit_olivia_inversion.sh"
```

`submit_olivia_inversion.sh` runs a login-node preflight
(`preflight_olivia_inversion.jl` through `run_olivia_login_preflight.sh`, with a
separate login-only Julia depot). The preflight prints the mode, files, regions and
sources, outputs and Slurm topology, then checks datasets, shapes, line lists, PSF
files, factory assets and the rank layout. `sbatch` is called only after it prints
`Sanity check OK`. Extra arguments (for example `--account=...`) pass through to
`sbatch`.

Inside the job, `run_olivia_inversion.sbatch` sources the run's
`olivia_runtime_environment.sh`, recreates a run-local Julia project, writes the MPI
preferences, compiles the Wittmann library natively, and prints
`FFNO_OLIVIA_JULIA_ENVIRONMENT_READY` before launching the distributed ranks. Do not
compile the library on the login node: login and accelerator nodes have different
CPU architectures. The default allocation is 8 nodes × 4 GPUs with two threaded
Julia ranks per node.

For inversion, add `inputs/observations.h5`, use `inversion.toml` (or
`inversion_z.toml`), and choose new output names if you want to keep the forward
result.

### Fresh Olivia installation

```bash
bash scripts/bootstrap_olivia_regression.sh --help
```

The bootstrap prepares a runtime-test directory, installs Julia packages on a
compute node and submits the full regression chain. See
[`docs/reports/olivia-runtime-test-guide.md`](docs/reports/olivia-runtime-test-guide.md)
and [`../examples/runtime_test_run`](../examples/runtime_test_run/README.md).

### GPU-service diagnostics

All off by default; set them when submitting:

| Variable | Effect |
| --- | --- |
| `FFNO_FSDP_DIAGNOSTICS=1` (or `2`) | per-rank logs of versions, input shapes and finiteness, GPU memory and progress through every model block (`2`: every submodule). Slow. |
| `FFNO_FSDP_CORE_DUMPS=1` | raise the core limit and run each rank in its own directory |
| `FFNO_POPULATION_MODEL_ORDER=H,CA` | order in which species are requested |
| `FFNO_FSDP_TRACEBACK_INTERVAL` | periodic Python stack dumps. Leave unset: it crashed every rank in job 2622023 |

## Physics and numerics

### Equilibrium

`src/Equilibrium.jl` is the only route to pressure, density, electron density and
the derived vertical coordinate. `prepare_equilibrium` runs once (collective);
`reconstruct_equilibrium!` runs every evaluation and fills `pgas`, `rho`, `ne`, `z`,
`logtau500` and `chi500` from T (and B) without touching T, B, velocities or vturb.
It returns an `EquilibriumOutcome` that is identical on every rank and does not throw
for physics failures (EOS envelope, non-finite or folded columns). Instead
`outcome.valid` is false. With `on_nonconvergence = "reject_trial"` the solver
rejects that trial step.

FFNO owns MPI and threading and passes HSEMHS its communicator. HSEMHS solves on
x slabs, which is why `decomposition = "x_slabs"` is required.

### Population mesh for log-tau atmospheres

The FFNO checkpoints were trained on simulation height grids: one set of heights for
every column, 12–30 km spacing in the photosphere and chromosphere, and bottoms
1–2.5 Mm below the surface. A 95-point log-tau column has 30 km to several Mm
spacing and a different height at every point, which is far outside that. Fed
directly, H n2 is about 100× off in the chromosphere.

`MeshPopulationModel` (`src/PopulationMesh.jl`) therefore maps each column onto a
fixed height mesh using its HSEMHS heights, calls the FFNO there, and interpolates
the populations back:

- inside the column: log T, log ne and log rho log-linear in z, velocities linear;
- above its top: the top values;
- below its bottom (log tau500 ≈ 1.4, 0.1–0.3 Mm deep): hydrostatic extension with
  d ln T / d ln P = `nabla`, rho and ne from the HSEMHS EOS table, velocities held;
- back to the column: populations log-linear in z.

The VJP is the transpose of these maps. Native-z atmospheres skip the mesh.

### Synthesis

`build_synthesis_setup` reads and caches all line physics outside the pixel loops:
Muspel atoms, continuum and Voigt tables, Kurucz line selections and the Wittmann
partition functions. Configured air wavelengths stay on the observation side and
are converted to vacuum for line physics. Muspel lines use the exact Voigt profile
outside the table range.

The formal solver (`src/Synthesis.jl`) is a short-characteristic solution with the
source function linear in τ, integrated from the bottom (I = S) to the top. Two
details matter for log-tau grids:

- On a native log10 tau500 grid the optical depth is integrated in ln τ500 using
  HSEMHS's chi500, not in height. A 0.1 dex step can span the whole transition
  region (9 600 km in one column of the 504 × 504 run), and the height trapezoid
  would assign the cool point's opacity to hot gas.
- Steps where ln χ or ln η changes by more than 1 are split into sub-steps (at most
  256) with χ and η log-linear inside, so the emergent intensity is not set by the
  coronal source function.

Columns are synthesized in parallel with one workspace per Julia thread.

### Observation model and objective

`GaussianPSFObservation` convolves the full spectral-spatial cube with Gaussian
kernels, halo-aware across MPI tiles, without resizing it. The data term is a
normalized chi-square with `(wavelength, Stokes)` and 2D spatial weights. Zero
weights exclude wavelengths, Stokes components or pixels. Regularization is added as
a separate, separately reported term.

### Inversion mapping

The optimizer parameters `p` are physical node values on coarse control maps. The
atmosphere at each trial is

```text
A = A_start + interpolate(p − p_initial)
```

so the first evaluation uses the unsmoothed input atmosphere and the nodes carry
only corrections. Each trial is rebuilt from an immutable, rank-owned copy of the
baseline, so updates do not accumulate. Full-grid bounds are checked before the
expensive synthesis. Infeasible trials are rejected, not clipped.
`refine_control_maps` starts a new stage from an accepted atmosphere with finer
control maps; use it with fresh optimizer history.

### Gradients

`HybridAdjointObjectiveGradient` assembles the gradient without a dense Jacobian:

- exact transposes of the distributed PSF and the formal solution;
- pullbacks through the FFNO line opacities and Kurucz LTE lines (with local
  fallbacks for the Muspel and Wittmann pieces);
- the population-mesh transpose and the FFNO VJP from the GPU service;
- equilibrium and regularization contributions by centred finite differences per
  control coordinate, one-sided near bounds (an implicit-function equilibrium VJP is
  HSEMHS-integration Phase 6).

`FiniteDifferenceObjectiveGradient` is the validation oracle;
`gradient_taylor_validation` and `dot_product_validation` check the adjoints.

### Checkpoints

Checkpoints are atomic and bound to the configuration, the baseline atmosphere
checksum, the node initialization and the correction representation. They restart
on a different MPI/thread topology. A checkpoint from an incompatible run is
refused, so start that run fresh.

## Parallel runtime

- **MPI ranks** own non-overlapping spatial tiles (x slabs with HSEMHS). Numeric
  payloads are scattered and gathered without Julia serialization. Spatial kernels
  use corner-complete halo exchange.
- **Julia threads** parallelize columns inside a rank. Set BLAS/OpenMP threads to 1.
  MPI calls stay on the initializing thread.
- **GPU control**: global rank 0 alone launches and talks to the FFNO service over a
  TCP status channel. Other ranks do not sit in an MPI collective while Slurm/NCCL
  start the GPU step. Non-root ranks have a status timeout, and one-rank failures
  abort the whole job instead of deadlocking.
- **FSDP service** ([`../ffno_fsdp_service.py`](../ffno_fsdp_service.py),
  [`../ffno_fsdp_runtime.py`](../ffno_fsdp_runtime.py), launched by
  `scripts/ffno_fsdp_service.sh`): one torchrun job holds the H and Ca models for the
  whole run. Parameters are `FULL_SHARD`-sharded, activations and FFTs are split
  into H slabs, and activation checkpointing keeps memory bounded. Only torchrun rank
  0 reads each checkpoint. The service feeds log10 T and velocities in km/s, the
  same as training. `run_inversion!` requires this backend with at least two GPU
  ranks; there is no other production FFNO route.

The scheduler chooses the rank count and Julia the threads per rank. For example, a
256-core allocation can run 16 ranks × 16 threads.

## Package layout

| File | Role |
| --- | --- |
| `src/Types.jl` | grids, vertical axes, atmospheres, magnetic fields, spectral and observation cubes |
| `src/Parallel.jl` | MPI context, tiles, process grids, scatter/gather, halos, rank-0 GPU coordination |
| `src/Diagnostics.jl` | GPU-control diagnostics and failure records |
| `src/Nodes.jl` | node fields and their expansion onto tiles |
| `src/EOS.jl` | Wittmann EOS adapter (synthesis), ideal-gas test EOS |
| `src/ContinuumOpacity.jl` | Julia continuum opacity |
| `src/PopulationModels.jl` | population interface, six-channel FFNO request, recorded/mock models |
| `src/PopulationMesh.jl` | height mesh for log-tau atmospheres and its VJP |
| `src/LinePhysics.jl` | FFNO transitions, Muspel line model, Kurucz LTE lines |
| `src/Synthesis.jl` | mixed-source synthesis setup, formal solver, synthesis VJP |
| `src/Observations.jl` | Gaussian PSF observation model, inversion weights |
| `src/ForwardModel.jl`, `src/Objectives.jl` | mock forward model and residual packing |
| `src/Regularization.jl` | vertical (seven-slot) and horizontal regularization |
| `src/DistributedForward.jl` | `HybridForwardModel`, `forward!`, distributed chi-square/regularization |
| `src/Equilibrium.jl` | HSEMHS equilibrium adapter |
| `src/FSDPService.jl` | client and launcher for the persistent GPU service |
| `src/Solvers.jl` | control maps, objective evaluation, pattern-search prototype |
| `src/Gradients.jl` | matrix-free linearization, finite-difference oracle, Taylor checks |
| `src/ProductionGradients.jl` | adjoint gradient of the full objective |
| `src/ScalableSolvers.jl` | bounded L-BFGS and its diagnostics |
| `src/IO.jl` | TOML configuration, dry-run summary |
| `src/Execution.jl` | HDF5 inputs/outputs, `run_inversion!`, `run_inversion_files!` |
| `ext/FFNOInversionMuspelExt.jl` | Muspel package extension |
| `scripts/` | `invert.jl`, `dry_run.jl`, Wittmann build, Olivia submission/preflight/runtime scripts, validation drivers |
| `configs/` | minimal example configuration for tests and dry runs |
| `benchmarks/` | mock forward benchmark |

## Tests and validation

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. benchmarks/mock_forward.jl

# 4 MPI ranks x 2 threads, hybrid runtime
julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=. --threads=2 test/mpi_hybrid_worker.jl`)'

# one-rank versus four-rank parity, and restart across topologies
julia --project=. scripts/validate_phase4_topology.jl
julia --project=. scripts/validate_phase5_mpi.jl
julia --project=. scripts/validate_phase6_mpi.jl
```

`test/runtests.jl` covers types and coordinates, nodes, the parallel runtime, I/O
and configuration, Wittmann and continuum opacity, population backends and the
population mesh, synthesis and Muspel integration, the HSEMHS equilibrium, the
Phase 5 and 6 solvers and gradients, the FSDP protocol, the correction mapping, and
the Olivia setup and preflight scripts. The `test/mpi_*_worker.jl` and
`test/olivia_*` files are driven by the validation scripts and the Olivia
regression chain (`scripts/run_olivia_runtime_tests.sbatch`), which also exercises
CUDA/NCCL probes, deliberate CPU and GPU stalls, timeouts and recovery.

Other checks: `scripts/validate_eos_consistency.jl` (HSEMHS table versus Wittmann
library), `scripts/validate_muspel_reference.jl` (synthesis against a Muspel
reference), and `../scripts/compare_service_with_ffnonet.py` (service populations
versus a direct `FFNONet.py` prediction on the same atmosphere; they should agree to
float32 round-off).

## Design documents and reports

- `docs/adr/`: architecture decisions. ADR-001 coordinates and units, ADR-002 FFNO
  population boundary, ADR-004 PRD/Stokes boundaries, ADR-005 grids, observation and
  objective, ADR-006 two-input/two-output contract, ADR-007 vertical regularization,
  ADR-008 mixed-line synthesis, ADR-009 hybrid MPI/threads/GPU, ADR-010 prototype
  solver, ADR-011 matrix-free L-BFGS, ADR-012 HSEMHS equilibrium (supersedes the
  force-balance ADR-003).
- [`docs/hsemhs-integration/`](docs/hsemhs-integration/README.md): the six-phase
  HSEMHS integration plan. Phases 1–3 are done; MHS modes, equilibrium-aware
  inversion examples and the implicit equilibrium VJP are Phases 4–6.
- `docs/reports/`: validation reports for each phase, the hybrid runtime, the Olivia
  runtime-test guide and the job 2622023 crash investigation.

## Known limitations

- **Region `normalization` is validated but not applied.** Synthetic intensities are
  in the solver's absolute units. Observations must be in the same units, or be
  scaled before they are written.
- **Region `psf_type`/`psf_file` are validated, and the preflight checks the file
  exists, but no per-region spectral PSF is applied.** The only instrumental profile
  is the global `[observation.gaussian_psf]`.
- Stokes I and non-PRD only.
- `equilibrium.model = "mhs"` is rejected until HSEMHS-integration Phase 4. A
  magnetic field is still used by `align = "ptot"` and by synthesis.
- Equilibrium derivatives are finite differences, which cost two equilibrium solves
  per control coordinate per gradient.
- FFNO lines are limited to H alpha and Ca II 8542, with the 6-level H and Ca II
  atoms used in training.
- The FSDP service needs CUDA (NVIDIA) GPUs and at least two GPU ranks.
