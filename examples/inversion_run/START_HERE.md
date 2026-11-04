# Quick start: atmosphere → FFNO populations → synthetic spectra

Use this guide on **Olivia**, with the updated FFNOML checkout. The entry point
is `FFNOInversion.jl/scripts/invert.jl` for both forward synthesis and inversion.
`forward.toml` selects forward mode: the supplied full atmosphere goes through
the HSEMHS equilibrium, FFNO3D populations, and spectral synthesis. No observations,
node interpolation, derivatives, or fitting are needed.

## 1. Make a separate run folder

Run these commands on the login node. Change the paths if your checkout or run
directory lives elsewhere.
Use a new run folder for each experiment; output files with the same names are overwritten.

```bash
repo=/cluster/projects/nn2834k/harshm/FFNOML
run=/cluster/work/projects/nn2834k/harshm/ffnoml_forward_001
mkdir -p "$run"
cp -R "$repo/examples/inversion_run/." "$run/"
```

Edit `$run/olivia_runtime_environment.sh` once if the Julia, Python, depot, or
Muspel paths differ. It contains all accelerator-runtime exports and is sourced
only after a Slurm job starts on an accelerator node. Do not source it on the
login node. This assumes the MPI/CUDA Python environment and cached Julia
dependencies from the successful Olivia tests already exist. For a new installation, first follow
the [Olivia environment/acceptance guide](../../FFNOInversion.jl/docs/reports/olivia-runtime-test-guide.md).
Its bootstrap submits setup **and five regression jobs**, not a synthesis run.

## 2. Supply the scientific files

The final layout must be as below. Files may be symlinks to permanent storage;
they must be readable from every allocated node. The template already creates
the subdirectories. Replace every `/path/to/...` below with a real file.

```text
ffnoml_forward_001/
├── forward.toml                 # atmosphere, wavelengths, physics, output paths
├── inversion.toml               # only needed later for fitting
├── model_factory.jl             # loads the physics and FFNO models
├── olivia_runtime_environment.sh # accelerator exports; never source on login
├── inputs/
│   ├── initial_atmosphere.h5
│   ├── kurucz_6302.list
│   ├── pf_Kurucz.input
│   ├── atoms/atom.h6_tiago2.yaml
│   ├── atoms/atom.ca2.yaml
│   └── wittmann/libwitt_ffno.so # generated after the production job starts
├── training_FFNO3D_zscale_expand_lognlte/
│   ├── 3D_sim_train_H.pt
│   └── 3D_sim_train_CA.pt
└── outputs/                     # created automatically, along with diagnostics/ and tmp/
```

```bash
atoms=/cluster/projects/nn2834k/harshm/multi3d/input/atoms
models=/cluster/projects/nn2834k/harshm/train_outputs/training_FFNO3D_zscale_expand_lognlte
ln -s /path/to/initial_atmosphere.h5 "$run/inputs/initial_atmosphere.h5"
ln -s /path/to/kurucz_6302.list "$run/inputs/kurucz_6302.list"
ln -s "$atoms/atom.h6_tiago2.yaml" "$run/inputs/atoms/atom.h6_tiago2.yaml"
ln -s "$atoms/atom.ca2.yaml" "$run/inputs/atoms/atom.ca2.yaml"
ln -s "$repo/scripts/pf_Kurucz.input" "$run/inputs/pf_Kurucz.input"
ln -s "$models/3D_sim_train_H.pt" "$run/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_H.pt"
ln -s "$models/3D_sim_train_CA.pt" "$run/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_CA.pt"
```

Both H and Ca checkpoints/atoms are required by this model factory, even for a
Ca-only spectrum. Raw Bifrost `mesh`/`atm3d` files cannot replace the HDF5 input.
Do not compile `libwitt_ffno.so` on the login node: Olivia's login and
accelerator nodes use different CPU architectures. The production job builds
the run-local library natively before launching distributed work and verifies
that its accelerator Julia process can load it.

## 3. Check the atmosphere and edit `forward.toml`

The reader expects these dataset names by default; change `[atmosphere.datasets]`
if your HDF5 uses different names. Set `storage_order = "python"` in
`[atmosphere]` for NumPy/h5py or C-written datasets; this reverses their
row-major dimensions before time/depth validation. Omit it, or use `"julia"`,
for files written with HDF5.jl.

| Dataset | Shape | Units |
| --- | --- | --- |
| `logtau_500` | `(z)`; strictly monotonic | log10 optical depth at 500 nm |
| `temperature` | `(time,z,y,x)` or `(z,y,x)` | K |
| `vx`, `vy`, `vz` | same as temperature; all three required | m/s |
| `vturb` (optional) | same as temperature | m/s |
| `Bx`, `By`, `Bz` (optional; supply all three) | same as temperature | tesla |

`time_index = 1` selects the first snapshot. Enable the `vturb` mapping to read
microturbulence; otherwise it is zero. Magnetic mappings are optional (absent means
B = 0). Unless you supply them (below), pressure, density, electron density and the derived coordinate (height for
this log-tau input) are reconstructed by the HSEMHS equilibrium configured in
`[equilibrium]`.

For other vertical axes start from a different template: `forward_logm.toml` for log10 column mass (dataset `log10m`, in kg m⁻²,
or g cm⁻² with a `units` attribute) and `forward_z.toml` for height (`z_m`, `x_m`, `y_m` in metres).

Before submission, set:

- `[equilibrium] eos_table_file` to the HSEMHS EOS table (e.g. a link to `HSEMHS/cache/eos_table_fullz.h5`). The pressure boundary is HSEMHS's: for native log-tau input the τ-hydrostatic top closure sets it (`[equilibrium.column_ltau]`), for native log-m input P = g m does (`[equilibrium.column_logm]`), for native-z input `[equilibrium.column_z] anchor` and `P_ref_pa` do. There is no top-pressure or top-density setting in FFNO.
- `storage_order = "python"` when the atmosphere file was written by NumPy/h5py or C; the supplied template uses this setting.
- `[grid] dx_m` and `dy_m` to the actual horizontal spacing in metres.
- `[[regions]]` air wavelengths (Å), counts and line lists. The example synthesizes H alpha and Ca II 8542 through FFNO and an LTE Fe 6302 region; only the last needs a Kurucz line list. Each region's `normalization` divides that region's spectra, written and observed, in the `[outputs] intensity_units` unit (`cgs` in the examples); overlapping regions must state the same value. Its `psf_type`/`psf_file` are checked but no per-region PSF is applied.
- `[observation.gaussian_psf]`: the example applies a **96 km spatial blur**. Set both spatial FWHM values to `0.0` for unblurred spectra. This section controls synthetic output even without observations.

How the run treats the atmosphere (details in the [package README](../../FFNOInversion.jl/README.md#physics-and-numerics)):

- **Pressure, density and electron density** come from HSEMHS: hydrostatic balance plus the EOS. An inversion always works this way.
- **A forward run may supply them:** add any of `ne` (m⁻³), `rho` (kg m⁻³), `pgas` (Pa) and `hydrogen_populations` (m⁻³, file axes `(level,z,y,x)`, last level the protons), in any combination, to `[atmosphere.datasets]`. Then HSEMHS isn't used: what you give is used as given, and the Wittmann EOS supplies what's missing. For a simulation, `ne`, `rho` and `pgas` together are the most faithful. `[equilibrium]` is then optional and its column sections are ignored, but the population mesh still needs `eos_table_file`, or `[population.mesh] enabled = false`. On log tau the axis must fit what you give: take it from a z run with the same input.
- **FFNO populations** are predicted on a fixed height mesh and interpolated back. This is on by default for log tau and log m; on z add `[population.mesh] enabled = true`, which you want for anything unlike a simulation grid, such as a 1-D model.
- **Above a column's top**, if the mesh goes higher, a corona is built: T rises to `corona_temperature_k` (1 MK) over `corona_transition_m` (240 km), hydrostatic from the top, with ρ and nₑ from the EOS and velocities held. `corona_temperature_k = 0` holds the top values instead.
- **Below a column's bottom**, the column is extended hydrostatically with d ln T / d ln P = `nabla` (0.13), with ρ and nₑ from the EOS and velocities held.
- Both extensions are the same for every species and axis, and feed only the FFNO: synthesis uses your levels.
- **Continuum:** one per region, Muspel's, with hydrogen from the EOS (a given atmosphere's own, when given) and no FFNO populations. `[physics] continuum_populations = "ffno"` is refused. Neutral H in line broadening comes from the same place.
- **Line centres:** FFNO lines are centred on H alpha 656.28 nm and Ca II 854.209 nm (air) by default. `[physics] line_center = "atom"` uses the atom file's level energies instead, which puts them about +0.16 and -0.03 Å away. Kurucz lines keep their list's wavelengths.
- **Wavelengths:** `start_angstrom` is always air and synthesis always runs on the vacuum grid. `[outputs] wavelength_frame` (`"air"`, the default, or `"vacuum"`) only chooses the written `wavelength_m`; the file attribute `wavelength_frame` records it.

Keep `mode = "forward"`, `stokes = ["I"]`, and `redistribution = "non_prd"`
for this starter run. Do not add nodes or a solver to `forward.toml`.

## 4. Preflight and submit

Use the submission helper once. It completes the CPU/login-node preflight
before it calls `sbatch`:

```bash
FFNOML_RUN_DIR="$run" FFNO_INVERSION_CONFIG="$run/forward.toml" \
  bash "$repo/FFNOInversion.jl/scripts/submit_olivia_inversion.sh"
```

The submission helper first reports the selected mode, files, spectral lines,
outputs, and SLURM resources. It validates the configuration, required HDF5
datasets and shapes, model-factory assets, and MPI topology; it calls `sbatch`
only after printing `Sanity check OK`. This preflight resets inherited Julia
settings, loads Olivia's `NRIS/Login` stack, and instantiates dependencies in
the separate login-only depot `/cluster/projects/nn2834k/harshm/julia-depot-login` (`FFNO_PREFLIGHT_JULIA_DEPOT` overrides it); it never
loads the accelerator runtime environment or depot. Accelerator jobs use `/cluster/projects/nn2834k/harshm/julia-depot-prod`
(`OLIVIA_JULIA_DEPOT`). Both are under the project because the home directory has no storage.

After the allocation starts, the same production job first recreates its
disposable Julia project, writes Olivia MPI preferences, instantiates and
precompiles dependencies, and builds and load-checks the native Wittmann EOS
library. Only after `FFNO_OLIVIA_JULIA_ENVIRONMENT_READY` appears in
`slurm-JOBID.out` does it launch the distributed Julia ranks. No separate setup
job is required. The local Muspel checkout must match the revision pinned in
`Project.toml`; keep the repository manifest for repeatability. The default
allocation is **8 nodes × 4 GPUs**, with two threaded Julia MPI ranks per node.
Another account can pass `--account=YOUR_ACCOUNT` to the submission helper.

## 5. Check the result; later switch to inversion

For the printed job ID, use `squeue -j JOB_ID` and, after completion,
`sacct -j JOB_ID --format=JobID,State,ExitCode`. Read `slurm-JOB_ID.out` and
`slurm-JOB_ID.err`; GPU details are under `diagnostics/slurm-JOB_ID/`.
A successful forward run writes:

- `outputs/synthesis.h5`: `intensity` with axes `(time,Stokes,wavelength,y,x)`, `wavelength_m` (air unless `[outputs] wavelength_frame = "vacuum"`; attribute `wavelength_frame`), and provenance. Spectra include the `[observation.gaussian_psf]` blur and are in `[outputs] intensity_units` (attribute `units`) divided by the region `normalization` (dataset `normalization`).
- `outputs/forward_atmosphere.h5`: atmospheric fields including `pgas`, `rho`, `ne`, `z`, and `populations/H` and `populations/CA` (population axes `(level,z,y,x)`). Both files record `execution_mode = "forward"`.

For **inversion**, add `inputs/observations.h5`, edit `inversion.toml` to match the
same atmosphere, wavelength grid and Gaussian PSF, then set controls,
bounds and a positive `solver.max_iterations`. Observations are in `[outputs] intensity_units` divided by the region `normalization`, and need `intensity`
and positive `sigma` with `(time,Stokes,wavelength,y,x)`, `wavelength_weights`
with `(wavelength,Stokes)`, and `spatial_weights` with `(y,x)`; the reader also
accepts the intensity/sigma arrays without time and weight arrays with time.
Set `FFNO_INVERSION_CONFIG="$run/inversion.toml"` and use the same submission
command. Choose distinct output paths if keeping the forward result.

The first synthesis preserves the full starting atmosphere; nodes parameterize
subsequent corrections. Setting `solver.max_iterations = 0` also selects forward
mode and ignores observations/nodes. See [the full example configuration](inversion.toml)
for fitting options.

## Off disk centre

To synthesize or invert at a viewing angle μ < 1, uncomment `[view]` in the configuration. A log-tau or log-m input must then
already be lines of sight along that angle, tagged with `view_*` attributes; a height-grid input is used as it is.
See [README.md](README.md#viewing-angle-off-disk-centre).

## GPUs too small for your cube (forward only)

On machines like eagle7, a cube such as 504 × 504 doesn't fit in GPU memory through the default service. The [batched service](../../FFNOInversion.jl/README.md#batched-forward-service-small-gpus) runs the same full-domain prediction from host RAM, more slowly, and can't be used for inversions. Set `FFNO_BATCHED_VERTICAL_NORM_SLABS` to the GPU count of an FSDP run you want to match exactly.

The template `model_factory.jl` always uses the Slurm service, so a run outside Olivia needs a factory that passes `FFNO_FSDP_LAUNCHER` to the launch call. `response_run/falc/model_factory.jl` and `run_falc_eagle7.sh` show this for eagle7.

## If the GPU service crashes

Set these when submitting (the job inherits them); all are off by default:

| Variable | Effect |
| --- | --- |
| `FFNO_FSDP_DIAGNOSTICS=1` | each GPU rank writes `diagnostics/slurm-JOB_ID/fsdp-ranks-JOB_ID/rank-NNNN.log`: versions, loaded libraries, input shapes/strides/dtypes and finiteness, GPU memory per request, and progress through scatter, each model block and gather (the GPU is synchronized at each block, so this is slow). Per-rank stdout/stderr go to `torchrun-JOB_ID/`. `2` logs every submodule. |
| `FFNO_FSDP_CORE_DUMPS=1` | raises the core-file limit and runs each rank in its own `core-rank-NNNN/` directory. |
| `FFNO_POPULATION_MODEL_ORDER=H,CA` | requests the population models in this order instead of alphabetically. |
| `FFNO_FSDP_TRACEBACK_INTERVAL=60` | periodic Python stack dumps from every rank. Leave unset: in job 2622023 the dump itself crashed every rank. |
| `CUDA_LAUNCH_BLOCKING=1`, `TORCH_SHOW_CPP_STACKTRACES=1` | standard CUDA/PyTorch settings for attributing a native failure (slow). |

For example:

```bash
FFNO_FSDP_DIAGNOSTICS=1 FFNO_FSDP_CORE_DUMPS=1 \
FFNOML_RUN_DIR="$run" FFNO_INVERSION_CONFIG="$run/forward.toml" \
  bash "$repo/FFNOInversion.jl/scripts/submit_olivia_inversion.sh"
```
