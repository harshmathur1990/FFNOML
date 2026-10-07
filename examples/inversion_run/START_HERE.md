# Quick start: atmosphere → FFNO populations → synthetic spectra

Use this guide on **Olivia**, with the updated FFNOML checkout. The entry point
is `FFNOInversion.jl/scripts/invert.jl` for both forward synthesis and inversion.
`forward.toml` selects forward mode: the supplied full atmosphere goes through
force balance/EOS, FFNO3D populations, and spectral synthesis. No observations,
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
if your HDF5 uses different names. Array order here is the order read by Julia's
HDF5 interface; check this when creating files using another language.

| Dataset | Shape | Units |
| --- | --- | --- |
| `logtau_500` | `(z)`; strictly monotonic | log10 optical depth at 500 nm |
| `temperature` | `(time,z,y,x)` or `(z,y,x)` | K |
| `vx`, `vy`, `vz` | same as temperature; all three required | m/s |
| `vturb` (optional) | same as temperature | m/s |
| `Bx`, `By`, `Bz` (optional; supply all three) | same as temperature | tesla |

`time_index = 1` selects the first snapshot. Enable the `vturb` mapping to read
microturbulence; otherwise it is zero. Omit magnetic mappings for HE3D; enabling
all three selects MHS. Pressure, density, electron density and height are derived
through force balance/EOS, rather than read as full input fields.

Before submission, set:

- `pressure_top_pa` in `[atmosphere]`, to an appropriate **top pressure for your atmosphere**; 0.1 Pa is an example. The current factory also accepts `FFNO_TOP_DENSITY_KG_M3` (default 1e-10 kg/m³) as an initialization guess, used at the top only on the first force-balance iteration; subsequent density comes from the EOS.
- `[grid] dx_m` and `dy_m` to the actual horizontal spacing in metres.
- `[[regions]]` wavelengths (Å), counts, normalization and line lists. The example synthesizes Ca II 8542 through FFNO and an LTE Fe 6302 region; only the latter needs a Kurucz line list.
- `[observation.gaussian_psf]`: the example applies a **96 km spatial blur**. Set both spatial FWHM values to `0.0` for unblurred spectra. This section controls synthetic output even without observations.

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
the separate login-only depot `~/julia-depot-ffno-login-1.12.2`; it never loads
the accelerator runtime environment or depot.

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

- `outputs/synthesis.h5`: `intensity` with axes `(time,Stokes,wavelength,y,x)`, `wavelength_m`, and provenance. Spectra include the configured normalization and PSF.
- `outputs/forward_atmosphere.h5`: atmospheric fields including `pgas`, `rho`, `ne`, `z`, and `populations/H` and `populations/CA` (population axes `(level,z,y,x)`). Both files record `execution_mode = "forward"`.

For **inversion**, add `inputs/observations.h5`, edit `inversion.toml` to match the
same atmosphere, wavelength grid, normalization and PSF, then set controls,
bounds and a positive `solver.max_iterations`. Observations need `intensity`
and positive `sigma` with `(time,Stokes,wavelength,y,x)`, `wavelength_weights`
with `(wavelength,Stokes)`, and `spatial_weights` with `(y,x)`; the reader also
accepts the intensity/sigma arrays without time and weight arrays with time.
Set `FFNO_INVERSION_CONFIG="$run/inversion.toml"` and use the same submission
command. Choose distinct output paths if keeping the forward result.

The first synthesis preserves the full starting atmosphere; nodes parameterize
subsequent corrections. Setting `solver.max_iterations = 0` also selects forward
mode and ignores observations/nodes. See [the full example configuration](inversion.toml)
for fitting options.
