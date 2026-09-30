# Quick start: atmosphere → FFNO populations → synthetic spectra

Use this guide on **Olivia**, with the updated FFNOML checkout. The entry point
is `FFNOInversion.jl/scripts/invert.jl` for both forward synthesis and inversion.
`forward.toml` selects forward mode: the supplied full atmosphere goes through
force balance/EOS, FFNO3D populations, and spectral synthesis. No observations,
node interpolation, derivatives, or fitting are needed.

## 1. Make a separate run folder

Run these commands in one Bash session on the login node. Change the paths if
your checkout or previously prepared Julia environment lives elsewhere.
Use a new run folder for each experiment; output files with the same names are overwritten.

```bash
repo=/cluster/projects/nn2834k/harshm/FFNOML
run=/cluster/work/projects/nn2834k/harshm/ffnoml_forward_001
tested=/cluster/work/projects/nn2834k/harshm/ffnoml_runtime_tests
mkdir -p "$run"
cp -R "$repo/examples/inversion_run/." "$run/"
export FFNOML_RUN_DIR="$run"
export OLIVIA_REPO_DIR="$repo/FFNOInversion.jl"
export OLIVIA_ENV_SCRIPT=/cluster/home/harshm/loadnvidiampi.sh
export OLIVIA_JULIA=/cluster/software/NRIS/neoverse_v2/software/Julia/1.12.2/bin/julia
export OLIVIA_PYTHON=/cluster/home/harshm/nvidiaenv/bin/python3
export OLIVIA_JULIA_DEPOT=/cluster/home/harshm/julia-depot-1.12.2
export JULIA_DEPOT_PATH="$OLIVIA_JULIA_DEPOT"
```

This assumes the MPI/CUDA Python environment and cached Julia dependencies from
the successful Olivia tests already exist. For a new installation, first follow
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
├── inputs/
│   ├── initial_atmosphere.h5
│   ├── kurucz_8542.list
│   ├── kurucz_6302.list
│   ├── pf_Kurucz.input
│   ├── atoms/atom.h6_tiago2.yaml
│   ├── atoms/atom.ca2.yaml
│   └── wittmann/libwitt_ffno.so
├── training_FFNO3D_zscale_expand_lognlte/
│   ├── 3D_sim_train_H.pt
│   └── 3D_sim_train_CA.pt
└── outputs/                     # created automatically, along with diagnostics/ and tmp/
```

```bash
atoms=/cluster/projects/nn2834k/harshm/multi3d/input/atoms
models=/cluster/projects/nn2834k/harshm/train_outputs/training_FFNO3D_zscale_expand_lognlte
ln -s /path/to/initial_atmosphere.h5 "$run/inputs/initial_atmosphere.h5"
ln -s /path/to/kurucz_8542.list "$run/inputs/kurucz_8542.list"
ln -s /path/to/kurucz_6302.list "$run/inputs/kurucz_6302.list"
ln -s "$atoms/atom.h6_tiago2.yaml" "$run/inputs/atoms/atom.h6_tiago2.yaml"
ln -s "$atoms/atom.ca2.yaml" "$run/inputs/atoms/atom.ca2.yaml"
ln -s "$repo/scripts/pf_Kurucz.input" "$run/inputs/pf_Kurucz.input"
ln -s "$models/3D_sim_train_H.pt" "$run/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_H.pt"
ln -s "$models/3D_sim_train_CA.pt" "$run/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_CA.pt"
c++ -O3 -std=c++17 -shared -fPIC -pthread "$repo/scripts/witt_eos_cpp.cpp" \
  -o "$run/inputs/wittmann/libwitt_ffno.so"
```

Both H and Ca checkpoints/atoms are required by this model factory, even for a
Ca-only spectrum. Raw Bifrost `mesh`/`atm3d` files cannot replace the HDF5 input.

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

- `pressure_top_pa` in `[atmosphere]`, and export `FFNO_TOP_DENSITY_KG_M3` in the shell, to appropriate **top boundary values for your atmosphere**. The example values, 0.1 Pa and 1e-10 kg/m³, are placeholders.
- `[grid] dx_m` and `dy_m` to the actual horizontal spacing in metres.
- `[[regions]]` wavelengths (Å), counts, normalization and line lists. The example synthesizes Ca II 8542 plus LTE blends, and an LTE Fe 6302 region.
- `[observation.gaussian_psf]`: the example applies a **96 km spatial blur**. Set both spatial FWHM values to `0.0` for unblurred spectra. This section controls synthetic output even without observations.

Keep `mode = "forward"`, `stokes = ["I"]`, and `redistribution = "non_prd"`
for this starter run. Do not add nodes or a solver to `forward.toml`.

## 4. Prepare Julia and submit

Reuse the validated dependency manifest and MPI preferences, with the current
package definition/source, in a run-local environment. This leaves the old test
environment untouched. Run once for this new folder:

```bash
export OLIVIA_JULIA_PROJECT="$run/julia-environment"
mkdir -p "$OLIVIA_JULIA_PROJECT"
cp "$tested/julia-environment/Manifest.toml" "$OLIVIA_JULIA_PROJECT/"
cp "$tested/julia-environment/LocalPreferences.toml" "$OLIVIA_JULIA_PROJECT/"
cp "$OLIVIA_REPO_DIR/Project.toml" "$OLIVIA_JULIA_PROJECT/"
for part in src ext scripts; do
  ln -s "$OLIVIA_REPO_DIR/$part" "$OLIVIA_JULIA_PROJECT/$part"
done
cd "$run"
sbatch "$OLIVIA_REPO_DIR/scripts/setup_olivia_environment.sbatch"
```

Wait for the setup job to finish successfully; its `.out` must contain
`FFNO_OLIVIA_JULIA_ENVIRONMENT_READY`. Then submit the actual synthesis:

```bash
# Replace this example density with your chosen physical boundary value.
export FFNO_TOP_DENSITY_KG_M3=1e-10
export FFNO_INVERSION_CONFIG="$run/forward.toml"
bash "$OLIVIA_REPO_DIR/scripts/submit_olivia_inversion.sh"
```

The default allocation is **2 nodes × 4 GPUs**, with one threaded Julia MPI rank
per node. Submission prints the job ID. Run on Slurm, not directly on the login
node. Another account can pass `--account=YOUR_ACCOUNT` to both submissions.

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
