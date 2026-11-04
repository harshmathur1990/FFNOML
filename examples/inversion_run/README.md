# FFNOInversion.jl run directory

**Start here:** [Short forward/inversion starter guide](START_HERE.md), including
the folder layout, scientific inputs, Olivia setup, submission, and outputs.

This is the per-run layout for a production inversion. The checked-in files are
a template; the scientific inputs must be symlinked from permanent storage.

Required links:

- `inputs/initial_atmosphere.h5`
- `inputs/observations.h5`
- `inputs/kurucz_6302.list` (for the LTE Fe 6302 region)
- `inputs/atoms/atom.h6_tiago2.yaml` and `atom.ca2.yaml`
- `inputs/pf_Kurucz.input`
- H and Ca checkpoints in `training_FFNO3D_zscale_expand_lognlte`

`inputs/wittmann/libwitt_ffno.so` is generated inside the production job before
distributed work begins.
Do not compile or link an x86-64 login-node copy into the run directory.

Prepare and submit:

```bash
repo=/path/to/permanent/FFNOML
run=/path/to/hot/ffnoml_inversion_run1

mkdir -p "$run"
cp -R "$repo/examples/inversion_run/." "$run/"

ln -s /permanent/inputs/initial_atmosphere.h5 "$run/inputs/initial_atmosphere.h5"
ln -s /permanent/inputs/observations.h5 "$run/inputs/observations.h5"
ln -s /permanent/multi3d/input/atoms/atom.h6_tiago2.yaml "$run/inputs/atoms/atom.h6_tiago2.yaml"
ln -s /permanent/multi3d/input/atoms/atom.ca2.yaml "$run/inputs/atoms/atom.ca2.yaml"
ln -s "$repo/scripts/pf_Kurucz.input" "$run/inputs/pf_Kurucz.input"
ln -s /permanent/models/3D_sim_train_H.pt "$run/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_H.pt"
ln -s /permanent/models/3D_sim_train_CA.pt "$run/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_CA.pt"
ln -s /permanent/inputs/kurucz_6302.list "$run/inputs/kurucz_6302.list"

cd "$run"
bash "$repo/FFNOInversion.jl/scripts/submit_olivia_inversion.sh"
```

Before calling `sbatch`, the submission helper prints the selected mode,
configuration and factory, HDF5 inputs, requested spectral regions and line
sources, outputs, and default SLURM topology. It checks the required datasets,
compatible input shapes, line lists, PSFs, factory assets, and rank layout.
Submission proceeds only after it prints `Sanity check OK`.
The helper uses the `NRIS/Login` stack and a separate persistent Julia depot for
this login-node preflight. Accelerator Julia, Python, MPI/CUDA, depot, Muspel,
and model initialization settings live in `olivia_runtime_environment.sh`; that file is
sourced only by the production job after Slurm starts it. The job prepares its
Julia environment and native EOS library before launching distributed ranks;
there is no separate setup submission.

The Olivia batch script derives the Julia thread count and the runtime
`threads_per_rank` value from SLURM's `cpus-per-task`. Do not hard-code
`threads_per_rank` in a copied run configuration; it is only needed for
non-SLURM launches where it defaults to Julia's active thread count.

`model_factory.jl` uses these relative paths through `FFNOML_RUN_DIR`. The
equilibrium (pressure, density, electron density, heights) comes from HSEMHS as
configured in `[equilibrium]`, unless a forward run supplies them; point `eos_table_file` at the HSEMHS EOS table.
The supplied configurations set `[atmosphere].storage_order = "python"` so
NumPy/h5py array dimensions are converted before validation and reading.

For atmosphere-to-spectrum synthesis without fitting, use `forward.toml`:

```bash
cd "$run"
FFNO_INVERSION_CONFIG="$run/forward.toml" \
  bash "$repo/FFNOInversion.jl/scripts/submit_olivia_inversion.sh"
```

This needs the atmosphere and model/physics assets listed above, but no
`inputs/observations.h5`. It returns `outputs/synthesis.h5` and
`outputs/forward_atmosphere.h5`. Alternatively set `max_iterations = 0` in an
inversion configuration; nodes and observations will be ignored. This is still
the equilibrium (HSEMHS, or a given atmosphere) followed by FFNO populations and synthesis,
with no optimization or gradient work. The example factory currently loads both
H and Ca assets even when only one species contributes to the configured lines.

## How the atmosphere is treated

- **P, ρ, nₑ:** from HSEMHS, by hydrostatic balance and the EOS. An inversion always works this way.
- **A given atmosphere:** in a forward run only, any of `ne`, `rho`, `pgas` and `hydrogen_populations` in `[atmosphere.datasets]`, in any combination and on any axis, replaces HSEMHS: what is given is used as given, and the Wittmann EOS supplies the rest. `[equilibrium]` is then optional and its column sections are ignored. The mesh still needs `eos_table_file`, or `[population.mesh] enabled = false`.
- **Population mesh:** the FFNO predicts on a fixed height mesh, on every axis and in both modes, unless `[population.mesh] enabled = false`.
- **Above a column's top:** where the mesh goes higher, a corona: T rises to `corona_temperature_k` (1 MK) over `corona_transition_m` (240 km), hydrostatic from the top, ρ and nₑ from the EOS, velocities held. `corona_temperature_k = 0` holds the top values.
- **Below a column's bottom:** a hydrostatic extension with d ln T / d ln P = `nabla` (0.13), ρ and nₑ from the EOS, velocities held.
- **Continuum:** Muspel's, with hydrogen from the EOS (a given atmosphere's own, when given). FFNO populations never enter it.
- **Line centres:** FFNO lines are centred on H alpha 656.28 nm and Ca II 854.209 nm (air) by default. `[physics] line_center = "atom"` uses the atom file's level energies instead, which puts them about +0.16 and -0.03 Å away. Kurucz lines keep their list's wavelengths.
- **Wavelengths:** `start_angstrom` is always air and synthesis always runs on the vacuum grid. `[outputs] wavelength_frame` (`"air"`, the default, or `"vacuum"`) only chooses the written `wavelength_m`; the file attribute `wavelength_frame` records it.

See [the package README](../../FFNOInversion.jl/README.md#population-mesh) for the details.

The rules are also written out in `forward.toml`, `forward_z.toml` and `forward_logm.toml`.

## Small GPUs (forward only)

For a cube whose activations don't fit on the GPUs, such as 504 × 504 on eagle7, use the
[batched service](../../FFNOInversion.jl/README.md#batched-forward-service-small-gpus). It is full-domain prediction from host RAM, not tiling,
and needs about 90 GB of RAM at that size. It answers forward runs only. Step-by-step guides: [eagle7](start_here_eagle7.md),
[any HPC with GPUs](start_here_general_hpc.md) and [a GPU host without a scheduler](start_here_local_cluster_gpu.md).

## Viewing angle (off disk centre)

Add a `[view]` section (commented templates in all five configurations) to synthesize or invert at μ = cos θ < 1;
without it, or at μ = 1, everything is the vertical run. Design, validation and the comparison with MULTI3D at
μ = 0.8: [`FFNOInversion.jl/docs/viewing-angle/README.md`](../../FFNOInversion.jl/docs/viewing-angle/README.md).

- **Native log tau** (`forward.toml`, `inversion.toml`): `inputs/initial_atmosphere.h5` must already be one line of
  sight per pixel: T and velocities on log10 tau500 along the ray, on a regular deprojected footprint grid, with the
  file attributes `view_mu`, `view_azimuth_deg` and `view_geometry` equal to the configuration (a file without them is
  accepted only at μ = 1). Set `[grid] dx_m`, `dy_m` to the footprint spacings (a sky pixel p becomes p across the
  tilt and p/μ along it). Use `geometry = "inclined"`: the FFNO then sees every ray point at its 3-D position.
- **Native log m** (`forward_logm.toml`): the same contract as native log tau, with log10 m along the ray.
- **Native z** (`forward_z.toml`, `inversion_z.toml`): the cube is used as it is; `inclined` synthesizes along the
  slanted rays through it. This is the direct route for an off-centre view of a simulation.
- **Inversions** at μ < 1 control `vlos` (n·v), not `vz` (see the comment in the inversion configurations).
- **PSF:** `[observation.gaussian_psf]` FWHMs stay in sky metres; the run stretches them by 1/μ along the tilt and
  accepts only azimuths 0, 90, 180 or 270° with a spatial PSF.
- Outputs carry `view_*` attributes. `model_factory.jl` predicts the populations on the Cartesian atmosphere and
  builds the synthesis setup on `synthesis_atmosphere(workspace, distributed)`, which is the footprint atmosphere for
  an inclined height-grid run; keep that if you write your own factory.

## Column response functions

After a forward run, `FFNOInversion.jl/scripts/column_response.jl` (Olivia: `run_olivia_column_response.sbatch`)
writes ∂I/∂v per column and log tau point for temperature, `vz`, `vlos`, `vturb`, `ne` and `pgas`;
`ffno_population_tangents.jl` (GPU) adds the response of the FFNO populations to temperature. See the package
README, "Column response functions".

