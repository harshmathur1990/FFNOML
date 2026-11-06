# FFNOInversion.jl

`FFNOInversion.jl` is the Julia forward-synthesis and inversion layer of ARUN.
It takes a 3D solar atmosphere (temperature, velocities, optional magnetic field),
obtains its pressure, density, electron density and geometry (from HSEMHS by hydrostatic balance, or in forward
synthesis from whichever of those the input supplies), predicts NLTE level populations
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
equilibrium                 Pgas, rho, ne, z (or log10 tau500) and chi500: HSEMHS from T (and B), or a given atmosphere (forward only)
   ▼
population mesh             columns mapped onto a training-like height mesh, extended above and below
   ▼
FFNO3D populations          persistent multi-GPU FSDP service, one model per species
   ▼
footprint columns (z, μ<1)  inclined view on a height grid: state and populations shifted onto the rays
   ▼
mixed synthesis             FFNO NLTE lines + Kurucz LTE lines + continuum, per region; Doppler n·v, path dz/μ
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
| Forward synthesis (`mode = "forward"`) | Production. Native log10 tau500, native log10 column mass or native z atmospheres. |
| Inversion (`mode = "inversion"`) | Implemented: bounded L-BFGS with matrix-free gradients, checkpoints, restart across topologies. |
| Equilibrium | HSEMHS column HSE (`column-hse-ltau`, `column-hse-logm`, `column-hse-z`). MHS Poisson modes (`model = "mhs"`) are rejected until HSEMHS-integration Phase 4. |
| Populations | FFNO3D through the persistent FSDP GPU service (H and Ca II, 6 levels each in the example factory; at least two GPU ranks), or for forward runs on small GPUs the [batched service](#batched-forward-service-small-gpus). |
| Spectral lines | FFNO: any line of the H and Ca II atoms, found from the region (Lyman, Balmer, Paschen and Brackett lines up to n = 5; Ca II H, K and the infrared triplet). Validated for H alpha and Ca II 8542 only. LTE: any Kurucz line list. |
| Viewing angle | `[view]` μ, azimuth, `plane_parallel` or `inclined`, all native coordinates, forward and inversion (ADR-013). Validated against MULTI3D at μ = 0.8. |
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
| `log10m` | `(z)` | log10 kg m⁻², or g cm⁻² with a `units` attribute | log-m runs only; strictly monotonic |
| `z_m` | `(z)` | m, positive upward | z runs only; either order |
| `x_m`, `y_m` | `(x)`, `(y)` | m | required for z runs, optional for log tau; uniform and increasing |
| `temperature` | `(time,z,y,x)` or `(z,y,x)` | K | required |
| `vx`, `vy`, `vz` | as temperature | m/s | all three required |
| `vturb` | as temperature | m/s | optional; zero when not mapped |
| `Bx`, `By`, `Bz` | as temperature | T | optional, all three or none; absent means B = 0 |
| `ne` | as temperature | m⁻³ | optional, forward only |
| `rho` | as temperature | kg m⁻³ | optional, forward only |
| `pgas` | as temperature | Pa | optional, forward only |
| `hydrogen_populations` | `(level,z,y,x)`, optionally with time first | m⁻³ | optional, forward only; the last level is the protons |

Dataset names are configurable in `[atmosphere.datasets]`. Set
`[atmosphere] storage_order = "python"` for files written by NumPy/h5py or C, so
row-major dimensions are reversed on read; use `"julia"` (the default) for
HDF5.jl files. `[inputs] time_index` (1-based) selects the snapshot.

Inputs may be stored bottom-first or top-first. They are reversed once on read so
that inside the package index `k = 1` is always the top of the atmosphere.

In an inversion, and in a forward run given none of the optional thermodynamic fields, pressure, density and electron
density are reconstructed by HSEMHS. Given any of them, a forward run uses them instead: see
[A given atmosphere](#a-given-atmosphere-nₑ-ρ-p-and-hydrogen-populations-any-axis). A starting atmosphere must already satisfy
the physical bounds configured for the controls.

[`../scripts/build_input_ltau.py`](../scripts/build_input_ltau.py) builds a native
log-tau input from a Bifrost snapshot. It writes `log10tau500`, `temperature`,
`velocity_{x,y,z}` and `B_{x,y,z}` in Python order, so map those names in
`[atmosphere.datasets]` and check that the units match the table above.
[`../scripts/build_input_logm.py`](../scripts/build_input_logm.py) does the same for a
native log column-mass input, with `log10m` in place of `log10tau500`.

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
| `wavelength_m` | concatenated region wavelengths (m), in the frame of `[outputs] wavelength_frame` |
| `objective_total`, `objective_data`, `objective_regularization` | inversion only |
| `execution_mode`, `axis_order`, `wavelength_frame`, `provenance_toml` | attributes |

### Atmosphere output

| Dataset / attribute | Content |
| --- | --- |
| `logtau_500` or `z_m`, `x_m`, `y_m` | native axes |
| `temperature`, `vx`, `vy`, `vz`, `vturb`, `Bx`, `By`, `Bz` | recovered (or input) fields, `(time,z,y,x)` |
| `pgas`, `rho`, `ne`, `z`, `logtau500`, `chi500` | equilibrium or given atmosphere, `(time,z,y,x)` |
| `column_mass` | runs with a given atmosphere only, kg m⁻², `(time,z,y,x)` |
| `populations/<species>` | FFNO populations, `(level,z,y,x)`, m⁻³ |
| `vertical_coordinate`, `vertical_unit`, `vertical_order = "top_first"` | attributes |
| `execution_mode`, `solver_termination`, `provenance_toml` | attributes |

`provenance_toml` records the configuration hash, parallel topology, equilibrium
mode and final status, EOS provenance, and the control mapping.

## Native vertical coordinates

`[atmosphere] vertical_coordinate` chooses the native grid. Everything else follows
from it:

| | `"log10tau500"` | `"log10m"` | `"z"` |
| --- | --- | --- | --- |
| Example configs | `forward.toml`, `inversion.toml` | `forward_logm.toml` | `forward_z.toml`, `inversion_z.toml` |
| Vertical dataset | `logtau500` | `log10m`: m in kg m⁻², or g cm⁻² with a `units` attribute | `z_m`, plus `x_m` and `y_m` |
| HSEMHS mode | `column-hse-ltau`: heights derived per column | `column-hse-logm`: heights and log10 tau500 derived per column | `column-hse-z`: log10 tau500 derived per column |
| Pressure boundary | HSEMHS tau-hydrostatic top closure (`[equilibrium.column_ltau]`) | P = g m, μ g m along the line of sight (`[equilibrium.column_logm]`) | `[equilibrium.column_z] anchor`, `P_ref_pa` |
| FFNO populations | on the `[population.mesh]` height mesh, then interpolated back, unless `enabled = false` | as log tau | as log tau |
| Formal solution | optical depth integrated on the native log tau500 grid using chi500 | optical depth integrated on the native log m grid using rho, dτ = (χ/ρ) dm | optical depth integrated in geometric height |
| Above and below the column | on the mesh: corona above, hydrostatic extension below ([Population mesh](#population-mesh)) | as log tau | as log tau |
| Control nodes | `log_tau_nodes` | `log_m_nodes` | `z_nodes_m` |
| Regularization length | `length_scale_dex` (default 1) | `length_scale_dex` (default 1) | `length_scale_m` (required for derivative types 1 and 4) |

A native height axis is never treated as log tau, and a column-dependent derived log
tau is never collapsed into one shared vector.

On a log m grid the axis is m along the line of sight, one ray per pixel, as on a log tau
grid: a file for μ < 1 must hold that axis and carry the view attributes. Both `z` and
`logtau500` are written as derived fields, and the axis is written as `log10m` [dex of kg m⁻²].

Without any given thermodynamics the axis sets the pressure, P = g m, and ρ comes from the EOS. With them, see below.

### A given atmosphere: nₑ, ρ, P and hydrogen populations, any axis

In forward synthesis, `datasets.ne` (m⁻³), `datasets.rho` (kg m⁻³), `datasets.pgas` (Pa) and `datasets.hydrogen_populations` (m⁻³, file axes
`(level,z,y,x)` optionally preceded by time, the last level the protons) are accepted in any combination, on any of the three axes. **If any of them is given,
HSEMHS is not used.** It solves only when none is given, and always in an inversion. `[equilibrium]` is then optional and its column sections are ignored.

A given field is never replaced. What is missing is found in this order, with Wittmann's composition, as in RH (`MULTIatmos`, `convertScales`):

| Missing | Comes from |
| --- | --- |
| ρ | the populations, n_H = Σ n, ρ = (mass per H) n_H; else, with nₑ and P, (mass per H)(P/kT − nₑ)/Σ abundances; else the EOS at (T, P) |
| P | (n_H Σ abundances + nₑ) k T when nₑ and n_H are known, n_H from the populations or ρ; else the P at which the EOS gives the given nₑ or ρ |
| nₑ | the EOS at (T, P) |
| neutral H, protons | the EOS at (T, P) |
| χ₅₀₀ | Muspel's continuum at 500 nm from T, nₑ, neutral H and protons |

nₑ is never derived from P and ρ by subtraction: in the photosphere it is a small difference of large numbers. The EOS here is the Wittmann library used by synthesis,
and it is LTE, so what it supplies can be far from a model's real values where the model is not in LTE. For a simulation's own thermodynamics, give
`ne`, `rho` and `pgas` together: the FFNO then receives its T, v, nₑ and ρ unchanged. Fields given together need not agree, and each is used as given; a difference above
0.3 dex between the given ρ or P and the value the other given fields imply is logged, as it usually means a mistake in the input.

The geometry follows from ρ and χ₅₀₀:

- Log m axis: dz = −2 μ Δm/(ρₖ₋₁+ρₖ), τ₁ = (χ₅₀₀/ρ) m, dτ = χ₅₀₀ |dz|/μ.
- Log τ axis: dz = −2 μ Δτ/(χ₅₀₀ₖ₋₁+χ₅₀₀ₖ). The axis must fit the atmosphere as given: take it from a z run with the same input, otherwise the heights are wrong.
- Height axis: the axis itself, τ₁ = χ₅₀₀ |Δz|/(2μ), 0 if that exceeds 1.
- On the mass and τ axes every column has z(τ = 1) = 0 (RH's rule, column by column, not HSEMHS's mean over the cube). Those columns are therefore not aligned with
  each other by pressure.

These runs also write `column_mass` [kg m⁻²] to the atmosphere output, along the line of sight, as in RH: P₁/(μ g) at the top and dm = (ρₖ₋₁+ρₖ)/2 · |dz|/μ below it. On a
log m axis it is the axis.

The continuum and neutral H in the broadening of FFNO and Kurucz lines use the given hydrogen when there is any, and Stark broadening uses nₑ. FFNO still predicts the line
populations, and the lower-level populations of Kurucz LTE lines still come from Wittmann at (T, P). The population mesh's corona and deep extension use the HSEMHS EOS
table: set `[equilibrium] eos_table_file`, or `[population.mesh] enabled = false`.

### An IRIS/MURaM FITS atmosphere

In forward synthesis, `[atmosphere] atmos_model_format = "iris"` with `atmos_directory = "..."` reads one snapshot's FITS cubes, `<code>_<run>_<quantity>_<snap>.fits`, instead of
`[inputs] initial_atmosphere_file` and `[atmosphere.datasets]`. The quantities are `lgtg`, `ux`, `uy`, `uz`, `lgr` and `lgp`, with heights in the `z-coordinate` extension and units from the
headers. It is read as `scripts/convert_iris_sim_fits_to_ffno_hdf5.py` reads it: index 1 is the top, and (x, y) of the file becomes (x, reversed y) with `uy` kept as it is, so results line up with
that script's predictions. It is a given atmosphere on a native z grid, with ρ and P as given and nₑ from the EOS at (T, P), so HSEMHS isn't used. The horizontal spacing comes
from the headers, and `[grid]`, if set, must agree. The population mesh needs `[equilibrium] eos_table_file`, and it gives the FFNO only the mesh's heights.
Cells colder than `[atmosphere] temperature_floor_k` (3250 K by default, as the script; 0 keeps the file's temperatures) are raised to it, and only the temperature changes. `storage_order` and `[atmosphere.datasets]` are ignored. The script's resampling is not applied: this runs on the cubes' own grid.

### A MULTI3D atmosphere

In forward synthesis, `[atmosphere] atmos_model_format = "multi3d"` with `atmos_directory = "..."` reads the snapshot directory of a MULTI3D run, with its `mesh` and `atm3d`, instead of
`[inputs] initial_atmosphere_file` and `[atmosphere.datasets]`. `atm3d` holds nₑ [cm⁻³], T [K], vx, vy, vz [km s⁻¹] and ρ [g cm⁻³], Float32 in Fortran order (x, y, z). When its size shows six more
fields, those are the hydrogen populations [cm⁻³], the last the protons, and they are used as well. Index 1 is the top, and (x, y) and the velocities stay as they are in the file, so results
line up with MULTI3D's own outputs for the snapshot and with the predictions of `ffno/pipeline.py`, with no y flip, unlike the IRIS route. In most snapshots the mesh's y falls with the index; only the size of its spacing is used, as in `pipeline.py`. It is a given atmosphere on a native z grid, so HSEMHS isn't used: P follows
from nₑ and ρ or the populations. Heights and the horizontal spacing come from `mesh`, and `[grid]`, if set, must agree. The population mesh needs `[equilibrium] eos_table_file`, and it gives the FFNO only the mesh's heights.
The file's temperatures are kept unless `[atmosphere] temperature_floor_k` asks for a floor, and only the temperature then changes. `storage_order` and `[atmosphere.datasets]` are ignored.
Positive `vz` is upward, the default of `vz_positive`.

## Configuration reference

The checked-in configs in [`../examples/inversion_run`](../examples/inversion_run)
are commented and complete. `scripts/dry_run.jl` prints the parsed configuration.
Unknown keys in `[equilibrium]` and its subsections are errors. With a given atmosphere, `[equilibrium]` is optional and its column sections are ignored.

### Top level, `[inputs]`, `[outputs]`

| Key | Meaning |
| --- | --- |
| `mode` | `"inversion"` (default) or `"forward"`. `[solver] max_iterations = 0` also selects forward. |
| `[inputs] initial_atmosphere_file`, `observation_file`, `time_index` | input paths (relative to the run directory) and 1-based snapshot |
| `[outputs] synthesis_file`, `atmosphere_file` | output paths |
| `[outputs] wavelength_frame` | `"air"` (default) or `"vacuum"`: the frame of the written `wavelength_m`, recorded in the file attribute `wavelength_frame`. A label only: `[[regions]] start_angstrom` stays air and synthesis runs on the vacuum grid in either frame |
| `[outputs] intensity_units` | `"si"` (J s⁻¹ m⁻² Hz⁻¹ sr⁻¹, RH's; default) or `"cgs"` (erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹, STiC's): the per-frequency unit (I_ν = I_λ λ²/c, vacuum λ) of the written synthesis and of the observation file's `intensity` and `sigma`, before the region `normalization` divides them. A written synthesis can be read back as an observation. The file records it in the `units` attribute of `intensity` and the divisor per wavelength in `normalization` |

### `[atmosphere]`

`vertical_coordinate`, `storage_order` and the `[atmosphere.datasets]` name map
(`logtau500` or `z_m`, `x_m`, `y_m`, `temperature`, `vx`, `vy`, `vz`, optional
`vturb`, `Bx`, `By`, `Bz`). There is no top-pressure or top-density option: configs
that still contain `pressure_top_pa` or `pressure_top_dataset` are rejected.

Optional `ne` (m⁻³), `rho` (kg m⁻³), `pgas` (Pa) and `hydrogen_populations` datasets in `[atmosphere.datasets]` are taken as the truth, **in forward synthesis only**: an inversion
always takes them from the EOS, and `load_config` refuses them there. The routes are in the table under
[A given atmosphere](#a-given-atmosphere-nₑ-ρ-p-and-hydrogen-populations-any-axis). See also [Equilibrium](#equilibrium).

`vz_positive` says what a positive `vz` means in the input files. `"up"` (default) is the code's own sign, as in the
Bifrost/MULTI3D files the FFNO was trained on (rising hot granules have positive `vz`). `"down"` negates `vz` on input
and on output (attribute `vz_positive`) and mirrors `vz` control bounds. Only that sign changes: z is positive upward
and τ increases downward in every run, and `vx`, `vy`, B, the grid, the `[view]` azimuth, observations and `vlos`
are untouched. Positive line-of-sight velocity (toward the observer) is a blue shift in every line model. Top-first or
bottom-first storage of the arrays needs no setting and no sign change.

### `[equilibrium]`

| Key | Meaning |
| --- | --- |
| `backend` | must be `"HSEMHS"` |
| `model` | `"column_hse"`; `"mhs"` is reserved for Phase 4 |
| `gravity_m_s2` | surface gravity (default 274.0) |
| `eos_table_file` | HSEMHS EOS table, e.g. a link to `HSEMHS/cache/eos_table_fullz.h5` |
| `on_nonconvergence` | `"reject_trial"` (an inversion trial is rejected) or `"abort"` |
| `[equilibrium.column_ltau] align`, `[equilibrium.column_logm] align` | `"ptot"` (P + B²/2μ₀ equal at one height) or `"none"` |
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
| `start_angstrom`, `step_angstrom`, `count` | uniform air-wavelength grid (Å), whatever `[outputs] wavelength_frame` writes |
| `normalization` | positive number: the observation file and the written synthesis both hold `[outputs] intensity_units` divided by it, for every Stokes parameter (1: plain SI or cgs). Overlapping regions must state the same value |
| `psf_type`, `psf_file` | `"none"`, `"fpi"` or `"file"`; a file is required unless `none` (see [Known limitations](#known-limitations)) |

Each region has one or more sources:

| `mode` | Keys | Meaning |
| --- | --- | --- |
| `"ffno"` | none | NLTE lines from FFNO populations through Muspel: every line of the factory's `ATOM_FILES` centred in the region |
| `"kurucz_lte"` | `linelist_file` | LTE lines from a Kurucz list, Wittmann EOS |

An FFNO source names no line. `ffno_region_lines` reads the atom files and takes every bound-bound transition whose
centre lies on the region's grid, within 20 × 2.5 km/s of its ends; its two population levels are the atom's levels
by increasing energy, which is the FFNO's level order. Preflight and production both choose through it, and preflight prints
what it found. A region with both Ca II H and K synthesizes both. A region with no line centre, such as a far wing alone,
is an error that lists the available lines. Below 200 nm the grid is in vacuum, as `_air_to_vacuum_nm` treats it. `species` and `line` keys in
older configs are ignored with a warning.

Resonance lines (lower level the lowest of the ion: Lyman lines, Ca II H and K) need PRD, which is not implemented,
so their core and near wings are not trustworthy; preflight marks them. Only H alpha and Ca II 8542 have been compared
with RH or MULTI3D; the other lines use the same populations without such a comparison.

Sources in one region share a single formal solution, so blends are synthesized
together: Kurucz lines inside an FFNO line's window (Fe I 656.92 nm next to H alpha,
Si I 853.62 nm next to Ca II 854.2 nm) add their opacity and emissivity to the FFNO
line's. Every model writes emissivity in one unit (kW m⁻³ sr⁻¹ nm⁻¹, Muspel's), so the
synthesized intensity is in kW m⁻² sr⁻¹ nm⁻¹ before the output conversion. Every region
has one continuum, which does not depend on the order of the sources: Muspel's with hydrogen from
the EOS, which reads no FFNO populations. A Kurucz line of the FFNO
transition's own ion within 0.1 nm of it (H alpha, Ca II 854.2 nm in a full Kurucz list)
is dropped and logged, so the FFNO line is not added a second time in LTE. A region
with only LTE sources does not call FFNO.

K94 damping constants of zero mean "not given", as in RH: radiative damping becomes
A_ul and Stark broadening is off. Van der Waals broadening of a Kurucz line is, in
this order (`KuruczLine.vdw_recipe`):

1. ABO (Anstee, Barklem & O'Mara) with σ and α given in the γ_vdW field as σ.α
   (a value ≥ 20: σ in a₀² is the integer part, α the fraction; RH's convention);
2. ABO from RH's tables (`data/barklem`, s-p, p-d and d-f) for a neutral atom whose
   level labels give the valence-electron orbitals (RH reads them the same way);
3. the list's log γ_vdW when non-zero;
4. Unsöld (γ = C T^0.3 n_H, collisions with H and He).

ABO gives γ = C(σ, α) T^((1−α)/2) n_H, as RH computes it. RH itself puts Unsöld
before the list's γ_vdW; here a value in the list is used.

FFNO atom lines (Muspel atom files) take the same theory through a
`{type: VanderWaals_ABO}` broadening entry. With `σ` and `α` (numbers or
`{value: ...}`) they are used as given, as in RH's `ABO_EXPL`. Without them they come
from the tables for a neutral atom whose level labels give s-p, p-d or d-f, as in RH's
`BARKLEM`; otherwise the entry falls back to Unsöld with a warning. As in RH, each ABO
entry adds Unsöld broadening by helium. Ca II and H I lines are outside the tables, so
for them σ and α must be given or Unsöld is used.

### `[population.mesh]`

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | true | predict the FFNO populations on the mesh, on every axis and in both modes; `false` feeds the FFNO the native grid |
| `bottom_m`, `top_m` | −1 Mm, 14 Mm | mesh limits |
| `spacing_m`, `uniform_top_m` | 20 km, 4.5 Mm | uniform spacing up to this height |
| `stretch`, `max_spacing_m` | 1.01, 100 km | spacing above it grows by this factor per point, up to this spacing |
| `nabla` | 0.13 | d ln T / d ln P of the extension below a column's bottom |
| `corona_temperature_k` | 1 MK | temperature of the corona above a column's top; 0 holds the top values instead |
| `corona_transition_m` | 240 km | height over which the corona reaches that temperature |

The defaults lie inside the range of the training grids. See [Population mesh](#population-mesh).

### `[population.given]` (forward only)

Populations read from files instead of predicted, for testing the synthesis and the rest of the code on populations known to be right, such as
MULTI3D's. Species not listed still come from the FFNO; when every species is listed, no GPU service starts.

```toml
[population.given]
format = "multi3d"          # MULTI3D out_pop; or "hdf5"
orientation = "auto"        # or identity, flip_x, flip_y, rotate_180, transpose, transpose_flip_x, transpose_flip_y, transpose_rotate_180
# tau500 = ".../tau500"     # default: MULTI3D's tau500 beside the first file, or in its directory's parent
# min_correlation = 0.5     # auto: the least correlation the chosen orientation may have

[population.given.files]
H = "/.../bifrost_data/cbh24/s0957/H/out_pop"
CA = "/.../bifrost_data/cbh24/s0957/CA/out_pop"
```

| Key | Default | Meaning |
| --- | --- | --- |
| `format` | `"multi3d"` | `"multi3d"`: `out_pop`, Float32 Fortran (x, y, z, level), top first, the NLTE levels [cm⁻³] first, what follows not read; the grid is `multi3d.input`'s beside the file, else the atmosphere's. `"hdf5"`: `populations/<species>` of a forward run's atmosphere output, (level, z, y, x) [m⁻³], checked with its `logtau500` |
| `files` | required | one file per species, by the model factory's species names; relative paths are taken from the configuration's directory |
| `orientation` | `"auto"` | how the file's columns map onto the atmosphere's: `auto` finds it from tau500; a named one is used as given, with a warning if another correlates better |
| `tau500` | beside the file | MULTI3D's tau500 cube (Float32 or Float64, (x, y, z)) for the orientation check |
| `min_correlation` | 0.5 | `auto` refuses unless the best orientation reaches this and leads the next by 0.1 |

The populations must be on the atmosphere's own grid: the same levels (a MULTI3D file needs a height axis, and a `mesh` beside it or in its parent
must have the atmosphere's heights) and the same columns, perhaps rotated or mirrored. Before anything starts, rank 0 checks each file's size
against its grid and the species' level count, then picks the orientation: at the level where the median tau500 is closest to 1 it correlates
log10 tau500 with the atmosphere's temperature under every rotation and mirror the grid shape allows (at a fixed height tau500 grows with the
temperature, through H⁻). A tau500 that falls with the index means the file is bottom first, and its levels are reversed. It then reads the
populations, refuses any that are not finite or negative, and prints every finding as `GIVEN_POPULATIONS ...` lines, which the outputs'
`provenance_toml` keeps under `given_populations`. Given populations skip the population mesh and `[physics] ffno_population_limit` and
`ffno_no_inversion`. The factory must build its populations with `launch_fsdp_population_models`, which hands them over; a run whose factory
does not is refused.

`scripts/preflight_given_populations.jl CONFIG.toml MODEL_FACTORY.jl` runs the checks alone, with `--read` also reading the populations.

### `[view]` (viewing angle)

`mu` (cos θ, 0 < μ ≤ 1, default 1), `azimuth_deg` (φ from +x toward +y, default 0) and `geometry`
(`"plane_parallel"`, default, or `"inclined"`). Without the section, or at μ = 1, every result is bit for bit the
vertical one. Design, decisions and validation: [`docs/viewing-angle/README.md`](docs/viewing-angle/README.md).

- The Doppler shift uses the line-of-sight velocity n·v. The `vlos` control moves v along n and keeps the
  perpendicular part at the starting atmosphere's; it cannot be combined with `vx`, `vy` or `vz` controls.
- **Native log τ:** the input columns must already be lines of sight: T on τ along the ray, on a regular deprojected
  footprint grid, tagged with `view_mu`, `view_azimuth_deg` and `view_geometry` (a file without the tags is accepted only
  at μ = 1). HSEMHS balances each ray. With `inclined`, the population mesh shifts every mesh level by −t·z to the
  Cartesian positions before the FFNO and the populations back by +t·z (t = tan θ (cos φ, sin φ), rays through the
  footprints at the run's z = 0).
- **Native z:** the cube is Cartesian and the FFNO sees it unchanged. The formal solution uses the path dz/μ;
  with `inclined` it runs on footprint columns: the state and populations shifted by +t·z per level on the root rank.
- The spatial PSF FWHMs are in sky metres and are stretched by 1/μ along the tilt; only φ = 0, 90, 180, 270° are
  accepted with a spatial PSF.
- Outputs record `view_*` attributes; `column_response.jl --variables vlos` gives the response to n·v (`vz` is the
  vertical component).

### `[physics]`, `[observation]`

| Key | Meaning |
| --- | --- |
| `[physics] redistribution` | `"non_prd"` |
| `[physics] line_center` | where FFNO line profiles are centred. `"reference"` (default): the laboratory wavelength in `FFNO_REFERENCE_NM`, H alpha at 656.28 nm and Ca II 8542 at 854.209 nm (air); a line not tabulated there uses the atom's. `"atom"`: the atom file's level energies, as RH with unmodified atoms. Kurucz lines keep their list's wavelengths |
| `[physics] ffno_population_limit` | largest total of a species' FFNO levels synthesis reads, as a multiple of what ρ holds (default `1.09`; `0` turns it off). See [FFNO population limits](#ffno-population-limits) |
| `[physics] ffno_no_inversion` | cap the upper level where the line opacity would turn negative (default `true`) |
| `[physics] continuum_populations` | only `"eos"` (default): one Muspel continuum per region, hydrogen from the EOS. `"ffno"` is refused. |
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
4. reconstructs the initial equilibrium, predicts populations once into
   `cartesian_populations(workspace)`, builds the footprint state (`footprint_state!`, a no-op unless the view is
   inclined on a height grid) and builds the cached synthesis setup (`build_synthesis_setup`) on
   `synthesis_atmosphere(workspace, distributed)`;
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
repo=/cluster/projects/nn2834k/harshm/ARUN
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

`src/Equilibrium.jl` holds the routes to pressure, density, electron density and
the derived vertical coordinate. With HSEMHS, `prepare_equilibrium` runs once (collective);
`reconstruct_equilibrium!` runs every evaluation and fills `pgas`, `rho`, `ne`, `z`,
`logtau500` and `chi500` from T (and B) without touching T, B, velocities or vturb.
It returns an `EquilibriumOutcome` that is identical on every rank and does not throw
for physics failures (EOS envelope, non-finite or folded columns). Instead
`outcome.valid` is false. A forward run then stops with an error, and so does an inversion for now: rejecting the
trial step, which `on_nonconvergence = "reject_trial"` will select, arrives with HSEMHS-integration Phase 4.

In forward synthesis, any of nₑ, ρ, P or the hydrogen populations in the input replaces all of this: see [A given atmosphere](#a-given-atmosphere-nₑ-ρ-p-and-hydrogen-populations-any-axis).

FFNO owns MPI and threading and passes HSEMHS its communicator. HSEMHS solves on
x slabs, which is why `decomposition = "x_slabs"` is required.

### Population mesh

The FFNO checkpoints were trained on simulation height grids: one set of heights for
every column, 12–30 km spacing in the photosphere and chromosphere, bottoms
1–2.5 Mm below the surface, and tops up to 14 Mm. A 95-point log-tau column has 30 km to several Mm
spacing and a different height at every point, which is far outside that. Fed
directly, H n2 is about 100× off in the chromosphere. A 1-D model on z has the same problem with its uneven levels.

`MeshPopulationModel` (`src/PopulationMesh.jl`) therefore maps each column onto a
fixed height mesh using its heights, calls the FFNO there, and interpolates
the populations back. It is on for every axis and in both modes, unless `[population.mesh] enabled = false`.
The rule is the same for every species and both axes:

- **inside the column:** ln T, log nₑ and log ρ log-linear in z, velocities linear;
- **above its top (corona):** only where the mesh must extrapolate. ln T rises linearly in z from the column's top value to
  `corona_temperature_k` over `corona_transition_m`, then stays there; a column already hotter keeps its temperature. P is
  integrated hydrostatically upward from the top state (RK4, g = `gravity_m_s2`), ρ and nₑ come from the HSEMHS EOS table, and
  velocities are held. Above the table's low-pressure end the last state is held. With `corona_temperature_k = 0` the top values
  are held instead;
- **below its bottom:** only where the mesh must extrapolate. Hydrostatic downward from the bottom state with
  d ln T / d ln P = `nabla`, ρ and nₑ from the HSEMHS EOS table, velocities held;
- **back to the column:** populations log-linear in z.

The VJP is the transpose of these maps and the JVP their tangent (finite differences of the top and bottom values in
the corona and the deep extension). The extensions exist only for the FFNO: synthesis uses the column's own levels.

The corona matters because the FFNO is not local in height: its deep populations respond to what fills the mesh above. For
FALC, which stops at 2.16 Mm, holding the 100,000 K top values to 14 Mm made total H up to 15× too high in the photosphere, and
changing a corona from 1 MK to 0.5 MK doubled it at 250 km. Columns that already reach the mesh top, like Bifrost cubes on log τ,
never use it.

### Viewing angle

`[view]` (configuration reference above; design, decisions and validation in
[`docs/viewing-angle/README.md`](docs/viewing-angle/README.md), ADR-013) sets the direction to the observer
n = (sin θ cos φ, sin θ sin φ, μ). The view lives in `Grid3D.view` (`ViewGeometry`), so every tile and output carries
it; `src/ViewAngle.jl` holds the line-of-sight velocity, the path factor and the periodic shift.

- **What changes.** HSEMHS balances along the line of sight (`ViewOptions`). Every Doppler shift uses n·v. Native
  log τ: the axis is τ along the ray, so the formal solution is unchanged; with `inclined` the population mesh
  shifts each mesh level by −t·z before the FFNO and the populations back by +t·z (t = tan θ (cos φ, sin φ)), so the
  network sees every ray point at its Cartesian position. Native z: the FFNO sees the cube unchanged; the formal
  solution uses the path dz/μ; with `inclined` the synthesis runs on footprint columns (state and populations
  shifted by +t·z per level, log for positive fields) — long characteristics sampled where each ray crosses each
  level plane.
- **Rays** are x = ξ + t (z − z_ref) with z_ref = 0 (HSEMHS's gauge for log τ, the grid's z = 0 for height grids),
  periodic; a warning is printed when they travel farther than the box.
- **Adjoints**: n·v̄ for velocities, the exact transpose of the shift (`shear_levels_transpose!`) for the population
  mesh and the footprint state, z̄/μ for the path.
- **Validation** (design doc §9): μ = 1 bit-identical to the vertical code on production data; adjoint against
  finite differences 4e-10 (log τ) and 3e-9 (z) at μ = 0.8; 1, 2, 3 MPI ranks identical; the footprint synthesis
  equals an independent ray tracer, and 4× finer sampling along the ray changes it by 0.6 % rms; FFNO H-alpha
  core against MULTI3D at μ = 0.8 r = 0.81 on the Bifrost cube's own height grid (0.82 at μ = 1), while
  plane-parallel gives 0.04.

### Synthesis

`build_synthesis_setup` reads and caches all line physics outside the pixel loops:
Muspel atoms, continuum and Voigt tables, Kurucz line selections and the Wittmann
partition functions. Configured air wavelengths stay on the observation side and
are converted to vacuum for line physics; `[outputs] wavelength_frame` only chooses the
frame written to the file. Muspel lines use the exact Voigt profile
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

Every region has one continuum, whatever its sources: Muspel's at the region's central wavelength, from T, nₑ and neutral H and protons
from the EOS at (T, P), forward and inversion alike. FFNO populations never enter it, and `[physics] continuum_populations = "ffno"` is
refused. The only FFNO populations synthesis reads are the line's own two levels: neutral H in van der Waals broadening comes from the EOS too.
In a given atmosphere both come from the input instead. Lower-level populations of Kurucz LTE lines stay Wittmann's.

### FFNO population limits

The FFNO can predict impossible populations in gas far from its training data: in `ar098192` at about 1 MK and high density, H totals up to 22,000 times the hydrogen present, and inverted Hα levels. Thin gas absorbs almost nothing but still emits, so these cells gave Hα up to 44 times the continuum, and near-zero opacity gave negative intensities. Before each FFNO line's opacity and emissivity, for Hα and Ca II, at every temperature:

- If the total of the species' levels exceeds `ffno_population_limit` × ρ/(mass per H) × abundance, all its levels are scaled down to that total. The mass per H is the Wittmann EOS's and the abundance the atom file's.
- With `ffno_no_inversion`, nᵤ is capped at nₗ Blu/Bul, where the line opacity is zero.

The inversion gradients follow both limits. They apply only to what synthesis reads: the FFNO's own populations are what the atmosphere file holds. `synthesis.h5` records both settings as attributes.

The default comes from the 49 MULTI3D outputs in `bifrost_data`. The H total reaches at most 1.086 of ρ/(mass per H) in 23 of the 25 H outputs. The other two, `en024031_by100_helium/109` and `qs006003_sap/689`, exceed it only in cool gas, up to 2.6e5 and 88 times, which looks like a problem in those runs and has not been traced. Ca is 1.031 everywhere. Hα is inverted in one cell of all 25, and Ca II 8542 in none.

FFNO line profiles are centred by `[physics] line_center`. By default that is the reference wavelength, such as H alpha 656.28 nm and Ca II
854.209 nm (air); only the profile's detuning moves, and the Doppler width and the transition's strength stay the atom's. With `"atom"` it is the atom
file's level energies, about +0.16 Å at H alpha and -0.03 Å at Ca II from those references, so results differ from RH's by the differences between the
two codes' atom files (+0.08 and -0.03 Å). Air wavelengths convert to vacuum with the same formula as RH's `air_to_vacuum`. RH's `vacuum_to_air`, which
labels its air output, is not its inverse: it differs by 3 mÅ at H alpha and 5 mÅ at Ca II, so comparisons at that level belong on vacuum axes.

Doppler widths include `vturb` in quadrature, in FFNO and Kurucz lines alike (`vturb` is not an FFNO input, so populations do
not respond to it). The Wittmann adjoint is local in depth: perturbing a level recomputes only that level.

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

### Column response functions

`scripts/column_response.jl` (Olivia: `scripts/run_olivia_column_response.sbatch`, submitted from a forward run
directory) writes R_v(λ, k; x, y) = ∂I(λ, x, y)/∂v(k, x, y) of the intrinsic intensity for every configured
wavelength and every native log τ500 point, per unit log τ500, for `temperature`, `vz`, `vlos` (n·v; equals `vz` at μ = 1), `vturb`, `ne` or `pgas`
(`--variables`, `--stride`, `--crop`). Populations and the HSE structure (pgas, rho, ne, z, chi500) are held at the
forward run's values, so each column is independent: one formal-solver adjoint sweep plus two opacity evaluations
per variable, with no FFNO or HSEMHS call. Size: 4 bytes × wavelengths × depths × columns × variables before
compression (2.7 GB for T and vz at stride 4 on 504²).

With `--population-tangents TANGENTS.h5` the temperature response also lets the FFNO populations respond.
`scripts/ffno_population_tangents.jl` (Olivia: `scripts/run_olivia_population_tangents.sbatch`, GPU) makes the file:
for every log τ500 point k0 it asks the FSDP service for one forward-mode JVP (dual tensors through the population
mesh and the FFNO, `population_jvp!`) seeded with 1 K at k0 in every column, and stores the population tangent
ṅ(k, x, y; k0) = Σ_{x',y'} ∂n(k, x, y)/∂T(k0, x', y') (Float32, 4 bytes × depths² × levels × columns per species,
55 GB for H on 504²). The seed in every column keeps the horizontal coupling of the Fourier layers in that sum without
storing the full Jacobian. `column_response.jl` then adds Σ_{k,l} G_l(k, λ) ṅ_l(k; k0) per column, with
G_l = ∂I/∂n_l from the same adjoint. pgas, rho, ne and z stay fixed. `test/ffno_jvp_cpu_check.py` checks the
distributed JVP against finite differences on CPU.

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

### Batched forward service (small GPUs)

[`../ffno_batched_service.py`](../ffno_batched_service.py) and [`../ffno_batched_backend.py`](../ffno_batched_backend.py) answer `PREDICT` for
cubes whose activations do not fit on the GPUs, such as 504 × 504 on eagle7's T4s. They speak the FSDP service's protocol from one process with no
torchrun, and hold the hidden states in host RAM (about 90 GB at 504 × 504 × 467) while the visible GPUs process batches. This is not tiling. The
layers are batched only where the operator allows it: kernel-1 layers anywhere, the 2D FFT by depth plane, the vertical branch by column, the 3×3×3
fusion convolution by depth with a one-cell overlap, and GroupNorm through streaming statistics in two passes. A batch that runs out of GPU memory is
split and retried. **VJP and JVP requests are refused, so this is for forward runs only.**

Use it by setting `FFNO_FSDP_LAUNCHER` to `scripts/ffno_batched_service_local.sh`, which the template `model_factory.jl` passes to the launch call,
with `CUDA_VISIBLE_DEVICES` choosing the GPUs, `FFNO_FSDP_PYTHON` a Python with CUDA torch and `[parallel] enabled = false`. Step-by-step guides:
[eagle7](../examples/inversion_run/start_here_eagle7.md), [any HPC with GPUs](../examples/inversion_run/start_here_general_hpc.md) and
[a GPU host without a scheduler](../examples/inversion_run/start_here_local_cluster_gpu.md).

| Variable | Default | Effect |
| --- | --- | --- |
| `FFNO_BATCHED_CHUNK_DEPTH`, `FFNO_BATCHED_CHUNK_ROWS`, `FFNO_BATCHED_VERTICAL_CHUNK` | 16, 16, 1024 | batch sizes: memory and speed only |
| `FFNO_BATCHED_VERTICAL_NORM_SLABS` | 1 | **changes the result**: see below |

The FSDP routes leave the vertical branch's `out_gn` as a plain GroupNorm, so its statistics are per GPU slab and their results depend on the number of
GPUs. `FFNO_BATCHED_VERTICAL_NORM_SLABS = N` reproduces an FSDP run on N GPUs, and 1 gives the volume statistic of `FFNO3D.forward`, on any number of GPUs.

Measured on H, with `scripts/compare_batched_with_reference.py` (4 November 2026):

| Case | Against | Largest difference |
|---|---|---|
| 256 × 256 × 350, 3 slabs | FSDP on 3 GPUs, same machine | 7.6e-6 dex (float32 rounding) |
| 504 × 504 × 467, 32 slabs | Olivia output, 8 nodes × 4 GPUs assumed | 0.0087 dex (median 0.00065) |

The 504² run took 428 s on 3 T4s, at 11 GB per GPU. `tests/test_batched_backend.py` checks equality with `FFNO3D.forward` for any batch sizes. A complete
run through this service and synthesis has been done on a 64 × 64 crop (spectra finite, line cores where expected). A complete run at 504 × 504, a comparison of the
final spectra with a reference and any comparison of the Ca predictions have not been done yet.

## Package layout

| File | Role |
| --- | --- |
| `src/Types.jl` | grids (with their `ViewGeometry`), vertical axes, atmospheres, magnetic fields, spectral and observation cubes |
| `src/ViewAngle.jl` | viewing angle: line-of-sight velocity, path factor, periodic shift and transpose, view attributes |
| `src/Parallel.jl` | MPI context, tiles, process grids, scatter/gather, halos, rank-0 GPU coordination |
| `src/Diagnostics.jl` | GPU-control diagnostics and failure records |
| `src/Nodes.jl` | node fields and their expansion onto tiles |
| `src/EOS.jl` | Wittmann EOS adapter (synthesis), ideal-gas test EOS |
| `src/ContinuumOpacity.jl` | Julia continuum opacity |
| `src/PopulationModels.jl` | population interface, six-channel FFNO request, recorded/mock models |
| `src/PopulationMesh.jl` | height mesh with its corona and deep extension, and their VJP and JVP |
| `src/GivenPopulations.jl` | `[population.given]`: populations from files, their preflight and orientation check (forward only) |
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
the Olivia setup and preflight scripts, the forward routes with a given density (`test/given_density.jl`), and the viewing angle (`test/view_angle.jl`: geometry, Doppler, shift and
transpose, slanted path, footprint synthesis, inclined population mesh, configuration, gradients at μ = 0.8 and the
μ = 1 identity), the native column-mass axis and given atmospheres (`test/log10m_mode.jl`), and the Muspel continuum
(`test/muspel_eos_continuum.jl`). `test/mpi_view_worker.jl` checks an inclined height-grid run on 1, 2 and 3 ranks
(`mpiexec -n R julia --project test/mpi_view_worker.jl OUT`). The `test/mpi_*_worker.jl` and
`test/olivia_*` files are driven by the validation scripts and the Olivia
regression chain (`scripts/run_olivia_runtime_tests.sbatch`), which also exercises
CUDA/NCCL probes, deliberate CPU and GPU stalls, timeouts and recovery.

Other checks: `scripts/validate_eos_consistency.jl` (HSEMHS table versus Wittmann
library), `scripts/validate_muspel_reference.jl` (synthesis against a Muspel
reference), and `../scripts/compare_service_with_ffnonet.py` (service populations
versus a direct `ffno/FFNONet.py` prediction on the same atmosphere; they should agree to
float32 round-off).

Comparison with RH on FALC (`../response_run/falc/build_falc_cmp.py`, `run_cmp_rh.sh`, `run_cmp_ffno.sh`,
`../figures/falc/plot_falc_cmp.py`): `FALC_82.atmos` and the centre pixel of `FALC_82_5x5.hdf5`, each with its own nₑ and hydrogen populations, on the mesh,
with `line_center = "atom"` and each code's atoms. RH gives 0.04% between the two atmospheres and FFNO about 1%. Against RH, on 4 November 2026, FFNO's median difference
is about 2% at H alpha and 5% at Ca II 8542, with cores about 10% and 6% deeper. RH treats H Lyman alpha and beta and Ca II H and K in PRD.

## Design documents and reports

- `docs/adr/`: architecture decisions. ADR-001 coordinates and units, ADR-002 FFNO
  population boundary, ADR-004 PRD/Stokes boundaries, ADR-005 grids, observation and
  objective, ADR-006 two-input/two-output contract, ADR-007 vertical regularization,
  ADR-008 mixed-line synthesis, ADR-009 hybrid MPI/threads/GPU, ADR-010 prototype
  solver, ADR-011 matrix-free L-BFGS, ADR-012 HSEMHS equilibrium (supersedes the
  force-balance ADR-003), ADR-013 viewing angle.
- [`docs/viewing-angle/`](docs/viewing-angle/README.md): viewing-angle design, the Codex review and the validation
  results.
- [`docs/hsemhs-integration/`](docs/hsemhs-integration/README.md): the six-phase
  HSEMHS integration plan. Phases 1–3 are done; MHS modes, equilibrium-aware
  inversion examples and the implicit equilibrium VJP are Phases 4–6.
- `docs/reports/`: validation reports for each phase, the hybrid runtime, the Olivia
  runtime-test guide and the job 2622023 crash investigation.

## Known limitations

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
- The FSDP service needs CUDA (NVIDIA) GPUs and at least two GPU ranks. The batched service works on one GPU or more, but is forward only.
- FSDP results depend on the number of GPUs, by up to about 0.3 dex at isolated points, because `out_gn` in the vertical branch is normalized per slab.
- Identical input columns do not give identical FFNO spectra: on the 5 × 5 FALC models they spread by about 4% of the maximum
  intensity. The cause has not been investigated.
- **Viewing angle:** one angle per run. Native log-τ inputs at μ < 1 must already be lines of sight on a deprojected
  footprint grid (resampling sky pixels and transforming B is the caller's job). A spatial PSF requires an azimuth of
  0, 90, 180 or 270°. The inclined shifts run on the root rank (a gather and a scatter per forward and per gradient).
  `plane_parallel` with log-τ inputs feeds the FFNO the rays as vertical columns, which does not reproduce 3-D NLTE
  at μ = 0.8.
