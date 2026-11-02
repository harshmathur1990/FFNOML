# Viewing angle (μ, azimuth) in FFNOInversion: design for review

*Status: approved 2 Nov 2026 after the Codex review (`codex-review-2026-11-02.md`, folded in below); implemented on branch
`inversions` (not committed); all of §9 done; ADR-013. Decisions in §11.*

Upstream: HSEMHS `embedding-api` 291fea3 ("Viewing angle (P1-P3)"), 0e4344a (viewing angle in the height modes,
checked on the 504² × 177 Bifrost cube) and its design
`MHS/design/tau/VIEWING_ANGLE_DESIGN.md`; README sections "What the τ-grid modes are and are not" and "Viewing angle";
report `HSEMHS/reports/mu_mismatch_2026-11-02/`. FFNOInversion pinned HSEMHS b73c863, which predates the viewing angle;
it now pins HSEMHS `main` 5c8444e (contains both).

---

## 1. What "HSEMHS supports μ" means, and what it does not

HSEMHS now takes a `ViewOptions(mu, azimuth_deg, geometry)` in `prepare_embedded` and a `[view]` section in its config.
Read carefully, it is a **statement about the input**, not a transformation:

- **It does not turn a disk-centre atmosphere into an off-centre one.** T(τ) and B(τ) are returned unchanged. Setting
  μ < 1 declares that the input columns *already are* lines of sight at that μ, with T on log τ counted along the ray
  (τ_LOS), on a regular deprojected footprint grid in the local solar frame.
- **What changes inside HSEMHS for `column-hse-ltau`:** each line of sight is balanced on its own with
  d ln P/d ln τ_LOS = μ g τ ρ/(χ P) and heights dz = −μ τ/χ d ln τ_LOS. P, ρ, nₑ and z are identical for both geometries;
  `inclined` only moves the nodes horizontally to x = ξ + t_x z, y = η + t_y z (t = tan θ (cos φ, sin φ)). Plane-parallel
  is exactly μ = 1 on the shifted axis log τ_vert = log τ_LOS + log μ.
- **`mhs-poisson-ltau`** has the slanted geometry in its force balance. FFNOInversion does not use the MHS modes yet
  (`equilibrium.model = "mhs"` is rejected), so this matters only later.
- **Height modes (`column-hse-z`, `mhs-poisson-z`): supported, with no extra input contract.** A height-grid cube is
  already placed in the local frame, so the view changes **only the optical depth**: P, ρ and χ₅₀₀ are bitwise
  independent of `[view]` (checked on the 504² × 177 Bifrost cube, 2 Nov).
  - `plane_parallel`: τ_LOS = τ_vertical/μ in each grid column.
  - `inclined`: χ₅₀₀ on each level is shifted to the footprint grid (periodic bilinear), each footprint column is
    integrated with path dz/μ, and τ is shifted back to the grid points (`ViewTau.jl`). `tau500`/`logtau500` are then
    τ_LOS **at the grid points**, along each point's own ray; down a grid column they need not increase (HSEMHS relaxes
    its fold check for this case). z(τ_LOS = 1) is reported **on the footprints**.
  - This is HSEMHS's route for an off-centre view of a simulation cube.
- **A mismatched μ is wrong, not approximate.** The 2 Nov report ran the vertical-τ halpha input as μ = 0.5 inclined:
  column HSE barely changed (P × 0.665), but MHS was badly wrong (460× the initial imbalance). An FFNO run at μ < 1 must
  therefore be fed τ_LOS columns, and building such an atmosphere from Bifrost needs a slanted-ray builder
  (`HSEMHS/scripts/viewing/make_slanted_tau.jl`), not the vertical τ file.

So "adding μ to FFNOInversion" is not one switch passed through to HSEMHS. HSEMHS covers the force balance; everything
else in the forward model that depends on the direction to the observer is FFNOInversion's job. §2 lists those pieces.

## 2. Where the direction to the observer enters the forward model

Direction to the observer **n** = (sin θ cos φ, sin θ sin φ, μ), μ = cos θ, φ from +x toward +y (HSEMHS convention).

| Piece | μ = 1 today | μ < 1 | Changes? |
|---|---|---|---|
| Atmosphere columns (native log τ) | vertical columns on τ_vert | one ray per footprint on τ_LOS | meaning only (input contract) |
| Equilibrium (column HSE) | HSEMHS | HSEMHS with `view` | pass `ViewOptions` |
| Formal solution, native log τ | per column on `logtau500`, weight τ/χ₅₀₀ | the same on τ_LOS | **no** (§3) |
| Formal solution, native z | per column in geometric z | path ds = dz/μ | **yes** (§3) |
| Doppler shift | `vz` | n·v = μ v_z + sin θ (cos φ v_x + sin φ v_y) | **yes** (§4) |
| FFNO populations | mesh columns at the pixel | 3-D NLTE populations at the node's true 3-D position | **yes**, inclined (§5) |
| Inversion velocity control `vlos` | alias of `vz` | n·v | **yes** (§4) |
| Spatial PSF | Gaussian on the sky = on the grid | sky Gaussian is stretched 1/μ along the tilt on the footprint grid | **yes** (§7) |
| Bottom boundary, Stokes I | Planck at the deepest point | the same | no |
| Magnetic field | local frame, HSEMHS only | the same (HSEMHS contract) | no (Stokes I only; Zeeman later needs B·n) |
| Native-z runs: equilibrium | HSEMHS | HSEMHS with `view` (P, ρ, χ₅₀₀ unchanged; τ_LOS) | pass `ViewOptions` |
| Native-z runs: synthesis | grid columns in z | plane-parallel: z/μ; inclined: footprint columns in z/μ | **yes** (§3, §6) |
| Native-z runs: populations | FFNO on the cube | the same (the cube is Cartesian) | no |

## 3. Formal solution

**Native log τ: unchanged.** The synthesis integrates χ_λ along each column on −ln10 log τ₅₀₀ with weight τ₅₀₀/χ₅₀₀
(`Synthesis.jl:134`). Along a ray, dτ_λ = χ_λ ds and dτ_LOS = χ₅₀₀ ds, so

    dτ_λ = (χ_λ/χ₅₀₀) dτ_LOS = (χ_λ τ_LOS/χ₅₀₀) d ln τ_LOS,

the present expression with τ_vert replaced by τ_LOS. The emergent intensity of a ray column is therefore computed by the
existing solver as long as `logtau500` holds τ_LOS (the input axis) and `chi500` the local opacity (HSEMHS returns it).
For a fixed T(τ_LOS) the spectrum changes with μ only through the equilibrium (P, ρ, nₑ), the populations and the Doppler
shift.

**Native z: path length dz/μ (Codex finding 1).** `_tau_integration` selects optical-depth integration only on a
native log-τ grid; on a height grid `formal_column!` integrates in geometric z, so HSEMHS's τ_LOS is never used by the
synthesis. The ray path is ds = |dz|/μ:

- `plane_parallel`: every grid column, path z/μ.
- `inclined`: the footprint columns of §6, path z/μ.

`formal_column!` passes z/μ to the solver; the adjoint scales z̄ by 1/μ. No τ is recomputed, so there is no new
derivative of τ or of its top closure, and 1/μ enters once. At μ = 1 the code path is the current one (no division).

## 4. Velocities

**Doppler.** Every Doppler shift reads `atmosphere.vz` today (`LinePhysics.jl`, `ext/FFNOInversionMuspelExt.jl`,
`ProductionGradients.jl`). It becomes v_LOS = n·v through one accessor, `los_velocity(atmosphere, k, x, y)`, which
returns `vz` itself when the view is vertical (bitwise μ = 1). The adjoints add n·v̄_LOS to v̄_x, v̄_y, v̄_z; the
numerical opaque-kernel VJP perturbs vx and vy as well as vz when μ < 1. The sign convention is the code's: the
production Muspel and Wittmann-Kurucz paths shift the line centre to λ₀(1 − v/c), i.e. positive v_LOS (toward the
observer, upward at μ = 1) is a blue shift. Recorded as the output attribute `view_doppler_sign`. The mock
`FFNOTransition` and the table-free `KuruczLTEModel` had the opposite sign (profile at λ(1 − v/c)); they now follow
the same convention, and a test checks the line centre at λ₀(1 − v/c) in every line model. The code frame is z up,
as the Bifrost data the FFNO was trained on (corr(T, vz) = +0.84 at z = 0); inputs whose positive vz means downward
set `[atmosphere] vz_positive = "down"`, which negates vz and nothing else.

**Control `vlos`.** The FFNO needs all three components, while the Doppler shift sees only n·v; transverse velocities
act weakly and indirectly through the FFNO populations (Codex finding 5: they are not independent measurements). The
control is the LOS component with the perpendicular part held at the prior:

    v = v_prior + δ n,        δ = v_LOS − n·v_prior,

rebuilt on every trial from the immutable prior (the starting atmosphere's vx, vy, vz), never accumulated. At μ = 1 this
is today's behaviour (vz = vz_prior + δ, vx and vy fixed). `vlos` together with any of vx, vy, vz as controls is
rejected (overlapping parameterisation). Regularisation of `vlos` acts on n·v; the gradient is n·v̄. Checkpoints record
the view and a restart with a different view is rejected.

## 5. FFNO populations at μ < 1 (native log τ)

The FFNO was trained on Cartesian cubes with full-angle 3-D NLTE radiation: populations do not depend on the viewing
angle; the observer selects rays through one population field. What changes for an inclined native-log-τ input is where a
node is: HSEMHS returns its height z, but node k of footprint (ξ, η) sits at (ξ + t_x z_k, η + t_y z_k, z_k). The
FFNO's horizontal layers must mix points at the same height and the same Cartesian x, y, not equally indexed footprints of
different rays (Codex: a coordinate conversion, not angle conditioning).

`MeshPopulationModel` maps each column onto a fixed height mesh z_m, calls the network and maps back. Every mesh level
has one height, so the ray ↔ Cartesian map is a uniform horizontal translation per level:

    F_cart(x, y, z_m) = F_ray(x − t_x z_m, y − t_y z_m, z_m),     n_ray(ξ, η, z_m) = n_cart(ξ + t_x z_m, η + t_y z_m, z_m).

Pipeline (inclined only): column → mesh per ray (unchanged) → shear the mesh features by −t z_m → FFNO → shear the mesh
populations by +t z_m → mesh → column per ray (unchanged).

**Variables (Codex finding 4).** The shear interpolates the mesh *interpolation variables*: ln T, vx, vy, vz, log10 nₑ,
log10 ρ (the same variables the column → mesh map interpolates; T is exponentiated afterwards), and ln n for the
populations (as the mesh → column back-map). Velocities are translated, never rotated: they stay local-frame components.

**Interpolation.** Periodic bilinear with HSEMHS's convention (`ViewTau.shear_levels`: the value at fractional cell
offset s·(t_x z/dx, t_y z/dy)). For a uniform shift the four weights are the same for every cell of a level, so they are
precomputed once per level and view. The transpose is the weighted scatter-add with the same weights (a correlation),
which is *not* the opposite shift; the opposite shift is not an exact inverse either. Bilinear interpolation of the log
variables keeps T, nₑ, ρ and n positive and introduces no overshoot (Fourier or cubic shifts could); its smoothing is
measured (§9 V3) before anything of higher order is considered.

**Derivatives.** The shear is linear in the interpolated variables, so the JVP shears the tangents of those variables and
the VJP applies the transpose; the existing exp/log chain rules of the mesh maps stay outside it. The mesh heights are
fixed, so the shifts do not depend on the solution.

**Where it runs.** `MeshPopulationModel` already runs on the GPU launcher rank with the *global* atmosphere
(`RootDistributedPopulationModel` gathers it), so the shear is a serial, threaded operation on global arrays there: no
new MPI, no second MPI context, and no layout adapter (Codex finding 7). Memory: one extra copy of the mesh features and
of the mesh populations on that rank. FFNOInversion therefore implements its own shear and transpose (small, testable on
CPU) instead of calling HSEMHS `shear_levels`, which needs an HSEMHS `ParCtx`, works on (nz, ny, nx) and has no
transpose; a test checks that both give the same values.

**Plane-parallel.** No shear: the network sees the ray columns as vertical at their footprints (heights compressed by μ
by HSEMHS). This is the 1.5-D convention applied to the population model, offered because HSEMHS offers it and to measure
the difference (§9 V6).

## 6. Native-z (height grid) runs

HSEMHS supports the view in both height modes (§1), and FFNOInversion runs `column_hse_z` on native-z grids, so this is
part of the implementation. It is also the direct route to an off-centre view of a simulation cube.

- **Populations: unchanged.** The atmosphere is a Cartesian cube; the FFNO sees it exactly as at μ = 1, once per
  forward, whatever the view.
- **Plane-parallel synthesis:** grid columns, path z/μ (§3), Doppler n·v.
- **Inclined synthesis: footprint columns.** The emergent intensity of footprint (ξ, η) is the formal solution along
  x = ξ + t z. A grid column is not a ray, so the synthesis runs on a *footprint atmosphere*: the state that opacity reads
  (T, vx, vy, vz, v_turb, P, ρ, nₑ and every species' populations) sheared by +t z_k per grid level, then the existing
  per-column opacity and formal solution with path z/μ. The synthetic cube lives on the footprint grid: the observation
  contract of §7.
  - Variables: ln T, ln P, ln ρ, ln nₑ, ln n (positive fields in log), velocities and v_turb linear.
  - Shearing the state and then computing opacity is not the same as shearing χ_λ and η_λ; it is the cheaper first
    implementation (about 20 planes per level instead of hundreds of wavelengths), accepted with a measured refinement
    error (§9 V5a, Codex finding 4).
  - Distribution: the synthesis is rank-local on x slabs, and a shift can cross any number of ranks. As for the
    populations, the state is gathered on the root rank, sheared there and scattered back (the existing
    `gather_atmosphere`/`distribute_field` pattern); the adjoint gathers the footprint cotangents, applies the
    transpose and scatters. Cost: one gather and one scatter of the state per forward and per gradient. Acceptable at
    first; replace by a distributed transpose if it shows up in the timings.
  - The footprint atmosphere is marked as such (`ViewGeometry.footprint = true`); the synthesis refuses an inclined
    native-z atmosphere that is not, so the grid columns can never be solved as rays by mistake.

## 7. Observations, coordinates and boundaries

**Footprint grid.** Contract taken from HSEMHS: pixels lie on a regular, deprojected footprint grid in the local frame;
dx, dy are footprint spacings (sky pixel p becomes p across the tilt and p/μ along it). Resampling sky pixels to
footprints, rotating to the box axes and transforming masks and noise is the caller's job, once per observation. If the
inversion keeps diagonal weights after that resampling, the induced noise correlation is an approximation the caller
records.

**Ray origin (Codex finding 6).** Rays are x = ξ + t_x (z − z_ref), y = η + t_y (z − z_ref) with **z_ref = 0**: the
height origin of the run. For native log τ that is HSEMHS's gauge (mean z(τ_LOS = 1) = 0); for native z it is the grid's
z = 0, as in HSEMHS's `ViewTau`. z_ref is fixed (HSEMHS uses the same origin, so making it configurable on one side only
would break registration) and recorded as `view_z_ref_m`. A gauge shift moves footprints by t·Δz; a test checks this.

**Periodic boundaries.** The shear wraps periodically in x and y (Bifrost/HSEMHS assumption), from the first wrap on.
The horizontal travel over the column, (z_top − z_bottom)·tan θ, is logged with the box size; a warning is printed
when it exceeds the box (repeated wraps), whatever μ.

**Spatial PSF.** `[observation.gaussian_psf]` FWHMs stay in sky metres (across-tilt footprint metres). On the footprint
grid the sky Gaussian is stretched by 1/μ along the tilt: FWHM_x/μ at φ ∈ {0°, 180°}, FWHM_y/μ at φ ∈ {90°, 270°}. Other
azimuths with a non-zero spatial PSF are rejected for now (the kernel would no longer be separable; a general kernel
transforms as A Σ_sky Aᵀ). The stretch is applied once, where the observation model is built; the discrete kernel is
normalised as now.

## 8. Configuration and metadata

```toml
[view]
mu = 1.0                     # cos θ, 0 < mu ≤ 1
azimuth_deg = 0.0            # φ from +x toward +y (ignored at mu = 1)
geometry = "plane_parallel"  # or "inclined"
```

- Passed to HSEMHS as `ViewOptions` in `prepare_embedded`. `[equilibrium.column_z] mu` (forwarded to HSEMHS
  `ColumnZOptions.mu`) is rejected when `[view]` is present with a different μ.
- Absent `[view]` = μ = 1: every existing config and output is unchanged.
- Validation: HSEMHS `option_errors`; a warning below μ = 0.2; the travel warning of §7.
- The view lives in the grid (`Grid3D.view::ViewGeometry`), so every local tile and gathered atmosphere carries it.
- Outputs (forward atmosphere and synthesis, inversion atmosphere, checkpoints, column responses, population tangents)
  record `view_mu`, `view_azimuth_deg`, `view_geometry`, `view_tx`, `view_ty`, `view_z_ref_m`, `view_doppler_sign`, and
  `view_columns` (`"rays"` for native log τ and footprint synthesis, `"cartesian"` for a native-z atmosphere).
- Input files: a native log-τ input at μ < 1 must carry `view_mu` (and azimuth, geometry) equal to the config, and is
  rejected otherwise (the mismatch of §1 is wrong, not approximate). A file without these attributes is a legacy
  disk-centre file and is accepted only at μ = 1. Native-z inputs need no tag (any μ).
- One angle per run.

## 9. Validation

| # | Test | Pass |
|---|---|---|
| V1 | μ = 1 (no `[view]`, and `[view]` with μ = 1 in both geometries): the existing test suite; forward spectra against the pre-change code | all tests; spectra bitwise |
| V2 | Doppler: signed shift for each velocity component at φ = 0°, 90°, 180°, 270°, 45°; adjoint against finite differences | sign exact; ≤ 1e-6 relative |
| V3 | Shear: integer and fractional shifts, periodic seams, unequal dx/dy against an analytic field; the same values as HSEMHS `shear_levels`; transpose dot-product test; shear∘unshear smoothing of Bifrost populations | ≤ 1e-12; dot test ≤ 1e-12 relative; smoothing reported |
| V4 | Uniform-atmosphere and constant-opacity slab: I = B(1 − exp(−χL/μ)) for both geometries and both native coordinates; plane-parallel equals inclined for a horizontally uniform atmosphere | ≤ 1e-10 |
| V5a | Native z, prescribed populations, Bifrost cube at μ = 0.8 inclined: footprint synthesis against an independent ray integrator (trilinear sampling at small ds) with the same populations and opacity code | geometry/interpolation error reported |
| V5b | Native z with FFNO populations against the MULTI3D μ = 0.8 rays `halpha_output/output_ray_mu08.hdf5` after its geometry is established (Codex finding 2: generating code, sheared input, φ, axis flips), next to FFNO μ = 1 against the vertical MULTI3D output | reported; the μ = 0.8 agreement compared with the μ = 1 agreement |
| V6 | Native log τ: the FFNO input maps (mesh features after the shear) at z = 0, 300, 1200 km against the Bifrost cube at the same heights, for a slanted τ_LOS atmosphere built from that cube; with and without the shear | maps shown |
| V7 | Gradients: dot-product and finite-difference checks of the full forward + observation chain at μ = 0.8 for temperature and `vlos`, both native coordinates | ≤ 1e-4 relative |
| V8 | MPI: 1 vs 2 vs 4 ranks, inclined with shifts crossing several ranks | equal to rounding |

### Results (2 Nov 2026)

| # | Result |
|---|---|
| V1 | Existing suite: 49 testsets pass (baseline 38 + the new ones). On a 6 × 6 crop of the production forward atmosphere (Muspel H-alpha and Ca II 8542 with the stored FFNO populations, Kurucz), the spectra and the synthesis VJP (T, vx, vz, χ₅₀₀, H and Ca populations) are **identical** to the pre-change code. |
| V2 | Doppler sign and projection for vx, vy, vz at φ = 0, 45, 90, 180, 270°: the opacity equals that of a vertical atmosphere moving at n·v (1e-12); adjoint through the gradient tests below. |
| V3 | Integer shifts are exact periodic translations; fractional shifts equal HSEMHS `pbilin` to 1e-14; transpose dot test 1e-13; the opposite shift is not the inverse (conserves level sums). Gauge covariance: raising the heights by Δ moves the footprints by t·Δ (test). Smoothing of shift-and-back on the MULTI3D H populations at 48 km (μ = 0.8): median |n′/n − 1| 0.1–2 % (level 3: 1.4 % at 300 km, 1.9 % at 1000 km), `response_run/view_check/v5a_ray_tracer.jl`. |
| V4 | Linear source along the ray: I = S₀(1 − e^−τ) + S₁(1 − e^−τ(1 + τ)) + S_b e^−τ with τ = χL/μ, to 1e-10 for μ = 1, 0.8, 0.45 in both geometries; a horizontally uniform native-z atmosphere gives the same spectrum inclined and plane-parallel (1e-12). |
| V5a | MULTI3D H populations on the Bifrost cube (every 4th column), μ = 0.8, φ = 180°, 400 random footprints: the library's footprint synthesis equals an independent ray tracer (state sampled at the true ray points, trilinear) at the grid levels exactly (0.0000); against the tracer at 4× finer vertical sampling it is off by 0.6 % rms (core 0.3 %, max 1.7 %). Shifting the opacities instead of the state is slightly worse (0.9 % rms, max 7.5 %); state vs opacity shift differ by 0.5 %. |
| V5b | **Done** (eagle7, FFNO H on the Bifrost cube every 4th column, native z, the simulation's thermodynamics; `response_run/view_check/forward_bifrost_view.jl`, `figures/viewing_angle/ffno_vs_multi3d_mu08.png`). H-alpha line-core map correlation (|Δλ| ≤ 0.03 nm) with MULTI3D: FFNO μ = 1 vs MULTI3D μ = 1 **0.82**; FFNO μ = 0.8 inclined vs MULTI3D μ = 0.8 **0.81**; plane-parallel vs MULTI3D μ = 0.8 0.04; cross pairs (μ = 1 vs 0.8) 0.07–0.13. Relative rms 0.08 (μ = 1) and 0.09 (inclined). The FFNO synthesis grid spans only about ±0.1 nm, so the wing statistics are partly clamped. Reference geometry: `output_ray_mu08.hdf5` is **MULTI3D** (rev 99acbd9, the same code, format and 467 levels as the vertical `Multi-3D-H_6_level_populations/output_ray.hdf5`) run on a sheared copy of snapshot 385 (`cb24bih_385_mu08.hdf5`): the matched μ = 0.8 counterpart of the vertical reference. It has `I[a, b]` = footprint (x = a, y = 503 − b) like the vertical MULTI3D output (continuum vs T at 80 km: r = 0.78 only with that flip), and its rays tilt toward −x: **φ = 180°** (H-alpha wing at Δλ = +0.04 … +0.11 nm vs T at 200 km: r = 0.48–0.51 at φ = 180° against 0.33–0.40 for 0, 90, 270°). Its `x` dataset (28.6 km spacing) does not match the 47.6 km grid it is registered to. |
| V6 | μ = 0.8, φ = 0°, Bifrost cube → τ_LOS atmosphere → HSEMHS column balance (3 s for 504²) → population mesh: correlation of the FFNO input with the Bifrost cube at the same height, with / without the shift: T 0.84/0.84 (0 km, no shift there), **0.96/0.68** (300 km), **0.55/0.02** (1200 km); v_z 0.98/0.98, 0.96/0.72, 0.79/0.29. ρ correlates weakly either way (HSE densities, not the simulation's). Figures `figures/viewing_angle/ffno_input_maps_*.png`. |
| V7 | Objective gradient (adjoint) against finite differences at μ = 0.8, φ = 30°, temperature and `vlos` controls: native log τ 4.4e-10 (both geometries), native z 2.6e-9 (both; inclined through the footprint state). Inclined population mesh: VJP and JVP against central differences 1e-5 / 1e-4. |
| V8 | Native z inclined (μ = 0.6, φ = 25°, NX = 11 on x slabs): 1, 2, 3 ranks give identical spectra, HSEMHS τ_LOS and pressure, and gradients equal to 2e-16; adjoint vs FD 1.8e-9 on each. |
| P | Production route (`run_inversion_files!`, eagle7: tagged τ_LOS input from the Bifrost cube every 4th footprint, `[view]`, HSEMHS column balance with the view, FFNO on the FSDP service through the inclined population mesh): H-alpha core vs MULTI3D μ = 0.8 r = 0.65 (rms 13 %); the same route at μ = 1 vs MULTI3D μ = 1 r = 0.67 (rms 13 %). The inclined view loses nothing; the gap to the native-z runs (0.81) is the log-τ/HSE route's own (HSE thermodynamics on 0.1 dex). `response_run/view_check/{build_slanted_input.jl,forward_view.toml,model_factory.jl}`. |

## 10. Plan

| Phase | Work |
|---|---|
| P0 | HSEMHS pin to `main` 5c8444e; existing tests (V1 baseline) |
| P1 | `ViewGeometry` in `Grid3D`; `[view]` config and metadata; `ViewOptions` to HSEMHS; `los_velocity` in every Doppler path and adjoint; `vlos` control (§4); native-z path z/μ; PSF stretch; V1, V2, V4 |
| P2 | Shear and transpose; inclined native-z footprint synthesis and its adjoint; inclined native-log-τ populations (predict, VJP, JVP); V3, V5a, V7, V8 |
| P3 | V6 maps; V5b against the RH15D μ = 0.8 rays; README and an ADR |

Nothing is committed without the user's approval.

## 11. Decisions

| Date | Decision | By |
|---|---|---|
| 2 Nov 2026 | HSEMHS `main` on the remote is the branch to pin and work against | User |
| 2 Nov 2026 | Both geometries, `plane_parallel` and `inclined`; velocity as §4; PSF as §7; validate against the available μ = 0.8 output | User |
| 2 Nov 2026 | Keep the population shear for inclined native log τ (§5); show the FFNO input maps at z = 0, 300, 1200 km (V6) | User |
| 2 Nov 2026 | Codex review accepted and folded in; implement and test | User |
| 2 Nov 2026 | Shear on the root rank on global arrays, FFNOInversion's own implementation with a transpose (§5, §6); state, not opacities, sheared for native-z synthesis | Design |
