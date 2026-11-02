# ADR-013: Viewing angle

Status: accepted, 2 November 2026. Design, review and validation: `docs/viewing-angle/README.md` (Codex review
`docs/viewing-angle/codex-review-2026-11-02.md`). HSEMHS pinned to `main` 5c8444e.

- **One view per run, in the grid.** `[view] mu, azimuth_deg, geometry` (`plane_parallel` | `inclined`) becomes
  `Grid3D.view::ViewGeometry`, so every tile, gathered atmosphere and output carries it. Absent or μ = 1 is the vertical
  view; every code path is then the pre-view one (spectra and adjoints bit-identical, checked on production data).
- **HSEMHS owns the force balance.** The view is passed as `ViewOptions` to `prepare_embedded`. A native log-τ input at
  μ < 1 must hold τ along the line of sight on a deprojected footprint grid and be tagged `view_*`; untagged files are
  disk-centre files, accepted only at μ = 1. Native-z inputs are Cartesian and need no tag.
- **Doppler.** Every opacity path reads `los_velocity` = n·v (vz at μ = 1); adjoints add n·v̄. The `vlos` control is
  v = v_prior + δ n, rebuilt from the immutable starting velocities on every trial, exclusive with `vx`/`vy`/`vz`
  controls; regularisation of `vlos` acts on n·v; checkpoints record the view off the vertical.
- **Formal solution.** Native log τ: unchanged (τ_LOS axis, local χ₅₀₀). Native z: path dz/μ in the geometric solver;
  `inclined` synthesizes a footprint atmosphere (state and populations shifted by +t·z per level, log for positive
  fields, gathered and shifted on the root rank) and the Cartesian columns are refused as rays. This is long
  characteristics for the emergent intensity: the ray sampled at its crossing with every level plane.
- **Populations.** The FFNO always sees a Cartesian cube. Native log τ `inclined`: the population mesh shifts its
  features by −t·z_m before the network and ln n by +t·z_m after it (predict, VJP, JVP). Plane-parallel feeds the ray
  columns as if vertical (the 1.5-D shortcut; it fails against MULTI3D at μ = 0.8, see the validation).
- **Ray origin.** x = ξ + t (z − z_ref), z_ref = 0: HSEMHS's gauge for native log τ, the grid's z = 0 for native z.
  Periodic wrap; a warning when the rays travel farther than the box.
- **Shift implementation.** FFNOInversion's own periodic bilinear `ShearPlan`/`shear_levels!` (HSEMHS's convention,
  equal to its `pbilin` to 1e-14) with an exact transpose `shear_levels_transpose!`; it runs on global arrays on the
  root rank, so it needs no second MPI context.
- **Observations.** Footprint grid supplied by the caller; the sky PSF is stretched by 1/μ along the tilt; only
  φ ∈ {0, 90, 180, 270}° with a spatial PSF.
