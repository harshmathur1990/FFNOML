# Equilibrium adjoint in FFNOInversion (Phase 7 of the HSEMHS plan)

Branch `equilibrium-adjoint` (FFNOML), started 24 Nov 2026. **Implemented 24 Nov 2026** (status at the end). The HSEMHS side is planned in HSEMHS
`design/adjoint/` (same branch name); this file covers what changes here once `HSEMHS.adjoint_embedded!` exists.

## Today

`objective_gradient!(::HybridAdjointObjectiveGradient, …)` (`src/ProductionGradients.jl`) assembles the gradient with
exact transposes for the PSF, formal solution, line opacities, population mesh and FFNO, ending with an
`AtmosphereCotangent` (temperature, vx, vy, vz, vturb, pgas, rho, ne, z, chi500, magnetic_field). The direct parts (T, v,
vturb, B) go to the control maps (`accumulate_control_vjp!`). The derived parts (pgas, rho, ne, z, chi500) and the
regularization go through `_force_regularization_vjp!`: for every control coordinate, two equilibrium reconstructions
and two regularization evaluations (central differences of ⟨bar, derived⟩ + regularization). It loops over all
coordinates, also those of variables the equilibrium does not depend on (vx, vy, vz, vturb), and differences the
regularization, which is an explicit function of the fields.

## Change

1. **Regularization gradient, analytic.** `Regularization.jl` penalties (vertical types 0–4 per slot, horizontal order 1
   or 2, scales) are quadratic forms in the fields; write `regularization_vjp!(bar, atmosphere, spec, dx, dy, context)`
   adding ∂penalty/∂field to `bar` (temperature, vlos/vz, vturb, B, and pgas for the `pgas_boundary` slot). Validated
   against central differences of `distributed_regularization_penalty`.
2. **Equilibrium VJP.** After the synthesis, population and regularization pull-backs, map `bar.pgas, rho, ne, z,
   chi500` from (z, x, y) to HSEMHS's (z, y, x) slab order (`_zxy_to_zyx`), call `HSEMHS.adjoint_embedded!` on the
   backend's workspace, and add `adj.lnT ./ T` to `bar.temperature` and `adj.B*` to `bar.magnetic_field` (when B is a
   control or present). Then the direct accumulation as now. The solve must be recorded: `reconstruct_equilibrium!` passes
   `record = true` when the gradient backend is the adjoint one, and the gradient is taken at the state of the last
   evaluation (`evaluate_objective!` already reconstructs at `parameters`; the adjoint uses that record).
3. **Remove the coordinate loop** from the production path. Keep the finite-difference equilibrium contribution as the
   validation oracle: `[solver] equilibrium_gradient = "adjoint"` (default) or `"finite_difference"` (today's behaviour,
   for checks).
4. **Given atmospheres** (forward only) are unaffected; the gradient path is never reached there.
5. **Logs and provenance.** Record the adjoint status (`adj.status`, and for Poisson modes the stationarity numbers) in the
   gradient stage log and in the checkpoint provenance.

## Validation

- `gradient_taylor_validation` and `dot_product_validation` (existing) with the adjoint backend against the
  finite-difference one on small cubes (hyades, CPU mock populations as in the e2e directory), for column-hse-ltau,
  -logm, -z, with and without B controls; 1 and 3 ranks.
- The same comparison inside one Olivia job on the 252² inversion setup, a few coordinates only (cost).
- Wall time per gradient: equilibrium part before and after.

## Docs to update

README "Known limitations" (the finite-difference equilibrium derivative), "Gradients", `examples/inversion_run`
comments, START_HERE, the HSEMHS pin in `Project.toml`.

## Status (24 Nov 2026)

Implemented as planned, with two differences: the configuration switch is the gradient backend's keyword
(`HybridAdjointObjectiveGradient(equilibrium = :adjoint | :finite_difference)`) rather than a TOML key, and backends
without an adjoint (`supports_equilibrium_adjoint` false: a given atmosphere, test stand-ins) fall back to the
finite-difference route automatically. `set_equilibrium_record!` makes the HSEMHS backend record its reconstructions; the
gradient turns it on. Tests: `test/equilibrium_adjoint.jl` (regularization gradient, equilibrium VJP for the three
column modes with and without B, objective gradient on τ and height grids against the finite-difference equilibrium
2e-10 / 1e-11 and the full oracle 3e-10), `test/mpi_equilibrium_worker.jl` + `mpi_equilibrium_compare.jl` (1, 2, 3 ranks
equal to 1e-10). HSEMHS and WittBifrostEOS come from local path sources until their `equilibrium-adjoint` branches are
pushed and pinned.
