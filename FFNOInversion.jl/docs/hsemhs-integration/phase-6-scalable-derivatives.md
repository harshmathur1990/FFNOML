# Phase 6: Scalable equilibrium derivatives and benchmarks

Repositories: HSEMHS (derivative operators on `embedding-api` or a follow-up branch), FFNOInversion.jl (gradient wiring).
Brief sections: §8 (JVP/VJP, warm starts, nonsmooth points), §10 step 6.
Depends on: Phase 5 (FD reference gradients and cost table).

## Goal

Replace the finite-difference equilibrium contribution with an implicit-function vector-Jacobian product (VJP). Its cost is a small number of linear solves per gradient, independent of the number of control coordinates. Validate it against the Phase 5 FD reference in all four modes. Optionally add warm starts with committed/trial workspaces. Only after this phase may production-scale performance be claimed.

## Scope

1. **Derivation document first** (`HSEMHS/reports/equilibrium_vjp_derivation.md`, reviewed by Codex before code). For each mode, write the defining problem and its derivative:
   - **Eligibility gate.** An implicit derivative is the derivative of a *stationary* solution, not of a finitely terminated solver. It is used only when the Phase 4 `implicit_derivative_eligible` flag is set (converged, stationarity norm below `implicit_stationarity_tol`). Otherwise the backend (a) tightens tolerances and re-solves once, (b) falls back to the validated FD reference for that gradient, or (c) reports an unsupported derivative, by configured policy. Every gradient records which path was used, plus adjoint linear-solve residual diagnostics. Acceptance of an approximate atmosphere for the objective and eligibility for an implicit gradient are reported separately.
   - **Anchor and damping must be derived, not reused.** `PoissonZ.Aop!` is the *damped increment* operator (`JᵀWJ + ereg + Aanc·anchor`), and the anchor is not part of Φ (Phase 1/4 documentation). Reusing it blindly as the adjoint system differentiates a different equation. The derivation states the actual stationary equation, which is ∇ᵤΦ = 0 with no anchor term. HSEMHS is the authority on the anchor (user decision, 24 October 2026), so no exact gauge constraint exists to differentiate. It then derives the adjoint operator from that equation and states how the null space or near-null space without the anchor (the constant-ln-P direction, if Φ is insensitive to it) is handled. For `PoissonTau`, the final LM μ is inspected and reported. The derivation explains whether μ affects only the step path or also the defining equation, and does not assume μ = 0 without checking.
   - First account for zero-B dispatch: the derivative backend follows the effective solver. An implicit derivative within a branch does not establish differentiability across the exact-zero HSE/nonzero-B MHS selection boundary. Detect branch crossings and document supported behavior; no unapproved smoothing or small-field threshold.

   - **column-hse-z:** per column, the ODE `du/dz = −g exp(Lρ(X,u) − u)` with anchor condition. Discrete map: RK4 with adaptive substeps (`ColumnHSE.jl`). Differentiate the discrete scheme as executed (discrete adjoint) at fixed substep counts. State that substep-count changes are treated as a nonsmooth point.
   - **column-hse-ltau:** `hse_tau!` (Heun substeps, `cap_per_dex` cap), followed by `align_ptot!` (global reduction setting a per-column shift so P + B²/2μ₀ matches at one height) and `gauge!` (global mean z(τ=1) = 0). The cap is a nonsmooth `min`. Document the active-set treatment (derivative of the active branch). The align/gauge steps are global but low-rank, and their adjoint is a reduction.
   - **mhs-poisson-z:** `solve_pressure` minimizes Φ(u; T, B) = Σ w H̄² |R|² below z_cut, then integrates hydrostatically above. The defining condition is stationarity ∇ᵤΦ = 0, **not** R = 0 (least squares, R ≠ 0 at the solution). The implicit derivative needs the Hessian of Φ in u: Gauss-Newton JᵀWJ plus residual-curvature terms Σ Rᵢ ∇²Rᵢ. State whether the second-order terms are dropped (Gauss-Newton approximation) and validate that choice numerically. H̄ and the weights are frozen from the start state. Document whether their dependence on T is differentiated or treated as fixed, and the consequence.
   - **mhs-poisson-ltau:** same structure, with unknowns (ln P below k_c, column height c), Levenberg-Marquardt damping (see the eligibility and damping notes above: inspect the final μ, do not assume it is zero), and the `dc_cap`/`du_cap` step scaling (affects path, not the stationary point).
   - **Hessian definiteness.** If residual-curvature terms Σ Rᵢ ∇²Rᵢ are included, the system matrix is not guaranteed SPD, so plain PCG may be invalid. Specify detection (CG curvature check pᵀAp ≤ 0, or a Lanczos estimate) and the response: a symmetric-indefinite solver (MINRES), or the Gauss-Newton matrix as a documented approximation, validated against FD. A singular or near-singular system (see the null-space note) gives an explicit status, not a silent result.

   For each mode, list the inputs whose sensitivity is propagated:
   - T (via ln T), and B (via L = J×B, and in ltau via the τ-surface geometry),
   - the EOS (`state`/`lnrho_dY` derivatives): the table uses **Hermite interpolation** (`hermite` in `eos/Table.jl`), so differentiate the actual Hermite interpolant, including the cell selection, the `BREAK_EPS` node pairs at breakpoints and the direct/blend/hot branch logic, and treat branch switches as nonsmooth points,
   - **order statistics in the weights:** `gmedians`/`gquantiles` (H̄ layer medians, `dzf`, the preconditioner medians) are order-statistic maps, not smooth reductions. If they are differentiated, specify routing to the selected sample(s), even-count averaging, ties, and a rank-independent adjoint. If they are frozen, label it (see below). Test median-order changes and changes of the `align_ptot!` reference column/height, not only RK and cap transitions,
   - z/τ mapping, gauge, alignment,
   - and how the output cotangents (P̄, ρ̄, n̄ₑ, z̄, log τ̄) flow back.

2. **HSEMHS operators** (new `src/EmbeddedAdjoint.jl`)
   - Per mode, `equilibrium_vjp!(ws, result, cot::(Pbar, rhobar, nebar, zbar, logtaubar)) -> (lnTbar, Bxbar, Bybar, Bzbar)`, plus `equilibrium_jvp!` for testing.
   - Reuse `PoissonZ.JtW!/Aop!/pcg!` and `PoissonTau.Jop!/JTop!/Aop!/Minv!`, but the brief warns these are not the full atmosphere-to-equilibrium derivative. The missing pieces must be added: ∂R/∂(ln T), ∂R/∂B, EOS chain rule, the above-cut hydrostatic integration, and the τ integration for the output fields.
   - The adjoint linear solve uses the same preconditioner. Its tolerance is a separate option (`adjoint_cg_tol`).
   - All collectives follow Phase 1 rules (injected communicator, main thread only).

3. **FFNO wiring** (`src/ProductionGradients.jl`)
   - New `AbstractEquilibriumDerivative`, with `FiniteDifferenceEquilibriumDerivative` (Phase 5 path, retained as the validation reference) and `ImplicitEquilibriumDerivative`. Selection via `[equilibrium] derivative = "implicit" | "finite_difference"`, default `"finite_difference"` until the validation in this phase passes for that mode. The report then states which modes default to implicit.
   - The implicit path: the cotangent of derived fields (`AtmosphereCotangent` pgas/rho/ne/z/logtau500), then the adapter pack, then `equilibrium_vjp!`, then the unpack into temperature/B cotangents, then add to the direct bars before `accumulate_control_vjp!`.
   - The regularization contribution, currently folded into the FD functional, gets its own analytic VJP. Regularization of **derived** fields (pgas, rho, ne, z) is not local or cheap to finite-difference, because those fields depend on the global equilibrium. Differentiate the regularizer with respect to its input fields, add those bars to the derived-field cotangent, and pass the sum through the single equilibrium VJP. If any FD remains, count its extra global solves in the benchmark.

4. **Warm starts (optional, only if Phase 5 timings show equilibrium solve count is the bottleneck after the VJP)**
   - Workspaces split into `committed` (state at the last accepted iterate) and `trial`. A trial solve starts from `committed`. Acceptance copies trial to committed. Rejection discards trial.
   - Checkpoints store the committed state so restart reproduces the warm-start policy. Otherwise restart re-initializes deterministically and the docs say results then differ within solver tolerance.
   - The FD reference path always uses deterministic initialization.

5. **Benchmarks.** `scripts/benchmark_hsemhs.jl` on eagle[7-8] measures, per mode and for 1/4/8/16 ranks (x slabs, within the allowed minimum columns per rank):
   - forward (stage split), FD gradient, implicit gradient,
   - memory high-water mark per rank (`distributed_memory_report` extended with HSEMHS workspace bytes),
   - strong scaling of the equilibrium solve alone.

   Decide from the numbers whether tile↔slab redistribution (brief §6) is needed. That work is not done here. If it is needed, the report recommends it as a follow-up phase.

## Out of scope

- Tile↔slab redistribution and 2D-tile Poisson (follow-up only if benchmarks justify it).
- `full-mhs-z`.

## Tests

| Test | What it proves | Pass criterion |
|---|---|---|
| JVP/VJP dot product, ×4 modes (HSEMHS level) | Adjoint consistency | ⟨J v, w⟩ = ⟨v, Jᵀ w⟩ to ≤ 1e-8 relative for random v (ln T, B) and w (P, ρ, nₑ, z, logτ) on 1 and 3 ranks. |
| JVP vs FD of the equilibrium map, ×4 | Derivative of the actual solver | Centered FD of `solve_*!` outputs along v (with tight solver tolerances) agrees with the JVP to ≤ 1e-5 (column modes) / ≤ the Phase 5 noise floor × 10 (MHS modes). |
| Gauss-Newton approximation check (MHS) | Brief §8 | Implicit derivative with and without the residual-curvature term, on a case with non-negligible final residual: the report states the difference. If GN-only is used, its error versus FD must be below the Phase 5 directional threshold. |
| Objective gradient: implicit vs FD reference, ×4 | End-to-end | The `objective_gradient!` vectors agree to ≤ 1e-4 relative (L2) for column modes, ≤ the Phase 5 MHS threshold for MHS modes. Coordinate-wise outliers are listed. |
| Taylor test with implicit gradient, ×4 | Gradient correctness | O(h²) remainder over ≥ 3 decades. |
| Nonsmooth points | Documented behavior | A column where `cap_per_dex` activates (ltau), an RK substep change (z), an EOS branch/breakpoint crossing, a layer-median order change, and an `align_ptot!` reference change: the derivative is the active-branch derivative and agrees with one-sided FD on the active side, or the crossing is detected and reported. |
| Eligibility gate | Implicit only at stationarity | An `accepted_approximate` atmosphere and a loosely converged one (stationarity above `implicit_stationarity_tol`) do not use the implicit path. The configured fallback (re-solve, FD, or unsupported) is taken and recorded. |
| Adjoint solve health | SPD and residuals | Adjoint linear-solve residual is reported and below `adjoint_cg_tol`. An injected indefinite system (curvature terms on a strongly non-converged state) is detected and handled by the documented response. |
| Frozen-weight label | Honest derivative | If weights/medians are frozen, the report quantifies the difference between the frozen-weight implicit gradient and FD of the full forward map (which recomputes them), and labels the gradient as approximate if that difference exceeds the directional threshold. |
| Rank independence | Collectives | The implicit gradient on 1, 2, 3 ranks agrees to ≤ 1e-8. |
| Warm start (if implemented) | Committed/trial contract | Rejected trials leave committed state bitwise unchanged. Accepted trials update it. Restart with the saved committed state reproduces a straight run bitwise. |
| eagle[7-8] inversion, implicit gradients, ×4 | Production | 20-iteration inversions in each mode, with objective reduction comparable to the Phase 5 FD-gradient runs over the same iterations (the report shows both curves). Wall time per iteration recorded. |

## Exit criteria

- The derivation document is reviewed before implementation. All tests pass.
- `docs/reports/hsemhs-phase-6-validation.md`:
  - benchmark tables (forward, FD gradient, implicit gradient, memory, scaling),
  - the default derivative per mode, with justification,
  - remaining limits (nonsmooth points, frozen weights, GN approximation),
  - the recommendation on tile↔slab redistribution.
- Final handoff (brief §10, last paragraph): changes in each repository, dependency revisions, runnable commands and example TOMLs, checks executed with results, measured costs, remaining limits.

## Points for the reviewer

- The frozen-weight question (H̄, wc, wf and the anchor setup, computed from the start state every forward call, then held fixed during the solve) is subtle. Because every forward recomputes them from controls, they are part of the actual forward map. Freezing them in the VJP is an approximation, which must be validated and labelled, never called an exact implicit gradient. The solver's stationary point depends on the weights, and the weights depend on T through the column-HSE start. Differentiating as if the weights were constants gives the derivative of a slightly different map. The derivation must state which map is differentiated, and the "JVP vs FD" test must perturb through the same path that the forward uses.
- If the MHS implicit derivative cannot meet the threshold, it is acceptable to ship implicit derivatives for the column modes only and keep FD for MHS, with the cost documented. The report must say so plainly.
