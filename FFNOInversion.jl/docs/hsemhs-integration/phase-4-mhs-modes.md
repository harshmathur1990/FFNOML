# Phase 4: MHS Poisson modes, convergence status, trial rejection

Repositories: FFNOInversion.jl, plus HSEMHS fixes on `embedding-api` if needed.
Brief sections: §2 (mode options), §4 (status), §7 (z/ltau boundary treatment), §9 (failure handling), §10 step 4.
Depends on: Phase 3.

## Goal

Enable `model = "mhs"`, giving `mhs-poisson-z` and `mhs-poisson-ltau` through the same adapter, with:

- per-evaluation convergence status,
- a configured, collective policy for unconverged or invalid reconstructions,
- exact restoration of the reference state after a rejected trial.

At the end of this phase all four modes run forward end to end. The inversion loop is wired to reject trials, but gradients are validated in Phase 5.

## Scope

1. **Configuration** (`src/IO.jl`)

   ```toml
   [equilibrium.poisson_z]       # mhs-poisson-z
   z_cut_m = 1.0e6
   max_steps = 30
   rtol = 1e-4
   n_consec = 3
   anchor_w = 1.0
   cg_tol = 1e-3
   cg_maxit = 80

   [equilibrium.poisson_ltau]    # mhs-poisson-ltau
   logtau_cut = -4.0
   max_steps = 60
   rtol = 1e-4
   n_consec = 3
   cg_tol = 1e-3
   cg_maxit = 80
   dc_cap_m = 3.0e5
   lm_mu0 = 0.0
   ```

   - The MHS modes also use the matching column section (`column_z` / `column_ltau`), because both Poisson drivers start from column HSE (`run_mhs_poisson_z` calls `column_hse` first; `run_mhs_poisson_ltau` calls `column_hse_tau` first).
   - `z_cut_m` and `logtau_cut` must lie inside the native range **and** leave at least each kernel's minimum number of retained levels (Phase 1 option validation). Checked at config/prepare time on every rank, never inside the solver (Phase 1 moved the `z_cut` error out of `solve_pressure`).
   - **Pressure anchoring for `mhs-poisson-z`: describe what the solver actually does, not a constraint it does not enforce.** At `b2401a3`, `anchor_w`/`Aanc` and `ereg` appear only in the increment operator `Aop!`, not in Φ or its right-hand side (Phase 1 gate report). They damp changes of the anchor-cell mean ln P per Gauss-Newton step. They do not hold it fixed. This phase documents:
     - the measured drift of the anchor-cell mean ln P relative to its column-HSE starting value (which itself comes from the column-HSE `anchor`/`P_ref`) on the test atmospheres and the example atmosphere,
     - how that drift depends on `anchor_w` and on the step count.
   - **Decision (user, 24 October 2026): HSEMHS is the authority.** The soft, damped anchor is the intended behaviour. FFNO does not add an exact mean-pressure constraint and does not ask HSEMHS for one. The drift measurement above is documentation, not a pass/fail limit. Options HSEMHS does not support (custom anchor-cell masks via config) are rejected.
   - B is required for MHS configurations. A config with `model = "mhs"` and no `Bx/By/Bz` datasets is rejected. Explicit zero B selects column HSE as specified below.
   - **User-required zero-B dispatch:** after applying controls, validate that all B components are finite, then collectively determine whether every owned value of Bx, By and Bz is exactly zero. For `model = "mhs"`, globally zero B selects `column-hse-z` or `column-hse-ltau` according to the native coordinate; otherwise select the matching Poisson mode. No implicit small-field threshold is introduced. A rank or column with zero B must still participate in the global MHS solve if B is nonzero anywhere in the domain. Prepare/cache both required workspaces and repeat the decision for every forward and gradient trial, because B controls can change it.
   - Record `requested_model`, `effective_mode`, and `selection_reason = :zero_magnetic_field` where applicable in status, diagnostics, and output. This branch is a model-selection requirement, not a claim that a general 3D zero-B Poisson solution equals independent column HSE.

2. **Adapter** (`src/Equilibrium.jl`)
   - `prepare_equilibrium` for the MHS modes caches the HSEMHS Poisson workspace (FFT plan, preconditioner buffers).
   - `reconstruct_equilibrium!` passes T and fixed B (the B controls from the inversion are applied before the call) and returns P, ρ, nₑ, z, logτ.
   - The status carries `steps`, `phi0`, `phi`, a safe `phi/phi0` (Phase 1 `phi0_zero` handling), `reason`, the stationarity norm, last linear-solve residual, final LM μ (ltau), and the column-HSE stage status.
   - **Three separate flags, never merged:**
     - `converged`: solver reason is `:converged` **and** the stationarity norm is below `stationarity_tol` (config, documented default from the Phase 1 fixtures). A run that stopped on small relative decreases but has a large stationarity norm is *not* converged.
     - `valid_fields`: EOS clean and every Phase 1 result-validity check passed (finite, positive, monotonic z/τ, above-cut continuation succeeded).
     - `accepted_approximate`: not converged, but `valid_fields` and `reason == :max_steps` and `phi/phi0 <= accept_phi_ratio`. `accept_phi_ratio` is `nothing` (disabled) by default. If set, it must satisfy `0 < accept_phi_ratio < 1`, which is validated at config time. An accepted approximate atmosphere keeps its actual stopping reason in status and output.
   - **Validity for the objective:** `valid_fields && (converged || accepted_approximate)`. Stated explicitly so "the solver exited" is never treated as convergence (brief §4.5).
   - **Implicit-derivative eligibility** (`implicit_derivative_eligible`) is a separate, stricter flag computed here and consumed in Phase 6: `converged` with stationarity norm below `implicit_stationarity_tol` (≤ `stationarity_tol`). An `accepted_approximate` atmosphere is never eligible for an implicit derivative.
   - B is preserved exactly. Assert bitwise equality of B before and after the call in debug mode.

3. **Failure policy and trial handling** (`src/Solvers.jl`, `src/ScalableSolvers.jl`, `src/DistributedForward.jl`)
   - `forward!` keeps the Phase 3 contract: on invalid equilibrium it short-circuits populations, synthesis and observation and returns the invalid status. The objective/solver owns reject vs abort.
   - `evaluate_objective!` maps an invalid equilibrium to:
     - `on_nonconvergence = "reject_trial"`: objective = `Inf` with `components.equilibrium_status` set, so the L-BFGS line search (`lbfgs_invert!`) and the prototype search treat it as a failed trial and backtrack,
     - `"abort"`: a collective `ArgumentError` naming the reason.
   - **Initial atmosphere:** an invalid equilibrium at the initial point is always a collective error, regardless of policy (brief §9).
   - The decision is collective. `reconstruct_equilibrium!` already returns globally reduced status (Phase 1). Add an `allreduce` assertion that all ranks agree on validity before branching.
   - **Restoration:** after a rejected trial, `_restore_trial_reference!` restores `pgas, rho, ne, z, logtau500` and the control fields. The next evaluation starts from the restored state. Because initialization is deterministic (column HSE then Poisson from scratch every call, with no warm start in this phase), the result of an evaluation depends only on its controls. Test this explicitly.
   - Never switch nonzero-field MHS to HSE on failure. Zero-B selection occurs before solving and is not a failure fallback. Add tests for both rules.
   - **Failed trials never replace accepted state.** A rejected or invalid trial must not overwrite: the last accepted prepared atmosphere and populations (`prepared_equilibrium`), the state written to output files, or the state written to checkpoints. Checkpoints are only written from accepted iterates.

4. **Diagnostics.** Each evaluation logs on root one line per call: mode, steps, phi/phi0, reason, EOS violations, seconds. This goes through the existing `diagnostic_event!` mechanism, not `println`. Output files record the final equilibrium status (Phase 5 extends the schema).

## Out of scope

- Solving for B (`full-mhs-z`).
- Warm starts (Phase 6).
- Gradient validation and restart (Phase 5).

## Tests

| Test | What it proves | Pass criterion |
|---|---|---|
| Adapter vs direct HSEMHS, ×2 MHS modes | Packing, options translation | Non-square, axis-varying synthetic atmosphere with non-trivial B (a flux tube with non-zero J×B). Adapter output equals direct `solve_mhs_poisson_*!` on hand-packed arrays to ≤ 1e-13. Steps and Φ identical. |
| Adapter vs standalone file run, ×2 | Brief §10 | The standalone `mhs-poisson-z` / `-ltau` driver on the same atmosphere matches to ≤ 1e-10 (fields) and gives identical step count and reason. |
| Zero-B selects column HSE | User requirement | With `model = "mhs"` and globally zero B, the Poisson solver is not called, effective mode is matching column HSE, and results equal an explicit column-HSE run for horizontally uniform and horizontally varying T. Test both native coordinates and multiple rank counts. |
| Mixed local zero/nonzero B | Collective selection | B is nonzero on only one rank: every rank selects and enters MHS. Repeat with a B control changing the domain from globally zero to nonzero and back. |
| Manufactured magnetized case | Independently checked physics | A horizontally uniform, height-varying Bx(z) on an isothermal atmosphere. The Lorentz force is purely vertical (−d(B²/2μ₀)/dz), so P + B²/2μ₀ follows hydrostatic balance. Compare P(z) with the 1D analytic or ODE solution to ≤ 1e-4 relative below z_cut. A second case: a nonzero current-free field (J×B = 0) on a **horizontally uniform** atmosphere, where column HSE is also a full 3D equilibrium, so the result equals column HSE. No equality is claimed for horizontally varying atmospheres, where zero Lorentz force does not make independent columns a 3D equilibrium. |
| Stationarity gate | Converged ≠ "stopped" | A case that stops on `n_consec` small decreases with a loose `cg_tol`: `converged = false` if the stationarity norm exceeds `stationarity_tol`, and the run is invalid unless `accepted_approximate` applies. With tight `cg_tol` the same case converges. |
| `accept_phi_ratio` | Range and flags | Values ≤ 0, ≥ 1 rejected at config. With a valid value, a max-steps run below the ratio is `accepted_approximate` with reason `:max_steps` recorded in status and output, and is never implicit-derivative eligible. Above the ratio it is invalid. |
| Anchor drift | Documented semantics | The anchor-cell mean ln P drift is reported for the manufactured and example cases at two `anchor_w` values. Pass = reported and consistent with the documented damping description. |
| Rank independence | MHS reductions/FFT | Each MHS mode on 1, 2, 3 ranks (NX=11): gathered P agrees to ≤ 1e-8 relative, step count identical or differs by ≤ 1 with Φ within rtol. Not bitwise. |
| Thread independence | Threaded kernels | 1 vs 4 Julia threads per rank: same agreement as above. |
| Unconverged → rejection | Policy | Set `max_steps = 1` on a case that needs more: `reject_trial` gives objective `Inf` on all ranks, and `abort` gives the same collective error on all ranks. |
| Initial invalid → error | Brief §9 | An initial atmosphere out of envelope under `reject_trial` still raises a collective error. |
| Restoration determinism | No state leakage | Evaluate at x₀, then an invalid trial x₁, then x₀ again. The second x₀ evaluation equals the first bitwise (deterministic init). Also: x₀ → valid x₂ → x₀ is bitwise equal. |
| Line search with rejection | Integration with L-BFGS | A mock forward model whose equilibrium is invalid for one coordinate above a threshold. `lbfgs_invert!` backtracks below the threshold and continues. All ranks take identical iteration paths (compare iteration records). |
| Line-search exhaustion | Accepted state preserved | Every trial in a line search is invalid until `maximum_line_search_trials` is hit: the solver terminates (or follows its documented policy), and prepared atmosphere/populations, output state and the last checkpoint are bitwise those of the last accepted iterate. |
| No failure-driven HSE fallback | Brief §9 | For globally nonzero B and `model = "mhs"`, the effective mode remains `mhs-poisson-*`, including after rejected trials. Globally zero B instead records the required column-HSE selection. |
| B preservation | Fixed-B contract | B arrays are bitwise unchanged by `reconstruct_equilibrium!`. |
| Repeated calls, no deadlock | Lifecycle | 100 forward calls, 3 ranks, alternating converged/unconverged/out-of-envelope. Statuses agree across ranks. No hang. |
| eagle[7-8] forward, all 4 modes | Production stack | A native-z and a native-logtau version of the example atmosphere (with B), forward in all four modes with the FSDP service on ≥ 4 ranks. Finite spectra, statuses and timings recorded. |

## Exit criteria

- All tests pass. eagle[7-8] runs complete for all four modes, with job IDs in the report.
- `docs/reports/hsemhs-phase-4-validation.md`: manufactured-case errors, per-mode step counts and timings (equilibrium vs rest of forward) on the reference grid at 1 and 4+ ranks, and the documented anchor/z_cut/logtau_cut semantics.
- ADR-012 updated with the validity rule and failure policy.

## Points for the reviewer

- `stationarity_tol` and `implicit_stationarity_tol` defaults come from the Phase 1 fixtures and must be reviewed against real-data statistics from the eagle[7-8] runs.
- `accept_phi_ratio` defaulting to "max_steps is invalid" is strict. It may cause many rejected trials if HSEMHS commonly hits max_steps on real data. The eagle[7-8] statistics in the report should show how often it happens before the default is relaxed.
- The ltau alignment (`align_ptot!`) and gauge (`gauge!`) are global reductions inside the equilibrium. They make every column's z depend on all columns. That is correct physics for these modes, but it matters for Phase 5's finite-difference noise and for Phase 6's derivative.
