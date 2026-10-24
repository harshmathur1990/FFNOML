# Codex review of the HSEMHS phase plans

Reviewed: README and phases 1–6, against FFNOML `07402ae` and HSEMHS `b2401a3`. This is a review of plans and their source assumptions, not validation of an implementation. No numerical tests were run. The phase ordering and native-z definition are sound; address the findings below before implementing the affected phases.

The README records user decisions made outside this chat about deleting the legacy backend, removing pressure controls, and cluster environments. This review treats those as the plan's stated decisions; it does not independently verify that conversation or cluster setup.

## High priority corrections

### 1. Resolved by user requirement: zero B selects column HSE

**Location:** Phase 4, test “Zero-B reduces to column HSE”; also the current-free-field test.

The user has clarified that globally zero B must select column HSE. Implement this as explicit, collective mode dispatch before solving, for both native coordinates, rather than testing whether a Poisson solver happens to return the column-HSE result. Phase 4 and the README now specify this requirement. Test horizontally varying columns as well as uniform ones and assert that the Poisson solver is not called.

Retain the distinction for nonzero current-free fields: zero Lorentz force does not generally make horizontally varying column HSE a full 3D equilibrium. Restrict that equality test to compatible horizontally uniform atmospheres. The zero-B selection requirement supersedes the original review recommendation for the zero-B test.

**Derivative consequence:** dispatch is repeated after B controls are applied, including FD perturbations. Exact-zero HSE versus arbitrarily small nonzero-B MHS can change the equilibrium map discontinuously for horizontally varying atmospheres. At such a point, do not silently return a smooth implicit or FD derivative across the branch. Detect and report the branch crossing; validate fixed-zero-B temperature inversions and define supported behavior for B inversions initialized at zero. Do not introduce a tolerance or smooth transition without an explicit scientific decision.

### 2. The implicit derivative needs a stationarity gate, not just small objective changes

**Locations:** Phase 4 validity rule; Phase 6 derivation and implicit VJP.

Current `PoissonZ.solve_pressure` and `PoissonTau.solve_tau` stop on consecutive small relative decreases in Phi, or terminate without descent. Small changes can indicate a tiny step or inaccurate linear solve, not stationarity. Phase 4 also allows opt-in acceptance of max-step results. An implicit derivative of a stationary solution is not generally the derivative of such a finitely terminated solver.

Add a globally normalized stationarity norm (the correct constrained/projected gradient where applicable), primal validity checks, and adjoint residual diagnostics. Define a stricter gate for using implicit differentiation. If it fails, tighten/retry the primal solve, use the validated FD reference, or report an unsupported derivative. Report acceptance of an approximate atmosphere separately from eligibility for an implicit gradient. Do not assume the final LM parameter is zero; inspect it and explain whether it affects steps only or the defining equation.

### 3. Derive the actual anchor and damping behavior before claiming a constraint

**Locations:** Phase 4 pressure-anchor paragraph; Phase 6 reuse of `Aop!` as the adjoint system.

Phase 4 says `solve_pressure` “holds the mean ln P” at its initial value. At this source baseline, `PoissonZ.Aop!` adds `ereg * v` and an `Aanc` rank-one term to the increment system, while `phi` contains only force residuals and the RHS is `-JtW!(R)`. There is no corresponding anchor residual in that objective. This is not evidence of an exactly enforced mean-pressure constraint.

Document whether these terms are increment damping, a soft penalty in a different objective, or a true constraint. Test the claimed mean-pressure behavior numerically. Derive the stationary/constrained adjoint operator independently: blindly reusing the damped increment operator differentiates a different equation. If a true gauge constraint is required, include its equation and multiplier/projection explicitly. Preserve standalone numerics unless a solver change is separately declared and validated.

### 4. Preparation and programmer errors can also deadlock MPI

**Locations:** Phase 1 status section and preparation API; Phase 3 EOS loading and hash checks.

The plan exempts shape/layout errors because preparation runs before collectives. A rank-local throw before other ranks enter `Comm_dup`, Cartesian construction, reductions, or FFT setup can still hang the job. Missing EOS files or mismatched hashes on one node are another example.

Specify a two-stage prepare protocol: perform local checks and catch expected validation/IO errors; exchange a compact status on an already valid caller communicator; only then enter collective construction. Use the same protocol between setup stages and in solve paths before their next collective. Include one-rank-only malformed-array, missing-file, and hash-mismatch tests followed by a successful parent-communicator operation. Do not promise recovery from arbitrary process death or failure inside MPI itself.

### 5. Matching partition functions alone does not establish EOS equivalence

**Locations:** README Q1; Phase 3 item 8.

Keep the user's Wittmann choice. However, the claim that consistency is “not a physics comparison” is too strong. HSEMHS table construction supports abundance inputs, direct/blended/hot branches, a switch temperature, fallback flags, and a `fix_gasc_h2` provenance value (`eos/BuildTable.jl`, `eos/Table.jl`). A shared partition-function hash does not establish compatibility of these settings with FFNO's compiled Wittmann backend.

Add a provenance comparison for abundances, library/source identity, relevant fixes, branch policy, and opacity convention. Sample rho, ne and overlapping opacity quantities from the table and direct synthesis backend over the actual example envelope, including the branch transition. Record expected interpolation/branch differences rather than demanding bitwise agreement. Test synthesis failure propagation when its direct EOS rejects a state that the HSEMHS hot branch accepts.

## Required additions by phase

### Phase 1: options, cutoff sizes, status, and ownership

- `ColumnZOptions` omits `P_ref`, although the pressure-anchoring contract requires a resolved value. Put it explicitly in the typed API and option-mapping tests.
- Validate the **number of retained levels**, not only whether a cutoff lies inside the coordinate range. `solve_pressure` uses `h[1]` and `h[end]`; a cutoff selecting only the bottom level fails. `solve_tau` similarly indexes face-spacing vectors. Define each kernel's minimum retained size and test both endpoints and grids with too few vertical levels.
- Handle an already equilibrated start explicitly. If `Phi0 = 0`, `Phi/Phi0` is undefined and strict decrease may be impossible. Add absolute tolerances, a zero-step success policy, and safe diagnostics. Test zero and near-zero initial residuals.
- Reject nonfinite inputs/results, nonpositive pressure/density/opacity, invalid or folded z(tau), and failed above-cut hydrostatic continuation. The latter's EOS/Newton failures need status coverage too, not just the inner Poisson iteration.
- Specify result buffer ownership: does `res` alias workspace arrays and become invalid on the next solve? This matters for restored trials, prepared results, and adjoints. Provide explicit copy/snapshot behavior where needed.
- Correct the claim that `NX div nr >= 2` rejects valid uneven splits: for standard `block_range`, the minimum slab size is exactly `floor(NX/nr)`. Keep collective checks for clearer diagnostics, but do not claim a mathematical bug there.
- Replace the “second allocation < 5% of first” gate with a warmed-up allocation budget versus local volume. Compilation/first-plan creation can make the current test pass even when every solve allocates a new volume. Report workspace and peak-memory bytes too.

### Phase 2: compatibility and coordinates

- Decide how the retained legacy constructor handles existing decreasing-logtau callers after making `VerticalAxis` top-first-only. A constructor cannot reverse the associated atmosphere fields. Normalize at a boundary that has the fields, or reject with an explicit migration path; add a direct-constructor regression.
- Require finite, strictly monotonic x/y and decide whether decreasing horizontal axes are normalized or rejected. Test y as well as x. Preserve origins and validate consistent coordinate metadata across ranks.
- Clarify the difference between the config key naming the native input axis and derived coordinate datasets in an output file. The reader currently forbids a logtau input mapping in z mode; an output round trip must still allow derived logtau to coexist in the file without treating it as a second native axis.
- Replace broad grep requirements with targeted checks for forbidden **native-axis assumptions**. Legitimate logtau accessors and config names necessarily remain.
- Do not use `@test_skip` to hide module-load errors. Keep each phase importable, or declare phases 2–3 an atomic integration before runnable acceptance.

### Phase 3: dependency and runtime lifecycle

- HSEMHS is under the Git repository's `HSEMHS/` subdirectory. The pinned Julia source declaration needs the package subdirectory as well as URL/revision; mirror that in portable setup instructions. Verify a clean checkout can instantiate without a development path.
- If FFNO calls FFTW configuration APIs directly, add FFTW as a **direct** dependency; a transitive dependency is not a supported import contract.
- For managed serial MPI, update finalization to depend on `owns_mpi`, not `context.enabled`. The existing `finalize_parallel!` checks both, which would leak MPI when the plan initializes it with `enabled=false`. Free backend communicators before finalizing, including failure/early-return paths, using structured cleanup.
- Decide behavior for `enabled=false` under a multi-rank launcher. Independent `COMM_SELF` solves can otherwise collide on output files and GPU launch. Reject this combination or explicitly manage independent runs.
- Extend the population-model audit beyond height zero: verify native-z spacing, vertical sample order, domain extent, resolution, and preprocessing against the trained checkpoint. If resampling is needed, include the full mapping and its VJP. Applying a model-only height offset must not mutate the native-z grid or violate `atm.z == broadcast(z_axis)`.

### Phase 4: failure and acceptance consistency

- Phase 3's failure test expects `forward!` to throw, while Phase 4 changes it to return invalid status. State the final contract: invalid equilibrium short-circuits populations, synthesis and observation; the objective/solver owns rejection versus abort.
- Constrain `accept_phi_ratio` to a documented valid range and test it explicitly. Track `converged`, `accepted_approximate`, and `valid_fields` separately; an accepted max-step atmosphere must retain its actual stopping reason.
- Ensure failed trials never replace the last accepted prepared atmosphere/populations, output state, or checkpoint state. Test line-search exhaustion as well as a single rejected trial.

### Phase 5: finite differences and reproducibility

- Do not mix equilibrium maps at different tolerances inside one finite difference. If perturbations use tighter tolerances, recompute the center and cotangent pairing at those tolerances too. Existing `f0` is taken from the current evaluation. Validate that the resulting gradient corresponds to the objective used by line search.
- When perturbations are invalid, first reduce the step with a bounded retry policy, then use a justified one-sided estimate. Test both sides invalid and near-bound controls without corrupting the base state.
- A `RecordedPopulationModel` verifies fixed input features and generally cannot support changing inversion controls. Use a differentiable mock or the real population service for inversion tests; reserve recorded populations for fixed-atmosphere parity tests.
- Store fingerprint components as well as the digest if restart errors must identify which component differs. Include EOS metadata not covered by `table_sha256`, synthesis EOS identity, coordinate normalization, model-height transform, and derivative settings that affect optimizer history. `table_sha256` currently hashes table arrays and flags, not all table metadata.
- Prefer an initial bounded, small-control real-GPU inversion gate before the proposed production-size FD runs. The plan itself projects thousands of Poisson solves per gradient; set a wall-time/control-count budget so Phase 5 does not block Phase 6 on an impractical benchmark.

### Phase 6: additional derivative terms and nonsmoothness

- The EOS table uses Hermite interpolation (`hermite` in `eos/Table.jl`), not generally piecewise-linear cells as the document states. Differentiate the actual interpolation and the branch/breakpoint logic.
- Global medians and quantiles used in weights are order-statistic maps, not merely smooth reductions. Specify selected-sample routing, even-count behavior, ties, and rank-independent adjoints. Test median-order changes and total-pressure alignment reference changes, not only RK/cap transitions.
- If start-state weights/anchors depend on controls, their derivatives belong in the derivative of the actual forward map. Freezing them in the VJP while recomputing them in every forward is an approximation; validate and label it rather than claiming an exact implicit gradient.
- Finite-differencing regularization of **derived** fields is not necessarily local/cheap: those fields depend on global equilibrium. Differentiate the regularizer with respect to its input fields and pass those bars through the equilibrium VJP, or count the additional global solves honestly.
- For a Hessian including residual-curvature terms, positive definiteness is not automatic. Specify how to detect non-SPD/singular systems and choose an appropriate transpose solver or documented approximation. Reusing PCG without this check is insufficient.

## Suggested disposition

Keep the six-phase structure. Resolve high-priority findings 1–5 in the plan first, then add the phase-specific acceptance tests while implementing. No need to preserve the legacy backend if its removal is the confirmed user decision. The key review gates are the standalone/embedded numerical comparison, collective-safe prepare/solve protocol, native-z forward contract, and the derivative of the actual accepted equilibrium map.
