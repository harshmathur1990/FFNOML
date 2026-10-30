# Phase 5: End-to-end inversions in all four modes

Repository: FFNOInversion.jl.
Brief sections: §8 (finite-difference reference), §9 (output, restart), §10 steps 5 and the validation list.
Depends on: Phase 4.

## Goal

Run real inversions (L-BFGS, with the FSDP population service) in all four modes, with:

- correct, validated gradients, where the equilibrium contribution is the finite-difference reference through HSEMHS,
- output and checkpoint files that record everything needed to reproduce and restart,
- restart that refuses incompatible state,
- runnable examples.

The goal is correctness, not speed. Cost is measured here and fixed in Phase 6.

## Scope

1. **Gradient path** (`src/ProductionGradients.jl`, `src/Gradients.jl`)
   - Honor the user-required zero-B dispatch on every perturbation. Fixed-zero-B temperature inversions differentiate the column-HSE path. A B perturbation from globally zero to nonzero can cross a discontinuity between HSE and MHS for horizontally varying atmospheres; detect and report unsupported derivatives rather than presenting a cross-branch difference as a smooth gradient. Add tests and document supported initialization for B inversions.
   - `_force_regularization_vjp!` → rename to `_equilibrium_regularization_vjp!`. It perturbs each scaled control coordinate, calls `_reconstruct_trial!` (restore reference, apply controls, then `reconstruct_equilibrium!`), and pairs the derived-field cotangent with the change in `pgas, rho, ne, z, logtau500`, plus regularization.
   - `_derived_pairing` includes `logtau500`. Check whether any downstream cotangent (`AtmosphereCotangent`) needs a `logtau500` slot. Add it only if a consumer reads logtau; otherwise document that it does not.
   - **Rename** `HybridAdjointObjectiveGradient.force_balance_step` → `equilibrium_step`. No default value is trusted for MHS. The step is chosen from a step-size sweep (test below), per mode, and recorded in the example configs.
   - Every plus/minus evaluation starts from the restored reference, and all ranks evaluate the same coordinate in the same order (the loop is already rank-synchronous).
   - **Invalid perturbations:** first retry with a reduced step (halve, at most `fd_max_retries`, default 3, with a minimum step tied to the solver tolerance), on both sides symmetrically. Only if one side stays invalid, use a one-sided difference toward the valid side, using the same machinery as the bound-limited steps, and record it in diagnostics. If both sides stay invalid, raise a collective error naming the coordinate. None of this may change the restored base state.
   - **One equilibrium map per difference.** Solver tolerance vs FD noise: the gradient may need tighter `rtol`/`cg_tol` than the forward. If so (`[equilibrium] gradient_rtol_factor`, added only if the sweep shows it is needed), the **center** value and the derived-field cotangent pairing are recomputed at the same tighter tolerances. Today `f0` is taken from the current evaluation, and mixing tolerances inside one difference is not allowed. The report shows that the resulting gradient is consistent with the objective the line search actually evaluates (directional check at forward tolerances).

2. **Output schema** (`src/Execution.jl: write_inversion_outputs`). Atmosphere file additions:
   - `z_m` or `logtau_500` (native axis), plus 3D `z`, `logtau500`, `pgas`, `rho`, `ne`,
   - attributes: `vertical_coordinate`, `vertical_unit`, `vertical_order`, `equilibrium_mode`, `equilibrium_backend = "HSEMHS"`, `hsemhs_revision`, `eos_table_sha256` (via HSEMHS `table_sha256`), `gravity_m_s2`, the full option struct as TOML, final `equilibrium_status` (reason, steps, phi/phi0, EOS violations), plus the ltau `align`/gauge settings or the z anchor settings.

   The synthesis file gets the same equilibrium provenance attributes.

3. **Checkpoint and restart** (`src/IO.jl`, `src/Solvers.jl`, `src/ScalableSolvers.jl`)
   - The checkpoint payload stores an `equilibrium_fingerprint` **as named components plus a digest**, so a mismatch can name the component. Components:
     - requested model and the effective-mode policy (zero-B dispatch), and the translated HSEMHS options (which include anchor, P_ref, z_ref, anchor_w, ltau top closure, cut, tolerances, `stationarity_tol`, `accept_phi_ratio`),
     - `eos_table_sha256` **and** the EOS provenance not covered by it (table metadata: abundances, `T_sw`, branch policy, `FIX_GASC_H2`, partition-function hash), plus the synthesis-side Wittmann library identity,
     - `hsemhs_revision`, gravity,
     - vertical axis kind and values, x/y, coordinate normalization (input order reversal applied),
     - any population-model height transform from Phase 3 item 7,
     - derivative settings that affect optimizer history (`equilibrium_step`, `gradient_rtol_factor`, `fd_max_retries`).
   - `_check_checkpoint_mapping` additionally requires an exact fingerprint match. A mismatch gives a collective error that lists which components differ.
   - `CapabilityManifest` `schema_version` → `2.0.0`. Old checkpoints are rejected with a "fresh run required" message (consistent with the existing correction-mapping policy).
   - Restart reproduces the deterministic initialization (no warm-start state is saved in this phase).

4. **Examples** (`examples/inversion_run/`, plus new `examples/inversion_run_z/`)
   - Four runnable TOMLs: `inversion_column_hse_ltau.toml`, `inversion_mhs_ltau.toml` (existing logtau input with B), and `inversion_column_hse_z.toml`, `inversion_mhs_z.toml` (native-z input: x_m, y_m, z_m, no logtau dataset).
   - Each has a matching `forward_*.toml`.
   - Updated `model_factory.jl` (shared), README sections, and eagle[7-8] sbatch/submit scripts accepting the config path.
   - A script `examples/make_native_z_input.py` (or Julia) that builds the native-z input from an existing simulation cube, with no logtau resampling. Document where it comes from.

5. **Regression scripts**
   - Update `test/olivia_phase1_to_phase6_regression.jl`, `scripts/validate_phase5_mpi.jl` and `scripts/validate_phase6_mpi.jl` to the new model factory and configs.
   - Add `scripts/validate_hsemhs_gradients.jl` (the step sweep and directional checks below) runnable on eagle[7-8].

## Out of scope

- An analytic or adjoint equilibrium derivative (Phase 6).
- Performance work beyond measurement.
- 2D tile ↔ slab redistribution.

## Tests

| Test | What it proves | Pass criterion |
|---|---|---|
| FD step sweep, ×4 modes | Choose a defensible step | On a small (e.g. 16×9×60) case, the directional derivative of the objective along a random unit direction (scaled controls) for steps h ∈ {1e-1, …, 1e-7}. Report the plateau. The chosen `equilibrium_step` lies in the plateau with relative variation < 1e-3 across one decade. A mode with no plateau is a blocking finding, reported with the equilibrium tolerance used. |
| Gradient Taylor test, ×4 modes | Gradient correctness | `gradient_taylor_validation` (existing): the first-order remainder decreases at O(h²) over ≥ 3 decades, for directions in T only, B only (MHS modes), and mixed. |
| Directional derivative vs full FD | Equilibrium term specifically | `centered_directional_validation` vs `objective_gradient!` ⋅ d: relative error < 1e-4 (column modes), < 1e-3 (MHS modes; the threshold is set from the sweep and justified in the report). |
| Restoration in gradient | No state leakage | The objective at x before `objective_gradient!` equals the objective at x after it, bitwise. |
| Invalid perturbation handling | Retry then one-sided | (a) + side invalid at full step but valid after one halving: a centered difference at the reduced step is used. (b) + side invalid at all retries: the − side one-sided difference is used, matches a one-sided FD reference, and all ranks agree. (c) both sides invalid: collective error naming the coordinate. (d) a near-bound coordinate. In every case the base-state objective afterwards is bitwise unchanged. |
| Tolerance consistency | One map per difference | With `gradient_rtol_factor < 1`, the center and pairing are recomputed at the tight tolerance, and the directional derivative of the forward-tolerance objective agrees with the gradient within the sweep threshold. |
| Rank independence of gradient | Collective order | The gradient on 1, 2, 3 ranks agrees to ≤ 1e-8 relative (column modes) / FD-noise level (MHS modes, threshold from sweep). |
| Short inversion, ×4 modes (local, mock populations) | Loop works end to end | A **differentiable** mock population model (`MockPopulationModel` with its `population_vjp!`). `RecordedPopulationModel` is not used here: it replays fixed input features and cannot follow changing controls, so it is reserved for fixed-atmosphere parity tests. Synthetic observation generated from a known atmosphere perturbed by +200 K in a blob. 10 L-BFGS iterations reduce the objective monotonically on accepted steps, and the recovered T moves toward the truth (error norm decreases). |
| Rejected-trial path in a real run | Phase 4 policy under L-BFGS | Force rejection (`max_steps` small) in one inversion. The run continues, and the iteration records show the backtracks. |
| Restart equivalence, ×4 modes | Brief §9 | 3 iterations, then checkpoint, then restart for 3 more, equals 6 straight iterations: parameters bitwise equal on the same rank/thread count. |
| Restart refuses mismatch | Fingerprint | Changing any one of: mode, EOS table file, EOS table metadata only (same arrays), gravity, z_cut, vertical axis kind, HSEMHS revision string, `equilibrium_step` → collective refusal naming the component. |
| Output schema | Provenance | Every new dataset and attribute is present. The EOS hash equals `table_sha256`. The native axis round-trips. |
| eagle[7-8] small-control GPU gate, ×4 modes | Bounded production check first | Real FSDP service, ≥ 4 ranks, a deliberately small control set (e.g. temperature only, 4 nodes × 2×2 maps = 16 coordinates; B added for MHS modes) and 3 L-BFGS iterations, within a wall-time budget stated in advance in the report (proposed: ≤ 2 h per mode). This gate must pass before the larger runs below. |
| eagle[7-8] short inversions, ×4 modes | Production | FSDP service, ≥ 4 ranks, 5 L-BFGS iterations from the example inputs in each mode, with the control count limited so the projected FD-gradient cost stays within the stated budget. Runs that would exceed it are not required for Phase 5 exit and are deferred to Phase 6. The objective decreases, outputs are written, and restart from iteration 3 works in at least one z and one ltau mode. Job IDs and wall times in the report. |
| Existing MPI/GPU regressions | Nothing broke | `olivia_phase1_to_phase6_regression.jl` passes with the new factory. So do `mpi_gpu_status_timeout_worker.jl` and `phase6_fsdp_protocol.jl`. |

## Cost measurement (required, no optimization)

Per mode on the example grid at eagle[7-8] scale, report:

- one forward: equilibrium / populations / synthesis / observation seconds,
- one gradient: total seconds, plus the FD equilibrium part separately, with the number of equilibrium solves (≈ 2 × number of control coordinates + 1),
- projected cost of a 50-iteration inversion.

These numbers decide Phase 6's priority. Phase 5 must not be blocked on an impractical production-size FD benchmark: the report states a wall-time and control-count budget up front, measures within it, and projects beyond it.

## Exit criteria

- All tests pass. The eagle[7-8] short inversions complete in all four modes.
- `docs/reports/hsemhs-phase-5-validation.md`: step-sweep plots/tables, Taylor-test slopes, directional-derivative errors, restart results, cost table, chosen `equilibrium_step` per mode.
- README (`FFNOInversion.jl/README.md`) documents the four modes, config keys, and example commands.
- At this point native z support may be called complete for correctness: no active path interprets z as logtau. The scalability claim waits for Phase 6.

## Points for the reviewer

- The FD equilibrium term costs about 2 × N_controls equilibrium solves per gradient. With e.g. 4 variables × 4 nodes × 8×8 control maps = 1024 coordinates, that is ~2000 Poisson solves per gradient, which is likely impractical for MHS at production size. Phase 5 accepts this for correctness on modest control counts. Phase 6 exists because of it.
