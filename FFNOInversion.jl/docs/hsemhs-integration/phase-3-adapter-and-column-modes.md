# Phase 3: Equilibrium adapter, MPI ownership, column modes, legacy removal

Repositories: FFNOInversion.jl (main work). HSEMHS only for fixes found during integration, committed on `embedding-api`.
Brief sections: §2 (`[equilibrium]` and mode table), §4, §5, §6, §7.
Depends on: Phase 1 (HSEMHS embedding API) and Phase 2 (native coordinates).

## Goal

Replace FFNO's native force balance with HSEMHS behind a single adapter, and make FFNO the sole owner of MPI and threads. A forward run in `column-hse-z` or `column-hse-ltau` then completes end to end: controls, then equilibrium, then populations, then synthesis, then observation, then output, on 1 and several ranks, with the FSDP GPU service. The legacy backend is deleted.

## Scope

1. **Dependency and environment**
   - Add HSEMHS to `FFNOInversion.jl/Project.toml` through `[sources]`: `HSEMHS = {url = "https://github.com/harshmathur1990/1DHSE3dMHS.git", rev = "<phase-1 commit>", subdir = "HSEMHS"}`. The package lives in the repository's `HSEMHS/` subdirectory, so `subdir` is required. For local development, `Pkg.develop(path=…)` may be used, but it must not be committed. Exit test: a clean clone of FFNOML on hyades and on eagle[7-8] instantiates with `Pkg.instantiate()` and no development path.
   - Add FFTW as a **direct** dependency of FFNOInversion, because FFNO calls `FFTW.set_num_threads` (Q4). A transitive dependency is not a supported import contract.
   - Resolve a single Manifest on Julia 1.12.3 (Q3: `julia/1.12.3` in `~/loadgcc.sh` on hyades/hyades[2-21], `~/loadnvidia_mpi.sh` on eagle[7-8], `~/loadamd_mpi.sh` on eagle9). One MPI.jl and one MPIPreferences across both packages, with FFTW added transitively.
   - **MPI binary:** system `openmpi/gcc13/5.0.5` on hyades (`~/loadgcc.sh`), eagle[7-8] (`~/loadnvidia_mpi.sh`) and eagle9 (`~/loadamd_mpi.sh`), selected with `MPIPreferences.use_system_binary()`. The environment-setup script generates `LocalPreferences.toml`. The report records `MPI.identify_implementation()` on each node group. Every MPI test runs on hyades, eagle[7-8] and eagle9. FSDP-service tests run on eagle[7-8] only (CUDA).
   - Update `scripts/prepare_olivia_environment.jl` / `initialize_olivia_environment.sh` if they enumerate packages.

2. **Delete the legacy backend.** Remove `src/ForceBalance.jl`, `src/LogPressure.jl`, `src/Opacity500.jl`, the reconstruction functions in `src/DistributedForward.jl` (`_local_boundary` through `reconstruct_force_balance_distributed!`, lines ~59–488 at `07402ae`), `HE3DBoundaryState` (`Types.jl`), their exports, `test/force_balance.jl`, `test/log_pressure.jl`, and the skips added in Phase 2. Keep `WittmannEOS` and whatever synthesis still uses (`WittmannKuruczState`, `ContinuumOpacity`): Q1 says synthesis stays on FFNO's Wittmann library. Delete `thermodynamics!`/`IdealGasEOS` if only the deleted backend used them. Delete anything that is no longer referenced. `ADR-003-force-balance.md` gets a "Superseded by HSEMHS integration" status line and a pointer to a new ADR-012.

3. **Configuration** (`src/IO.jl`)

   ```toml
   [equilibrium]
   backend = "HSEMHS"            # only accepted value
   model = "column_hse"          # "column_hse" | "mhs" (mhs is accepted in Phase 4)
   gravity_m_s2 = 274.0
   eos_table_file = "inputs/eos_table.h5"
   eos_pf_file = "inputs/pf_Kurucz.input"   # if HSEMHS load_table needs it
   on_nonconvergence = "reject_trial"       # "reject_trial" | "abort"

   [equilibrium.column_z]        # only for column-hse-z / mhs-poisson-z
   anchor = "top"                # top | bottom | height
   z_ref_m = 0.0                 # for anchor = height
   tau_top = "exp_tail"
   # rk options...

   [equilibrium.column_ltau]     # only for *-ltau
   cap_per_dex = 50.0
   align = "ptot"
   ```

   - Derive the mode from `(vertical_coordinate, model)` using brief §2's table. No user-facing mode key.
   - Sections irrelevant to the derived mode are rejected (`[equilibrium.column_ltau]` in a z run, for example).
   - **Pressure boundary (Q2: HSEMHS is the final authority):**
     - Remove `pressure_top_pa` and `pressure_top_dataset` from `[atmosphere]`. A config containing either fails with "the pressure boundary is set by HSEMHS; configure [equilibrium.column_z] anchor/P_ref".
     - Delete `AtmosphereInputConfig.pressure_top`, `InversionInputBundle.pressure_top`, the `distribute_field(... pressure_top ...)` call in `run_inversion!`, and the `local_pressure_top` argument of `InversionModelFactory.build_model` (the signature becomes `build_model(config, distributed, workspace, context)`). Update `scripts/preflight_olivia_inversion.jl` and the example factory.
     - z modes: `[equilibrium.column_z]` carries HSEMHS's `anchor`, `P_ref` and `z_ref` with HSEMHS names, defaults and validation. Whatever HSEMHS requires (`column_hse.P_ref` is REQUIRED for z modes in `Config.jl`) is required here too.
     - ltau modes: no anchor keys exist. The τ-hydrostatic top closure in HSEMHS sets the pressure.
   - All translation to HSEMHS option structs lives in one function, `_hsemhs_options(config)`, which also calls the HSEMHS validators from Phase 1.
   - HSEMHS modes require `parallel.decomposition = "x_slabs"`. `cartesian_2d` is rejected for them with a clear message.

4. **MPI and thread ownership** (`src/Parallel.jl`)
   - `initialize_parallel`: if FFNO initializes MPI, use `MPI.Init(threadlevel=:funneled)` and verify `provided >= MPI.THREAD_FUNNELED`. If MPI is already initialized, query `MPI.Query_thread()` and require ≥ FUNNELED. Fail collectively otherwise.
   - `serial_context` (`enabled = false`): **decision — the serial path is a managed single-rank MPI path.** If `[equilibrium]` is configured, `initialize_parallel` with `enabled=false` still initializes MPI (FUNNELED) and passes `MPI.COMM_SELF` to HSEMHS. FFNO's own code keeps `context.enabled = false` semantics (no FFNO collectives). This avoids maintaining a second non-MPI HSEMHS path. HSEMHS already runs "every run through the MPI layer". The ParallelContext records the backend communicator separately.
   - **Finalization depends on ownership, not `enabled`.** `finalize_parallel!` currently finalizes only when `context.enabled && context.owns_mpi`, which would leak MPI in the managed serial path. Change it to depend on `owns_mpi` alone. `run_inversion_files!` frees backend communicators (`close_equilibrium!`) before `finalize_parallel!`, in a `try`/`finally` that covers failure and early-return paths.
   - **`enabled = false` under a multi-rank launcher is rejected.** If `MPI.Comm_size(MPI.COMM_WORLD) > 1` while `parallel.enabled = false`, every rank fails with a message saying to set `enabled = true`. Otherwise independent `COMM_SELF` runs would collide on output files and GPU-service launch.
   - **Q4: FFNO owns threading.** BLAS and FFTW threads are set once at startup by FFNO from `threads_per_rank` (BLAS = 1 inside threaded column loops; FFTW threads documented). HSEMHS sets none. The report gives the values used on hyades and eagle[7-8].

5. **Adapter** (new `src/Equilibrium.jl`)

   ```julia
   abstract type AbstractEquilibriumBackend end
   struct HSEMHSBackend <: AbstractEquilibriumBackend ... end      # holds mode, options, EOS table, ParCtx, workspace

   backend = prepare_equilibrium(config, distributed, context)        # collective, once per run
   result  = reconstruct_equilibrium!(backend, distributed, context)  # collective, every evaluation
   close_equilibrium!(backend)                                        # frees backend-owned communicators
   ```

   - `prepare_equilibrium` follows the Phase 1 two-stage protocol: local checks (EOS table path exists and is readable on this node, hashes, option validity, tile shape) are caught and agreed on `context.comm` before any HSEMHS collective construction. A missing file on one node gives the same error on every rank.
   - `prepare_equilibrium`:
     - builds `ParCtx` from `context.comm` (or `COMM_SELF`), duplicated by HSEMHS,
     - checks HSEMHS's slab ownership equals FFNO's `x_slabs` tile for this rank (global x ranges, not rank order) and fails collectively otherwise,
     - loads the EOS table once per rank,
     - validates EOS coverage of the input temperature range,
     - prepares the HSEMHS workspace for the derived mode.
   - `reconstruct_equilibrium!`:
     - packs `distributed.local_atmosphere` T (and B, for the ltau align step) from FFNO `(k,x,y)` to HSEMHS `(k,y,x)`,
     - calls the mode's `solve_*!`,
     - unpacks into `pgas`, `rho`, `ne`, `z` (3D) and `logtau500` (3D) of the local atmosphere.
     - It does **not** modify T, B, v or vturb. It returns `EquilibriumResult(status, diagnostics)`, with timing.
   - **Coordinate rules:**
     - native z: `atm.z` = broadcast native axis, and `atm.logtau500` = column-dependent result.
     - native logtau: `atm.logtau500` = broadcast native axis, and `atm.z` = HSEMHS reconstructed heights after HSEMHS's documented gauge (mean z(τ=1) = 0) and alignment.
     - Both are asserted with `check_coordinate_consistency` from Phase 2.
   - EOS flags: global counts of `EOS_OUT_OF_ENVELOPE`, `EOS_DIRECT_FAILED` and `OPACITY_INCOMPLETE` go into the status. Any out-of-envelope point makes the result invalid.

6. **Call-site replacement**
   - `HybridForwardModel`: replace the fields `eos, opacity, boundary, force_options` with `equilibrium::AbstractEquilibriumBackend` (keep `eos::WittmannEOS` for synthesis, Q1). `prepared_force_balance` becomes `prepared_equilibrium`.
   - `forward!` calls `reconstruct_equilibrium!` and records `HybridForwardTimings.equilibrium_seconds` (renamed from `force_balance_seconds`).
   - **Final failure contract (fixed now, so Phase 4 does not change it).** `reconstruct_equilibrium!` returns a globally agreed status. On invalid status, `forward!` short-circuits (no populations, synthesis or observation), leaves the last accepted prepared atmosphere/populations untouched, and returns the invalid status in its result. The caller decides what happens next: `run_inversion!` in forward mode raises a collective `ArgumentError`. The objective/solver policy (reject vs abort) arrives in Phase 4.
   - `Solvers.jl`: `_ThermodynamicTrialReference` gains `logtau500`. `_restore_trial_reference!` restores it.
   - `ProductionGradients._reconstruct_trial!` calls `reconstruct_equilibrium!`. Gradients are only exercised in Phase 5, but the call path is switched now so nothing references deleted code.
   - `examples/inversion_run/model_factory.jl` builds the backend with `prepare_equilibrium`, and drops `HE3DBoundaryState`, `ForceBalanceOptions` and `FFNO_TOP_DENSITY_KG_M3`.
   - **Phase ordering on every rank:** apply controls, then equilibrium (HSEMHS collectives), then populations (rank-0 GPU service, non-root ranks in status wait), then synthesis (threads), then observation (halo exchange), then objective. HSEMHS collectives never run inside `Threads.@threads` or during the GPU status-wait phase. Add an assertion in `reconstruct_equilibrium!` that it runs on `context.initializing_thread`.

7. **Population and synthesis contracts**
   - `population_features` takes 3D z (already true). For ltau runs, z is now corrugated and gauge-dependent. **Audit the trained checkpoint's full input contract** against what the inversion will feed it, and record the results in the report:
     - height zero point (which z = 0 the training data used, from the training/data-preparation scripts) vs the HSEMHS ltau gauge (mean z(τ=1) = 0) and vs native-z input files,
     - vertical sample order, spacing (uniform or not, value), number of levels, vertical extent and horizontal resolution the checkpoint was trained on, and any `zscale` normalization (`training_FFNO3D_zscale_expand_lognlte`),
     - the preprocessing applied to each channel (`FFNO_INPUT_CHANNELS`) and to z.
   - If anything differs, add an explicit mapping in the population-model path (not in the equilibrium): a model-only height offset or a resampling operator, each with its VJP. The mapping must not modify the atmosphere's native z grid and must not break `atm.z == broadcast(z_axis)` for native-z runs. Applying a mapping requires a test. If no faithful mapping exists (for example, unsupported extent), the run is rejected with a clear message. Never feed a different coordinate convention to the trained model silently.
   - Muspel formal solve receives 3D z (already true). Confirm top-first order (Phase 2 audit).

8. **Wittmann EOS compatibility (Q1).** Both sides use the Wittmann EOS, but a shared partition-function file does not by itself make the HSEMHS table and FFNO's compiled Wittmann library equivalent. The table build has abundance inputs, direct/blended/hot branches, a switch temperature `T_sw`, fallback flags and the `FIX_GASC_H2` build option (`eos/BuildTable.jl`, `eos/Table.jl`), and `table_sha256` hashes only the table arrays and flags, not this metadata.
   - **Provenance comparison** at `prepare_equilibrium`, reported and checked: partition-function file hash, abundance set, Wittmann library source/build identity and `FIX_GASC_H2` status on both sides, the table's `T_sw` and hot-branch policy, and the χ500 convention (m⁻¹ extinction, absorption + scattering). Hard mismatches (partition functions, abundances) fail collectively. Others are recorded in provenance.
   - **Sampled comparison** (`scripts/validate_eos_consistency.jl`): ρ and nₑ from the table (`state`) and from FFNO's `WittmannEOS` over the actual (T, P) envelope of the example atmospheres after equilibrium, deliberately including points across the `T_sw` branch transition and near the table edges. Report max/p99 relative differences per region (cold direct, blend, hot branch). The expectation is agreement within the table's validated interpolation error (HSEMHS `reports/gate1_table*.md`) in the direct region, with documented, not bitwise, differences in blend/hot regions.
   - **Failure propagation:** a state that the HSEMHS hot branch accepts but FFNO's direct Wittmann call (synthesis) rejects must produce a collective, clearly attributed synthesis failure, not NaNs. This is tested.

## Out of scope

- MHS modes (Phase 4). `model = "mhs"` is rejected with "available in Phase 4".
- Inversion runs, gradients correctness and restart (Phase 5).

## Tests

| Test | What it proves | Pass criterion |
|---|---|---|
| Config: mode derivation | Brief §2 table | All four (coordinate, model) pairs derive the right mode. Irrelevant sections and anchors rejected. `backend ≠ "HSEMHS"` rejected. `cartesian_2d` + HSEMHS rejected. |
| Adapter vs direct HSEMHS (column-hse-z, column-hse-ltau) | Packing and units | On a non-square synthetic atmosphere (NX=12, NY=7, NZ=40, T varying independently in x, y, z), FFNO `reconstruct_equilibrium!` output equals a direct HSEMHS `solve_*!` call on hand-packed arrays to ≤ 1e-13 relative, for P, ρ, nₑ, z, logτ. |
| Adapter vs standalone file run | End-to-end equivalence (brief §10) | The same atmosphere written in HSEMHS input format and run with the standalone driver gives fields matching the adapter to ≤ 1e-10. |
| Vertical order | Rule 4 | Bottom-first input produces the same equilibrium (after reversal) as top-first input. |
| Coordinate semantics | No relabeling | z run: `atm.z` equals the axis exactly and `atm.logtau500` varies by column (test atmosphere with horizontal T variation). ltau run: the reverse. |
| Zero-B hydrostatic manufactured case | Physics sanity | Isothermal T = 6000 K, ideal-gas test EOS (`eos/TestEOS.jl: IdealGasMu`) if the adapter accepts it, else the table. P(z) = P₀ exp(−z/H) to ≤ 1e-6 relative, for both z and ltau modes (ltau: compare P(τ) against the analytic relation for constant χ/ρ if the test EOS supports it, otherwise column-HSE-z vs column-HSE-ltau consistency at matched τ). |
| MPI ownership | Brief §5 | (a) FFNO initializes MPI with FUNNELED and verifies it. (b) Pre-initialized MPI with SINGLE is rejected on all ranks. (c) No HSEMHS call creates or finalizes MPI (mock by counting via `MPI.Initialized` before/after, plus grep). (d) `close_equilibrium!` frees communicators, and 20 prepare/close cycles leave no leak warnings. |
| Serial path | Single-rank MPI with `COMM_SELF` | `enabled=false` + `[equilibrium]` runs one forward. Result equals 1-rank `enabled=true`. No FFNO collectives are called (a mock context counts them). |
| Rank independence | Decomposition safety | A forward with `MockPopulationModel`/recorded populations on 1, 2, 3 ranks (`x_slabs`, NX=11 for the uneven split) gives a gathered spectrum and atmosphere agreeing to ≤ 1e-10 relative. |
| Repeated calls, no deadlock | Lifecycle | 100 consecutive `forward!` calls on 3 ranks with alternating valid/out-of-envelope temperature perturbations on one rank. All ranks report the same status each time. No hang (CI timeout 10 min). Steady-state allocation per call < 5 % of the first. |
| Failure is collective | Brief §9 | Out-of-envelope T on rank 2 only. `reconstruct_equilibrium!` returns invalid status on every rank. `forward!` short-circuits and returns the invalid status on every rank, populations are not called, prepared state is unchanged, and `run_inversion!` (forward mode) raises the same `ArgumentError` on every rank. |
| Prepare failure is collective | Two-stage protocol | EOS table missing on one rank only (path override on that rank), and a provenance mismatch on one rank only: every rank throws the same error, and a parent-communicator `Barrier` afterwards succeeds. |
| Serial lifecycle | Ownership | `enabled=false` + `[equilibrium]`: MPI is initialized and finalized exactly once, backend communicators are freed first, including when the forward throws. `enabled=false` under `mpiexec -n 2` is rejected on both ranks. |
| Clean instantiate | Dependency spec | Clean clone on hyades and eagle[7-8], `Pkg.instantiate()`, `using FFNOInversion, HSEMHS` succeeds without a dev path. |
| Population input contract | Item 7 | If a height offset or resampling is applied, a test checks the model receives the transformed z while `atm.z` stays equal to the native axis, and the mapping's VJP passes a dot-product test. |
| EOS failure propagation | Q1 | A hot-branch-accepted, direct-rejected state yields a collective, attributed synthesis error. |
| GPU-service protocol unaffected | Brief §5 | Existing `test/mpi_gpu_status_timeout_worker.jl`, `test/phase6_fsdp_protocol.jl` and `test/fsdp_only_architecture.jl` pass with the HSEMHS model factory. |
| Legacy removed | Decision | Grep for `ForceBalance\|HE3DBoundaryState\|reconstruct_force_balance\|LogPressure\|Opacity500` in `src/ ext/ scripts/ test/ examples/` returns no matches outside ADR history. |
| Wittmann compatibility | Q1 | Mismatched partition-function file or abundance set rejected on all ranks. Matching inputs accepted. Provenance recorded. Sampled ρ/nₑ comparison reported per branch region and within the expectation above. |
| Ptop removed | Q2 | Configs with `pressure_top_pa` or `pressure_top_dataset` are rejected with the HSEMHS message. `build_model` has no pressure argument. Grep for `pressure_top` in `src/ scripts/ examples/` finds only the rejection check. |
| eagle[7-8] forward smoke | Production stack | `examples/inversion_run/forward.toml`, converted to `[equilibrium] model="column_hse"` (logtau input), runs on eagle[7-8] with the FSDP service and 4 ranks. A native-z forward with a z version of the same atmosphere also runs. Both outputs are written, and the spectrum is finite. Synthesis is plotted next to the old example output for a qualitative check only. |

## Exit criteria

- All tests above pass. The eagle[7-8] smoke runs complete, with job IDs in the report.
- `docs/reports/hsemhs-phase-3-validation.md` includes: dependency versions and the pinned HSEMHS revision, the EOS provenance comparison and the per-region sampled ρ/nₑ differences, the population-model input-contract audit, the z-convention finding for the population model (item 7), per-stage timings for one forward on the reference grid (equilibrium vs populations vs synthesis), and the BLAS/FFTW thread settings.
- New `docs/adr/ADR-012-hsemhs-equilibrium.md` records: HSEMHS as sole backend, mode derivation, HSEMHS as sole authority on the pressure boundary (Q2), serial = single-rank MPI, FFNO ownership of MPI and threads (Q4), `x_slabs` requirement, and the Wittmann EOS split between equilibrium (HSEMHS table) and synthesis (FFNO library) (Q1).

## Points for the reviewer

- Deleting the legacy backend and switching to HSEMHS in the same phase means there is no intermediate state with two backends. Validation relies on the adapter-vs-standalone-HSEMHS equivalence instead of old-vs-new.
- The population model's z convention (item 7) is the most likely source of a silent scientific error in ltau runs. Codex should check the report's finding carefully.
