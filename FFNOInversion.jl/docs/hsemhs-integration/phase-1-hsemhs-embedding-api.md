# Phase 1: HSEMHS embedding API

Repository: HSEMHS (`/Users/harshm/Documents/WorkRepo/1DHSE3dMHS/HSEMHS`), on a new branch `embedding-api` from `b2401a3`.
Brief sections: §1 (HSEMHS seams), §4, §5 (HSEMHS side), §6 (slab constraints).
Depends on: nothing. Runs in parallel with Phase 2.

## Goal

Make every numerical path of the four fixed-B modes callable in process, with:

- arrays already in memory,
- a communicator supplied by the caller,
- no file IO, MPI lifecycle calls or global state mutation,
- a structured status result,

so that FFNO can embed HSEMHS in Phase 3. The standalone file-driven runs must execute the same kernels through this API and produce unchanged results.

## Scope

1. **Communicator injection** (`Parallel.jl`, `Slab.jl`)
   - `init_parallel(comm, NX, NY, NZ; px, py, periodic, debug)` takes a parent communicator. A convenience method without `comm` keeps today's `COMM_WORLD` behaviour for the standalone drivers only.
   - `slab_par(comm, NX, NY, NZ)` likewise. It must use `MPI.Comm_size(comm)`, not `COMM_WORLD`.
   - `init_parallel` duplicates the parent (`MPI.Comm_dup`) before `Cart_create`, so HSEMHS messages cannot match caller messages. `ParCtx` records that it owns the duplicate and the Cartesian communicator. A new `free_parallel!(par)` frees both. It never finalizes MPI.
   - Grep audit: after this phase, `MPI.COMM_WORLD` appears only in `ensure_mpi`-adjacent standalone entry points (`Driver.jl`, `scripts/`, the no-`comm` convenience constructors). `ensure_mpi()` is not called from any embedded entry point.
   - Ownership checks use the global index ranges in `Block`, not the assumption that Cartesian rank equals x-slab index.
   - The slab precondition `NX ÷ nr >= 2` is mathematically correct: the smallest `block_range` block is exactly `floor(NX/nr)`. It is still a rank-local `error` before any collective, so it moves into the two-stage prepare protocol (item 2) to give a collective, consistent diagnostic. The accepted set of decompositions does not change.

2. **Structured status and collective-safe failure**
   - Add `EquilibriumStatus` (or extend `SolverStatus`/`Status.jl`) carrying: `status::SolverStatus`, `reason::Symbol` (`:converged`, `:stalled`, `:max_steps`, `:no_descent`, `:eos_out_of_envelope`, `:invalid_input`), `steps`, `phi0`, `phi`, `eos_violations::Int` (global count), `column_flags_summary`, `messages`.
   - Every embedded path returns status instead of throwing on physics failures. Today these throw from inside collective sequences, and some do so non-collectively:
     - `PoissonTau.column_hse_tau` (`error("column HSE in τ left the EOS table…")`),
     - `PoissonTau.solve_tau` (`error("start state outside the EOS table")`),
     - `PoissonZ.solve_pressure` (`error("z_cut … lies below the grid")`, which belongs in input validation).
   - **Two-stage prepare protocol (also used before every later collective).** A rank-local throw, even for a programmer error, can hang the job if other ranks have already entered `Comm_dup`, `Cart_create`, a reduction or FFT setup. So every `prepare_*` and `solve_*!`:
     1. runs all local checks (shapes, layout, finiteness, option validity, file existence/readability, hash comparisons) inside `try`, catching expected validation and IO errors into a compact local status code and message,
     2. exchanges the status on the **caller's already valid communicator** (`Allreduce(max)` of the code, then a broadcast of the first failing rank's message),
     3. only enters collective construction or the solver if every rank passed. Otherwise every rank throws the same `ArgumentError` (prepare) or returns the same invalid status (solve).
   - The same pattern separates setup stages (communicator, then EOS table, then FFT plan, then buffers). It does not promise recovery from process death or failures inside MPI itself, which remain job-level failures.
   - Logging goes through a caller-supplied `logf`, with the default a no-op. Embedded paths must not `println`.

3. **In-memory API, one entry per mode.** New file `src/Embedded.jl`, exported:

   ```julia
   ws = prepare_column_hse_z(par, z, x, y, eos, opts)            # validates, allocates
   res = solve_column_hse_z!(ws, lnT)                            # (vertical,y,x) slab arrays in, result out

   ws = prepare_column_hse_ltau(par, logtau, x, y, eos, opts)
   res = solve_column_hse_ltau!(ws, lnT, Bx, By, Bz)

   ws = prepare_mhs_poisson_z(par, z, x, y, eos, opts)
   res = solve_mhs_poisson_z!(ws, lnT, Bx, By, Bz)

   ws = prepare_mhs_poisson_ltau(par, logtau, x, y, eos, opts)
   res = solve_mhs_poisson_ltau!(ws, lnT, Bx, By, Bz)
   ```

   Exact names may change. The contract may not:
   - **Stationarity and convergence measures in the status (needed for Phase 4 validity and Phase 6 implicit-derivative eligibility).** The Poisson solvers currently stop on `n_consec` consecutive small relative decreases in Φ, or on "no descent". Neither proves stationarity. Each solve additionally reports:
     - a globally normalized stationarity norm: ‖∇ᵤΦ‖ = ‖JᵀW R‖ for `PoissonZ`, and the corresponding gradient over the (ln P below k_c, column height c) unknowns for `PoissonTau`, normalized by Φ and the total weight, plus the projected version if any cap or bound was active on the last step,
     - the last linear-solve relative residual and iteration count,
     - the final LM parameter μ (`PoissonTau`),
     - absolute as well as relative Φ.
   - **Already-equilibrated starts.** If Φ₀ = 0 or is below an absolute tolerance, `Φ/Φ₀` is undefined and strict decrease may be impossible. Add an absolute tolerance, return `:converged` with zero steps when the start already meets it, and make every ratio in diagnostics safe (report `NaN`-free values with an explicit `phi0_zero` flag).
   - **Result validity checks.** Before returning valid, verify: all fields finite; P, ρ, χ500 > 0; z(τ) and τ(z) strictly monotonic per column (no folded heights in ltau modes); the above-cut hydrostatic continuation in `solve_pressure` (its Newton loop and EOS calls) succeeded in every column. Each check has its own `reason` and global count.
   - **Result buffer ownership.** State in the docstring whether `res` arrays alias workspace buffers. Proposed: they do, for zero-copy use, and are overwritten by the next `solve_*!` on the same workspace. `snapshot(res)` returns an owned deep copy. FFNO copies into its own atmosphere arrays immediately (Phase 3), and anything retained across solves (restored trials, prepared results, adjoint state) uses `snapshot`.
   - **Single layout.** All four take and return rank-local `(vertical, y, x)` arrays, top-first vertical order, on the slab owned by `par`. `column-hse-z` currently runs on `Field` (`(x,y,z)` with halos) via `init_parallel` with a 2D process grid. The embedded variant packs into and out of `Field` internally, or the column kernel `hse_column!` is driven directly per column (preferred: `hse_column!` is already column-local and needs no halos).
   - **Result fields**, all on the native grid: `lnP` (or `Pgas`), `rho`, `ne`, `chi500`, `tau500`, `logtau500` (3D), `z` (3D: the broadcast native axis for z modes, reconstructed and corrugated for ltau modes), `z_tau1` (2D), `eos_flags` (3D `UInt8`), plus `EquilibriumStatus`.
   - **One EOS call per point.** Density, electron density and χ500 for the final state all come from `state(tab, lnT, lnP)` on the same table. Today `PoissonZ.tau_columns` recomputes ρ, `write_tau_grid` recomputes ρ via `lnrho_dY`, and nₑ is not returned by the Poisson modes. The embedded result computes all three in one pass.
   - **χ500 units.** Confirm that `chi500` is extinction per length in m⁻¹ (`eos/Opacity500.jl:chi500_m` multiplies the CGS result by 100), and document it in the result docstring. τ integration (`tau_column!`) must consume m⁻¹ with z in m.
   - **Workspaces** cache the FFT plan (`SlabFFT`), the preconditioner buffers, the `Prob`/`TP` arrays, the per-thread column scratch, and the band/weight vectors that depend only on the grid. Weights that depend on the starting state (`H̄` layer medians, `wc`, `wf`) are recomputed each solve, as they are today. Repeated `solve_*!` calls must not allocate O(volume) beyond the result buffers. This is verified by an allocation test.
   - **No global mutation (Q4: FFNO owns MPI and threading).** No `BLAS.set_num_threads` inside library calls (remove it from `__init__` and from the `run_mhs_poisson_*` drivers' library path; standalone entry points may set it themselves), no FFTW thread setting, no `ENV`, no `Pkg`, no MPI init/finalize.

4. **Pressure anchoring stays HSEMHS-owned** (cross-phase Q2). HSEMHS is the final authority on the pressure boundary. The embedded API takes the anchor through the option structs below (`anchor`, `P_ref`, `z_ref`, `anchor_w`, the ltau top closure) with exactly the semantics of the standalone config (`Config.jl` schema). No per-column map or other new anchor semantics are added. `P_ref = "from_input"` is a file-driver feature. The embedded API takes the resolved number, and the gate report states how an embedding caller should supply it.

5. **Boundary-condition options exposed as typed structs**, with all option names in one place:
   - `ColumnZOptions` (anchor, **P_ref** (resolved number, required as in the standalone schema), z_ref, n_sub, adapt, rk_tol, n_sub_max, tau_top, H_max, mu),
   - `ColumnTauOptions` (cap_per_dex, align ∈ {"ptot","none"}),
   - `PoissonZOptions` (z_cut_m, max_steps, rtol, n_consec, anchor_w, cg_tol, cg_maxit, tau options),
   - `PoissonTauOptions` (logtau_cut, max_steps, rtol, n_consec, cg_maxit, cg_tol, dc_cap_m, du_cap, lm_mu0).

   Validation of each struct is a pure function, so FFNO can call it at config time. Validation includes the **number of retained vertical levels**, not only whether a cutoff lies inside the range: `solve_pressure` indexes `h[1]` and `h[end]` and needs at least 2 levels at or below `z_cut` (a cut selecting only the bottom level fails), and `solve_tau` indexes face-spacing vectors (`dzf`, `wf`) with the same kind of minimum. Each kernel's minimum retained size is documented, and grids with fewer total levels than a kernel needs (`NZ < 3` already, plus the per-kernel minimums) are rejected.

   **Anchor and damping semantics (`PoissonZ`).** At `b2401a3`, `Aop!` adds `ereg .* v` and a rank-one `Aanc` term over the anchor cells to the Gauss-Newton increment system, while `phi` contains only force residuals and the right-hand side is `-JᵀW R`. There is no anchor residual in Φ. So the anchor and `ereg` are increment damping in the linear step, not an enforced constraint: the mean ln P over the anchor cells is discouraged from changing per step but is not held exactly. Phase 1 does not change these numerics. The gate report must (a) state this precisely, (b) measure the drift of the anchor-cell mean ln P from start to finish on the mode fixtures, and (c) give the equivalent statement for `PoissonTau` (`ereg`, LM damping `μ Dlm`). Phase 4 and Phase 6 rely on this description.

6. **Standalone drivers on the API.** `run_column_hse`, `run_column_hse_ltau`, `run_mhs_poisson_z` and `run_mhs_poisson_ltau` become: read file, then `prepare_*`, then `solve_*!`, then write file. Output files keep the same datasets and attributes. `full-mhs-z` and the M5/M6 experimental modules are untouched.

7. **Package hygiene for embedding**
   - Add a `[compat]` entry for every dependency.
   - Check `Project.toml` resolves and the suite passes on Julia 1.12.3 (Q3: the hyades `julia/1.12.3` module with `openmpi/gcc13/5.0.5`). Use no Julia 1.13 features. Regenerating the committed 1.13.1 Manifest is left to the user. Report the 1.12.3 resolution result.
   - Record the MPI.jl binary/ABI preference HSEMHS was tested with.

## Out of scope

- Any FFNO change (Phase 3).
- Derivatives (`Jop!`/`JTop!` exposure for the VJP is Phase 6).
- `full-mhs-z`, staggered, Gauss-Newton and joint-GN experiments.
- Non-periodic horizontal boundaries.

## Tests

All new tests go in `test/test_embedded.jl` (single rank) and `test/mpi/mpi_tests.jl` (2, 3, 4 ranks; 3 exercises an uneven split).

| Test | What it proves | Pass criterion |
|---|---|---|
| Embedded vs file driver, ×4 modes | The API is the same numerics as the standalone runs | On the existing fixtures/configs in `config/modes/`, every output field (P, ρ, τ, z, z_tau1) matches the pre-refactor `b2401a3` driver output to ≤ 1e-12 relative (1 rank). Status and step counts are identical. |
| Pre-refactor regression files | Refactoring changed nothing | Before editing, run the four mode configs at `b2401a3` and store the outputs under `test/fixtures/embedded_ref/` (small grids only, < 5 MB total). Compare after the refactor. |
| Rank-count independence | Reductions and FFT are decomposition-safe | For each mode on 1, 2, 3 and 4 ranks, gathered fields agree to ≤ 1e-10 relative (not bitwise; sums reorder). |
| Uneven x split | `block_range` is honoured | NX = 11 on 3 ranks runs and matches 1 rank. NX = 5 on 3 ranks is rejected on all ranks with the same message. |
| Injected communicator | No hidden `COMM_WORLD` | Run on a 4-rank world split into two 2-rank sub-communicators (`MPI.Comm_split`), each solving a different atmosphere at the same time. Both results equal the corresponding 2-rank standalone solve. Then a world `Barrier` succeeds (no stray messages). |
| Communicator lifetime | No leaks | 50 prepare/solve/free cycles on a sub-communicator. The communicator count (via `MPI.Comm_free` bookkeeping, or no `MPI_ERR_COMM` at finalize) is unchanged, and no warnings appear. |
| One-rank-only prepare failures | Two-stage protocol | On 3 ranks, separately: a malformed (wrong-shape) array on rank 1 only; a missing EOS table path on rank 2 only; a table/partition-function hash mismatch on rank 0 only. Every rank throws the same `ArgumentError`, and a following `Barrier` on the parent communicator succeeds. |
| Retained-level limits | Cutoff validation | `z_cut` at the top level, at the bottom level (rejected), one level above the bottom (minimum accepted); the same for `logtau_cut`; a grid with too few levels for each kernel. Rejections are collective, acceptances run. |
| Zero and tiny initial residual | Φ₀ handling | A column-HSE start fed to Poisson with B ≡ 0 (Φ₀ ≈ 0) and with B scaled to 1e-12 T: `:converged`, zero or few steps, finite diagnostics, `phi0_zero` set appropriately. |
| Invalid results | Validity checks | Inputs forcing (a) a folded z(τ) column in ltau, (b) a failed above-cut continuation, (c) non-positive pressure: each yields its specific `reason` on all ranks. |
| Stationarity norm | Status content | On a converged fixture, the reported stationarity norm is below a documented threshold and matches an independent computation of ‖JᵀW R‖ (z) to 1e-10. On a `max_steps = 1` run it is larger and reported. |
| Anchor drift | Documents damping semantics | Reports the anchor-cell mean ln P change between start and finish for the `mhs-poisson-z` fixtures. Pass = number reported and consistent with the gate-report description (no claim of exact preservation). |
| Result aliasing | Buffer contract | After a second solve, a retained `snapshot` of the first result is unchanged and the un-snapshotted result reflects the second solve. |
| Collective failure | Physics failure does not deadlock | A temperature cube that leaves the EOS envelope on one rank only. Every rank returns `reason = :eos_out_of_envelope` with the same global `eos_violations`, and a following `Barrier` completes. Repeat for a `solve_tau` start-state failure. |
| Layout | `(vertical,y,x)` contract | NX ≠ NY ≠ NZ, T = f(k) + g(j) + h(i) with distinct f, g, h. Column HSE output in column (i,j) depends only on that column's T, checked against a direct `hse_column!` call. |
| Anchor options pass-through | Q2 | For each anchor option, the embedded solve with the option struct equals the standalone run whose TOML sets the same keys (≤ 1e-12). |
| χ500 / τ units | Units contract | For a hydrostatic column, τ from the result equals the trapezoid of `chi500` (m⁻¹) over z (m) with the documented top closure, to ≤ 1e-12. |
| Allocations | Workspace reuse | After warm-up (compilation and first plan creation excluded), the 3rd–10th `solve_*!` calls allocate no more than a documented budget of `c × local_volume × 8` bytes per call, with `c` stated per mode (target `c ≤ 1`, i.e. at most one volume-sized temporary). The report also gives workspace bytes and peak RSS per rank. |
| No global mutation | Library hygiene | `BLAS.get_num_threads()` is unchanged across `using HSEMHS` and every `solve_*!`. |

The existing test suite (`test/runtests.jl`, including MPI ranks) must still pass.

## Exit criteria

- All tests above pass on 1–4 ranks locally. The multi-rank suite also passes on hyades (`source ~/loadgcc.sh`, system OpenMPI via MPIPreferences), or the report explains why not.
- The `COMM_WORLD` grep audit result is included in the report.
- `HSEMHS/reports/embedding_api_gate.md` lists: API signatures, result and status fields, the per-mode option mapping table (option, unit, default, valid range), the measured per-solve wall time on a reference grid for 1 and 4 ranks, and the allocation numbers.
- Commit on branch `embedding-api`. Phase 3 pins this commit.

## Risks and points for the reviewer

- `column-hse-z` currently uses the 2D Cartesian `Field` path with halos. Moving it onto x slabs changes its decomposition. The values must not change, since `column_hse` claims "bitwise independent of the thread count and of the MPI decomposition". The test above checks this.
- Whether `align_ptot!` (ltau) and `gauge!` are differentiable enough for Phase 6 is not decided here, but their exact semantics (which height, which reduction) must be documented in the gate report because Phase 5/6 depend on them.
