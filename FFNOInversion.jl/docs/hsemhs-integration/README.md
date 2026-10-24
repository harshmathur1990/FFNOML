# HSEMHS integration: phase plan

Parent specification: `../HSEMHS-integration-implementation-plan.md` (the "brief").
Source baseline: FFNOML `07402ae` (branch `inversions`), HSEMHS `b2401a3` (branch `main`, `git@github.com:harshmathur1990/1DHSE3dMHS.git`), 24 October 2026.

## Decision that changes the brief

The native force-balance backend in `FFNOInversion.jl` does not work and is **removed, not preserved**. This supersedes brief §2 ("Preserve the legacy backend during migration with explicit selection and regression tests"). Specifically:

- There is no `backend = "legacy"` option and no legacy regression suite. HSEMHS is the only equilibrium implementation.
- `ForceBalance.jl`, `LogPressure.jl`, `Opacity500.jl`, the reconstruction half of `DistributedForward.jl`, `HE3DBoundaryState`, `ForceBalanceOptions`, `ForceBalanceDiagnostics`, `reconstruct_force_balance!`, `reconstruct_force_balance_distributed!`, `lorentz_force!` and their tests (`test/force_balance.jl`, `test/log_pressure.jl`) are deleted in Phase 3.
- Old logtau TOML files keep parsing where unambiguous, but they run HSEMHS physics. Results are not expected to match old outputs, and no test asserts that they do.
- `[equilibrium] backend` is still required in config, but `"HSEMHS"` is its only accepted value. Keeping the key leaves room for a later backend without a schema change.

Everything else in the brief stands.

## Phases

| # | Document | Repository | Depends on | Delivers |
|---|---|---|---|---|
| 1 | [phase-1-hsemhs-embedding-api.md](phase-1-hsemhs-embedding-api.md) | HSEMHS | — | An in-memory, communicator-injected, status-returning API for all four modes. Standalone drivers rewired onto it. |
| 2 | [phase-2-ffno-native-coordinates.md](phase-2-ffno-native-coordinates.md) | FFNOInversion.jl | — | Vertical-axis descriptor, z/logtau nodes and interpolation (with transpose), regularization, config, IO, fingerprints, `x_slabs` decomposition. |
| 3 | [phase-3-adapter-and-column-modes.md](phase-3-adapter-and-column-modes.md) | both | 1, 2 | `Equilibrium.jl` adapter. FFNO-owned MPI with injected communicator. `column-hse-z` and `column-hse-ltau` running forward end to end. Legacy backend deleted. |
| 4 | [phase-4-mhs-modes.md](phase-4-mhs-modes.md) | both | 3 | `mhs-poisson-z` and `mhs-poisson-ltau` through the adapter. Convergence status, collective trial rejection, restoration. |
| 5 | [phase-5-end-to-end-inversions.md](phase-5-end-to-end-inversions.md) | FFNOInversion.jl | 4 | Finite-difference reference gradients through the backend, line-search rejection, checkpoint/restart schema, runnable examples for all four modes, eagle[7-8] regressions. |
| 6 | [phase-6-scalable-derivatives.md](phase-6-scalable-derivatives.md) | both | 5 | Cost benchmarks. Implicit-function VJP for the equilibrium, validated against the Phase 5 reference. Optional committed/trial warm starts. |

Phases 1 and 2 touch different repositories and share no code, so they can run in parallel. Phase 3 is the first point where the two meet.

```text
Phase 1 (HSEMHS API) ──┐
                       ├──> Phase 3 (adapter + column modes) ──> Phase 4 (MHS) ──> Phase 5 (inversions) ──> Phase 6 (VJP)
Phase 2 (FFNO coords) ─┘
```

## Rules that apply to every phase

1. **Native z is real z.** No scientific path may interpret a native height axis as logtau, or collapse a column-dependent derived logtau into a shared vector.
2. **Zero-B dispatch (explicit user requirement).** After applying controls, a globally zero magnetic field selects the matching column-HSE mode, including when `model = "mhs"`. This is intentional dispatch, not recovery from a failed MHS solve. Decide collectively on every evaluation; a zero local tile alone must not select HSE. Record requested and effective modes and the selection reason. Nonzero-field MHS failures never switch to HSE. An HSEMHS call never silently uses `COMM_WORLD`. Out-of-range or unsupported options raise errors instead of being ignored.
3. **Explicit layout at boundaries.** FFNO stores `(vertical, x, y)`. HSEMHS slab kernels use `(vertical, y, x)`, and the column-HSE `Field` is `(x, y, vertical)` with halos. All packing happens in one adapter, and tests use non-square grids with fields that vary independently along every axis.
4. **Canonical vertical order inside FFNO is top-first** (`k = 1` is the top, z decreasing, logtau increasing). This matches HSEMHS (`PoissonTau.read_tau_input` requires increasing logtau, and `PoissonZ` documents "z top first"). Inputs in the other order are reversed once on read, with every field reversed together. Phase 2 confirms the existing synthesis, formal solver and population paths agree with this order.
5. **Collectives are symmetric.** Every rank enters every MPI collective in the same order. Local validation failures, including programmer errors, missing files and hash mismatches, are caught and agreed on an already valid communicator before the next collective (two-stage protocol, Phase 1). `error()` is never called on a subset of ranks inside a collective sequence.
6. **Each phase ends with a validation report** in `FFNOInversion.jl/docs/reports/hsemhs-phase-N-validation.md` (and in `HSEMHS/reports/` for HSEMHS-side work). A report lists commits, commands run, test results with numbers, deviations from the phase document, and remaining limits. A phase is not complete until Codex has reviewed its report.

## Review history

- **Codex review, 24 October 2026** (`codex-review-2026-10-24.md`). Codex added the zero-B dispatch rule directly (README rule 2, brief, Phases 4–6). The other findings were checked against the source and incorporated:
  - (2) stationarity gate → Phase 1 status fields, Phase 4 `converged`/`valid_fields`/`accepted_approximate`/implicit eligibility, Phase 6 eligibility gate,
  - (3) anchor/damping semantics → Phase 1 documentation and drift test, Phase 4 anchor paragraph, Phase 6 adjoint derivation,
  - (4) collective-safe prepare → Phase 1 two-stage protocol, Phase 3 prepare and tests,
  - (5) EOS compatibility → Q1 below and Phase 3 item 8,
  - per-phase additions → the respective phase documents.
  - Corrections to the first draft that Codex identified: the `NX ÷ nr` slab check is mathematically correct (now only made collective); the EOS table is Hermite-interpolated, not piecewise-linear; `table_sha256` does not cover table metadata.

## Resolved cross-phase decisions

These were open questions in the first draft and have now been answered by the user (24 October 2026). The phase documents reference them by number.

- **Q1. EOS: both sides use the Wittmann EOS.** The HSEMHS `EOSTable` is built from Wittmann (`eos/WittmannDirect.jl`), and FFNO synthesis (Kurucz LTE via `WittmannKuruczState`, continuum) calls the Wittmann library. Equilibrium ρ, nₑ and χ500 come from HSEMHS. Synthesis-only quantities keep coming from FFNO's Wittmann library. The user's decision to use Wittmann on both sides stands. A shared partition-function file alone does not prove the HSEMHS table and FFNO's compiled library are equivalent, though: the table build has abundances, branch policy (`T_sw`, hot branch), fallback flags and `FIX_GASC_H2`, and `table_sha256` does not hash this metadata. So Phase 3 compares provenance, samples ρ/nₑ from both over the example envelope (including the branch transition), and tests failure propagation when synthesis rejects a state the table accepts (Codex review, finding 5).
- **Q2. HSEMHS is the final authority on the pressure boundary.** FFNO stops accepting a top pressure:
  - `[atmosphere] pressure_top_pa` and `pressure_top_dataset` are removed from the schema, and a config containing either is rejected with a message pointing to `[equilibrium]`.
  - `InversionInputBundle.pressure_top`, the distributed `pressure_field` and the `local_pressure_top` argument of `build_model` are deleted.
  - Pressure anchoring is configured only through HSEMHS's own options (`column_hse.anchor`, `P_ref`, `z_ref`, the Poisson `anchor_w`, the ltau top closure), passed through with HSEMHS names, defaults and semantics. FFNO adds no anchor semantics of its own and does not extend HSEMHS for this (the earlier per-column `P_ref` proposal is dropped).
  - The pressure boundary is not an inversion control.
  - This includes `PoissonZ`'s anchor. It damps the anchor-cell mean ln P per step rather than holding it fixed, and that is accepted as intended. No exact mean-pressure constraint is added (user decision after the Codex review, 24 October 2026).
- **Q3. Julia version: the cluster modules.** Each node group has its own environment script in the home directory:

  | Nodes | Script | Julia | MPI modules loaded |
  |---|---|---|---|
  | hyades, hyades[2-21] | `~/loadgcc.sh` | `julia/1.12.3` | `openmpi/gcc13/5.0.5`, `gcc/13.3.1` |
  | eagle[7-8] (NVIDIA GPU) | `~/loadnvidia_mpi.sh` | `julia/1.12.3` | `openmpi/gcc13/5.0.5`, `gcc/13.3.1` (plus the same `~/packages` HDF5/PnetCDF/NetCDF as `loadgcc.sh`) |
  | eagle9 (AMD GPU, 2× MI100) | `~/loadamd_mpi.sh` | `julia/1.12.3` | `openmpi/gcc13/5.0.5`, `gcc/13.3.1` (plus the same `~/packages` HDF5/PnetCDF/NetCDF as `loadgcc.sh`) |

  All three use Julia 1.12.3, which matches the FFNOInversion manifest. Both packages resolve in one environment on 1.12.3. HSEMHS (manifest currently 1.13.1) must use nothing newer than 1.12.
  - **MPI binary: system OpenMPI 5.0.5 everywhere we run MPI.** `~/loadnvidia_mpi.sh` (created 24 October 2026) is `loadnvidia.sh` plus the OpenMPI/gcc modules and HDF5/NetCDF exports from `loadgcc.sh`. On eagle7 it gives `/usr/local/opt/gcc13/openmpi-5.0.5/bin/mpiexec` (Open MPI 5.0.5), Julia 1.12.3, and the T4 GPUs visible. `~/loadamd_mpi.sh` does the same for `loadamd.sh`, and on eagle9 it gives the same OpenMPI 5.0.5, Julia 1.12.3 and the two MI100 GPUs. The original `loadnvidia.sh` and `loadamd.sh` are unchanged. hyades, eagle[7-8] and eagle9 therefore all use the same OpenMPI build, and MPI.jl is configured with `MPIPreferences.use_system_binary()` against it. The preference is identical on all node groups, so it can live in one `LocalPreferences.toml` generated by the environment-setup script. Both FFNO and HSEMHS use this binary.
  - **Hardware.** The existing FSDP GPU service is CUDA (NVIDIA). GPU production runs target eagle[7-8]. eagle9 (AMD MI100) runs every MPI/CPU test (equilibrium, adapter, mock-population forwards and inversions). FSDP-service runs there are out of scope until the population service supports ROCm.
- **Q4. FFNO controls MPI and threading.** FFNO initializes and finalizes MPI, owns the communicator handed to HSEMHS, and sets Julia, BLAS and FFTW threading. HSEMHS library code makes no MPI lifecycle calls and sets no thread counts: `__init__`'s `BLAS.set_num_threads(1)` and the drivers' BLAS toggles are removed from library paths in Phase 1. Standalone HSEMHS drivers may still set them in their own entry points.
