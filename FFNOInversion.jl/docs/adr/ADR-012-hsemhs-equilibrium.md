# ADR-012: HSEMHS equilibrium backend

Status: accepted, 24 October 2026 (HSEMHS integration Phase 3). Supersedes ADR-003.

- **One backend.** Pressure, density, electron density and the derived vertical coordinate come from HSEMHS, called
  in process through `src/Equilibrium.jl` (`prepare_equilibrium`, `reconstruct_equilibrium!`, `close_equilibrium!`).
  `[equilibrium] backend = "HSEMHS"` is the only accepted value; the abstract interface admits test doubles only. The
  former force-balance backend (ForceBalance.jl, LogPressure.jl, Opacity500.jl, `HE3DBoundaryState`) is deleted.
- **Mode derivation.** The HSEMHS mode follows from the native vertical coordinate and `[equilibrium] model`:
  z + column_hse → column-hse-z, log tau + column_hse → column-hse-ltau (mhs → mhs-poisson-z/-ltau from Phase 4).
  There is no separate mode key. Sections for the other coordinate are rejected.
- **HSEMHS is the authority on the pressure boundary.** FFNO accepts no top pressure (`pressure_top_*` are rejected).
  Native z: `[equilibrium.column_z]` carries HSEMHS's `anchor`, `P_ref_pa` (required), `z_ref_m` and column options
  with HSEMHS names and semantics. Native log tau: no anchor; the τ-hydrostatic top closure sets the pressure;
  `[equilibrium.column_ltau]` carries `align`. (HSEMHS no longer caps the rise of ln P in the τ-grid column HSE;
  heights strictly decrease and a column leaving the EOS table is reported invalid.) HSEMHS's PoissonZ anchor damps rather than fixes the
  anchor-cell mean ln P; that is intended (user decision).
- **Coordinates.** Native z: `atmosphere.z` is the axis broadcast, `logtau500` is HSEMHS's column-dependent result.
  Native log tau: `logtau500` is the axis broadcast, `z` is HSEMHS's reconstructed height (aligned on P + B²/2μ₀ and
  gauged to mean z(τ = 1) = 0). Missing magnetic data means B = 0.
- **MPI and threads are FFNO's.** `initialize_parallel` initializes MPI with THREAD_FUNNELED (or verifies the supplied
  level), finalizes only what it initialized (serial contexts too), and sets BLAS and FFTW to one thread per rank
  (Julia threads parallelize the column/slab loops). A serial run is a managed single-rank MPI run; the backend uses
  `COMM_SELF`. `enabled = false` under a multi-rank launch is rejected. HSEMHS gets the inversion communicator (it
  duplicates it) and makes no MPI lifecycle or thread-count calls.
- **Decomposition.** HSEMHS solves on x slabs; multi-rank runs require `parallel.decomposition = "x_slabs"`, and the
  backend checks that HSEMHS's slab equals each rank's tile.
- **Failure contract.** `reconstruct_equilibrium!` returns a rank-identical `EquilibriumOutcome` and never throws for
  physics failures. On an invalid outcome `forward!` skips populations, synthesis and observation, keeps the previous
  spectrum and returns `valid = false`; forward mode raises a collective error; until Phase 4 the objective does too.
  Local preparation failures (missing table, wrong grid kind, provenance mismatch) are agreed collectively before any
  HSEMHS collective (HSEMHS's two-stage protocol) and raise the same `ArgumentError` on every rank.
- **EOS split.** Equilibrium ρ, nₑ, χ₅₀₀ come from the HSEMHS Wittmann table (one `state` call per node); synthesis
  keeps FFNO's Wittmann library. Both are built from the same Wittmann source and partition functions; the table uses
  tighter solver tolerances, FIX_GASC_H2 and a direct/blend/hot branch policy. `prepare_equilibrium(...; synthesis_eos)`
  rejects a table built with different partition functions or abundances and records the table provenance; the
  quantitative comparison is `scripts/validate_eos_consistency.jl`.
