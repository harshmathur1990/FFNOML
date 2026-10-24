# HSEMHS integration, Phase 3: equilibrium adapter, column modes, legacy removal — validation

Specification: `docs/hsemhs-integration/phase-3-adapter-and-column-modes.md`. Baseline: `000a737` (Phase 2).
HSEMHS: `f342f6b5e4058923ec0b171bcc8f3450644060a9` (branch `embedding-api`, pushed). Environment: hyades login node, Julia
1.12.3 (node-local `/tmp/julia-1.12.3`), the package Manifest, MPI.jl's bundled MPICH 5.0.2, `CXX=g++`.

## Changes

| Area | Change |
|---|---|
| Dependencies | `Project.toml`: `HSEMHS = {url = "https://github.com/harshmathur1990/1DHSE3dMHS.git", rev = "f342f6b5…", subdir = "HSEMHS"}`, FFTW as a direct dependency; Manifest resolved on Julia 1.12.3 from GitHub (no development path). |
| `src/Equilibrium.jl` (new) | `AbstractEquilibriumBackend`, `EquilibriumOutcome`, `EquilibriumConfig` and its `[equilibrium]` parser, `HSEMHSBackend` with `prepare_equilibrium` / `reconstruct_equilibrium!` / `close_equilibrium!` (ADR-012). |
| Legacy removed | `ForceBalance.jl`, `LogPressure.jl`, `Opacity500.jl`, the reconstruction half of `DistributedForward.jl`, `HE3DBoundaryState`, `ForceBalanceOptions`/`Diagnostics`, their exports and tests (`force_balance.jl`, `log_pressure.jl`). `WittmannEOS`/`thermodynamics!` (synthesis, EOS checks) and `IdealGasEOS` (fixture) kept. ADR-003 marked superseded. |
| Configuration | `[equilibrium]` required: `backend = "HSEMHS"` (only value), `model` (`column_hse`; `mhs` rejected until Phase 4), `gravity_m_s2`, `eos_table_file` (relative to the config), `on_nonconvergence`; `[equilibrium.column_z]` (z: HSEMHS `anchor`, `P_ref_pa` required, `z_ref_m`, column options) or `[equilibrium.column_ltau]` (log tau: `cap_per_dex`, `align`); the other coordinate's section, unknown keys and invalid values rejected. `pressure_top_pa` / `pressure_top_dataset` rejected ("set by HSEMHS"). Multi-rank runs require `decomposition = "x_slabs"`. |
| Execution | `InversionInputBundle(atmosphere, observation)`; `build_model(config, distributed, workspace, context)`; forward mode raises a collective error on an invalid equilibrium; the run closes the equilibrium backend (after the population service) in `finally`; equilibrium provenance (backend, mode, EOS table hash and metadata, final status) in the output provenance. |
| Forward model | `HybridForwardModel(populations, redistribution, synthesizer, observation, equilibrium, capabilities; prepared_equilibrium)`. `forward!` returns `(spectrum, equilibrium, timings, valid)`; an invalid equilibrium skips populations, synthesis and observation and keeps the previous spectrum. `HybridForwardTimings.equilibrium_seconds`. Trial restoration includes `logtau500`; `_reconstruct_trial!` uses the backend. |
| MPI and threads | `initialize_parallel` initializes MPI with THREAD_FUNNELED (verifies a supplied level), also for serial runs (single-rank MPI, backend on `COMM_SELF`); rejects `enabled = false` under a multi-rank launch; sets BLAS and FFTW to one thread. `finalize_parallel!` depends on `owns_mpi` only. |
| Population runtime (Python) | `ffno_fsdp_runtime.py` divides z by 1e6 before the model (see the population audit); z remains the autograd leaf in metres. |
| Examples, docs | `configs/example_intensity_nonprd.toml`, `examples/inversion_run/{forward,inversion}.toml` (`[equilibrium]`, `x_slabs`); `model_factory.jl` prepares the HSEMHS backend with the synthesis EOS and reuses the first reconstruction; `FFNO_TOP_DENSITY_KG_M3` removed; README/START_HERE updated; ADR-012. |
| Scripts | `scripts/validate_eos_consistency.jl` (new); preflight checks the EOS table instead of a pressure map; `validate_phase4_topology.jl` compares equilibrium verdicts and modes. |
| Tests | New: `test/hsemhs_equilibrium.jl` (single rank, real table), `test/mpi_equilibrium_worker.jl`, `test/test_equilibrium.jl` (deterministic hydrostatic test double for the interface, used where a test concerns orchestration, not equilibrium physics). Updated: phase 4/5/6 tests and MPI workers, configuration tests, Wittmann test (now compares the library with the HSEMHS table). |

## Results

**Full suite** (continue-on-error harness, all files of `runtests.jl`): all 24 files pass (including the Muspel adapter test and the new `hsemhs_equilibrium.jl`).

**`test/hsemhs_equilibrium.jl`: 59/59.** Configuration (mode derivation for both coordinates; `mhs` rejected with the
Phase 4 message; non-HSEMHS backend, missing `P_ref_pa` for z, the other coordinate's section, unknown keys
including `pressure_top_pa`, invalid `align`/`anchor` rejected; `pressure_top_pa` in `[atmosphere]`, `cartesian_2d` with
MPI and a missing `[equilibrium]` rejected by `load_config`). Adapter vs direct HSEMHS on a 40×12×7 atmosphere with T
varying independently along all axes: P, ρ, nₑ, z, log τ agree to ≤ 1e-13 for column-hse-z and column-hse-ltau (with
B); native coordinate field equals the axis, the derived one varies by column; T and B untouched. Hydrostatic physics
on a 400-level height grid: |dP/dz + ρg|/ρg < 2e-3 everywhere (centred differences of the HSEMHS column), top anchor
exact. Bottom-first and top-first native-z input files give bitwise identical pressure. Out-of-envelope temperature:
invalid outcome with `:eos_out_of_envelope`, fields untouched, the next valid solve recovers; forward short-circuits
and keeps the previous spectrum. Provenance: table partition-function hash equals the FFNO `scripts/pf_Kurucz.input`
hash; a synthesis EOS with the same partition functions is accepted, a modified file rejected; a missing table and a
grid of the wrong kind rejected before any solver state exists. 20 prepare/reconstruct/close cycles; close is
idempotent.

**`test/mpi_equilibrium_worker.jl`, 1/2/3 ranks, x slabs, NX = 11 (uneven):** 8/8 on each. Gathered P, ρ, nₑ, z,
log τ and spectrum versus one rank: column-hse-z **bitwise identical** on 2 and 3 ranks; column-hse-ltau bitwise except
z, 1.1e-13 relative (gauge sum). Out-of-envelope temperature on the last rank only: the same invalid verdict and
reason on every rank. 100 consecutive forwards alternating valid/invalid: identical verdicts, no hang. `enabled =
false` under `mpiexec -n 2`: rejected on both ranks.

**Existing MPI validation scripts** (`validate_phase4_topology`, `validate_phase4_restart`, `validate_phase5_mpi`,
`validate_phase6_mpi`, `mpi_hybrid_worker`; 1 and 4 ranks): all pass. The phase 5/6 objectives are identical to the
Phase 2 values (their fixtures use the test double and synthesizers that do not read pressure); phase 4 topology
parity holds between 1 and 4 ranks.

**Wittmann EOS compatibility (Q1)** — `scripts/validate_eos_consistency.jl`, 7200 points (3000–30000 K × 1e-3–1e5 Pa):

| region | ρ table vs library | nₑ table vs library |
|---|---|---|
| direct (T < T_sw = 15020 K), 5040 pts | max 3.2e-3, p99 1.4e-3, median 3.1e-6 | max 4.9e-6, p99 4.4e-6 |
| blend, 300 pts | max 6.8e-6 | max 1.9e-6 |
| hot, 1860 pts | max 5.6e-6 | max 1.5e-6 |

No point where the table is valid and the library fails. Provenance: same Wittmann source (SHA-256 `28f905a4…`), same
partition functions (`a7d15b25…`), abundances `wittmann_default`; the table's shim uses outer tolerance 1e-10,
5000 iterations and FIX_GASC_H2 = 1, the FFNO library the unmodified source. The direct-region ρ differences up to
0.3 % at cool, dense points are attributed to FIX_GASC_H2 and the tolerance difference (H₂-dominated region); they are
the expected size of the "same EOS, different build" difference, not interpolation error (table midpoint error
≤ 7e-7 in ln ρ per its metadata).

## Population-model input contract (item 7)

Read from `pipeline.py`, `FFNONet.py`, `models/ffno_model.py`, `ffno_fsdp_runtime.py` and the training mesh
`bifrost_data/en024048_hion/385/mesh`:

* **Units — a defect, fixed.** Training and offline prediction feed z in **Mm** (`FFNONet._normalize_z_scale`: z[m] /
  1e6) and dx, dy in **m**. The FSDP runtime used by the inversion passed z in **metres** straight to the model — a
  factor 1e6 in the first z feature channel (absolute z) for every inversion call. Fixed in `ffno_fsdp_runtime.py`
  (z / 1e6 inside the graph; gradients still per metre).
* **Convention.** The model's z features are: absolute z (Mm), z normalized to the column's own [min, max], |dz| over
  its column mean, and the column span. Training z: Bifrost/MULTI3D mesh, 467 levels, +14.37 Mm (top) to −1.02 Mm,
  top first, positive upward, non-uniform (Δz 18.6–97 km), z = 0 near the Bifrost photosphere.
* **Native-z runs** taking z from the same simulation family match the sign, order and zero point; HSEMHS z inputs
  (`input-mhs-hse.nc`: +2.98 to −0.49 Mm, 177 levels) are a **crop**, so z_norm, dz_norm and the span feature differ
  from training.
* **Native log-tau runs:** HSEMHS heights are gauged to mean z(τ = 1) = 0 (positive upward, top first) — the same
  sign/order and approximately the Bifrost zero point; the log-tau grids used so far (log τ −8 … 1.4) span roughly
  −0.1 to 2–3 Mm, again far less than the training span. (The removed legacy backend set z = 0 at the top and negative
  below, i.e. a different zero point and sign of the absolute-z feature as well.)
* No resampling, padding or offset is applied; `atm.z` stays the native axis (native z) or the HSEMHS height (log tau).
  **Decision (user, 24 October 2026):** FFNO3D is grid agnostic by design — the reason it was chosen is to generalize
  across vertical grids — so inversion columns with a different extent and spacing from the training columns are
  passed as they are. Only the unit (Mm) and convention (top first, positive upward, photospheric zero) must match,
  which they now do.

## Deviations from the phase document

* **eagle[7-8] forward smoke runs with the FSDP service not yet done.** The example run directory has no atmosphere,
  atom or EOS inputs; assembling a native-log-tau and a native-z input from the Bifrost data and launching the
  population service on eagle7's GPUs (no Slurm there) is a separate step, pending.
* **System OpenMPI not set up.** All MPI runs used MPI.jl's bundled MPICH on one node; installing `hpcx` into
  `~/packages` for `libhcoll` (system OpenMPI on every node) and multi-node runs are pending.
* Tests of phases 4–6 run on a deterministic hydrostatic test double (`test/test_equilibrium.jl`) instead of HSEMHS:
  they test orchestration (optimizer, FSDP protocol, PSF, gradients, restart), and the HSEMHS path is covered by its own
  single- and multi-rank tests with the real table. The deleted force-balance internals tests have no counterpart.
* The zero-B hydrostatic check uses the real Wittmann table and a centred-difference residual instead of an ideal-gas
  analytic profile (HSEMHS's ideal-gas test EOS is not accepted by the embedded API, which takes an `EOSTable`).
* The Phase 4 policy hook `on_nonconvergence` is parsed and recorded but not used yet; until Phase 4 any invalid
  equilibrium in the objective is a collective error.

## Remaining limits

* `model = "mhs"` (Poisson modes) is Phase 4.
