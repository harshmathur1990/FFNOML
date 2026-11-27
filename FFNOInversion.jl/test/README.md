# FFNOInversion tests

The regression suite of FFNOInversion: what it runs, how to run it, and what each test checks.

- **Regression suite:** `test/runtests.jl`. It runs on one process with no GPU and no FFNO network: the FFNO is replaced
  by stand-ins. It needs the HSEMHS EOS table and builds the Wittmann EOS library.
- **Multi-rank and Olivia tests:** the `mpi_*` workers and the `olivia_*` probes. They run separately, see
  [below](#tests-outside-the-regression-suite).

**Rule (26 Nov 2026):** every code change is followed by the full regression suite before it is committed, pushed or
proposed as a PR. Every bug fix comes with a test that fails without the fix. A change only to docs is exempt, and the
PR says so.

## Running

```bash
cd FFNOInversion.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

On hyades:

```bash
source ~/loadgcc.sh
JULIA_PKG_USE_CLI_GIT=true CXX=g++ CC=gcc /tmp/julia-1.12.3/bin/julia --project=. --threads=8 -e 'using Pkg; Pkg.test()'
```

- `CXX=g++ CC=gcc`: `loadgcc.sh` sets MPI compiler wrappers, and they break the build of the Wittmann EOS library
  (`libwitt`) that several tests do.
- `JULIA_PKG_USE_CLI_GIT=true`: Julia's built-in git sometimes fails to fetch from GitHub ("packfile trailer mismatch").
- Node-local Julia (`/tmp/julia-1.12.3`): Julia started from StorNext can fail with "failed to map segment".

Inputs and timing:

- **The HSEMHS EOS table:** `HSEMHS_EOS_TABLE`, default `/mn/stornext/d9/data/harshm/hsemhs_data/cache/eos_table_fullz.h5`.
  Without it the HSEMHS tests are skipped.
- **The NLTE table:** `HSEMHS_NLTE_EOS_TABLE`. It is optional, and without it the NLTE equilibrium tests are skipped
  with a note.
- **Time:** the full suite takes about 30 to 40 minutes with 8 threads on hyades. The HSEMHS backend tests (7 min) and the
  given-density tests (4 min) take longest.

Running one file:

```bash
# only the equilibrium-adjoint tests
julia --project=. -t 16 test/run_equilibrium_adjoint.jl
```

For any other file, write a small runner: `using Test, FFNOInversion` followed by `include("test/<file>.jl")`. Files
that use the HSEMHS table need `include("test/hsemhs_equilibrium.jl")` first; it defines `HSEMHS_TABLE` and `H`.

## What the suite checks

**Last full run:** 27 Nov 2026 on hyades. 74 test sets and 16 134 checks passed, with branch
`soft-stop-restart`. The one skipped check is the optional comparison with STiC (STiC is not installed).

The count column is the number of checks in that run.

### Olivia setup and submission

| File | Count | What it checks |
|---|---:|---|
| `olivia_environment_setup.jl` | 42 | The Olivia scripts read as text, so they run on any machine. **Thread topology:** the number of threads per rank is set in one place (`SLURM_CPUS_PER_TASK`) and reaches Julia and the FFNO service. **Login preflight vs compute-node runtime:** they use separate Julia installations and depots (`julia-depot-login` vs `julia-depot-prod`), and the login-node script never reads the compute-node settings. **Per-job environment:** `prepare_olivia_environment.jl` builds the job's Julia project from the checkout, links `src/`, `ext/` and `scripts/`, and never overwrites a directory it did not make. |
| `submission_preflight.jl` | 44 | The check `submit_olivia_inversion.sh` runs before `sbatch`. It must refuse a bad configuration. For a good one it prints the full report: date, host and Julia; the FFNOML commit with the HSEMHS and WittBifrostEOS pins; mode, atmosphere shape and storage order, spectral regions and lines; the equilibrium with the **EOS model (LTE or NLTE)** and the **EOS table's path, size and date**; atmosphere datasets, population mesh, physics, view, grid, observation model, controls, solver, regularization and parallel settings; outputs and the Slurm layout; the configuration file word for word; and "Sanity check OK". **Saved report:** the real submit helper is run with a stand-in preflight and a fake `sbatch`. The report must be saved as `<run_dir>/preflight/preflight-<time>.log` and `preflight-<job id>.log`, with the sbatch arguments and the job number. A failed preflight must keep its log and not submit. |

### Basic types, coordinates and configuration

| File | Count | What it checks |
|---|---:|---|
| `unit_types.jl` | 4 | The basic types refuse invalid input: unsorted grids, negative temperatures, and Stokes components in the wrong order. |
| `unit_nodes.jl` | 4 | Control nodes are expanded onto the full 3-D grid: constant nodes give a constant field, and linear nodes give the expected sums. |
| `unit_coordinates.jl` | 113 | The two native vertical axes, height z in metres and log τ₅₀₀. Covered: the rules for each axis, coordinate consistency, node interpolation on the native axis, and a dot-product check of its transpose. Also: regularization length scales, configuration rules, reading native-z input in either storage and vertical order with explicit x/y, output round trips and fingerprints, and the split of the box into x slabs. |
| `atom_lines.jl` | 21 | The FFNO lines are found from the spectral region and the atom files instead of a hand-written table. Checks H-alpha, Ca II 8542, 8662, 8498, H and K, and Lyman alpha (vacuum wavelengths below 200 nm). A region holding two lines gets both, and the line centres match the old table. |
| `io_config.jl` | 54 | Reading the configuration file. Covers dataset names, vertical axis, storage order, equilibrium mode and alignment, observation and weight files, and outputs. Also writing and reading back checkpoints, and the one-line run summary. |
| `parallel_runtime.jl` | 20 | The parallel runtime without MPI: reductions and broadcast on one rank, and the process grid (16 ranks on 200×200 gives 4×4). Tiles cover every column exactly once, a local atmosphere is the right part of the global one, and gather and halo padding work. |
| `extension_contracts.jl` | 2 | Placeholders for future PRD and full-Stokes work: the output cube has the right shape, and an unsupported combination is refused instead of silently computed. |

### Equation of state and continuum opacity

| File | Count | What it checks |
|---|---:|---|
| `wittmann_production.jl` | 11 | The Wittmann EOS library builds and loads. At photospheric points its ρ and nₑ agree with the HSEMHS equilibrium table, which is built from the same Wittmann source: ρ within 10⁻³ and nₑ within 2 % (relative). |
| `continuum_opacity.jl` | 13 | The Julia continuum opacity gives finite, positive density, electron density, partial pressures and extinction. If the STiC source is present, it matches STiC's `cop.cc` to 2×10⁻¹² (relative). That comparison is skipped otherwise. |
| `continuum_source.jl` | 25 | Every region uses one continuum: Wittmann's, with hydrogen from the EOS. It never reads the FFNO hydrogen populations, and it does not depend on the order of a region's sources. |
| `continuum_adjoint.jl` | 51 | The Wittmann opacity at one depth depends only on that depth. So its derivative is computed by perturbing and recomputing one level, and must equal perturbing and recomputing the whole column. The EOS backend is reused, not rebuilt per call. |

### Populations (FFNO side)

| File | Count | What it checks |
|---|---:|---|
| `population_backend.jl` | 8 | FFNO inputs: six channels in a fixed order (T, vx, vy, vz, log nₑ, log ρ). A recorded prediction replays exactly. A changed atmosphere or inconsistent metadata is refused. |
| `population_mesh.jl` | 240 | The population mesh shared by all columns, for log τ input. **Inside a column**, interpolation is exact for quantities linear in z. **Below a column**, it is hydrostatic with a fixed d ln T/d ln P. **Back to the column**, it is exact for populations log-linear in z. Its derivative matches central differences in T, velocities, nₑ, ρ and heights, and a native height grid skips the mesh. **Explicit and adaptive meshes:** the configuration and defaults, the bounds, and the grading (neighbouring spacings differ by at most 1.2×). The adaptive criterion holds on every step above the minimum spacing, and the mesh is deterministic. A steeper atmosphere gets more heights and a looser criterion fewer. The gradient uses the mesh of its own prediction. `extend = false` stays within the columns' own heights, and an explicit mesh equal to the adaptive one gives the same populations. |
| `z_mesh.jl` | 174 | The mesh on a native height-grid atmosphere: on by default and off on request. The populations come back on the input's heights, and columns are never sheared. **Corona above a column's top:** a transition region to the coronal temperature, hydrostatic, with nₑ and ρ from the EOS. Its tangent and transpose agree, and a column already hotter than the corona keeps its temperature. |
| `population_limits.jl` | 31 | The `[physics]` options `ffno_population_limit` and `ffno_no_inversion`. Off, the raw values are used. With the conservation limit, every level is scaled so the total stays at the limit. With no inversion, a capped line gets zero opacity. Line opacity is never negative. The Ca limit is smaller by its abundance, and the gradient matches finite differences in every level and in ρ. |
| `given_populations.jl` | 38 | `[population.given]`: populations read from a MULTI3D file, which may be mirrored, rotated or bottom first. The preflight must find the right orientation among the eight possible, use a configured one even when another correlates better, and refuse ambiguous cases. Populations from an HDF5 forward output work too. When every species is given, no FFNO service is started. |

### Synthesis (the forward spectrum)

| File | Count | What it checks |
|---|---:|---|
| `phase3_synthesis.jl` | 29 | **Mixed synthesis:** FFNO and other opacity contributions add before one formal solution. **Regions:** each region is computed on its vacuum grid, the output axis is the regions' air grids joined, and the gradient equals the per-region gradients. **Sub-stepping:** an optically thick step across a jump (a hot transparent top over cool opaque gas) is sub-stepped to match a finely resolved reference, while smooth columns are unchanged. **Log τ grid:** with a source function linear in τ₅₀₀ the emergent intensity is exact. |
| `muspel_phase3_integration.jl` | 14 | The production adapter to Muspel's line profiles. Outside Muspel's table the Voigt function is the exact Faddeeva function. The old linear extrapolation went negative for large damping and gave negative line opacity in collapsed columns. |
| `phase4_integrated.jl` | 24 | The whole forward chain: equilibrium, then populations, synthesis and observation. If the equilibrium is invalid, the rest is skipped and the previous spectrum is kept. |
| `mixed_region_units.jl` | 122 | **Kurucz lines inside FFNO regions:** one emissivity unit for every model, RH's damping defaults, and no FFNO line counted twice. Covers the output unit and region normalization, observations read in the output unit, and microturbulence in the FFNO line opacity. |
| `vz_positive.jl` | 25 | `vz_positive = "down"` changes only the sign of vz in the files, and its control bounds. z stays positive upward and τ increases downward. |
| `view_angle.jl` | 402 | Viewing angle μ < 1. Covers the geometry and the line-of-sight Doppler shift, and the periodic shear of the box, with a dot-product check of its transpose (292 cases). Also the slanted path length and footprint synthesis on height grids, the population mesh on inclined rays, and configuration and inputs. The full gradient off the vertical matches finite differences for both axes and both geometries (plane-parallel and inclined). μ = 1 takes the vertical code path exactly. Doppler sign: positive v, toward the observer, is a blue shift in every line model. |

### Equilibrium (HSEMHS)

| File | Count | What it checks |
|---|---:|---|
| `hsemhs_equilibrium.jl` | 13 519 | The HSEMHS backend on one rank, with the real EOS table. Covered: configuration; FFNO's wrapper against direct HSEMHS calls for every mode; hydrostatic physics with B = 0 on the height grid; and the input's vertical order. Also: failures and provenance, the forward model with HSEMHS, and 20 prepare/solve/close cycles without leaks. **Given electron density:** 13 454 point checks that a given nₑ gives HSEMHS's state at (T, nₑ). The NLTE-table tests run only if that table exists. |
| `log10m_mode.jl` | 122 | The column-mass axis (log₁₀ m): axis rules, configuration and input, and the formal solution and its derivative on the mass axis. `column-hse-logm` through HSEMHS gives P = g m, and P = μ g m along the line of sight. The full gradient matches finite differences at μ = 1 and 0.8, and a wrong grid is refused. Also a given nₑ alone on the mass axis (EOS at T, nₑ), and a given atmosphere (nₑ and hydrogen populations) with no HSEMHS at all. |
| `given_density.jl` | 600 | A given atmosphere with nₑ and ρ and no HSEMHS. The input ρ and nₑ reach the FFNO unchanged, and hydrogen comes from the EOS at (T, P). Run for each vertical axis and each combination of given fields, with the LTE and the NLTE EOS. |
| `equilibrium_adjoint.jl` | 33 | The equilibrium adjoint in the gradient, each part against central differences. **Regularization:** its analytic gradient, for each order, kind and viewing angle. **HSEMHS equilibrium adjoint:** checked for all three axes, with and without B. **The full gradient** matches finite differences of the equilibrium and the full finite-difference oracle (to 10⁻⁵ and 10⁻⁴). **First gradient after a prepared solve:** the model factory's prepared equilibrium is not recorded, so the adjoint must solve again. This check reproduces Olivia job 3175806 and fails without the fix in `objective_gradient!`. |
| `nlte_eos_synthesis.jl` | – | Synthesis with the NLTE EOS: the continuum hydrogen and the neutral H in broadening come from the NLTE table, and the Kurucz metals are in LTE at the atmosphere's nₑ. Their adjoints and the full gradient are checked against finite differences. Runs only with the NLTE table. |
| `muspel_eos_continuum.jl` | 26 | The Muspel continuum with hydrogen from the EOS at (T, P), in the forward and the inversion. No FFNO population enters it, and the gradient matches finite differences. |

### Inversion (gradients and optimizer)

| File | Count | What it checks |
|---|---:|---|
| `phase5_inversion.jl` | 31 | Optimizer bookkeeping, with a simple ideal-gas equilibrium (`test_equilibrium.jl`). Covers bounded controls and refining the control maps. **Prepared solve used once:** the factory's prepared solve is used once, on the first evaluation, then never again. Recovery of a known atmosphere from its own spectrum, a directional derivative, and a restart from a checkpoint. |
| `phase6_gradients_solver.jl` | 22 | The matrix-free node gradient against finite differences. The bounded L-BFGS recovers a known answer, rejects infeasible trial steps without corrupting its state, and restarts. |
| `phase6_production_gradient.jl` | 9 | The production gradient back through the observation (PSF, weights) and the formal solver, including sub-stepped steps where χ and η jump by e³ to e¹⁴. Then the complete objective with mixed NLTE and Kurucz lines. |
| `correction_mapping.jl` | 17 | Controls are corrections to the starting atmosphere. Zero corrections reproduce it exactly, a correction never accumulates across trials, and a node bump that would break a bound is rejected. Finite differences at a bound use one-sided steps, and the production adjoint includes this mapping. |
| `soft_stop_resume.jl` | 66 | Soft stops and resuming. A stop file or the job deadline ends the run cleanly at an iteration boundary, with the checkpoint saved and the outputs written; resuming continues exactly as an uninterrupted run; resuming a converged run does not iterate again; `--finish` writes the outputs of a checkpoint without iterating; the `[solver]` keys and the batch script's job-end export. |
| `integration_mock_forward.jl` | 3 | A deterministic mock forward model: two runs give identical results, the second allocates no more than the first, and a perfect fit has zero residual. |
| `observation_regularization.jl` | 17 | The PSF applied on the full grid, wavelengths and pixels with zero weight being excluded, and the regularization penalty: zero for a constant atmosphere, positive for a gradient, refused for an unsupported field. |

### Production structure and the FFNO service

| File | Count | What it checks |
|---|---:|---|
| `phase6_execution.jl` | 28 | A production inversion is only accepted with a multi-GPU FSDP population service. **The executable path:** two inputs and two outputs. Forward mode needs no observation or controls, ignores optimizer settings, and equals an inversion with zero iterations. |
| `phase6_fsdp_protocol.jl` | 15 | The protocol of the persistent FFNO service, using a stand-in server: requests, a composite shutdown shared by several models, and the order of model requests. **A service that dies mid-request** (as in job 2622023): the prediction error is reported, and cleanup still closes the socket, the launcher, the log and the equilibrium. |
| `fsdp_only_architecture.jl` | 17 | Reads the source code: there is exactly one production FFNO backend, the FSDP service (full sharding, multi-GPU required, protocol version 2, a gloo control group). The old PythonCall route is gone. |

### Input formats

| File | Count | What it checks |
|---|---:|---|
| `iris_atmosphere.jl` | 43 | A small IRIS/MURaM FITS cube written and read back. The top of each output column is the file's last level at reversed y. The temperature floor (3250 K by default) changes only T, the mesh can be turned off, `[grid]` must agree, and bad inputs are refused. |
| `multi3d_atmosphere.jl` | 56 | A small MULTI3D mesh and `atm3d` written and read back. Fields are converted to SI with nothing flipped; with populations, the last level is the protons and ρ stays as given. Covers the temperature floor and `vz_positive`, a falling y mesh (only the spacing is used), `[grid]` agreement, and refusals. |

### Helper files (no tests of their own)

- `helpers.jl`: shared fixtures.
- `test_equilibrium.jl`: an ideal-gas hydrostatic stand-in for the equilibrium. Each rank uses the same formula for its
  columns, so results do not depend on the decomposition. It can be made to fail on every rank above a temperature, to
  test failure handling.
- `cop_oracle.cpp`: the bridge to STiC's `cop.cc` for the optional comparison.

## Tests outside the regression suite

These need several MPI ranks, a GPU or Olivia. Run them when the code they cover changes.

| File | How to run | What it checks |
|---|---|---|
| `mpi_hybrid_worker.jl` | `mpiexec -n 4 julia --project=. --threads=2 test/mpi_hybrid_worker.jl` | Threads touch only rank-local memory, and the forward on rank-owned tiles reassembles the same cube for any decomposition. |
| `mpi_coordinates_worker.jl` | `mpiexec -n R … test/mpi_coordinates_worker.jl` (R = 1, 2, 3) | Distribute and gather round trips of native-z and native-log τ atmospheres, with their 3-D coordinate fields. |
| `mpi_equilibrium_worker.jl` + `mpi_equilibrium_compare.jl` | `mpiexec -n R … test/mpi_equilibrium_worker.jl OUT` for R = 1, 2, 3, then `julia --project test/mpi_equilibrium_compare.jl OUT` | The HSEMHS column modes across ranks, with NX = 11 for uneven splits. 1, 2 and 3 ranks must agree. Also failures seen by every rank, and repeated calls. |
| `mpi_view_worker.jl` | `mpiexec -n R … test/mpi_view_worker.jl OUT` (R = 1, 2, 3) | An inclined view on a height grid, with shifts that cross several ranks. The spectrum, gradient and HSEMHS τ along the line of sight must agree across rank counts. |
| `mpi_phase4_worker.jl`, `mpi_phase4_failure_worker.jl`, `mpi_phase4_restart_worker.jl` | `scripts/validate_phase4_topology.jl`, `scripts/validate_phase4_restart.jl` | The forward on 1 vs 4 ranks. Rank 0 gets the complete FFNO input before scattering populations back. An equilibrium failure on one rank is the verdict on all ranks, and a restart works across topologies. |
| `mpi_phase5_worker.jl` | `scripts/validate_phase5_mpi.jl` | The inversion on 1 vs several ranks gives the same result. |
| `mpi_phase6_worker.jl` | `scripts/validate_phase6_mpi.jl` | The production gradient path: gather, adjoint on rank 0, scatter. Only rank 0 owns the model. |
| `mpi_gpu_status_timeout_worker.jl` | `scripts/validate_gpu_control_timeout.jl` | A GPU service that stops answering is detected by timeout. |
| `olivia_*.jl`, `olivia_*.py` | `scripts/run_olivia_runtime_tests.sbatch` on Olivia | The Olivia runtime chain. Covers MPI, GPU and NCCL/gloo probes, the FFNO service, the Phase 6 solver and memory, and deliberate CPU and GPU stalls with timeouts and recovery. `olivia_phase1_to_phase6_regression.jl` runs phases 1–6 on the compute nodes. |
| `ffno_jvp_cpu_check.py` | `python -m torch.distributed.run --nproc_per_node=2 FFNOInversion.jl/test/ffno_jvp_cpu_check.py` | The FFNO's forward-mode derivative (JVP) through the distributed H-slab adapters, on CPU with no GPU or FSDP. |

Other checks: `scripts/validate_eos_consistency.jl` (HSEMHS table vs Wittmann library), `scripts/validate_muspel_reference.jl`
(synthesis vs a Muspel reference), and the Python tests of the FFNO network and its service in the repository's `tests/`, such as
`test_batched_backend.py`, `test_height_operator.py` and `test_fsdp_service_launcher.py` (`python -m pytest tests`).
