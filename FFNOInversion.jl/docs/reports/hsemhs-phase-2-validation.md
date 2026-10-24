# HSEMHS integration, Phase 2: native vertical coordinates — validation

Specification: `docs/hsemhs-integration/phase-2-ffno-native-coordinates.md`. Baseline: FFNOML `07402ae` (branch
`inversions`). Environment: hyades login node, Julia 1.12.3 (node-local copy `/tmp/julia-1.12.3`; the stornext install
intermittently fails to map its shared libraries), the package's own Manifest, MPI.jl's bundled MPICH, `CXX=g++` (the
system OpenMPI wrapper `mpic++` that `~/loadgcc.sh` sets links `libhcoll`, which most nodes lack; the Wittmann test build
needs no MPI).

## Changes

| Area | Change |
|---|---|
| `src/Types.jl` | `VerticalAxis(kind, values)` with `kind` `:z_m` (m, strictly decreasing) or `:log10tau500` (dex, strictly increasing): top first, never reordered. `Grid3D(vertical, x, y)`; x, y finite and strictly increasing. Legacy `Grid3D(log_tau500, x, y)` builds a log-tau axis and rejects a decreasing vector with a migration message. Field `grid.log_tau500` removed; accessors `vertical_coordinate`, `native_depth`, `nlevels`, `log_tau500_axis`, `z_axis` (the last two throw naming kind and unit), `vertical_unit`. `Atmosphere3D` gains a 3-D `logtau500`; `check_coordinate_consistency`. |
| `src/Nodes.jl`, `src/Gradients.jl` | `NodeField(values, nodes, kind=:log10tau500)`; `expand_nodes` (global and tile) and the node-expansion VJP interpolate on `native_depth` and throw on a kind mismatch. |
| `src/Solvers.jl`, `src/ScalableSolvers.jl` | `ControlMapSpec.nodes` + `kind`; layout signature includes kind and unit (checkpoints of another kind are rejected); `refine_control_maps(...; nodes=...)` interpolates in the controls' own coordinate; diagnostics write `nodes`, `node_kind`, `node_unit`. |
| `src/IO.jl` | `[atmosphere] vertical_coordinate = "z" \| "log10tau500"` (legacy inference: `datasets.logtau500` without `z_m`); z requires `z_m`, `x_m`, `y_m` and forbids `logtau500`; log tau requires `logtau500`, forbids `z_m`, `x_m`/`y_m` optional (both or neither). Controls take exactly the matching key (`z_nodes_m` or `log_tau_nodes`). `[grid] dx_m/dy_m` optional with explicit axes (spacing read from the file; both given must agree to 1e-9). Non-uniform or decreasing x/y rejected. `[regularization.vertical] length_scale_m` / `length_scale_dex`; types 1/4 on a z grid without `length_scale_m` are an error. `[parallel] decomposition = "x_slabs"`, `min_columns_per_rank` (default 2). Legacy positional `ControlMapConfig` constructor kept (log tau). `dry_run_summary` prints `vertical_coordinate`, `vertical_dataset`, `nodes_unit`. |
| `src/Execution.jl` | Reads the native axis and explicit x/y (origin preserved, length and spacing checked); bottom-first input is reversed once, axis and every 3-D field together; native z also sets `atmosphere.z` (the broadcast axis). Output writes the native axis (`z_m` or `logtau_500`), `x_m`, `y_m`, the 3-D `z` and `logtau500` when present, attributes `vertical_coordinate`, `vertical_unit`, `vertical_order = "top_first"`, `native_axis_dataset`. The reference atmosphere ID hashes the axis kind and values. |
| `src/Regularization.jl`, `src/DistributedForward.jl` | Vertical penalties use the native axis ÷ length scale (`_vertical_spacing_axis`); with log tau and the default 1 the penalty is bit-identical. Distribution and gathering carry `VerticalAxis` and `logtau500`. |
| `src/Parallel.jl` | `ParallelOptions.min_columns_per_rank`; `process_grid(nranks, nx, ny, :x_slabs)` = `(nranks, 1)`, rejecting fewer than `min_columns` x columns on any rank (computed identically on every rank); `local_tile` uses the configured decomposition; `initialize_parallel` accepts `x_slabs`. |
| legacy force balance (`ForceBalance.jl`, reconstruction in `DistributedForward.jl`) | Mechanical switch to `log_tau500_axis(grid)`; still log-tau only, deleted in Phase 3. |
| `scripts/preflight_olivia_inversion.jl` | Checks the configured vertical dataset and explicit x/y datasets. |
| tests | New `test/unit_coordinates.jl` (106 tests, in `runtests.jl`), `test/mpi_coordinates_worker.jl`; three tests updated for renamed fields (`io_config.jl`, `force_balance.jl`, `phase6_gradients_solver.jl`). |
| ADRs | ADR-001 (vertical axis, top-first, accessors, coordinate fields, node kinds, horizontal axes), ADR-007 (length scales). |

## Results

**Full suite, baseline vs Phase 2.** Run with a harness that includes `runtests.jl`'s files in order in one process
and continues past a failing file (plain `Pkg.test()` stops at the first erroring file): `07402ae` — 23 of 24 files
pass, `muspel_phase3_integration.jl` errors because its atom file
`/mn/stornext/d9/data/harshm/multi3d/input/atoms/atom.h6_tiago2.yaml` does not exist (environment). Phase 2 — the same
23 files pass, the same Muspel error, and `unit_coordinates.jl` passes (106/106; `io_config.jl` 54/54 with the new
coordinate assertions). No test changed state. Afterwards the Muspel test's hard-coded atom path was corrected to the
real location `/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms` (overridable with `FFNO_ATOM_DIR`, as in
`scripts/validate_muspel_reference.jl`); it now passes (2/2), so all 24 original files pass.

**MPI validation scripts, baseline vs Phase 2** (`validate_phase4_topology`, `validate_phase4_restart`,
`validate_phase5_mpi`, `validate_phase6_mpi`, `mpi_hybrid_worker` on 1 and 4 ranks): all pass on both, and the printed
checksums, regularization values and objectives are **identical** (e.g. `MPI_PHASE4_OK checksum=5.458701579229605e12
reg=0.00040000000000000533`, `MPI_PHASE6_OK objective=5.17093124925378e-10` on 4 ranks) — the log-tau path is unchanged.

**Coordinate round trip** (`test/mpi_coordinates_worker.jl`, 1/2/3 ranks × {cartesian_2d, x_slabs} × {z, log tau}):
all pass; gathered atmospheres equal the input bitwise including 3-D `z` and `logtau500` and B; coordinate metadata is
identical on every rank (hash reduction); x-slab tiles span all of y.

**`test/unit_coordinates.jl` coverage** (all pass): axis rules (orders, length, finiteness, unknown kind); accessors
and wrong-kind messages with units; legacy constructor rejects a decreasing log-tau vector with the migration message;
decreasing x and y rejected; coordinate consistency; z-node interpolation exact for a linear profile on a non-uniform
z grid, clamped outside the nodes; z nodes on a log-tau grid and log-tau nodes on a z grid rejected (global and tile);
node-expansion transpose dot products ≤ 1e-12 for both kinds, control maps 1×1, 3×1, 3×2, non-square grid, serial and
x-slab tiles; z-unit change (m → mm with the length scale) leaves the penalty unchanged to 1e-12; missing
`length_scale_m` rejected; the log-tau penalty equals the hand-computed historical value exactly; config matrix
(z with logtau dataset, z without x_m, inconsistent dx, wrong node key, missing length scale, x_slabs, legacy example
configs parse as log tau); native-z input identical across both storage orders and both vertical orders, x origin
preserved, (z,y,x) → (z,x,y), non-uniform x rejected with the reason; output round trip of a native-z atmosphere with
a derived 3-D logtau500; reference ID and layout signature change with the coordinate kind; x-slab tiling and
rejection.

## Consumer audit (item 8)

| Module | Vertical coordinate use | Order assumption |
|---|---|---|
| `PopulationModels.population_features`, `RecordedPopulationModel` | 3-D `atmosphere.z` (required) | none (cube) |
| `FSDPService` (PREDICT/VJP payloads) | 3-D `z` as Float32 plus uniform `dx`, `dy` from `grid.x/y` | none |
| `Synthesis.formal_solve!`, `synthesize!` | `atmosphere.z[:, x, y]` per column | top first: integrates from k = nz (bottom) up to k = 1 |
| `ext/FFNOInversionMuspelExt` | 3-D `z` to `Muspel.Atmosphere3D`; `piecewise_1D_linear!` returns the intensity at index 1 | top first |
| `ProductionGradients` | `z` cotangent; `nlevels` | top first (formal-solver VJP) |
| `LinePhysics`, `ContinuumOpacity`, `EOS`, `Diagnostics`, `Observations`, `Objectives`, `ForwardModel` | none | — |
| legacy `ForceBalance`, reconstruction in `DistributedForward` | `log_tau500_axis(grid)` (log tau only) | top first now enforced; deleted in Phase 3 |

No consumer reads the native axis by position as log tau. The top-first convention (README rule 4) matches every
order-sensitive consumer.

## Deviations from the phase document

* The phase document asked for a grep audit of `log_tau500`; per the review it is replaced by targeted tests. The
  remaining `log_tau500` text in `src/` is the accessor `log_tau500_axis`, the legacy constructor argument and the
  legacy force balance (which uses the accessor).
* "Native-axis assumption sweep" is covered by the consumer audit above plus the wrong-kind tests of every function
  that interpolates or regularizes on the axis, not by calling every exported function with a z grid: most exported
  functions require a full forward model (populations, synthesis caches) that Phase 3 builds.
* Decreasing x/y are rejected rather than normalized (as specified). Single-point x or y remain allowed (existing tests
  and the normalized control-map coordinates use them).
* For native z, `read_inversion_inputs` fills `atmosphere.z` with the broadcast axis already in Phase 2 (populations and
  synthesis need it); the derived `logtau500` is left for the Phase 3 equilibrium.

## Remaining limits

* Equilibrium physics still runs only on log tau (legacy backend); a native-z configuration reads, distributes,
  regularizes and writes correctly but cannot run a forward model until Phase 3.
* `[equilibrium]` and the removal of `pressure_top_*` are Phase 3.
