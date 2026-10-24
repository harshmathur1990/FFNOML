# Phase 2: FFNO native vertical coordinates

Repository: FFNOInversion.jl (FFNOML, branch `inversions`).
Brief sections: §1 (FFNO seams), §2 (atmosphere/controls schema), §3, §6 (`x_slabs`), §9 (schema parts not tied to equilibrium).
Depends on: nothing. Runs in parallel with Phase 1.

## Goal

Make the vertical coordinate an explicit, typed property of the grid and of every control. An atmosphere can then be read, distributed, parameterized, regularized, written and fingerprinted on either native z or native log10 τ500, with no path quietly assuming logtau. Equilibrium physics is not touched in this phase.

## Scope

1. **Vertical axis type** (`src/Types.jl`)
   - `struct VerticalAxis{T}`, holding `kind::Symbol` (`:z_m` or `:log10tau500`) and `values::Vector{T}`, strictly monotonic and top-first (README rule 4).
   - `Grid3D` holds `vertical::VerticalAxis{T}`, `x`, `y`. The constructor `Grid3D(log_tau500, x, y)` is kept and builds a logtau axis, so existing call sites compile. Because a grid constructor cannot reverse the atmosphere fields that go with it, the legacy constructor **rejects a decreasing (bottom-first) logtau vector** with a migration message ("reverse the vertical axis and every field together, or read through `read_inversion_inputs`, which normalizes"). Normalization happens only at boundaries that hold the fields (the input reader). A regression test calls the legacy constructor directly with both orders.
   - x and y must be finite and strictly increasing. Decreasing horizontal axes are rejected rather than normalized, because normalizing them would also require flipping observations, PSFs and spatial weights. The same applies to y as to x, and both are tested.
   - Accessors: `vertical_coordinate(grid)` returns the kind, `native_depth(grid)` returns the values, and `log_tau500_axis(grid)` / `z_axis(grid)` return the values or throw `ArgumentError` naming the actual kind and unit.
   - Remove the field `grid.log_tau500`. Every remaining use becomes an error fixed by choosing the correct accessor. Acceptance is by targeted tests of **native-axis assumptions**, not by grep: each scientific function that takes a grid is exercised with a z grid and must either work on z or throw the wrong-kind error (see "Native-axis assumption sweep" in Tests). Legitimate logtau accessors and config names remain.
   - `Atmosphere3D`: keep `z::Union{Nothing,A}` (3D) and add `logtau500::Union{Nothing,A}` (3D, derived). Invariant, checked by a helper `check_coordinate_consistency(atm)`:
     - native z: `atm.z` is `nothing`, or equals the broadcast axis exactly,
     - native logtau: `atm.logtau500` is `nothing`, or equals the broadcast axis exactly.

2. **Nodes and interpolation** (`src/Nodes.jl`, `src/Gradients.jl`)
   - Rename `NodeField.log_tau_nodes` to `nodes`, and add `kind::Symbol`. The constructor rejects mismatched kinds at use, not at construction: `expand_nodes(field, grid)` throws if `field.kind != vertical_coordinate(grid)`.
   - `expand_nodes` (both global and tile overloads), `_node_expansion_vjp_local!` and `node_expansion_vjp` interpolate on `native_depth(grid)`. Same piecewise-linear-with-clamp scheme as today.
   - Error messages include the unit (`m` or `dex`).

3. **Configuration** (`src/IO.jl`)
   - `[atmosphere] vertical_coordinate = "z" | "log10tau500"`. If absent and `datasets.logtau500` is present while `datasets.z_m` is absent, infer `log10tau500` (legacy). Any other absent or ambiguous combination is an error.
   - `[atmosphere.datasets]`:
     - z requires `z_m`, `x_m`, `y_m` and forbids `logtau500`.
     - log10tau500 requires `logtau500` and forbids `z_m`. `x_m`/`y_m` are optional; if absent, x and y come from `[grid] dx_m/dy_m` as today.
   - `[[inversion.controls]]`: exactly one of `z_nodes_m` or `log_tau_nodes`, matching the coordinate. The wrong key for the coordinate is an error naming both.
   - `[grid] dx_m/dy_m`: optional when explicit `x_m`/`y_m` are given. If both are given, they must agree with the coordinate spacing to 1e-9 relative.
   - `AtmosphereInputConfig` gains `vertical_coordinate`, `vertical_dataset`, `x_dataset`, `y_dataset`. `ControlMapConfig.log_tau_nodes` becomes `nodes` plus `kind`.
   - `dry_run_summary` prints `vertical_coordinate=… vertical_dataset=… nodes_unit=…`.
   - `[parallel] decomposition = "cartesian_2d" | "x_slabs"`.
   - `[regularization.vertical] length_scale_m` (z) / `length_scale_dex` (logtau, default 1.0), see item 5.
   - Equilibrium config (`[equilibrium]`) is **parsed and validated in Phase 3**, not here. Removing `pressure_top_*` (Q2) also happens in Phase 3, because the legacy backend that consumes it is deleted there.

4. **Input reading** (`src/Execution.jl: read_inversion_inputs`, `scripts/preflight_olivia_inversion.jl`)
   - Read the vertical, x and y vectors with the declared `storage_order`. Validate their lengths against field dimensions after the `(z,y,x)` to `(z,x,y)` permutation. Require uniform x and y spacing, and reject non-uniform input explicitly, because `FSDPService._spacing` and the PSF kernels require it.
   - Preserve physical origins: x and y come from the file, not `0:nx-1`.
   - Bottom-first vertical input is reversed to top-first, with every 3D field reversed in the same operation. A test checks a non-symmetric field.
   - Preflight mirrors the same shape checks.

5. **Regularization** (`src/Regularization.jl`)
   - `_vertical_penalty(a, axis, typ)` uses `native_depth(grid)` divided by a length scale: `length_scale_m` for z, `length_scale_dex` for logtau.
   - With a logtau axis and the default `length_scale_dex = 1.0`, results are identical to today's.
   - `length_scale_m` is required when vertical regularization types 1 or 4 are active on a z grid. There is no silent default, because metres and dex are not interchangeable.

6. **Control initialization and application** (`src/IO.jl: _sample_control_field`, `src/Solvers.jl`)
   - Sample at `nodes` on `native_depth`.
   - `apply_control_maps!`, `parameter_nodefield`, `refine_control_maps` and `_layout_signature` carry the node kind. The signature string includes the kind and unit, so a checkpoint from a logtau layout is rejected by a z layout.

7. **Distribution and output** (`src/DistributedForward.jl: distribute_atmosphere/gather_atmosphere`, `src/Parallel.jl`, `src/Execution.jl`)
   - Broadcast the `VerticalAxis` and the explicit x and y vectors. Distribute and gather the 3D `z` and `logtau500` fields when present.
   - `x_slabs` decomposition: `process_grid` returns `(nranks, 1)`, and each rank's tile spans all of y. Reject collectively if any rank would own fewer than `min_columns_per_rank` (config, default 2 to match HSEMHS) x columns. Verify that the existing halo users (PSF `exchange_halos`, regularization) work with `py = 1`.
   - `cartesian_2d` keeps working for now. Phase 3 makes HSEMHS modes require `x_slabs`.
   - `write_inversion_outputs` writes the native axis under its own name (`z_m` or `logtau_500`) plus 3D `z` / `logtau500` when present. File attributes: `vertical_coordinate`, `vertical_unit`, `vertical_order = "top_first"`, `x_m`, `y_m`.
   - **Native axis vs derived datasets.** The config key `[atmosphere.datasets] logtau500`/`z_m` names the *native input axis*. An output file may legitimately contain both a native `z_m` axis and a derived 3D `logtau500` (or the reverse). The reader identifies the native axis only from the `vertical_coordinate` attribute (when reading FFNO output) or from the config (when reading input), never from which datasets exist. A round-trip test writes a native-z atmosphere with derived logtau and reads it back as native z.
   - Coordinate metadata (axis kind, values, x, y) is identical on all ranks after distribution. Assert this with a hash allreduce.
   - `_atmosphere_reference_id` hashes the axis kind and values and x and y, instead of `grid.log_tau500`.

8. **Audit of other coordinate consumers.** Read and record (in the phase report) the vertical-coordinate contract of `PopulationModels.population_features` (needs 3D z), `FSDPService` payloads, `Synthesis.formal_solve!`, `ext/FFNOInversionMuspelExt.jl` (passes `atmosphere.z` to Muspel), `LinePhysics`, `ContinuumOpacity`, `Diagnostics`, and `Observations`. For each, state whether it consumes z, logtau, index or nothing vertical, and whether top-first is assumed. No consumer may read the native axis by position assuming logtau. Fix any that do.

## Out of scope

- Any equilibrium or HSEMHS code, and deleting the legacy backend (Phase 3). The legacy backend is broken but must stay **importable** in this phase: its uses of `grid.log_tau500` are switched mechanically to `log_tau500_axis(grid)`. No `@test_skip` is used to hide load errors. Record each legacy test's pass/fail state at `07402ae` before editing. After Phase 2, each one must have the same state (a legacy test that already failed may keep failing, but none may newly fail to load). Phase 3 deletes them.
- Gradients beyond the node-expansion transpose.
- Restart schema fields for equilibrium (Phase 5).

## Tests

New tests go in `test/unit_coordinates.jl`. `test/unit_nodes.jl`, `test/unit_types.jl`, `test/io_config.jl` and `test/observation_regularization.jl` are extended.

| Test | What it proves | Pass criterion |
|---|---|---|
| Axis validation | Typed axis | Non-monotonic, non-finite or length < 2 rejected. Wrong-kind accessor throws an error naming kind and unit. |
| Interpolation on z | Nodes really live on z | z nodes `[-2e5, 0, 3e5, 7e5]` m and a field linear in z expand exactly (≤ 1e-12) on a non-uniform z grid. The same node values interpreted on a logtau grid with identical numeric values throw. |
| Node-kind mismatch | No relabeling | `expand_nodes(NodeField(…, kind=:log10tau500), z_grid)` throws. |
| Interpolation transpose | Adjoint correctness | Dot-product test ⟨E v, w⟩ = ⟨v, Eᵀ w⟩ to ≤ 1e-12 relative, for z and logtau, global and tile overloads, ncx/ncy ∈ {1, 3}, non-square grids. |
| Legacy logtau parity | Nothing changed for logtau | For existing `test/io_config.jl` and `test/unit_nodes.jl` fixtures, expanded fields, control layouts and regularization penalties are bitwise equal to the `07402ae` results (stored before editing). |
| Config matrix | Schema rules | Every forbidden combination (z with logtau dataset, z with `log_tau_nodes`, logtau with `z_nodes_m`, missing `x_m` for z, inconsistent `dx_m`, z grid + vertical type 1 without `length_scale_m`) raises an error with a specific message. Legacy example TOMLs (`examples/inversion_run/*.toml`) still parse unchanged. |
| Native-z read | Real z input | A synthetic HDF5 with `z_m`, `x_m` (origin 1.2e6 m), `y_m` and no logtau dataset, in both storage orders and both vertical orders, gives identical in-memory atmospheres. x and y origins are preserved. A non-symmetric T profile ends up top-first. |
| Non-uniform x rejected | Explicit constraint | Non-uniform `x_m` errors with a message naming FSDP/PSF. |
| Regularization scale | Unit safety | On a z grid, scaling z by 1000 (m to mm) and `length_scale_m` by 1000 leaves the penalty unchanged to ≤ 1e-12. |
| Legacy constructor order | No silent reversal | `Grid3D(increasing_logtau, x, y)` works. `Grid3D(decreasing_logtau, x, y)` throws the migration message. |
| Horizontal axes | Validation | Decreasing, non-finite or non-uniform x and, separately, y are rejected with specific messages. |
| Native-axis assumption sweep | No logtau assumption | Every exported function that takes a `Grid3D` or `Atmosphere3D` (enumerated in the report) is called with a native-z grid. It either produces a result consistent with z (checked against a hand computation where it is numeric) or throws the wrong-kind error. A list of functions that legitimately require logtau is included. |
| `x_slabs` | Decomposition | For nx ∈ {8, 11}, nranks ∈ {1, 2, 3, 4}, tiles cover x exactly once and y fully. nx = 5 on 3 ranks with min 2 columns is rejected on all ranks. |
| Distribute/gather round trip | Axis and derived fields travel | MPI worker (extend `test/mpi_phase4_worker.jl` or add `mpi_coordinates_worker.jl`) on 1, 2, 3 ranks × {cartesian_2d, x_slabs} × {z, logtau}. The gathered atmosphere is bitwise equal to the input, including the 3D `z`/`logtau500`. |
| Output round trip | Schema | Write, then read back, gives axis kind, values, x, y, and the 3D derived fields. Attributes present. |
| Fingerprint | Provenance | Changing only the axis kind (same numbers) changes `_atmosphere_reference_id` and `_layout_signature`. |

The full existing test suite (`julia --project -e 'using Pkg; Pkg.test()'`) runs to completion with no load errors. Every test has the same pass/fail state as the `07402ae` baseline except tests that this phase intentionally changes, and those are listed in the report with a reason.

## Exit criteria

- All tests pass on 1–3 ranks locally.
- The native-axis assumption sweep, the legacy-test baseline comparison and the consumer audit (item 8) are in `docs/reports/hsemhs-phase-2-validation.md`.
- An ADR update: `ADR-001-coordinates-units.md` gains the vertical-axis and top-first decisions, and `ADR-007-vertical-regularization.md` gains the length-scale rule.

## Points for the reviewer

- Removing the `grid.log_tau500` field instead of keeping a deprecated alias is deliberate. An alias would let a z grid silently answer "logtau" questions.
- Whether `NodeField` should carry `kind` or be parameterized by a coordinate type: this document chooses a runtime `Symbol` to keep serialization and checkpoint compatibility simple. Either is acceptable if mismatches are rejected.
