# Implementation brief for Claude: native z inversions and HSEMHS integration

## Objective and scope

Extend `FFNOInversion.jl` on the `inversions` branch to invert atmospheres on either geometrical height z or log10(tau500), with either column HSE or fixed-magnetic-field Poisson MHS reconstruction. Integrate the Julia package at `/Users/harshm/Documents/WorkRepo/1DHSE3dMHS/HSEMHS` in process, using the inversion's MPI ranks and Julia threads.

This document is an implementation specification, not a report of implemented functionality. Source inspection baseline: FFNOML `07402ae`, HSEMHS `b2401a3`, 24 October 2026. Recheck the interfaces against the current source before editing. The external repository is outside the current workspace's writable roots; changes there require the applicable filesystem approval or a separately authorized checkout.

Deliver all four combinations end to end, including forward evaluation, inversion controls, gradients, output, restart, and MPI operation. Do not implement z support by relabeling a logtau vector. Keep `full-mhs-z`, which also solves for B, outside this scope.

**User definition of native z support:** the input atmosphere supplies x, y, and z coordinates and atmospheric fields sampled on that spatial grid, instead of an atmosphere sampled on logtau. Inversion nodes are placed at specified z positions and interpolated in z. A native-z input must not require a logtau dataset. Derived tau may be computed for downstream physics, but it does not replace z as the input or control coordinate.

## 1. Existing seams to replace or generalize

| Source | Current constraint / required work |
| --- | --- |
| `src/Types.jl` | `Grid3D` has a mandatory `log_tau500` vector; `Atmosphere3D` arrays are `(nz,nx,ny)` and `z` is optional. Introduce explicit native-coordinate semantics. |
| `src/Nodes.jl` | `NodeField.log_tau_nodes`, interpolation, and transpose interpolation assume logtau. Generalize all three together. |
| `src/IO.jl` | Config types, parsing, control initialization, and summaries assume logtau and top pressure. Add coordinate and equilibrium configuration. |
| `src/Execution.jl` | Input loading, grid distribution, output, and fingerprints assume a shared logtau vector. Update serialization and provenance. |
| `src/Regularization.jl` | Vertical spacing is taken from `grid.log_tau500`. Define coordinate-aware penalties and scaling. |
| `src/ForceBalance.jl`, `src/DistributedForward.jl` | Existing equilibrium implementation and distributed reconstruction. Introduce backend dispatch and replace call sites. |
| `src/ProductionGradients.jl` | `_reconstruct_trial!` directly calls the old distributed reconstruction. The force contribution is finite-differenced per control coordinate. |
| `src/Parallel.jl` | MPI lifecycle ownership exists, but initialization uses `MPI.Init()`; decomposition selection currently accepts only `cartesian_2d`. |
| `src/PopulationModels.jl`, `src/FSDPService.jl`, synthesis and extensions | Audit their actual coordinate contracts. Population inference explicitly requires the geometrical-height cube; do not assume it requires a shared logtau grid. |

HSEMHS seams:

- `Parallel.jl`: `ensure_mpi`, `init_parallel`, `ParCtx`; currently builds from `MPI.COMM_WORLD`.
- `Slab.jl`: `slab_par`, distributed reductions, halo operations, and distributed FFT; currently uses world size.
- `PoissonTau.jl`: `column_hse_tau`, `solve_tau`, internal Jacobian/transpose operators.
- `PoissonZ.jl`: `solve_pressure`, `lorentz`, `tau_columns`.
- `ColumnHSE.jl`, `RunColumnHSE.jl`: extract the height-column numerical path from the run driver.
- `RunModes.jl`: useful mode orchestration reference; do not call file-oriented drivers on each inversion evaluation.
- `eos/Table.jl`: `state(tab, lnT, lnP)` returns `(lnrho, lnne, lnchi500, status)`.

## 2. User-facing configuration

Implement this conceptual schema; align type names with repository conventions:

```toml
[atmosphere]
vertical_coordinate = "z"          # "z" or "log10tau500"
storage_order = "python"

[atmosphere.datasets]
x_m = "x_m"                       # native-z input x coordinates, metres
y_m = "y_m"                       # native-z input y coordinates, metres
z_m = "z_m"                       # required for native z
# logtau500 = "logtau_500"         # required for native logtau
temperature = "temperature"
vx = "vx"
vy = "vy"
vz = "vz"
Bx = "Bx"
By = "By"
Bz = "Bz"

[equilibrium]
backend = "HSEMHS"
model = "mhs"                     # "column_hse" or "mhs"
gravity_m_s2 = 274.0
eos_table_file = "inputs/eos_table.h5"
on_nonconvergence = "reject_trial"

[parallel]
enabled = true
decomposition = "x_slabs"

[[inversion.controls]]
variable = "temperature"
z_nodes_m = [-200000.0, 0.0, 300000.0, 700000.0]
control_nx = 8
control_ny = 8
lower = 2500.0
upper = 12000.0
scale = 1000.0
```

The example illustrates new fields; retain the existing required inputs, regions, outputs, and solver sections in runnable examples. Existing logtau configurations keep `log_tau_nodes`. Require exactly one matching node-coordinate key; reject `z_nodes_m` for logtau controls and vice versa. Infer legacy logtau coordinate only when the new field is absent and the old schema is unambiguous. Preserve the legacy backend during migration with explicit selection and regression tests; document the final default rather than silently changing old run physics.

Derive the mode, without a second contradictory user-selectable mode key:

**User-required override:** after controls are applied, globally zero B selects the matching column-HSE mode even when `model = "mhs"`. Use a collective exact-zero check over finite owned B components, repeat it for every evaluation, and record requested/effective modes and the selection reason. A locally zero tile does not select HSE when B is nonzero elsewhere. This deliberate dispatch is distinct from switching to HSE after an MHS failure. See phase 4 and the review addendum for dispatch tests and derivative behavior at zero-B branch crossings.

| Native coordinate | Model | Derived HSEMHS mode |
| --- | --- | --- |
| z | column_hse | column-hse-z |
| log10tau500 | column_hse | column-hse-ltau |
| z | mhs | mhs-poisson-z |
| log10tau500 | mhs | mhs-poisson-ltau |

Add typed mode-specific options for pressure anchors, tau top treatment, height alignment/gauge, z or logtau cutoff, outer stopping rules, and inner linear-solver tolerances. Translate to actual HSEMHS option names in one place. Reject options irrelevant to the selected mode. Do not assume the existing `pressure_top_pa` has the same semantics as each HSEMHS anchor; verify each solver and explicitly implement, translate, or reject that boundary control.

## 3. Native coordinate model

Use an explicit coordinate descriptor, for example `VerticalAxis(kind, values)`, within `Grid3D`. Provide `native_depth(grid)`, `vertical_coordinate(grid)`, and explicit accessors that error on the wrong coordinate. Migrate scientific code away from direct `grid.log_tau500` access. Preserve the old constructor for logtau callers where practical.

For this implementation, native z is a common one-dimensional Cartesian height vector across columns, in metres, positive upward. Arbitrary per-column native z grids are outside scope. Keep derived coordinates as three-dimensional atmospheric fields:

Read native-z x, y, and z vectors from the input file and validate their lengths against the corresponding field dimensions after applying the declared storage order. Preserve their physical origins and spacing. Do not reconstruct x/y from `dx_m`/`dy_m` when explicit coordinates are supplied. Existing scalar grid spacing options may remain for legacy input; if also supplied with explicit coordinates, validate consistency. If an existing numerical operator requires uniform horizontal spacing, check that requirement and reject unsupported nonuniform input explicitly.

- Native z: `atmosphere.z[k,i,j]` equals the broadcast native z axis; derived `logtau500[k,i,j]` can differ between columns.
- Native logtau: the shared native axis is log10(tau500); reconstructed `atmosphere.z[k,i,j]` can be corrugated.

Do not collapse a column-dependent derived logtau field into a shared vector. Define one canonical vertical ordering at the adapter boundary and preserve the solver's expected ordering explicitly. Physical z increases upward, while a top-first solver array generally has decreasing z. Reverse every associated field together when needed.

Generalize node interpolation and its transpose to the native coordinate. Coordinate units must appear in config validation and errors. Update node initialization from the input atmosphere, bound checks, control-map distribution, and derivative paths.

Vertical regularization uses the native coordinate with explicit normalization scales: metres and logtau dex are not interchangeable. Retain legacy logtau penalty behavior by default. Add documented z length scales so a change in coordinate units cannot arbitrarily change regularization strength.

## 4. In-memory equilibrium API

Add a dedicated adapter such as `src/Equilibrium.jl` and an HSEMHS backend implementation. Suggested contract:

```julia
workspace = prepare_equilibrium(grid, tile, options, eos, context)
result = reconstruct_equilibrium!(workspace, atmosphere, boundary)
```

The precise API may differ, but enforce these responsibilities:

1. Preparation validates coordinate, topology, boundaries, EOS coverage, and available solver options collectively. Load the EOS table once per rank and cache communication plans and reusable numerical buffers.
2. Reconstruction accepts rank-local native T and B and boundary controls. Velocities and microturbulence remain inversion-controlled fields.
3. Return native-grid Pgas, rho, ne, z, derived tau/logtau, and structured status. Preserve native T and B for these fixed-B modes. Explicitly flag any scientifically required remapping rather than silently changing controls.
4. Use the same EOS table for equilibrium density, electron density, and chi500. Check and propagate EOS flags. Verify chi500's units and convention in the implementation before integrating tau; do not confuse extinction per length with opacity per mass.
5. Return iterations, stopping reason, objective/force measures, EOS violations, and validity independently. HSEMHS documents that process exit zero does not prove convergence.
6. No per-evaluation HDF5 round trip, external launcher, package activation, global environment mutation, or MPI lifecycle call.

Refactor HSEMHS drivers to wrap this API so standalone and embedded paths execute the same numerical kernels. Pin the external package revision in the integration environment; use a local development path only for development, not as the portable production dependency specification. Reconcile Julia/package versions and use the same MPI implementation and MPI.jl preferences in the single resolved environment.

## 5. MPI and threading ownership

FFNO owns initialization, finalization, job launch, and the overall phase sequence. Request and verify at least `MPI.THREAD_FUNNELED`; when MPI is already initialized, query and validate the supplied thread level. Only the initializing/main thread performs MPI calls.

Refactor HSEMHS `init_parallel` and `slab_par` to accept a supplied communicator. Embedded calls must never fall back to world behind the caller's back. Construct its Cartesian topology from an agreed parent communicator, optionally duplicated to isolate messages. Check local ownership against global ranges rather than assuming Cartesian rank ordering. Cache and free backend-owned communicators at preparation/teardown; do not finalize caller-owned MPI.

All participating ranks execute the same sequence:

```text
restore reference / apply controls
  -> equilibrium reconstruction
  -> population prediction / synthesis
  -> objective and regularization
  -> gradient phases, if requested
  -> collective trial acceptance or rejection
```

Keep HSEMHS collectives outside FFNO threaded column loops and outside the non-root GPU status-wait phase. Retain FFNO's rank-0 GPU service protocol. Reuse Julia's process-wide thread pool; create no second nested column thread pool. Allocate mutable EOS/synthesis scratch according to the actual thread scheduling and thread-ID contract. Set BLAS and FFT threading deliberately to avoid oversubscribing allocated cores.

For serial FFNO operation, provide a genuine local backend path or a deliberately managed single-rank MPI path. Do not accidentally call `COMM_WORLD` from `serial_context`. Document which path is implemented and test it.

## 6. Spatial decomposition strategy

Implement a shared x-slab path first: FFNO owns `(nranks,1)` horizontal process topology, complete vertical columns, complete y extent. Add `x_slabs` as an explicit decomposition choice. The HSEMHS Poisson modes and tau-column path already require this structure; z-column HSE can use the compatible process layout.

Verify compatible local extents, uneven x partitions, minimum columns per rank, PSF halo widths, regularization halos, and global GPU input assembly. HSEMHS currently has stricter minimum local-size rules than FFNO. Reject impossible allocations collectively with an actionable message.

Do not reuse FFNO halo semantics for HSEMHS blindly. HSEMHS uses periodic horizontal boundaries; FFNO's existing halo consumers include nonperiodic behavior. Each operator must retain its specified boundary convention. Changing the physical equilibrium boundary is a separate numerical feature, not a communication adjustment.

If 2D FFNO tiles are needed later, add cached tile-to-slab and slab-to-tile redistribution using distributed typed buffers and ownership intersections, with bounded memory and uneven partition support. Include the transpose redistribution in derivative paths. Do not gather the global volume on root for each solve. Porting the Poisson and FFT machinery to 2D tiles is a later alternative, not a prerequisite for the first integration.

## 7. Array and downstream contracts

FFNO uses `(vertical,x,y)` while HSEMHS slab kernels use `(vertical,y,x)`. Pack/unpack explicitly at the adapter boundary; do not reinterpret array storage. Test with nonsquare spatial dimensions and fields that vary independently along all axes. File layout is a separate contract and must not be inferred from kernel layout.

Audit population-model preprocessing, GPU payloads, synthesis, line opacity, diagnostics, and all extension code. Keep their existing physical input contract when it accepts z directly. If any component requires a fixed sampling grid, implement an explicit resampling operator and its derivative/transpose. Specify extrapolation or reject insufficient coverage. Never feed a trained model a new coordinate encoding merely because tensor sizes agree.

For z runs, integrate tau using the selected EOS opacity and documented upper-boundary treatment, then retain the column-dependent result. For logtau runs, retain the HSEMHS height gauge and alignment convention consistently. Verify whether alignment and pressure anchoring are compatible with any inverted boundary parameter before enabling that control.

## 8. Gradient correctness and solver state

Route forward reconstruction, `_reconstruct_trial!`, finite-difference references, and any prepared-state reuse through the same backend. Include backend configuration and coordinate identity in cache keys.

Initially preserve FFNO's finite-difference equilibrium contribution as a correctness reference. Every plus/minus evaluation must enter collectives in identical global order and start from the same restored state. Solver tolerances must make reconstruction noise small relative to the perturbation signal. Check directional derivatives across several step sizes; do not assume the existing default step is appropriate for MHS.

This reference costs approximately two global reconstructions per unconstrained control coordinate plus restoration; benchmark it before calling the integration production-ready. Do not drop pressure, density, ne, z, tau, alignment, or resampling derivatives to improve speed.

For scalable production gradients, implement a backend JVP/VJP. HSEMHS's internal `Jop!`/`JTop!` are useful components but are not a complete atmosphere-to-equilibrium derivative. Derive the derivative of the actual defining problem: for a stationarity equation `F(s,a)=0`, solve the transpose linearized system and propagate through inputs, boundaries, EOS, and mappings. A least-squares equilibrium generally requires differentiating stationarity, rather than assuming the residual itself is exactly zero. State and validate any Gauss-Newton approximation.

Use deterministic initialization first. Add warm starts only with explicit committed/trial workspaces: rejected trials cannot mutate committed state, and restart must reproduce the initialization policy. Gauge changes, total-pressure alignment, cutoffs, clipping, and active bounds may introduce nonsmooth behavior; document and validate their treatment.

## 9. Failure handling, output, and restart

Distinguish invalid physics, nonconvergence, and infrastructure failure. An invalid or unconverged trial follows the configured collective rejection policy; an invalid initial atmosphere reports a clear collective error. Do not silently switch MHS to HSE. Collectively propagate known local validation failures before entering the next collective. Do not claim a reduction can recover from a rank that has already failed inside a blocking MPI operation; retain the job-level failure boundary.

Update input/output and checkpoint schemas to record native coordinate kind and values, derived coordinate fields, mode, boundary parameters, gauge/alignment, gravity, solver tolerances/status, backend revision, and EOS table identity/hash. Include these in restart compatibility checks and fingerprints. Preserve existing SI conventions and storage-order options. Keep legacy logtau files readable where unambiguous. Restart should reject a mismatched coordinate or backend rather than reinterpret saved controls.

## 10. Implementation sequence and completion criteria

1. **Coordinate foundation:** generalize grid, nodes, interpolation transpose, regularization, config, IO, distribution, and fingerprints. Prove z controls interpolate on z and preserve legacy logtau behavior.
2. **Embedding foundation:** expose HSEMHS numerical API and injected communicators; add shared slab selection and backend lifetime management. Verify no hidden world collectives or per-evaluation setup.
3. **Column modes:** implement both z and logtau column HSE through the adapter, including derived coordinates and consistent EOS fields.
4. **MHS modes:** integrate Poisson-z and Poisson-ltau, native boundary controls, convergence status, collective rejection, and trial restoration.
5. **End-to-end inversions:** update population/synthesis contracts and gradient paths; supply runnable examples for all four coordinate/model combinations.
6. **Scalability:** benchmark reconstruction and gradient cost; implement/validate a scalable derivative path before making production-scale performance claims. Add tile/slab redistribution only if measured needs justify it.

Required meaningful validation:

- Compare embedded and standalone HSEMHS results from identical inputs/options for all four modes, including EOS fields, coordinate mappings, and status.
- Test z and logtau increasing/decreasing input order, nonsquare dimensions, explicit axis permutations, uneven slab partitions, and invalid rank counts.
- Use zero-B hydrostatic and manufactured magnetized cases with known or independently checked behavior. Compare with tolerances, not bitwise identity across rank counts.
- Run multiple ranks and thread counts, including one rank, and verify MPI lifecycle ownership and deadlock-free repeated objective/gradient calls.
- Check interpolation/resampling transpose dot products and objective directional derivatives for temperature, magnetic controls, and supported pressure-boundary controls.
- Exercise EOS out-of-envelope states, nonconvergence, rejected line-search trials, restored caches, and restart equivalence.
- Complete end-to-end forward and short inversion runs for all four modes. Run the existing MPI/GPU-service regression checks affected by the new phase integration.

Final Claude handoff should list actual changes in each repository, dependency revisions, runnable commands and example TOMLs, checks executed and results, measured reconstruction/gradient costs, and any remaining limits. Do not mark native z support complete while any active scientific path still interprets native height as logtau.
