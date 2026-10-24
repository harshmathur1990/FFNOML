# Force-balance diagnostics

Both reconstruction functions return these measurements in `ForceBalanceDiagnostics`.
They evaluate the final returned state, after best-iterate restoration or fallback.
They do not change pressure updates, convergence criteria, or acceptance decisions.
The distributed path also emits a `FORCE_BALANCE event=final_diagnostics` log line.

| Fields | Meaning |
| --- | --- |
| `force_residual_x`, `force_residual_y`, `force_residual_z` | Maximum normalized force mismatch along x-, y-, and depth-index edges. Their maximum agrees with `force_residual` for every field value, including zero. |
| `force_integrability_max`, `force_integrability_rms` | Maximum and RMS normalized circulation around physical grid faces. Nonzero circulation indicates that the prescribed discrete force cannot exactly equal a scalar pressure gradient on all edges. |
| `force_integrability_faces` | Number of faces contributing to circulation statistics. Each face is counted once across MPI tile boundaries. |
| `eos_density_relative_error`, `eos_electron_relative_error` | Maximum cell-wise relative discrepancy between stored density/electron density and a fresh EOS evaluation at the final temperature and gas pressure. |

For each edge from a to b, define q = F_L - rho*g*e_z. The directional
residual is abs(P_b - P_a - average(q) dot (r_b-r_a)) / (edge_length * scale),
where scale = max(g*max(abs(rho)), max(norm(F_L))) + machine epsilon.
The same whole-domain scale is used on every MPI rank.
Horizontal edges can have a vertical displacement on the corrugated mesh;
these are edge-direction diagnostics, not Cartesian force components at fixed z.

For a face, the compatibility statistic is abs(signed_sum(edge_force_integrals)) /
(sum(abs(edge_force_integrals)) + machine epsilon). RMS is weighted by face count,
not by rank count. A domain with no faces reports zero maximum/RMS and zero count.
Near-zero circulation is a necessary compatibility check, not proof of equilibrium
or EOS consistency. Small individual forces can still have large normalized circulation.

EOS errors use abs(stored - expected) / max(abs(expected), machine epsilon).
The expected arrays are diagnostic scratch arrays; the returned atmosphere is not
replaced by those arrays. This check can reveal inconsistencies from blending or
retaining earlier iterates, even when the force residual is small.

Validation covers manufactured corrugated HSE and MHS equilibria, an incompatible
force field, known EOS discrepancies, and distributed/whole-domain agreement.
`julia --project=FFNOInversion.jl FFNOInversion.jl/scripts/validate_phase4_topology.jl`
from the repository root also checks
one-rank/four-rank parity and faces crossing both horizontal tile seams.

## One MHS iteration

There is one solver and no preliminary HSE solve. `mode` always records `MHS`.
Initialization evaluates EOS(T,P), opacity and the optical-depth height integral
once at the supplied pressure (or a constant boundary-pressure initial guess).
It does not update pressure. The main iteration starts from that state.

Every iteration evaluates Lorentz force, vertical pressure seeding, and the
same coupled x/y/z pressure relaxation. There is no test of B values to select
updates, convergence criteria, or acceptance. Missing magnetic storage is
normalized to zero arrays for evaluating the same algebra.

For B=0, J cross B vanishes automatically, leaving grad(P)=-rho*g*e_z.
The horizontal equations remain dP/dx=dP/dy=0 at fixed physical height; independent
columns with horizontal pressure gradients are not solutions of this system.
All three directional residuals enter convergence for every B value.

Physical edge integrals on corrugated surfaces retain vertical force times the
edge's height displacement. A constant nonzero B also has zero Lorentz force.
`iterations` counts the single solver loop. Best-force state selection and
backtracking are the same for all B values. An improving but unconverged state
may still be returned; `accepted_3d` is not a physical acceptance criterion.

## Initial checkpoint

`initial_callback=snapshot -> ...` replaces the removed bootstrap callback.
It runs after initialization and before the first pressure update, receiving
independent copies of `pgas`, `rho`, `ne`, `z`, and initial force residual metadata.
In MPI it receives local arrays and failures are coordinated across ranks.
`bootstrap_iterations`, `bootstrap_*` diagnostics, and `bootstrap_callback` have
been removed. Historical run artifacts retain their original schemas and do
not represent the current implementation.

## Consistent trial experiment

`ForceBalanceOptions(enforce_trial_consistency=true)` blends pressure, then
recomputes EOS, opacity and height at that trial pressure. It defaults to false.
Initialization always evaluates dependent fields at its starting pressure;
that one-time evaluation is necessary to compute geometry and Lorentz force.
The experiment does not guarantee force balance or convergence.

## FIRTEZ-style log-pressure step

Pressure now uses u=ln(P/1 Pa) and h=(F_L-rho*g*e_z)/P_old. The continuum equation
is Laplacian(u)=div(h). The implementation adapts its divergence formulation to
our existing physical-edge grid: D' W D u = D' W I(h), where I(h) is the trapezoidal
line integral along each edge and W is inverse squared physical edge length.
Horizontal weights retain the existing continuation ramp. This graph projection
is not an exact metric discretization of FIRTEZ's Cartesian FISHPACK Laplacian.

There is no vertical pressure seed. Jacobi-preconditioned conjugate gradients
solves the coupled log equation with local MPI halos and scalar reductions.
The prescribed pressure plane is Dirichlet in u. Other exterior edges retain our
natural force boundary rather than FIRTEZ's six-face Dirichlet pressures.

The log candidate is clipped to [ln(0.5*minimum(boundary pressure)),
ln(max(log_pressure_max_pa,maximum(boundary pressure)))]. The default configurable
upper numerical bound is 1e6 Pa; it is not FIRTEZ's empirical bottom boundary.
Pressure backtracking blends logarithms and exponentiates. Dependent-field
blending and optional trial consistency remain as before.

`log_pressure_iterations=1000` and `log_pressure_tolerance=1e-8` control PCG.
The legacy neighbor-relaxation `pressure_sweeps`, `pressure_max_sweeps`, and
`pressure_tolerance` settings do not control the active log solve.
`last_pressure_iterations`, `last_pressure_residual`, `last_pressure_converged`,
and `last_pressure_clipped_cells` describe the last attempted inner solve, not
necessarily the returned best outer state. The PCG residual is recomputed before
clipping. Clipping may invalidate its equation; the outer physical force
residuals still determine force convergence. A converged divergence solve alone
does not prove the vector MHS equation.

This imports the log-pressure equation and log-space update, not FIRTEZ's
beta-dependent current limiting, derivative attenuation, empirical boundary
pressures, sensitivity-region rules, temperature/B remapping, or stopping test.
Source inspected: local `firtez-dz-mhs-master/src/mhs_tools.f90`, MHS_SOLVER,
its HW3CRT call and logarithmic clipping/relaxation block.
