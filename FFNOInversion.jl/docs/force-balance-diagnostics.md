# Force-balance diagnostics

Both reconstruction functions return these measurements in `ForceBalanceDiagnostics`.
They evaluate the final returned state, after best-iterate restoration or fallback.
They do not change pressure updates, convergence criteria, or acceptance decisions.
The distributed path also emits a `FORCE_BALANCE event=final_diagnostics` log line.

| Fields | Meaning |
| --- | --- |
| `force_residual_x`, `force_residual_y`, `force_residual_z` | Maximum normalized force mismatch along x-, y-, and depth-index edges. Their maximum agrees with the existing `force_residual`, up to rounding. |
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

## Bootstrap checkpoint

The `bootstrap_*` fields measure the returned 1-D HSE initialization immediately
before 3-D relaxation. They include iteration count, convergence of the existing
pressure/density/height change test, directional/full force residuals, face
circulation, and EOS consistency. The distributed log emits them in
`event=bootstrap_diagnostics`. The bootstrap uses zero magnetic force, including
when it initializes an MHS run. Final unprefixed fields evaluate the best returned
3-D state (or the restored bootstrap if no improvement was accepted).

A converged bootstrap need not be in 3-D equilibrium: its vertical residual tests
column-wise hydrostatics, while its x/y residuals reveal horizontal imbalance.
Do not interpret `bootstrap_converged` as passing the 3-D force tolerance.

## Saving the bootstrap atmosphere

Both reconstruction functions accept `bootstrap_callback=snapshot -> ...`.
It runs once after bootstrap diagnostics, before 3-D relaxation. The snapshot
contains independent copies of `pgas`, `rho`, `ne`, and `z`, plus the immutable
bootstrap diagnostics. Saving or modifying those copies does not affect the
solver. Temperature, velocities and grid coordinates remain available from the
input atmosphere. In MPI, the callback receives each rank's local arrays; use
separate rank-specific outputs. Callback failures are coordinated across ranks.

The crop run script in `outputs/hse3d_bootstrap_relaxation32_20261012` saves
`atmosphere_bootstrap.h5` at this checkpoint, with Python-visible `(z,y,x)` axes
and SI unit attributes, before saving the final `atmosphere_hse3d.h5`.
