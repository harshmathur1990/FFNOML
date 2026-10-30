# ADR-010: Phase 5 bounded derivative-free prototype

## Status

Accepted for the Phase 5 exact-model prototype.

## Decision

The Phase 5 solver uses deterministic best-improvement coordinate pattern search in dimensionless control coordinates. Each parameter block declares physical lower/upper bounds and a positive scale. Rank 0 proposes and accepts trials; the small coarse control vector is broadcast, expanded directly onto every rank's owned atmospheric tile, and evaluated by the existing hybrid distributed forward/objective route.

The prototype intentionally does not introduce a dense Jacobian, AD dependency, L-BFGS implementation, or serial application route. Centered finite differences in normalized directional coordinates are retained as the Phase 6 gradient oracle. Coarse maps can be interpolated into a finer layout between externally managed inversion cycles.

## Objective convention

The data term is global weighted chi-square divided by the number of positive-weight residual samples. Vertical and horizontal regularization are evaluated on the reconstructed atmosphere and added without hiding their individual values. Every iteration records total, data, regularization, accepted coordinate/direction, evaluation count, and current maximum scaled step.

## Restart and ownership

Only coarse controls and solver state are replicated. Atmosphere, population, and spectrum arrays remain MPI tile-local. Checkpoints are written atomically by rank 0, carry the capability manifest and a stable control-layout signature, and are read/broadcast at restart. A checkpoint may resume under another compatible rank/thread topology because it contains no tile decomposition.

## Consequences

- Pattern search is deliberately expensive and is restricted to small/coarse Phase 5 problems.
- Production-size optimization, VJPs, custom adjoints, L-BFGS/trust-region logic, and scalable gradients remain Phase 6.
- Exact-model recovery proves orchestration and identifiability of the fixture, not observational validity or independence from the forward surrogate.
