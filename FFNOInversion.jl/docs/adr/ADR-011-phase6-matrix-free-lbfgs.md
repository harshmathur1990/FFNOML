# ADR-011: Matrix-free gradients and bounded L-BFGS

## Status

Accepted. The production pullback chain and bounded L-BFGS implementation are complete. Olivia job 2072078 passed the target solver, recovery, layout-memory and CUDA/NCCL infrastructure gates; the updated real-checkpoint FFNO VJP case remains a post-change target regression.

## Decision

Gradient implementations extend `AbstractObjectiveGradient` and return the objective, a gradient in dimensionless scaled-control coordinates, and the number of complete forward evaluations consumed. This contract does not expose or allocate a dense spectral Jacobian. Module linearizations expose only JVP and VJP actions and are validated with dot products and Taylor remainders.

The control-map adjoint reverses the exact trilinear interpolation used by `expand_nodes`. Every MPI rank accumulates cotangents from its owned atmospheric tile and an all-reduction produces the replicated coarse-control gradient. The physical-to-scaled control factor is part of this pullback.

The scalable reference optimizer is bounded limited-memory BFGS. Rank 0 constructs and broadcasts the search direction and accepts or rejects Armijo trials. All ranks execute the same distributed objective and gradient route. Active-bound projected gradients define convergence. Rejected line-search sequences explicitly reevaluate the last accepted controls before checkpointing or returning.

L-BFGS checkpoints contain only coarse controls, scaled gradients, limited-memory vector pairs, convergence records, and capability/layout metadata. They contain no tile-owned atmosphere, population, or spectrum arrays and can restart with another compatible MPI/thread topology.

## Validation boundary

The exact-model test supplies a hand-derived VJP for the deterministic intensity fixture. It validates the gradient contract, distributed control-map adjoint, Taylor behavior, synchronized optimizer, rejection recovery, restart, and forward-call efficiency. `FiniteDifferenceObjectiveGradient` is retained for tests and debugging, not as the production large-control gradient.

The enabled chain now includes rank-0 FFNO VJPs, distributed observation and node adjoints, scalar transfer, FFNO/Kurucz line physics, Muspel/Wittmann local fallbacks, and a force-balance/regularization composite action. The complete mixed objective is tested against the centered oracle and by Taylor remainder. GPU checkpoint behavior remains guarded by the bounded Olivia production-VJP regression.
