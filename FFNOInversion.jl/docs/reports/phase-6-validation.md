# Phase 6 validation - in progress

Date: 2026-08-26

## Implemented

- Generic matrix-free JVP/VJP actions with dot-product validation.
- Exact distributed adjoint of coarse-node expansion, including MPI reduction and physical-to-scaled control conversion.
- Bounded finite-difference objective gradient retained as an oracle.
- First-order Taylor-remainder validation for complete objective gradients.
- Synchronized bounded L-BFGS with projected gradients, limited memory, Armijo backtracking, forward-call accounting, safe rejected-trial restoration, atomic checkpoint/restart, and TOML/CSV diagnostics.
- Solver configuration supports `lbfgs` and the retained `prototype_pattern_search` baseline without obsolete method-specific keys in the example configuration.

## Local evidence

The complete local suite passes 206 assertions. The node-expansion pullback passes its dot-product test below `1e-12` relative error. The six-control exact-model objective pullback agrees with centered finite differences and its Taylor remainder converges at second order.

Bounded L-BFGS recovers all four temperature and two velocity controls to below `1e-6` absolute error, reaches an objective below `1e-14`, and uses fewer complete forward evaluations than the Phase 5 pattern-search baseline. A deliberately reversed gradient causes every line-search trial to be rejected; the solver restores the accepted atmosphere and exits with `line_search_failed`. Interrupted execution reproduces uninterrupted controls, gradients, history, objective, and forward-evaluation count.

The MPI validator passes for one and four ranks. Both topologies make the same line-search decisions. Normal MPI reduction roundoff is below the asserted tolerance. A one-rank checkpoint restarts on four ranks and reproduces the uninterrupted one-rank result and forward-evaluation count.

Commands:

```text
julia --project=. test/runtests.jl
julia --project=. scripts/validate_phase6_mpi.jl
```

## Remaining before Phase 6 acceptance

- Wire configuration, two-input ingestion, solver selection, final two-output writing and the production gradient backend into one executable inversion entry point. At present `lbfgs_invert!` is called only by tests and validators.
- Implement and test the rank-0 FFNO GPU VJP service through the existing GPU control/status protocol.
- Implement and chain pullbacks for force balance/EOS/opacity, Muspel non-LTE synthesis, Kurucz LTE synthesis, the distributed observation PSF, weighted data objective, and vertical/horizontal regularization.
- Run a Taylor and dot-product test for every enabled custom pullback and a complete real-physics gradient against the centered finite-difference oracle.
- Run the solver on the real small-volume mixed FFNO plus Kurucz path and retain a forward-call comparison against Phase 5.
- Demonstrate full-domain peak memory and GPU/MPI runtime behavior on Olivia target hardware, including rejected-trial and failure recovery.

## Olivia tests prepared

The runtime harness now has a dedicated `phase6` allocation containing four
tests: the distributed exact-model L-BFGS solver with checkpoint/restart,
rejected-trial restoration,
a configurable 800 by 800 full-layout rank-owned memory allocation, and a
rank-0-controlled CUDA autograd VJP with an NCCL checksum. Every case has a
hard external timeout, five-second scheduler/heartbeat sampling, orphaned-step
cleanup and a diagnostic archive.

These tests provide target-runtime evidence for implemented infrastructure.
They deliberately do not count as FFNO, force-balance, Muspel, Kurucz, PSF or
regularization gradient evidence because those production pullbacks do not yet
exist.

Phase 6 has begun and its solver/gradient architecture is operational, but the phase is not yet complete.
