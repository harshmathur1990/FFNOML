# Phase 5 validation: small-volume inversion prototype

## Outcome

Phase 5 is implemented on the single hybrid MPI/thread application path. The solver optimizes bounded and scaled coarse temperature/velocity control maps, applies them directly to rank-owned atmospheric tiles, evaluates normalized weighted chi-square plus separately retained regularization components, writes restartable checkpoints, and emits a TOML/CSV convergence bundle.

## Local acceptance evidence

```sh
julia --project=. --threads=4 test/runtests.jl
julia --project=. scripts/validate_phase5_mpi.jl
```

The complete Phase 0-5 suite passes 180/180 assertions in 14 test sets. The Phase 5 exact-model fixture is a small 3D atmosphere with four independent spectral responses to upper/lower temperature and line-of-sight velocity. Six controls encode a two-point horizontal temperature mode and a two-node vertical velocity mode.

From both 4500 K and 7000 K temperature initializations with zero velocity, the prototype reduces the objective by more than six orders of magnitude and recovers all six controls exactly on the configured 500-unit search lattice. The two runs finish with identical parameters. Centered directional estimates at steps `1e-3` and `5e-4` agree to the asserted `1e-8` relative tolerance.

An interrupted three-iteration run resumes to eight iterations with parameters, objective history, accepted decisions, and evaluation count identical to an uninterrupted eight-iteration run. Atomic checkpoint replacement and capability/layout mismatch rejection are exercised.

The MPI validator runs an eight-iteration inversion with one rank and four ranks. It obtains identical parameters and accepted decisions, with only `4e-15` absolute objective reduction-order roundoff. A checkpoint written after three iterations with one rank resumes with four ranks and reproduces the uninterrupted result and evaluation count.

## Scope boundary

This is an exact-model/inverse-crime orchestration test using a deterministic intensity synthesizer. It establishes that the inversion loop, distributed ownership, parameter recovery, objective accounting, finite-difference oracle, and restart behavior work. It does not validate recovery through the real FFNO/Muspel/Kurucz physics, noise robustness, independent RH/MULTI3D observations, GPU gradients, or production-volume performance. Those remain the Phase 5 deployment extension and Phases 6-7 scientific gates.
