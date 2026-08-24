# ADR-004: PRD and Stokes extension boundaries

Status: accepted for Phase 0.

Release 1 implements intensity only and no PRD. These are capability selections, not array or optimizer assumptions.

- `SpectralCube` always contains a Stokes axis; `StokesSet` declares active components.
- `AbstractRedistributionModel` isolates non-PRD and future PRD state/iterations.
- `AbstractSynthesizer` returns the same spectral type for scalar and polarized solvers.
- Observation and residual layouts are component-aware; future Mueller response belongs to the observation adapter.
- `CapabilityManifest` is stored with checkpoints and unsupported requested physics fails before inversion.
- CI uses mock PRD and IQUV implementations to prove substitutability without claiming scientific correctness.
