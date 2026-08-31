# ADR-002: Canonical FFNO population boundary

## Status

Superseded in runtime details by the Phase 6 multi-GPU FSDP service. The data contract remains accepted.

## Retained decision

The canonical FFNO input channel order is `temperature, vx, vy, vz, log10_ne, log10_rho`, stored as `(channel,nz,nx,ny)`. Corrugated geometrical height is passed separately as `(nz,nx,ny)` together with physical `dx` and `dy`. FFNO returns positive finite linear populations in m^-3 as `(nz,nx,ny,nlevels)`.

Checkpoint identity, channel order, level names and output representation are validated at the Julia/Python boundary. `RecordedPopulationModel` serializes an exact request/response case for deterministic CPU tests and checks both atmospheric features and the complete three-dimensional height field before replaying it.

The earlier embedded Python bridge and its Julia extension were removed when FSDP became mandatory. The application-level `run_inversion!` entry point now accepts only `RootDistributedPopulationModel` or composite backends whose root models are `FSDPFFNOModel` instances connected to a service with at least two GPU ranks. This is a positive capability requirement, not a blacklist of old backends.

Mocks and recorded populations remain test fixtures for isolated numerical components. They cannot cross the production inversion boundary.
