# Phase 2 validation report

Date: 2026-08-31

## Retained implementation

- Canonical six-channel FFNO request construction with explicit axes and units.
- Positive/finite density and electron-density validation before logarithms.
- Checkpoint identity, channel order, level map and output-representation metadata.
- Canonical positive linear population output `(nz,nx,ny,nlevels)` in m^-3.
- Serializable record/replay fixtures that validate both features and the complete geometrical-height cube.

## Production-backend decision

Phase 2's embedded Python implementation has been deleted. There is no compatibility execution option in the inversion package. Production population inference and VJP use only the persistent multi-GPU FSDP service implemented in Phase 6. Test mocks and recorded responses remain available below the application boundary for deterministic CPU validation.

The cumulative local Phase 1-6 suite includes all eight Phase 2 population-contract assertions. The separate FSDP-only architecture test verifies that the deleted extension/runtime files are absent, the project has no PythonCall dependency, `run_inversion!` requires `FSDPFFNOModel`, and the service advertises the `FULL_SHARD_H_SLAB` capability.
