# Phase 2 validation report

Date: 2026-08-23

## Implemented

- Canonical six-channel FFNO request construction with explicit axis and unit contracts.
- Positive/finite density and electron-density validation before logarithms.
- Persistent PythonCall backend loaded through an optional Julia extension.
- Python runtime that loads checkpoint, normalization and model weights once and performs repeated in-memory inference.
- SHA-256 checkpoint, channel order, level map and output-representation metadata.
- Canonical linear population output `(nz,nx,ny,nlevels)` in m^-3.
- Serializable record/replay backend that validates request features and geometrical height.
- Separate Python-free unit tests and PythonCall CPU integration test.

## Verification

The Phase 2 population tests remain 8/8 within the expanded 105/105 core suite, which runs without importing PythonCall. The optional CPU bridge fixture passes 3/3 tests and proves that two inference calls use one factory/backend construction. The available H and Ca checkpoints both load, produce finite positive `(4,8,8,6)` cubes, and reuse one loaded backend across two calls. Phase 1's independent Python suite remains 4/4 passing.

For the H checkpoint, the persistent runtime is bitwise identical to the existing `FFNONet.ffno_predict_populations(..., tiled=false)` path after applying the documented legacy `(nx,ny,nz,nlevels)` to canonical `(nz,nx,ny,nlevels)` permutation: maximum absolute and relative differences are both zero.

Checkpoint evidence:

- H: SHA-256 `901dcd28a6ee651c12a26a60effdd28c7ea211b596a30b87654435e87803c755`, epoch 171, validation loss 0.005883101594708233.
- Ca: SHA-256 `7a9365d26543dcb9b94da29dc574ae1e18b879f0818909e9d579a7aa4af9b760`, epoch 252, validation loss 0.007309474505186349.
- Both declare `Cin=6`, `Cout=6`, and six-element input/output normalization arrays.
- On this CPU-only Apple environment, the first and repeated H calls on a `(4,8,8)` cube take approximately 7.6 s each; Ca takes approximately 7.4 s. H checkpoint construction/load takes approximately 0.9 s.

## Acceptance status

Phase 2 is accepted for functional integration: real-checkpoint parity, persistence, metadata validation, H/Ca loading, and CPU separation all pass. CUDA/MPS is unavailable on this host, so GPU latency and memory evidence is deferred until execution on target GPU hardware; this does not change the backend interface.

The synthetic uniform input used for runtime parity verifies software equivalence, not scientific population accuracy. Recorded realistic-atmosphere validation remains part of later independent scientific validation.
