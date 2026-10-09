# ADR-001: Coordinates, axes, and units

Status: accepted for Phase 0.

- Julia atmospheric arrays use `(nz, nx, ny)`.
- Spectral arrays use `(n_lambda, n_stokes, nx, ny)` even when only Stokes I is active.
- The inversion coordinate is a strictly monotonic `log_tau500` vector.
- SI units are used at public boundaries: K, m/s, Pa, kg/m^3, m^-3, m, tesla, and wavelength in m.
- Axis permutations and training-specific unit conversions belong in adapters and must be asserted.
