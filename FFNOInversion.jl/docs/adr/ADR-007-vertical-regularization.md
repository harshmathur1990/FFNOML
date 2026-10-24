# ADR-007: Seven-slot vertical regularization

Status: accepted for Phase 0.

Vertical regularization follows the STiC-style fixed parameter order:

1. temperature;
2. line-of-sight velocity (`vz` for the initial disk-center implementation);
3. microturbulence;
4. magnetic-field strength;
5. magnetic inclination;
6. magnetic azimuth;
7. gas-pressure boundary parameter.

Each slot selects one type: 0 none, 1 first derivative with respect to the native vertical coordinate, 2 deviation from its column depth mean, 3 deviation from zero, or 4 second derivative with respect to the native vertical coordinate. Type 4 is included because the referenced STiC implementation defines it as the second-derivative penalty. Temperature types 2 and 3 are normalized to type 0 because those constraints are not meaningful for temperature.

The effective slot strength is `regularize * regularization_weights[i]`. Normalization scales are independent and prevent units from setting the numerical balance. For `pgas_boundary`, type 1 follows the STiC boundary-enhancement convention and penalizes deviation from one after division by the configured reference pressure scale. Magnetic strength, inclination, and azimuth are derived from `Bx/By/Bz`; requesting them without an available magnetic field is an explicit configuration/runtime error. Horizontal regularization remains a separate per-variable system.

Length scales (HSEMHS integration, Phase 2, 24 October 2026): derivatives are taken with respect to the native
coordinate divided by a length scale, `regularization.vertical.length_scale_dex` (default 1, which reproduces the
historical log-tau penalty exactly) on log10tau500 grids and `regularization.vertical.length_scale_m` on height grids.
`length_scale_m` has no default: types 1 or 4 on a height grid without it are a configuration error, because metres
and dex are not interchangeable and a change of coordinate unit must not change the regularization strength.
