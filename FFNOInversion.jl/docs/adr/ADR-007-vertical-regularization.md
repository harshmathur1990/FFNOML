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

Each slot selects one type: 0 none, 1 first derivative in `logtau_500`, 2 deviation from its column depth mean, 3 deviation from zero, or 4 second derivative in `logtau_500`. Type 4 is included because the referenced STiC implementation defines it as the second-derivative penalty. Temperature types 2 and 3 are normalized to type 0 because those constraints are not meaningful for temperature.

The effective slot strength is `regularize * regularization_weights[i]`. Normalization scales are independent and prevent units from setting the numerical balance. For `pgas_boundary`, type 1 follows the STiC boundary-enhancement convention and penalizes deviation from one after division by the configured reference pressure scale. Magnetic strength, inclination, and azimuth are derived from `Bx/By/Bz`; requesting them without an available magnetic field is an explicit configuration/runtime error. Horizontal regularization remains a separate per-variable system.
