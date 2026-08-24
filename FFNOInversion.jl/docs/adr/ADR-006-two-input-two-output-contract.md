# ADR-006: Two-input/two-output inversion contract

Status: accepted for Phase 0.

Every inversion run exposes exactly two scientific inputs:

1. one observation file containing intensity, sigma, `(wavelength, Stokes)` weights, and a 2D spatial-weight map;
2. one initial-atmosphere file containing `logtau_500`, temperature, velocities, top pressure, and optional magnetic field.

Every run produces exactly two primary scientific outputs:

1. one synthesis file containing the final PSF-degraded full-grid spectrum, wavelength coordinates, residual and chi-square diagnostics, and provenance;
2. one atmosphere file containing the recovered atmospheric variables, HE3D/MHS-derived variables, predicted populations, convergence history, and provenance.

Checkpoints and logs are operational sidecars, not additional scientific inputs or primary products. Dataset names remain configurable, but observation weights must live in the observation file rather than a third weights file.
