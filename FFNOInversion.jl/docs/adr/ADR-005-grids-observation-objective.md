# ADR-005: Synthesis grid, observation grid, weights, and regularization

Status: accepted for Phase 0.

- The input atmosphere supplies `logtau_500`, temperature, `vx/vy/vz`, and top pressure (a scalar in Pa or a named 2D dataset). It may optionally supply all of `Bx/By/Bz`.
- The single observation file contains the spectral cube, uncertainty, `(wavelength, Stokes)` weights, and spatial weights.
- `dx_m` and `dy_m` describe the high-resolution spatial grid. Repeatable spectral-region records describe simultaneous wavelength windows and associate each one with its continuum normalization, PSF type, and PSF file.
- Release 1 applies separable Gaussian wavelength/x/y PSFs without resizing or sampling the cube.
- Chi-square participation is controlled only by weights. `wavelength_weights` has shape `(n_lambda,n_stokes)`; an all-zero Stokes column disables inversion of that component. A zero entry in the 2D spatial map excludes that pixel.
- Wavelength-Stokes and spatial weights are multiplied on the full cube. Noise sigma remains separate.
- Atmospheric regularization is evaluated on the full inversion result. Vertical and horizontal terms have independent per-variable strengths, normalization scales, and derivative orders.
- Future nonseparable PSFs and full wavelength-Stokes-spatial weight tensors belong behind the observation/objective interfaces rather than in the forward physics.
