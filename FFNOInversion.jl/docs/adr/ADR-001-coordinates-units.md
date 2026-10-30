# ADR-001: Coordinates, axes, and units

Status: accepted for Phase 0.

- Julia atmospheric arrays use `(nz, nx, ny)`.
- Spectral arrays use `(n_lambda, n_stokes, nx, ny)` even when only Stokes I is active.
- The inversion grid has a native vertical coordinate, `VerticalAxis(kind, values)`, with `kind` `:z_m`
  (geometrical height in metres, positive upward) or `:log10tau500` (dex). It is shared by all columns and stored top
  first: index 1 is the top, so z strictly decreases and log10 tau500 strictly increases. Readers that hold the fields
  reverse bottom-first input once, axis and every field together; `VerticalAxis` and `Grid3D` never reorder. The legacy
  `Grid3D(log_tau500, x, y)` constructor builds a log-tau axis and rejects a decreasing vector. (Updated for the HSEMHS
  integration, Phase 2, 24 October 2026.)
- Scientific code reads the axis only through `vertical_coordinate`, `native_depth`, `nlevels`, `log_tau500_axis` and
  `z_axis`; the last two throw on the wrong kind. There is no `grid.log_tau500` field.
- `Atmosphere3D.z` and `Atmosphere3D.logtau500` are 3-D coordinate fields. The one matching the native kind equals the
  axis broadcast over columns (`check_coordinate_consistency`); the other is derived and may vary by column (a log-tau
  field is never collapsed into a shared vector).
- Control nodes (`NodeField`, `ControlMapSpec`) carry their coordinate kind; expanding nodes of one kind onto a grid
  of the other throws. Nodes may be given in either monotonic order.
- Horizontal axes are finite and strictly increasing. Native-z input supplies explicit `x_m`/`y_m` (origin preserved);
  they must be uniform because the FSDP population payload and the PSF kernels use one spacing.
- SI units are used at public boundaries: K, m/s, Pa, kg/m^3, m^-3, m, tesla, and wavelength in m.
- Axis permutations and training-specific unit conversions belong in adapters and must be asserted.
