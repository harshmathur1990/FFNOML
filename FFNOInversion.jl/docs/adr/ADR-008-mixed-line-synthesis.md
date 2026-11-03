# ADR-008: mixed line synthesis

Date: 2026-08-23

## Decision

A spectral region does not have one exclusive LTE or NLTE mode. It owns an ordered list of opacity sources. An FFNO source identifies a species and transition and carries its already-computed population cube. A Kurucz LTE source identifies a line-list file that is parsed and window-selected during setup. All sources add extinction and emissivity on the same high-resolution wavelength/depth grid. One formal solver is then called for the combined coefficients.

The stable seams are `AbstractLineOpacityModel`, `add_opacity_emissivity!`, `AbstractFormalSolver`, and `formal_solve!`. Release 1 validates Stokes I and `NonPRD`; polarized and PRD implementations can add methods without changing the region, observation, or objective contracts.

## Consequences

- Halpha (H FFNO), Ca II 8542 (Ca FFNO), and Kurucz LTE blends may be active simultaneously.
- LTE-only regions never need a population-model call.
- Line lists and other atomic data are setup-time caches; synthesis-column loops perform no file I/O.
- PSF application and wavelength/spatial weights remain downstream of the full-resolution formal solution.
- One continuum per region: a dedicated Wittmann contributor, with hydrogen from the EOS, so it reads no FFNO populations and does not depend on source order; every FFNO/Kurucz source adds line opacity only. `[physics] continuum_populations = "ffno"` has the region's first FFNO source supply it instead, with FFNO hydrogen.
- Configured STiC-style wavelengths are retained as air wavelengths for observation/output; line physics uses the RH air-to-vacuum conversion internally.
