# Phase 1 validation report

Date: 2026-08-23

## Implemented

- Explicit HE3D selection when magnetic data are absent; magnetic arrays and Lorentz work are skipped.
- Explicit MHS selection when all three magnetic components exist.
- Nonuniform-grid curl, `J=curl(B)/mu0`, and all three components of `JxB`.
- A coupled 3D least-squares pressure-gradient solve, not independent 1D pressure columns.
- Fixed-point updates of pressure, EOS density/electron density, 500 nm opacity, and corrugated height.
- Independent convergence diagnostics for pressure, density, force residual, and height displacement.
- Exact preservation of temperature and optional B on the target `logtau500` grid.
- Julia FFI adapter to the repository's Wittmann C++ implementation for `(T,Pgas) -> (rho,ne)`.
- Production 500 nm mass opacity from STiC `cop`, with explicit Angstrom/cgs-to-SI conversion.

## Verification

```text
JULIA_DEPOT_PATH=/private/tmp/ffnoinversion-julia-depot julia --project=. test/runtests.jl
Phase 1 manufactured force balance     | 12 / 12 passed
Wittmann/STiC production opacity       |  6 /  6 passed
Entire Julia suite                         79 / 79 passed
```

The C++/Python Wittmann parity suite passed 4/4 cases under the bundled Python 3.12 runtime, including the independent 500 nm continuum-opacity comparison. A direct Julia call at 5770 K returned positive finite values for pressures 1 Pa and 100 Pa:

```text
rho = [2.6548446764615586e-8, 2.6791996934445493e-6] kg m^-3
ne  = [1.2755734320425938e17, 1.3683526228412273e18] m^-3
```

The 500 nm mass opacity agrees with independent `scripts/witt.py` values at five representative `(T,Pgas)` points from 3500-12000 K and 0.1-1000 Pa to a maximum relative difference below 3e-4. Both HE3D and MHS converge using the production opacity backend. The MHS fixture produces a nonzero Lorentz force, changes the pressure solution relative to HE3D, produces horizontal pressure structure, and preserves every B component exactly.

## Acceptance status

Phase 1 is accepted. `WittmannOpacity500` is the production backend; `ReferenceOpacity500` remains an explicitly named manufactured-test fixture.

## Known discretization limit

Curl uses x/y coordinates and the horizontally averaged geometrical-height coordinate. This is an orthogonal-grid Phase 1 stencil. A metric-aware operator must be validated before interpreting strongly corrugated MHS volumes scientifically.
