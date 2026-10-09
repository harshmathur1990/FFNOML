# Phase 3 validation report

Date: 2026-08-31

## Implemented

- Separate redistribution, opacity/emissivity, formal-solver, and synthesis interfaces.
- Mixed Stokes-I synthesis that sums any number of NLTE and LTE contributors before one formal solution.
- FFNO transition contributor consuming canonical `(nz,nx,ny,nlevels)` populations.
- Portable Kurucz line-list parser, wavelength-window selector, and immutable setup cache.
- RH K94 fixed-width parser with air-to-vacuum conversion, random lower/upper-level ordering, statistical weights, Einstein coefficients, and tabulated radiative/Stark/van-der-Waals damping.
- Wittmann partition-function/Saha backend for LTE ion and lower-level populations, neutral hydrogen, and wavelength-dependent continuum opacity.
- Armstrong Voigt implementation ported from RH and velocity/Doppler/damping terms following `RLKProfile`.
- Persistent Wittmann backend: partition data is opened once during setup, never in wavelength or pixel loops.
- Production Muspel opacity/emissivity and piecewise-linear formal-solver adapter, pinned to Git commit `01ec68d` with native three-dimensional height support.
- Region configuration with repeatable `[[regions.sources]]` entries (`ffno` or `kurucz_lte`).
- Explicit rejection of PRD and non-I synthesis in the release-1 mixed backend.

## Verification

The cumulative Phase 1-6 Julia suite passes 242 assertions. Phase 3 tests cover portable and real K94 parsing, air-to-vacuum conversion, species/stage decoding, window selection, persistent Wittmann LTE synthesis for the supplied Fe I 6301/6302 list, mixed opacity, positive finite output, contributor interaction, LTE-only setup without FFNO, and PRD rejection. Configuration tests verify that one region can contain both an FFNO Ca II 8542 source and a Kurucz LTE blend source. The mandatory Muspel integration test constructs `Muspel.Atmosphere3D` with a full three-dimensional `z` array and passes that array through the adapter before comparing with Muspel's direct calculation.

The earlier production Muspel baseline was evaluated against the supplied snapshot 385 reference files using the exact Tiago H and Ca atomic models. For column `(x=1,y=1)`, both Halpha and Ca II 8542 were bitwise identical to the stored reference intensities: maximum absolute error `0.0` and maximum relative error `0.0`. After pinning the newer native-3D-height revision, the local app could not reopen the sibling 1.3 GB atmosphere because macOS returned `Operation not permitted`. Therefore the pinned revision's full H/Ca reference rerun remains a required Olivia regression case; it is not represented as new local evidence.

## Acceptance status

Phase 3 is complete for the current scope: scalar intensity, complete redistribution/non-PRD, FFNO Halpha and Ca II 8542 populations, and simultaneous RH-style K94 LTE lines. H and Ca have independent supplied-spectrum parity. The Kurucz path is a source-level port validated with the real 6301/6302 list and manufactured-atmosphere tests; no independent stored Kurucz reference spectrum was supplied, so its regression baseline is the ported RH equations and tests rather than an external golden cube.

PRD, Stokes Q/U/V, Zeeman components, isotopic/hyperfine splitting, and RH scattering (`RLK_SCATTER`) remain explicitly outside this release and are rejected by capability validation.
