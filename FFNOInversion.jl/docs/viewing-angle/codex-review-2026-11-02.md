# Viewing-angle design review

Reviewed 2 November 2026 against `README.md` in this directory, current FFNOInversion synthesis/population/control code, and upstream HSEMHS `ViewTau.jl` and viewing-angle design. This is a design review; no implementation or numerical validation was performed. Existing working-tree edits were left untouched.

## Main recommendation

Keep viewing direction outside the FFNO input channels. For a fixed physical atmosphere and the current scalar-population, non-PRD model, the observer selects a ray through the population field; changing the observer does not require a different population solution. Native-z Cartesian input can use one population prediction for multiple synthesis directions. Inclined native-log-tau input must be placed on the Cartesian mesh before predicting populations and sampled back along rays afterward.

The proposed shear is needed for native-log-tau inclined columns: heights alone give z, not the x/y position of each node. At footprint (xi, eta), a node at z lies at (xi + tx*z, eta + ty*z). A horizontal network layer must mix points at the same physical height and Cartesian x/y, rather than equally indexed footprints along different rays. This is a coordinate conversion, not angle conditioning of the trained operator.

## Changes required before implementation

### 1. Correct native-z formal-solution assumptions (high priority)

The document's section 6 calls plane-parallel native-z synthesis unchanged because HSEMHS supplies tau_LOS. Current `Synthesis.jl::_tau_integration` selects optical-depth integration only for a native `:log10tau500` grid. For native-z, `formal_column!` calls `formal_solve!` with geometrical z. Updating atmosphere.logtau500 alone therefore leaves the path length vertical.

Define a ray integration contract explicitly. For native-z a straightforward route is geometric integration with ds = |dz|/mu, after shearing state for inclined geometry and without shearing for plane-parallel geometry. Alternatively construct a ray-view workspace explicitly marked for tau_LOS integration. Do not change the atmosphere's native coordinate simply to trigger the existing branch.

If integrating in recomputed tau_LOS, the derivative must include the dependence of tau_LOS and its top closure on chi500, and any dependence on the state. The existing native-log-tau adjoint treats its input axis as fixed and cannot automatically provide that new derivative. Audit formal-coordinate/weight derivatives and the return path to original atmosphere variables. Avoid counting 1/mu twice.

### 2. Verify what the mu=0.8 reference actually represents (high priority)

`output_ray_mu08.hdf5` exists, but the filename does not prove true inclined 3D ray tracing. Standard RH 1.5D solves independent columns. Establish the generating executable/version, original or pre-sheared atmosphere, ray direction, phi, axis flips, population source and spectral-physics settings. Check run inputs/logs or source before interpreting continuum alignment. Map correlation can help registration but cannot establish the physical reference geometry.

Use two validation levels:

1. An independent formal ray integrator with identical prescribed populations, opacity physics and boundaries, to isolate geometry/interpolation error.
2. RH/MULTI3D spectra as a broader scientific comparison, with differences in population solutions and line physics reported separately.

Do not use agreement in z(tau=1) from a shared implementation as the only independent geometry test. Compare complete tau profiles and an analytic/manufactured optical-depth integral.

### 3. Make the input coordinate/frame contract machine-checkable

Store and validate, when relevant: vertical versus LOS tau convention, input mu/azimuth/geometry, Cartesian versus footprint spatial axes, local Cartesian vector frame, footprint reference height, periodic boundaries and pixel spacings. Reject incompatible tagged inputs before equilibrium reconstruction. Define a backward-compatible interpretation for legacy disk-centre files and require an explicit convention for off-centre log-tau inputs.

Native-z accepts an unchanged Cartesian atmosphere; do not impose the native-log-tau LOS-column input requirement on it. Record the projection direction and sign convention, including the Doppler sign, in output metadata.

### 4. Specify interpolation variables and their derivatives

Document whether the shear acts on T or lnT/logT, ne/rho or their logarithms, and populations or log populations. These choices differ numerically. Existing population-mesh features mix temperature in physical units, logarithmic densities and linear velocities; do not describe the operation simply as the six raw network inputs without checking the Julia/Python normalization boundary.

A fixed bilinear shear is linear in its supplied array. If transformations are applied before/after it, JVP/VJP need their chain rules. The transpose of bilinear interpolation is weighted scatter-add; it is not the opposite-sign shear. Opposite-sign shearing is also not an exact inverse. Define positivity and conservation checks and evaluate smoothing against resolved spatial structure before choosing higher-order shifts. Fourier/cubic interpolation can overshoot positive physical fields.

Shearing state and then computing opacity is not equivalent to shearing opacity/emissivity. The former is a reasonable first implementation only with a quantified refinement/interpolation test, especially across sharp thermal structures and Doppler gradients. Preserve vector components in the local frame; translating them is not rotating them.

### 5. Complete the velocity-control definition

The proposed fixed-prior perpendicular component is a sensible default. Reconstruct all three velocity components from an immutable prior on every trial; avoid accumulating updates from the previous trial. Reject overlapping `vlos` and Cartesian velocity controls unless a precise nonredundant parameterization is provided.

Update control expansion, pullback, regularization, bounds/scales, checkpoint/restart validation and response/tangent tools together. Reject a restart with a different viewing direction unless an explicit conversion is implemented.

Correct the statement that Stokes I constrains only n dot v: its direct Doppler response depends on n dot v, but this coupled model can respond indirectly to transverse velocities through FFNO populations. Those indirect constraints may be weak and should not be presented as independent velocity measurements.

### 6. Fix the spatial origin and projection conventions

Write rays as x = xi + tx*(z-z_ref), y = eta + ty*(z-z_ref), with z_ref stored explicitly. A change of height origin must be accompanied by the corresponding footprint translation; otherwise shifting the height gauge changes image registration. Add a gauge/footprint covariance test, especially for cross-comparing HSEMHS reconstructed heights and Bifrost absolute coordinates.

The stated 1/mu PSF stretch is appropriate for the chosen axis-aligned footprint convention. Keep PSF widths explicitly in sky units, normalize the transformed discrete kernel, and distinguish angular units from physical metres. For rotated images define the sky-to-footprint affine map; a general PSF covariance transforms as A*Sigma_sky*A'. User-side image resampling also changes masks and potentially noise covariance. Record that approximation if the inversion retains diagonal weights. Do not apply the geometric stretch twice.

Periodicity of the ray sampler is an explicit boundary condition, not a consequence of FFT layers alone. State where periodic wrapping begins and estimate horizontal travel (z_top-z_bottom)*sqrt(1-mu^2)/mu; a warning based only on mu misses repeated wraps in narrow/tall boxes.

### 7. Clarify distributed implementation and cost

Current upstream ViewTau supports arbitrary 2D block decompositions, but takes HSEMHS ParCtx and Float64 arrays in (nz,ny,nx). FFNO arrays use (nz,nx,ny). Specify an adapter using the existing owned communicator/block layout; do not initialize another MPI runtime. A shear transpose must accumulate overlapping contributions globally and return owned cells without halo duplication. Keep MPI calls on the initializing thread and make errors collective-safe.

Precompute level shifts and interpolation weights for fixed mesh/view, share them across fields and JVP/VJP calls, and benchmark the actual forward plus adjoint allocation/communication cost. The document's approximately 10 GB estimate needs a stated meaning: payload gathered per rank, total collective traffic, or memory. These are different, rank-dependent quantities. Batch fields/levels only within a bounded memory budget.

## Suggested implementation order and acceptance tests

1. Pin an exact full HSEMHS main commit, rather than a moving branch, and run existing mu=1 regressions.
2. Add input/view contracts, LOS projection, control/regularization semantics and native-z plane-parallel path scaling. Test both fixed-input coordinates separately.
3. Implement shared shear, transpose and layout adapter; then native-z inclined synthesis with prescribed populations. Validate this before coupling FFNO and equilibrium.
4. Add native-log-tau ray-to-Cartesian population reconstruction and its complete JVP/VJP chain.
5. Add sky PSF transformation and combined observation/inversion tests.
6. Compare matched native-z and slanted-log-tau representations of the same prescribed physical atmosphere. Test HSE-reconstructed versions separately: equilibrium reconstruction changes the atmosphere and is an additional source of error.

Additional acceptance tests:

- Uniform atmosphere: inclined and plane-parallel agree at a fixed mu; known constant-opacity slab follows exp(-chi*L/mu).
- Structured analytic field: known translations at phi=0/90/180/270 and a non-axis-aligned phi; signed Doppler tests for each velocity component.
- Projection/sampling reversal tests, periodic seams, integer/fractional shifts and unequal dx/dy.
- Dot-product and directional finite-difference tests of the complete forward/observation/regularization chain in both native coordinates, including thermodynamic and LOS controls; state numerical tolerances rather than calling a floating-point dot test exact.
- One-rank versus 2D-tile and x-slab MPI layouts, shifts crossing multiple ranks, and serial error propagation.
- mu=1 explicit identity fast paths for bitwise compatibility; mu near 1 and small mu for finite values and meaningful geometry diagnostics.
- Spectral convergence under horizontal and vertical refinement and matching top/bottom closures. Keep observer-angle support explicitly scoped to current Stokes-I/non-PRD physics.

## Answers to the remaining design questions

The native-log-tau inclined population path does need the coordinate shear (or an equivalent direct remapper). HSEMHS heights alone do not replace it. Prefer the Julia side first because the interpolation and its transpose can be validated independently of the GPU service. For native-z, leave FFNO population prediction Cartesian and shear only the synthesis inputs. Start with state interpolation, provided its error is measured against independent ray sampling/refinement. Native-z geometry should be validated first because it avoids mixing ray tracing with equilibrium reconstruction.

Reference for the RH validation caveat: https://github.com/ITA-Solar/rh/blob/master/doc/source/input.txt
