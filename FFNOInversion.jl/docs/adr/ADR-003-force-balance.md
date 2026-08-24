# ADR-003: HE3D/MHS mode selection

Status: Phase 1 accepted.

- If magnetic field data are absent, select HE3D and allocate no dummy magnetic arrays.
- If magnetic field data are present, select MHS and include `J x B`, with `J = curl(B)/mu0`.
- Geometrical height increases upward. The top boundary is `z=0`; deeper layers have negative `z`, and gravity has magnitude 274 m/s^2 downward. The implemented column equation is `dP/dz = -rho*g + (JxB)_z`.
- Scalar or `(nx,ny)` pressure and density boundary values are accepted at the top or bottom. A full-boundary solve is rejected until its mathematical meaning is fixed.
- Curl uses centered differences in the interior and one-sided differences at boundaries on the declared x/y coordinates and horizontally averaged z coordinate. This orthogonal Phase 1 stencil records the limitation; a metric-aware curl is required for strongly corrugated MHS volumes.
- HE3D never allocates or evaluates magnetic data. MHS computes all components of `J=curl(B)/mu0` and `JxB`; the vertical component enters pressure integration and the full magnitude is diagnosed.
- The pressure update is not a set of independent 1D columns. A vertical integration supplies the initial iterate, followed by a 3D least-squares gradient relaxation using x, y and z neighbors with the declared pressure boundary fixed. The fixed-point iteration alternates EOS density/electron density, 500 nm opacity, z reconstruction and this coupled pressure solve with under-relaxation. It requires pressure, density, normalized force, and height changes to pass independently.
- Temperature and optional B are held exactly on the target log-tau grid, so their remap errors are zero by construction. Pressure positivity and finite EOS results are hard failure conditions.
- `WittmannEOS` calls the repository's compiled Wittmann implementation for both rho and ne. `WittmannOpacity500` uses the STiC `cop` continuum routine at exactly 5000 Angstrom and converts volume extinction from cm^-1 to mass extinction in m^2 kg^-1. `ReferenceOpacity500` remains only for manufactured tests. Representative opacity points agree with the independent Python Wittmann implementation to a maximum relative difference below 3e-4.
