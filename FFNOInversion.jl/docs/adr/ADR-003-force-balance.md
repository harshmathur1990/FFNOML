# ADR-003: Force balance with automatic magnetic coupling

Status: **Superseded** (24 October 2026) by ADR-012: the force-balance backend did not work and was removed; HSEMHS is the equilibrium backend. Kept for history.
Previously: updated 2026-10-12; superseded the earlier HE3D/MHS mode selection.

- There is one reconstruction pipeline. Missing magnetic data represents B=0
  by evaluating the same algebra on zero arrays. Explicit zero magnetic arrays have identical
  behavior. There are no public `HE3DMode`, `MHSMode`, or mode-selector APIs.
- There is no HSE bootstrap. Initialization computes EOS, opacity and geometry
  once from initial pressure without updating pressure. One MHS iteration loop
  then updates pressure and dependent fields.
- Every B value uses the same Lorentz evaluation, coupled x/y/z pressure
  relaxation, residual, backtracking and convergence equations. No B-value
  branch selects an algorithm. Missing storage is normalized to zero arrays.
- For B=0, magnetic force vanishes algebraically, leaving full hydrostatic
  equilibrium including zero horizontal pressure gradients. This is not a
  collection of unrestricted independent HSE columns.
- The physical equation is grad(P) = J cross B - rho*g*e_z, with
  J = curl(B)/mu0. Physical x/y forces are solely the Lorentz components.
  Edge integrals across corrugated optical-depth surfaces include the vertical
  force multiplied by the edge's height change; this is part of the coordinate
  geometry, not an extra horizontal force.
- Height increases upward. The top optical-depth sample is z=0 per column;
  deeper layers have negative z. Gravity is 274 m/s^2 downward. Scalar or mapped
  top/bottom pressure boundaries are supported; full boundaries are rejected.
- Curl uses physical derivatives on the corrugated grid, obtained through the
  optical-depth-to-height chain rule. Prescribed B remains fixed on log-tau
  samples; reconstruction does not enforce div(B)=0 or solve for B.
- Reconstruction solves a coupled log-pressure graph divergence equation using
  PCG, clips the log candidate to explicit numerical bounds, and relaxes pressure
  in logarithmic space. There is no vertical pressure seed. See the diagnostics
  document for differences from FIRTEZ's FISHPACK discretization.
  Pressure positivity and finite EOS results remain hard requirements.
- A uniform nonzero B enters magnetic reconstruction but has zero curl and
  Lorentz force; its presence does not guarantee convergence.
- Diagnostic `mode` always records MHS. Iterations count the single solver loop.
  Bootstrap options, routines, diagnostic fields and callbacks are removed;
  `initial_callback` can checkpoint the state before the first pressure update.
  All field values retain the same best-force selection, with fallback to the
  initial state.
- Production thermodynamics and opacity remain WittmannEOS and
  WittmannOpacity500. The optional consistent-trial experiment remains disabled
  by default; this policy change does not add any new trial enforcement.
