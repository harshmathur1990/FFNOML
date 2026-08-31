# Phase 6 validation - implementation complete, target GPU regression pending

Date: 2026-08-31

## Implemented production path

- Generic matrix-free JVP/VJP actions with dot-product validation.
- Exact distributed adjoint of coarse-node expansion, including MPI reduction and physical-to-scaled control conversion.
- Exact transposes for the distributed spectral/spatial Gaussian observation operator and the scalar formal solver.
- Analytic opacity/emissivity pullbacks for FFNO transitions and the Julia Kurucz LTE path.
- A cell-local numerical pullback for the Muspel opacity primitives and a column-local numerical fallback for the opaque Wittmann Kurucz C backend. Neither fallback repeats the formal solution, force balance, FFNO inference or complete forward model.
- Persistent PyTorch FFNO VJP support in `ffno_runtime.py`, exposed through PythonCall and the existing rank-0 GPU control/status service.
- Separate H and Ca population workspaces and composite distributed population backends, allowing simultaneous FFNO species plus Kurucz LTE contributors.
- A complete `HybridAdjointObjectiveGradient` that chains weighted chi-square, PSF, formal transfer, line physics, population inference, force reconstruction and regularization into scaled coarse-control gradients.
- The HE3D/MHS plus regularization composite uses bounded centered perturbations of the CPU reconstruction only. It performs one complete forward/GPU evaluation per gradient and does not construct a dense Jacobian.
- Synchronized bounded L-BFGS with projected gradients, limited memory, Armijo backtracking, safe rejected-trial restoration, atomic checkpoint/restart and forward-call accounting.
- Canonical HDF5 input/output and one executable MPI route from the two scientific inputs to the two scientific outputs. File axes `(time,z,y,x)` and `(time,Stokes,wavelength,y,x)` are converted to the internal canonical layouts without interpolation.

## Local evidence

The complete local suite passes 220 assertions. The production PSF and formal-solver pullbacks pass directional dot-product checks. A simultaneous two-species FFNO-transition plus Kurucz-LTE objective gradient agrees with the complete centered finite-difference oracle to `3.67e-10` relative error. Its Taylor remainder is second order, and bounded L-BFGS reduces the mixed objective in 22 complete forward evaluations.

The HDF5 executable test reads one atmosphere file and one observation file, runs the sole MPI/hybrid inversion path, and writes one synthesis file and one recovered-atmosphere file. It verifies the declared time/Stokes/wavelength/y/x axis contract and population output.

The MPI validator passes for one and four ranks. It now additionally exercises the production gather -> rank-0 population VJP -> scatter route with no population model on non-root ranks. Both topologies return the exact cotangent, make the same optimizer decisions, and a one-rank checkpoint restarts on four ranks with the uninterrupted result and forward-evaluation count.

Commands:

```text
julia --project=. test/runtests.jl
julia --project=. scripts/validate_phase6_mpi.jl
julia --project=. scripts/invert.jl CONFIG.toml MODEL_FACTORY.jl
```

## Olivia evidence received

Job `2072078` exited normally and passed all four original Phase 6 target-runtime cases:

- distributed L-BFGS and checkpoint/restart;
- rejected-trial restoration;
- configurable 800 by 800 rank-owned layout memory;
- CUDA autograd VJP plus NCCL checksum.

The job completed in 154 seconds with a maximum reported step RSS of 10.7 GiB under a 24 GiB per-rank test limit. The harness produced periodic diagnostics and a compressed evidence archive.

## Post-implementation Olivia regression prepared

The Phase 6 Olivia group now contains a fifth bounded test, `phase6_production_ffno_vjp`. It loads the real H checkpoint into the persistent GPU backend, applies the new population VJP, and compares a directional derivative with centered finite differences. It has a 600 second external timeout, periodic Python tracebacks and the existing scheduler cleanup/diagnostic archive.

Run:

```text
bash scripts/submit_olivia_phase6_tests.sh
```

Required new marker:

```text
OLIVIA_PRODUCTION_FFNO_VJP_OK
```

## Phase assessment

Phase 6 implementation is complete: the production gradient, scalable solver, MPI rank-0 VJP service, mixed NLTE/LTE test, executable two-input/two-output route and local/MPI validation artifacts exist. Final target-hardware sign-off requires rerunning the updated Olivia Phase 6 group once and retaining the new real-checkpoint VJP marker. No further Phase 6 source implementation is currently identified.
