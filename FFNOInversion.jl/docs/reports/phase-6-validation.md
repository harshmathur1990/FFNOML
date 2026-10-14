# Phase 6 validation - multi-GPU FSDP implementation complete, target acceptance pending

Date: 2026-08-31

## Implemented production path

- Generic matrix-free JVP/VJP actions with dot-product validation.
- Exact distributed adjoint of coarse-node expansion, including MPI reduction and physical-to-scaled control conversion.
- Exact transposes for the distributed spectral/spatial Gaussian observation operator and the scalar formal solver.
- Analytic opacity/emissivity pullbacks for FFNO transitions and the Julia Kurucz LTE path.
- A cell-local numerical pullback for the Muspel opacity primitives and a column-local numerical fallback for the opaque Wittmann Kurucz C backend. Neither fallback repeats the formal solution, force balance, FFNO inference or complete forward model.
- Persistent multi-node PyTorch service launched only by Julia MPI rank 0. Every torchrun rank participates in `FULL_SHARD` FSDP and owns one FFNO H slab; the same service handles repeated predictions and VJPs without per-call process or HDF5 staging.
- Differentiable distributed FFT all-to-all, global GroupNorm reductions and H-halo exchange, plus eval-mode activation checkpointing for production-size VJPs.
- The embedded Python runtime and Julia extension are deleted. `run_inversion!` positively requires a persistent `FSDPFFNOModel` service with at least two GPU ranks; mocks and recorded populations cannot enter the executable inversion route.
- Separate H and Ca population workspaces and composite distributed population backends, allowing simultaneous FFNO species plus Kurucz LTE contributors.
- A complete `HybridAdjointObjectiveGradient` that chains weighted chi-square, PSF, formal transfer, line physics, population inference, force reconstruction and regularization into scaled coarse-control gradients.
- The HE3D/MHS plus regularization composite uses bounded centered perturbations of the CPU reconstruction only. It performs one complete forward/GPU evaluation per gradient and does not construct a dense Jacobian.
- Synchronized bounded L-BFGS with projected gradients, limited memory, Armijo backtracking, safe rejected-trial restoration, atomic checkpoint/restart and forward-call accounting.
- Canonical HDF5 input/output and one executable MPI route from the two scientific inputs to the two scientific outputs. File axes `(time,z,y,x)` and `(time,Stokes,wavelength,y,x)` are converted to the internal canonical layouts without interpolation.

## Local evidence

The complete local suite passes 242 assertions. The production PSF and formal-solver pullbacks pass directional dot-product checks. A simultaneous two-species FFNO-transition plus Kurucz-LTE objective gradient agrees with the complete centered finite-difference oracle to `3.67e-10` relative error. Its Taylor remainder is second order, and bounded L-BFGS reduces the mixed objective in 22 complete forward evaluations. A pure-Julia fake service verifies persistent binary prediction/VJP reuse and shutdown. Architecture tests verify the absence of the deleted runtime, reject every non-FSDP application backend, reject a one-rank service, and require the `FULL_SHARD_H_SLAB` protocol capability.

The Phase 3 production adapter is now part of the same suite. Muspel is pinned to Git commit `01ec68d`; the test constructs `Muspel.Atmosphere3D` with an actual three-dimensional height array and then verifies the synthesized spectrum against direct Muspel line preparation. The app could not reopen the sibling 1.3 GB Bifrost atmosphere for the full stored-spectrum rerun (`Operation not permitted`), so that pinned-revision reference case is mandatory in the cumulative Olivia job and remains external evidence.

The HDF5 executable test reads one atmosphere file and one observation file, runs the sole MPI/hybrid inversion path, and writes one synthesis file and one recovered-atmosphere file. It verifies the declared time/Stokes/wavelength/y/x axis contract and population output.

The MPI validator passes for one and four ranks. It exercises the outer MPI gather/control/scatter population seam with no population model on non-root Julia ranks. Both topologies return the exact cotangent, make the same optimizer decisions, and a one-rank checkpoint restarts on four ranks with the uninterrupted result and forward-evaluation count. Real FSDP/NCCL execution cannot be validated on the local non-CUDA host.

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

## Cumulative Olivia acceptance prepared

The only public Olivia submission command now launches one cumulative Phase 1-6 regression chain. Its first allocation runs all local Julia tests, real Muspel atmosphere/population/intensity parity, the Phase 4-6 MPI topology/restart validators, all earlier MPI/GPU health and recovery cases, and the following two production FSDP gates:

1. `phase6_production_fsdp_ffno_vjp`: all eight torchrun ranks construct the distributed FFNO, only global rank 0 reads the full H checkpoint, `FULL_SHARD` leaves every rank with fewer local parameters than the complete model, H-slab prediction and VJP run collectively, and the VJP is compared with centered finite differences.
2. `phase6_persistent_fsdp_service`: the exact outer Julia MPI route launches one persistent FSDP service, performs prediction and VJP through the in-memory socket protocol, proves the same service was reused, and shuts it down collectively.

They have 600 and 900 second external timeouts, periodic Julia/Python tracebacks and the existing scheduler cleanup/diagnostic archive.

The remaining four allocations isolate internal-timeout containment, recovery, external-watchdog containment and a second recovery. Phase-specific submission scripts have been removed so newly developed tests must be appended to this cumulative suite.

Run from any directory:

```text
bash /cluster/work/projects/nn2834k/harshm/FFNOML/FFNOInversion.jl/scripts/submit_olivia_regression.sh
```

Required new marker:

```text
OLIVIA_PRODUCTION_FSDP_FFNO_VJP_OK
OLIVIA_FSDP_SERVICE_INTEGRATION_OK
```

## Phase assessment

Phase 6 source implementation is complete for the corrected architecture: the production gradient, scalable solver, persistent rank-0-controlled multi-GPU FSDP service, distributed FFNO activation/VJP path, mixed NLTE/LTE test, executable two-input/two-output route and local/MPI validation artifacts exist. Phase 6 is not target-accepted until all five jobs from one cumulative Olivia regression chain exit normally, including both FSDP markers above.
