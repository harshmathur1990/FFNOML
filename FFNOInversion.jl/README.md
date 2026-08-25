# FFNOInversion.jl

Phase 0 scaffold for the spatially coupled Julia inversion layer described in the project planning PDF.

## Run contract

One inversion run has exactly two scientific inputs and two scientific outputs:

1. observation input file - intensity, uncertainty, `(wavelength, Stokes)` weights, and spatial weights;
2. initial-atmosphere input file - `logtau_500`, temperature, velocities, top pressure, and optional magnetic field;
3. synthesis output file - final full-grid synthetic spectrum after the configured PSF, wavelengths, residual/chi-square diagnostics, and provenance;
4. atmosphere output file - recovered atmosphere plus derived pressure, density, electron density, height, populations, convergence history, and provenance.

The TOML file configures these four paths but is not itself a scientific data input.

## Current capabilities

- canonical 3D grids, atmospheres, optional magnetic fields, observations and node fields;
- automatic HE3D selection when B is absent and MHS selection when B is present;
- Stokes-aware spectral cubes with intensity-only/non-PRD Release 1 defaults;
- stable population, redistribution, synthesis and observation interfaces;
- a two-input/two-output run contract plus configuration for `logtau_500`, temperature, velocities, top pressure, `dx/dy`, and any number of simultaneously inverted spectral regions;
- Gaussian spectral/spatial PSFs on the full grid and zero-capable wavelength/2D-spatial chi-square weights;
- vertical and horizontal regularization of selected recovered atmospheric variables;
- deterministic mock forward model, weighted residual packing, configuration dry-run and restart checkpoints.

Phase 1 provides iterative HE3D/MHS reconstruction, an in-memory Wittmann EOS adapter, Lorentz-force diagnostics, and a production 500 nm mass-opacity backend using STiC's continuum routine. The constant-opacity backend remains only for manufactured tests.

Phase 2 adds a persistent PythonCall extension for FFNOML inference plus a Python-free recorded request/response backend. The production runtime loads a checkpoint once, validates normalization and channel/level metadata, and accepts in-memory `(T,vx,vy,vz,ne,rho,z)` volumes on repeated calls. The supplied H and Ca checkpoints pass real-checkpoint integration and persistence tests.

Phase 3 adds production mixed intensity synthesis. Each region may combine FFNO Halpha or Ca II 8542 with RH K94 LTE lines before one formal solution. `build_synthesis_setup` caches Muspel atom/continuum/Voigt data, K94 lists, and the persistent Wittmann partition-function backend outside the pixel loops. Exactly one contributor owns continuum, so blended sources do not double count it. LTE-only regions do not invoke FFNO. Configured STiC-style air wavelengths are retained for observations and converted to vacuum for line physics.

The Phase 4 runtime foundation uses hybrid MPI plus Julia threading. MPI ranks own non-overlapping 2D spatial tiles; typed numeric payloads are scattered/gathered without Julia object serialization; coupled spatial kernels use corner-complete halo exchange; and `Forward.jl`-style column synthesis uses one mutable workspace per Julia thread. Global rank 0 alone controls the persistent or nested GPU launcher through a TCP status channel, so non-root ranks do not hold an MPI collective open during an overlapping Slurm/NCCL launch. MPI calls remain on the initializing Julia thread.

Olivia deployment validation is provided by `scripts/run_olivia_runtime_tests.sbatch`.
It runs small MPI/CUDA/NCCL probes, intentional CPU and GPU-rank stalls with
bounded timeouts, Slurm-step cleanup, failure propagation, post-timeout recovery,
and diagnostic collection without loading the scientific forward model. See
`docs/reports/olivia-runtime-test-guide.md` for submission and return-artifact
instructions.

The production GPU control plane also supports an internal non-root status
timeout and charge-style periodic per-rank diagnostics. Configure these through
the `parallel` TOML keys or the `FFNO_GPU_*` environment overrides documented in
ADR-009. The external Slurm watchdog remains necessary for a rank-0 process that
cannot be interrupted safely.

Phase 4 integrates that runtime into the sole application-level forward path: distributed node expansion, HE3D/optional-B MHS and EOS, rank-0 population inference with tile redistribution, threaded local synthesis, halo-aware spatial PSF, distributed observations and weights, global chi-squared/regularization reductions, and final atmosphere/spectrum collection. One-rank development runs and multi-rank production runs invoke the same `HybridForwardModel` and `forward!` implementation.

The scheduler chooses rank count and Julia chooses threads per rank. For example, a 256-core allocation can start with 16 ranks and 16 threads per rank. Set BLAS/OpenMP thread counts to one when Julia threads own column-level parallelism.

## Grid and objective convention

The atmosphere file supplies the complete `logtau_500` vector and the initial temperature and velocity cubes. Repeatable `[[regions]]` tables configure simultaneous spectral windows with their synthesis grids, continuum normalization, PSF type, and PSF file; `dx_m`/`dy_m` describe the spatial forward grid. The observation layer convolves the full spectral-spatial cube without resizing it. Chi-square uses a `(wavelength, Stokes)` weight matrix and a general 2D spatial-weight map. Zero entries exclude wavelengths, Stokes components, or pixels; an all-zero Stokes column disables inversion of that component.

Regularization is evaluated on the full recovered atmosphere independently of spectral chi-square weights. Vertical regularization uses seven fixed slots `(Temp, Vlos, vturb, B, inc, azi, pgas_boundary)`, one global multiplier, seven relative weights, and per-slot types: 0 none, 1 first derivative, 2 deviation from depth mean, 3 deviation from zero, and 4 second derivative. Temperature types 2/3 normalize to 0. Horizontal strengths and derivative order remain separately configurable.

## Verify

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. scripts/dry_run.jl configs/example_intensity_nonprd.toml
julia --project=. benchmarks/mock_forward.jl

# Local hybrid integration example: 4 MPI ranks x 2 Julia threads
julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=. --threads=2 test/mpi_hybrid_worker.jl`)'

# Full Phase 4 one-rank versus four-rank topology parity
julia --project=. scripts/validate_phase4_topology.jl
```
