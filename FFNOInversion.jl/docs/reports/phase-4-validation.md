# Phase 4 validation: unified distributed 3D forward model

## Outcome

Phase 4 is implemented through one application-level hybrid path. MPI owns 2D spatial tiles, Julia threads own independent local-column work, and global rank 0 alone controls the persistent/nested GPU population service. A one-rank run uses the same distributed types and `forward!` method as a multi-rank run.

## Integrated path

1. Rank 0 distributes the initial atmosphere into typed, rank-owned tiles.
2. Node maps expand directly onto each tile using global normalized coordinates.
3. HE3D or optional-B MHS reconstructs local pressure, density, electron density and height with MPI halos and global convergence/force reductions.
4. Rank 0 alone invokes the population service; returned population volumes are redistributed to tile owners.
5. Every rank performs synthesis with thread-private workspaces.
6. Spectral PSF work is local; spatial PSF work uses corner-complete MPI halos.
7. Observation cubes and weights are distributed without interpolation.
8. Chi-squared and regularization are reduced globally while residual and atmospheric arrays remain tile-local.
9. Final spectrum and atmosphere products are collected on rank 0 for the current output adapter.
10. Stage timings, per-rank hot-path array ownership, and rank/thread/tile provenance are available as machine-readable records.

## Evidence

```sh
julia --project=. --threads=4 test/runtests.jl
julia --project=. scripts/validate_phase4_topology.jl
julia --project=. scripts/validate_phase4_restart.jl
julia --project=. scripts/benchmark_phase4_local.jl
julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=. --threads=2 test/mpi_phase4_failure_worker.jl`)'
```

The 142/142 unit/integration assertions include HE3D, optional-B MHS, node expansion, observation distribution, rank-local population prediction, threaded synthesis, spatial/spectral PSF, chi-squared, vertical/horizontal regularization, final collection, stable workspace identities, deterministic repeated spectra, stage timings, memory reporting and TOML provenance output.

The topology validator runs the complete application path with one MPI rank and four MPI ranks (two Julia threads per rank) on an uneven `nz=4, nx=11, ny=7` grid. Ten complete fields - spectrum, populations, temperature, pressure, density, electron density, height and all three magnetic components - agree at `rtol=1e-12, atol=1e-14`; regularization agrees at `rtol=1e-13`. The globally coupled population fixture depends on the complete volume mean and therefore checks rank-0 staging rather than only a pointwise mock. The four-rank memory audit reports a maximum hot-path owned-array total of 15,744 bytes and asserts that no rank owns the complete spatial domain.

Checkpoint portability passes in both directions (write with one rank and restore with four; write with four and restore with one). An injected rank-0 GPU-launcher error reaches all four ranks, and a subsequent control launch succeeds, proving communicator/control recovery.

The local equal-core smoke benchmark produced the same spectrum checksum for `1 rank x 4 threads`, `2 x 2`, and `4 x 1`. Instrumented forward times were approximately 0.695, 1.018 and 1.049 seconds respectively; maximum owned arrays decreased from 47,728 to 25,824 to 14,016 bytes. This tiny fixture is communication/latency dominated and is not a production topology recommendation.

The GPU callback is a deterministic globally coupled population fixture because GPU hardware is unavailable in the local validation environment. The MPI launch, rank-0-only control socket, atmospheric staging and population redistribution are real.

## Deployment note

The current final-output adapter gathers products to rank 0. The scientific hot path retains only rank-owned arrays outside the transient rank-0 FFNO staging operation. Collective HDF5 hyperslab output remains an output-backend optimization; it does not change the Phase 4 scientific or parallel architecture. Target GPU, multi-node scheduler/network, per-process RSS/device-memory and production-size rank/thread scaling evidence cannot be produced on this workstation and remain the deployment acceptance boundary.
