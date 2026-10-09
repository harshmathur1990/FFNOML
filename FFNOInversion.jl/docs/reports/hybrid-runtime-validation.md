# Hybrid runtime validation

## Implemented

- `ParallelContext` with serial and MPI modes, topology validation and controlled finalization.
- Near-square 2D process-grid selection and deterministic tile ownership.
- `DistributedField` with typed scatter/gather for arbitrary leading axes and trailing x/y axes.
- Corner-complete halo exchange with replicated physical boundaries.
- Rank-local atmosphere extraction, including optional magnetic and derived fields.
- Thread-private synthesis workspaces and static threaded column scheduling.
- Scalar MPI sum/max reductions.
- Rank-0-only GPU coordinator using the non-MPI TCP control protocol from `charge`.
- Configuration keys for MPI enablement, decomposition, threads per rank and GPU launcher rank.

## Validation commands

```sh
julia --project=. --threads=4 test/runtests.jl
julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=. --threads=2 test/mpi_hybrid_worker.jl`)'
```

The first command passes the complete Phase 0-3 suite plus serial decomposition, local atmosphere, thread-workspace and halo tests. The second passes with four MPI ranks and two Julia threads per rank. It verifies typed scatter/gather, 2D/corner halo values, threaded local reductions, distributed forward-model parity against a monolithic result, exactly one GPU launcher invocation, and communicator recovery after the GPU control stage.

## Superseded status note

The integrations originally listed here as pending are now implemented and validated in `phase-4-validation.md`. Collective HDF5 hyperslab output remains a later output-backend optimization; the current Phase 4 adapter collects final products on rank 0.
