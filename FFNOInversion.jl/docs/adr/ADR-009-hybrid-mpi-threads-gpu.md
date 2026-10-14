# ADR-009: Hybrid MPI, Julia threads, and rank-0 GPU control

## Status

Accepted and implemented as the Phase 4 runtime foundation.

## Decision

- MPI owns inter-process and inter-node decomposition. The spatial domain is represented by non-empty 2D Cartesian tiles.
- Julia threads own independent column work inside a rank. Mutable synthesis scratch arrays are private to a Julia thread; immutable atomic and Kurucz caches may be shared.
- MPI calls are restricted to the thread that initialized the parallel context.
- Global MPI rank 0 alone launches and controls GPU work. A TCP status channel, adapted from the `charge` branch, prevents non-root ranks from remaining inside an MPI collective during nested Slurm/NCCL execution. The launched population service is one persistent multi-node `torchrun` job, not one Python process per inversion evaluation.
- Every allocated GPU rank constructs an FSDP wrapper. Only torchrun rank 0 reads the full checkpoint; `sync_module_states` initializes the other ranks and `FULL_SHARD` owns parameter shards between wrapped-block evaluations.
- FFNO activations are decomposed into H slabs across the same GPU ranks. Distributed FFT transposes, global normalization reductions and halo exchanges have explicit autograd paths, so prediction and VJP use the same global 3D operator. FSDP alone is not considered sufficient because it would still replicate the dominant full-volume activations.
- The TCP control setup and non-root status wait have configurable internal timeouts. A status timeout closes the waiting socket, records the last durable phase, and fails the affected outer rank so Slurm can terminate a launcher rank that remains blocked.
- Optional per-rank diagnostics write durable phase events and periodic CPU/memory/resource samples. Events cover endpoint setup, peer connections, pre-launch barrier, launcher entry/return, status wait/receipt/timeout, post-launch barrier, completion and failure.
- Atmospheric and spectral payloads use typed MPI buffers. Julia serialization is limited to small control messages on the GPU status socket.
- Spatially coupled operations use x/y halo exchange. X is exchanged first and y second including x halos, so corners are correct.
- Rank/thread topology is a launch-time decision. Scientific code must give the same result for compatible topologies.

## Timeout boundary

`gpu_connect_timeout_seconds`, `gpu_status_timeout_seconds`,
`gpu_diagnostic_interval_seconds`, and `gpu_diagnostics_directory` are part of
`ParallelOptions`; `FFNO_GPU_CONNECT_TIMEOUT`, `FFNO_GPU_STATUS_TIMEOUT`,
`FFNO_GPU_DIAGNOSTIC_INTERVAL`, and `FFNO_GPU_DIAGNOSTICS_DIR` override them at
launch time. Zero disables the corresponding status timeout or monitor.
When no diagnostics directory is configured, Slurm runs automatically use
`gpu-control-diagnostics-slurm-JOBID` in the working directory; non-Slurm runs
remain quiet unless a directory is requested explicitly.

The internal status timeout is not a safe mechanism for interrupting an
arbitrary rank-0 CUDA or Python call. It releases non-root socket waiters and
causes the outer Slurm step to fail. The batch watchdog remains the final
authority for killing an unresponsive rank-0 process and any nested GPU step.

## Memory ownership

After initial distribution, every Julia MPI rank owns only its tile and
workspaces. Rank 0 may transiently hold a complete input while
streaming/scattering it, but callers must release that input after distribution.
No GPU owns the global FFNO activation volume: each owns one H slab. No GPU
retains the complete parameter set: FSDP all-gathers only the currently executing
wrapped unit and reshares it afterward. The service remains alive across all
objective and VJP calls and is shut down collectively when the inversion ends.

## Consequences

HE3D/MHS, spatial PSFs and horizontal regularization must be written against distributed fields and halo APIs. Column synthesis, Kurucz LTE work and column-local EOS operations can use rank-local Julia threads. Optimizer scalars use MPI reductions; atmospheric gradients remain tile-local.
