# Olivia hybrid-runtime test guide

This harness tests only MPI, Julia-thread, rank-0 launcher, CUDA/NCCL, timeout,
cleanup and recovery behavior. It does not load an atmosphere, FFNO checkpoint,
atom, Kurucz list or synthesis code.

Inversion MPI jobs source `/cluster/home/harshm/loadnvidiampi.sh` by default.
Set `OLIVIA_ENV_SCRIPT` only when an alternate MPI-compatible module setup is
required. Non-inversion training and prediction jobs are unaffected.

The harness enables the production control-plane diagnostics every 5 seconds,
a 60-second peer-connection timeout and a 180-second normal status timeout.
The internal-timeout case overrides the status timeout to 20 seconds; the
external-watchdog case sets it to zero deliberately. The relevant overrides are
`FFNO_GPU_CONNECT_TIMEOUT`, `FFNO_GPU_STATUS_TIMEOUT`,
`FFNO_GPU_DIAGNOSTIC_INTERVAL`, and `FFNO_GPU_DIAGNOSTICS_DIR`.

## Submit

From the `FFNOInversion.jl` package directory on Olivia:

```sh
sbatch scripts/run_olivia_runtime_tests.sbatch
```

Submitting from the package's `scripts` directory is also supported:

```sh
sbatch run_olivia_runtime_tests.sbatch
```

The default package root is
`/cluster/work/projects/nn2834k/harshm/FNOML/FFNOInversion.jl`, so the script
may be submitted from any directory when invoked by its absolute path. The
harness validates `Project.toml` and `src/FFNOInversion.jl` before running. An
explicit `OLIVIA_REPO_DIR` takes precedence only when deliberately testing a
different checkout; submission-directory and parent-directory discovery remain
fallbacks if the default checkout is unavailable.

The package environment must already be instantiated in the selected depot:

```sh
JULIA_DEPOT_PATH=/cluster/work/projects/nn2834k/harshm/julia-depot-1.12.2 \
    julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Before allocating any MPI step, the batch script now imports `FFNOInversion`
and `MPI` from the resolved project. It first configures MPIPreferences to use
the system OpenMPI selected by `loadnvidiampi.sh`, with `srun` as the launcher
and the OpenMPI ABI. This is required because MPI.jl otherwise defaults to
MPICH_jll, which rejects Slurm's PMIx runtime during `MPI_Init_thread`. A missing
package, bad depot, system-library problem, or wrong MPI preference therefore
stops once with exit 2 and leaves the exact error in `preflight.txt`, instead of
producing the same immediate failure for all nine runtime cases. The resulting
`LocalPreferences.toml` is checkout-local and ignored by git.

The defaults request two `accel` nodes, four GPUs per node, one outer MPI rank
per node and four Julia threads per rank. Override paths at submission time only
when the checkout or environments differ:

```sh
sbatch --export=ALL,OLIVIA_JULIA_DEPOT=/cluster/work/projects/nn2834k/harshm/julia-depot-1.12.2,OLIVIA_PYTHON=/cluster/home/harshm/nvidiaenv/bin/python /cluster/work/projects/nn2834k/harshm/FNOML/FFNOInversion.jl/scripts/run_olivia_runtime_tests.sbatch
```

By default the Julia project is the package directory. Set
`OLIVIA_JULIA_PROJECT` only if that alternate environment already contains a
`Pkg.develop` entry for this checkout.

Use at least two nodes: the purpose is to exercise Olivia's inter-node PMIx,
Slingshot and NCCL paths. The probes transfer only scalar tensors.

The outer Julia/OpenMPI step is launched with `CUDA_VISIBLE_DEVICES=-1` even
though the batch allocation contains GPUs. This prevents the CUDA-aware
OpenMPI/OFI runtime from retaining a context on the default GPU of every node;
the observed symptom was `cudaErrorDevicesUnavailable` only for torchrun
`local_rank=0` on every node. The overlapping nested `srun --gpus-per-node`
step receives a fresh, step-local `CUDA_VISIBLE_DEVICES` mapping from Slurm.
The nested launcher validates and records that mapping before starting
torchrun, so a missing remap fails explicitly instead of appearing as an NCCL
problem.

## Test sequence

1. Healthy multi-node MPI initialization, barrier and scalar allreduce.
2. Intentional CPU-rank stall. The wrapper must time out, kill numbered Slurm
   steps and retain per-rank last-event/heartbeat logs.
3. Healthy MPI rerun after the timeout.
4. Healthy rank-0-controlled nested `torchrun`, CUDA allocation and NCCL scalar
   allreduce across every allocated GPU.
5. Intentional GPU-process failure. Every outer MPI rank must receive the same
   launcher error, then the control path must recover.
6. Intentional GPU-rank stall with the internal status timeout enabled. A
   non-root inversion rank must record `gpu_status_timeout`; Slurm then cleans
   up the failed outer and nested steps.
7. Healthy hybrid GPU rerun after the internal timeout.
8. Intentional GPU-rank stall with the internal timeout deliberately disabled.
   The batch wrapper must time out and kill the outer and nested steps. Python
   faulthandler writes periodic stack traces while the stall is active.
9. Healthy hybrid GPU rerun after the external timeout.

Every healthy case must contain its explicit MPI or GPU completion marker. The
injected GPU-failure/recovery case must contain
`intentional_gpu_process_failure`; an unrelated CUDA/NCCL initialization error
is not accepted as the requested injected failure. An intentional stall is
counted as passing only when GNU `timeout` terminates it and the rank logs
contain the exact injected-stall marker. A timeout during MPI, CUDA or NCCL
initialization is therefore a failure, not a false positive. An early crash is
also a failure because it did not exercise timeout containment.

## Send back

The final log prints paths like:

```text
Diagnostics: .../olivia-runtime-evidence-JOBID
Archive: .../olivia-runtime-evidence-JOBID.tar.gz
```

Please provide:

- `olivia-runtime-JOBID.out` and `.err`;
- the generated `olivia-runtime-evidence-JOBID.tar.gz` archive;
- `sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList`.

The archive contains the command and stdout/stderr for each case, scheduler
snapshots sampled during each run, cleanup actions, post-case `nvidia-smi`, MPI
rank heartbeats, GPU-rank heartbeats and Python stack dumps.
