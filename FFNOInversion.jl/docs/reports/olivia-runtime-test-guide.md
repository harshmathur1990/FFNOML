# Olivia hybrid-runtime test guide

This harness tests only MPI, Julia-thread, rank-0 launcher, CUDA/NCCL, timeout,
cleanup and recovery behavior. It does not load an atmosphere, FFNO checkpoint,
atom, Kurucz list or synthesis code.

Inversion MPI jobs source `/cluster/home/harshm/loadnvidiampi.sh` by default.
Set `OLIVIA_ENV_SCRIPT` only when an alternate MPI-compatible module setup is
required. Non-inversion training and prediction jobs are unaffected.
The harness resolves the module-provided Julia and Python commands to absolute
paths on the batch host and verifies both executables on every allocated node
before launching MPI. `OLIVIA_JULIA` and `OLIVIA_PYTHON` are optional explicit
overrides; normal runs do not need either variable.

Each batch worker enables the production control-plane diagnostics every 5 seconds,
a 60-second peer-connection timeout and a 180-second normal status timeout.
The internal-timeout case overrides the status timeout to 20 seconds; the
external-watchdog case sets it to zero deliberately. The relevant overrides are
`FFNO_GPU_CONNECT_TIMEOUT`, `FFNO_GPU_STATUS_TIMEOUT`,
`FFNO_GPU_DIAGNOSTIC_INTERVAL`, and `FFNO_GPU_DIAGNOSTICS_DIR`.

After a recoverable timeout or nonzero step exit, the worker cancels orphaned
numeric Slurm steps and performs bounded multi-node PMIx readiness probes.
Hard-killing a stalled nested GPU step was observed to make Olivia's job-level
Slingshot VNI unusable for the remainder of that allocation: repeated new MPI
steps reported `Error configuring interconnect` for many minutes. Therefore,
the internal and external GPU-deadlock cases are terminal cases in separate
allocations. Their dependent successor allocations prove fresh-start recovery;
the harness does not claim that a destroyed job VNI can be repaired in place.

## Submit the full validation chain

From the `FFNOInversion.jl` package directory on Olivia, use the submission
helper rather than submitting the batch worker directly:

```sh
bash scripts/submit_olivia_runtime_tests.sh
```

It can also be invoked from the package's `scripts` directory:

```sh
bash submit_olivia_runtime_tests.sh
```

The helper submits four dependency-chained allocations: safe/recoverable
tests, internal-timeout containment, external-watchdog containment, and final
fresh-allocation recovery. It prints every job ID and the final job to monitor.
Additional `sbatch` options supplied to the helper are forwarded to all four
jobs. Directly submitting `run_olivia_runtime_tests.sbatch` runs only the
non-destructive `safe` group and is intended for focused debugging, not full
acceptance evidence.

The default package root is
`/cluster/work/projects/nn2834k/harshm/FNOML/FFNOInversion.jl`. The submission
helper may be invoked from any directory through its absolute path. Each batch
worker validates `Project.toml` and `src/FFNOInversion.jl` before running. An
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
OLIVIA_JULIA_DEPOT=/cluster/work/projects/nn2834k/harshm/julia-depot-1.12.2 \
OLIVIA_PYTHON=/cluster/home/harshm/nvidiaenv/bin/python \
    bash /cluster/work/projects/nn2834k/harshm/FNOML/FFNOInversion.jl/scripts/submit_olivia_runtime_tests.sh
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

## Test sequence across allocations

1. Healthy multi-node MPI initialization, barrier and scalar allreduce.
2. Intentional CPU-rank stall. The wrapper must time out, kill numbered Slurm
   steps and retain per-rank last-event/heartbeat logs.
3. Healthy MPI rerun after the timeout.
4. Healthy rank-0-controlled nested `torchrun`, CUDA allocation and NCCL scalar
   allreduce across every allocated GPU.
5. Intentional GPU-process failure. Every outer MPI rank must receive the same
   launcher error, then the control path must recover.
6. In a separate allocation, intentionally stall a GPU rank with the internal
   status timeout enabled. A
   non-root inversion rank must record `gpu_status_timeout`; Slurm then cleans
   up the failed outer and nested steps.
7. Rerun the healthy hybrid GPU case in a fresh dependent allocation after the
   internal timeout.
8. In that allocation, intentionally stall a GPU rank with the internal timeout
   deliberately disabled.
   The batch wrapper must time out and kill the outer and nested steps. Python
   faulthandler writes periodic stack traces while the stall is active.
9. Rerun the healthy hybrid GPU case in a fresh dependent allocation after the
   external timeout.

Every healthy case must contain its explicit MPI or GPU completion marker. The
injected GPU-failure/recovery case must contain
`intentional_gpu_process_failure`; an unrelated CUDA/NCCL initialization error
is not accepted as the requested injected failure. An intentional stall is
counted as passing only when GNU `timeout` terminates it and the rank logs
contain the exact injected-stall marker. A timeout during MPI, CUDA or NCCL
initialization is therefore a failure, not a false positive. An early crash is
also a failure because it did not exercise timeout containment.

## Send back

Each allocation's final log prints paths like:

```text
Diagnostics: .../olivia-runtime-evidence-JOBID
Archive: .../olivia-runtime-evidence-JOBID.tar.gz
```

The complete validation passes only when all four logs end with their matching
`OLIVIA_RUNTIME_TESTS_OK group=...` marker.

Please provide:

- `olivia-runtime-JOBID.out` and `.err`;
- the generated `olivia-runtime-evidence-JOBID.tar.gz` archive;
- `sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList`.

The archive contains the command and stdout/stderr for each case, scheduler
snapshots sampled during each run, cleanup actions, post-case `nvidia-smi`, MPI
rank heartbeats, GPU-rank heartbeats and Python stack dumps.
