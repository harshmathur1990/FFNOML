# Olivia hybrid-runtime test guide

This harness tests MPI, Julia threads, the rank-0 launcher, CUDA/NCCL, timeout,
cleanup and recovery behavior. Its Phase 6 group additionally runs the small
exact-model distributed L-BFGS fixture and allocates a configurable full-layout
memory model. It does not load a real atmosphere, FFNO checkpoint, atom or
Kurucz list, and it does not claim a production-physics gradient test.

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

The helper submits five dependency-chained allocations: safe/recoverable
tests, Phase 6 solver/GPU prerequisites, internal-timeout containment,
external-watchdog containment, and final fresh-allocation recovery. It prints
every job ID and the final job to monitor.
Additional `sbatch` options supplied to the helper are forwarded to all five
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
producing the same immediate failure for all fourteen runtime cases. The resulting
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
Slingshot and NCCL paths. The GPU collective probes transfer only scalar
tensors; the full-layout memory case allocates rank-local arrays without
communicating model payloads.

No `testdata` directory is needed. The Phase 6 solver fixture is generated in
memory. The memory-layout probe directly constructs rank-owned arrays and does
not read an input model.

## Phase 6 allocation

The dedicated `phase6` allocation runs four bounded cases:

1. The exact-model six-control L-BFGS fixture through the normal multi-node MPI
   route. It checks convergence, control recovery, a forward-call ceiling and
   interrupted-versus-restarted checkpoint identity.
2. A deliberately reversed gradient. All Armijo trials must be rejected and
   the last accepted atmosphere must be restored on every rank.
3. A full-layout allocation probe. Defaults are `NZ=64`, `NX=NY=800`, 134
   wavelengths and 10 population levels in Float64. Every rank allocates only
   its tile. The reported owned arrays and peak rank RSS must remain below 24
   GiB. Override `PHASE6_MEMORY_NX`, `PHASE6_MEMORY_NY`,
   `PHASE6_MEMORY_NZ`, `PHASE6_MEMORY_NLAMBDA`,
   `PHASE6_MEMORY_LEVELS`, or `PHASE6_MEMORY_LIMIT_GIB` when the production
   layout changes.
4. A rank-0-controlled nested CUDA/NCCL probe that computes a known PyTorch
   vector-Jacobian product on every allocated GPU and verifies the distributed
   checksum.

The memory probe validates the currently implemented rank-owned array layout,
not the final production-gradient peak. The CUDA VJP probe validates Olivia's
autograd and launcher environment, not the FFNO checkpoint pullback. These
distinctions are intentional: the production FFNO and CPU-physics pullbacks
must be implemented before their scientific tests can exist.

Run only the Phase 6 allocation with:

```sh
bash scripts/submit_olivia_phase6_tests.sh
```

As with the full-chain helper, this command may be invoked through its absolute
path from any directory. Additional `sbatch` options are forwarded. For
example, a deliberate alternate checkout can still be selected with
`OLIVIA_REPO_DIR`; the established Olivia checkout remains the default.

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
6. Multi-node Phase 6 L-BFGS convergence and forward-call check.
7. Multi-node rejected-trial restoration.
8. Configured full-layout rank-owned allocation and RSS check.
9. Rank-0-controlled CUDA VJP and NCCL checksum.
10. In a separate allocation, intentionally stall a GPU rank with the internal
   status timeout enabled. A
   non-root inversion rank must record `gpu_status_timeout`; Slurm then cleans
   up the failed outer and nested steps.
11. Rerun the healthy hybrid GPU case in a fresh dependent allocation after the
   internal timeout.
12. In that allocation, intentionally stall a GPU rank with the internal timeout
   deliberately disabled.
   The batch wrapper must time out and kill the outer and nested steps. Python
   faulthandler writes periodic stack traces while the stall is active.
13. Rerun the healthy hybrid GPU case in a fresh dependent allocation after the
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

The complete validation passes only when all five logs end with their matching
`OLIVIA_RUNTIME_TESTS_OK group=...` marker.

Please provide:

- `olivia-runtime-JOBID.out` and `.err`;
- the generated `olivia-runtime-evidence-JOBID.tar.gz` archive;
- `sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList`.

The archive contains the command and stdout/stderr for each case, scheduler
snapshots sampled during each run, cleanup actions, post-case `nvidia-smi`, MPI
rank heartbeats, GPU-rank heartbeats and Python stack dumps.
