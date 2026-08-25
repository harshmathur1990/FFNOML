# Olivia hybrid-runtime test guide

This harness tests only MPI, Julia-thread, rank-0 launcher, CUDA/NCCL, timeout,
cleanup and recovery behavior. It does not load an atmosphere, FFNO checkpoint,
atom, Kurucz list or synthesis code.

Inversion MPI jobs source `/cluster/home/harshm/loadnvidiampi.sh` by default.
Set `OLIVIA_ENV_SCRIPT` only when an alternate MPI-compatible module setup is
required. Non-inversion training and prediction jobs are unaffected.

## Submit

From the `FFNOInversion.jl` package directory on Olivia:

```sh
sbatch scripts/run_olivia_runtime_tests.sbatch
```

The package environment must already be instantiated in the selected depot:

```sh
JULIA_DEPOT_PATH=/cluster/work/projects/nn2834k/harshm/julia-depot-1.12.2 \
    julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The defaults request two `accel` nodes, four GPUs per node, one outer MPI rank
per node and four Julia threads per rank. Override paths at submission time when
the checkout or environments differ:

```sh
sbatch --export=ALL,OLIVIA_REPO_DIR=/cluster/work/projects/nn2834k/harshm/FNOML/FFNOInversion.jl,OLIVIA_JULIA_DEPOT=/cluster/work/projects/nn2834k/harshm/julia-depot-1.12.2,OLIVIA_PYTHON=/cluster/home/harshm/nvidiaenv/bin/python scripts/run_olivia_runtime_tests.sbatch
```

By default the Julia project is the package directory. Set
`OLIVIA_JULIA_PROJECT` only if that alternate environment already contains a
`Pkg.develop` entry for this checkout.

Use at least two nodes: the purpose is to exercise Olivia's inter-node PMIx,
Slingshot and NCCL paths. The probes transfer only scalar tensors.

## Test sequence

1. Healthy multi-node MPI initialization, barrier and scalar allreduce.
2. Intentional CPU-rank stall. The wrapper must time out, kill numbered Slurm
   steps and retain per-rank last-event/heartbeat logs.
3. Healthy MPI rerun after the timeout.
4. Healthy rank-0-controlled nested `torchrun`, CUDA allocation and NCCL scalar
   allreduce across every allocated GPU.
5. Intentional GPU-process failure. Every outer MPI rank must receive the same
   launcher error, then the control path must recover.
6. Intentional GPU-rank stall after CUDA/NCCL initialization. The wrapper must
   time out and kill the outer and nested steps. Python faulthandler writes
   periodic stack traces while the stall is active.
7. Healthy hybrid GPU rerun after the timeout.

An intentional stall is counted as passing only when GNU `timeout` terminates
it and the rank logs contain the exact injected-stall marker. A timeout during
MPI, CUDA or NCCL initialization is therefore a failure, not a false positive.
An early crash is also a failure because it did not exercise timeout containment.

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
