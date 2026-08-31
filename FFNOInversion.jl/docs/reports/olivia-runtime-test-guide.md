# Olivia cumulative Phase 1-6 regression guide

This is the only Olivia acceptance workflow for the inversion package. It runs every retained Phase 1-6 test, all newly added FSDP tests, and the earlier MPI/GPU timeout and recovery tests. Phase-specific submission helpers have been removed; future tests must be added to this cumulative suite.

## Submit

The helper can be invoked from any working directory:

```sh
bash /cluster/work/projects/nn2834k/harshm/FFNOML/FFNOInversion.jl/scripts/submit_olivia_regression.sh
```

The default checkout is `/cluster/work/projects/nn2834k/harshm/FFNOML/FFNOInversion.jl`; `OLIVIA_REPO_DIR` is unnecessary for that checkout. Set it only to test another package directory. Additional `sbatch` arguments passed to the helper are forwarded to every allocation.

The helper submits five `afterok`-chained allocations:

1. the full old-plus-new Phase 1-6 regression;
2. internal GPU-status-timeout containment;
3. healthy recovery in a fresh allocation;
4. external-watchdog timeout containment;
5. healthy recovery in another fresh allocation.

All five jobs must exit normally. If one fails, later `afterok` jobs do not provide acceptance evidence.

## Environment and data prerequisites

Inversion jobs source `/cluster/home/harshm/loadnvidiampi.sh` by default. Set `OLIVIA_ENV_SCRIPT` only for an alternate MPI-compatible environment. The worker resolves Julia and Python to absolute paths and verifies them on every node before testing.

Instantiate the package environment once on a login node. The manifest pins Muspel to Git commit `01ec68d`, which supports a three-dimensional height array:

```sh
JULIA_DEPOT_PATH=/cluster/work/projects/nn2834k/harshm/julia-depot-1.12.2 \
    julia --project=/cluster/work/projects/nn2834k/harshm/FFNOML/FFNOInversion.jl \
    -e 'using Pkg; Pkg.instantiate()'
```

No `testdata` directory and no `config.py` change are required. The real-reference case uses these existing paths by default:

- `/cluster/work/projects/nn2834k/harshm/bifrost_data/en024048_hion/385/{mesh,atm3d}`;
- the H and Ca population/intensity files under `/cluster/work/projects/nn2834k/harshm/FFNOML/training_FFNO3D_zscale_expand_lognlte`;
- `3D_sim_train_H.pt` in that same training directory;
- atoms under `/cluster/work/projects/nn2834k/harshm/multi3d/input/atoms`.

Set `FFNO_REFERENCE_ATMOSPHERE_DIR` only if the Bifrost snapshot is stored elsewhere.

The worker configures MPI.jl to use the system OpenMPI selected by `loadnvidiampi.sh`, with `srun` and the OpenMPI ABI. It keeps outer Julia ranks CUDA-blind with `CUDA_VISIBLE_DEVICES=-1`; only the overlapping torchrun step receives GPUs from Slurm.

## Full-regression allocation

The first allocation runs 17 bounded cases:

1. the complete local Julia `runtests.jl` suite for Phases 1-6, including the FSDP-only architecture guard and native three-dimensional Muspel height test;
2. real H and Ca Muspel parity using the supplied atmosphere, population and trusted-intensity files;
3. Phase 4 one-rank/four-rank full-field topology parity;
4. Phase 4 cross-topology checkpoint restart;
5. Phase 5 MPI topology and restart parity;
6. Phase 6 gradient/L-BFGS MPI topology and restart parity;
7. healthy multi-node MPI initialization and reduction;
8. intentional CPU-rank stall with bounded cleanup;
9. healthy MPI reuse after that timeout;
10. healthy rank-0-controlled nested CUDA/NCCL execution;
11. injected GPU-process failure and recovery propagation;
12. distributed Phase 6 L-BFGS and restart;
13. rejected-trial atmosphere restoration;
14. configurable 800 by 800 rank-owned memory layout;
15. CUDA autograd VJP and NCCL checksum;
16. real-checkpoint FFNO `FULL_SHARD` FSDP VJP versus finite differences;
17. the outer Julia MPI to persistent FSDP predict/VJP/shutdown route.

Cases 16 and 17 are the production FFNO acceptance gates. They require every torchrun rank to participate, at least two GPU ranks, exactly one full-checkpoint reader, parameter shards smaller than the full model, H-slab-distributed activations, persistent service reuse and clean collective shutdown.

The memory test defaults can be changed with `PHASE6_MEMORY_NX`, `PHASE6_MEMORY_NY`, `PHASE6_MEMORY_NZ`, `PHASE6_MEMORY_NLAMBDA`, `PHASE6_MEMORY_LEVELS` and `PHASE6_MEMORY_LIMIT_GIB`.

## Timeout allocations and diagnostics

Internal and external GPU-stall tests run in separate allocations because force-killing a nested GPU step can leave the allocation's Slingshot VNI unusable. Each is followed by a fresh-allocation healthy GPU case. An intentional stall passes only if the expected injected-stall marker was written before the bounded timeout; initialization failures do not count.

The worker records scheduler snapshots, five-second Julia/Python heartbeats, stack traces, cleanup actions, interconnect readiness, node GPU health and case stdout/stderr. Important overrides are `FFNO_GPU_CONNECT_TIMEOUT`, `FFNO_GPU_STATUS_TIMEOUT`, `FFNO_GPU_DIAGNOSTIC_INTERVAL` and `FFNO_GPU_DIAGNOSTICS_DIR`.

## Acceptance evidence to return

Every allocation writes:

```text
olivia-runtime-JOBID.out
olivia-runtime-JOBID.err
olivia-runtime-evidence-JOBID/
olivia-runtime-evidence-JOBID.tar.gz
```

The complete regression passes only when all five logs end with `OLIVIA_RUNTIME_TESTS_OK group=...`. The first log must also contain:

```text
OLIVIA_PHASE1_TO_PHASE6_JULIA_TESTS_OK
PHASE3_MUSPEL_REFERENCE_OK
PHASE4_FULL_FIELD_PARITY_OK
PHASE4_RESTART_TOPOLOGY_PARITY_OK
PHASE5_MPI_TOPOLOGY_RESTART_OK
PHASE6_MPI_TOPOLOGY_RESTART_OK
OLIVIA_PRODUCTION_FSDP_FFNO_VJP_OK
OLIVIA_FSDP_SERVICE_INTEGRATION_OK
```

Return the five `.out` and `.err` files, the five evidence archives, and:

```sh
sacct -j JOB1,JOB2,JOB3,JOB4,JOB5 \
    --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList
```
