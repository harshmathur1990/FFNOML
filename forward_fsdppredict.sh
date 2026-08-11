#!/bin/bash
# Rank-0 FFNoML launcher used by Forward.jl --fsdp-launcher.
# Contract: PREDICTION_NAME SOLVE_H5 PREDICTION_OUTPUT.
# The outer invocation creates exactly one Slurm step. That step starts one
# torchrun launcher per node; torchrun then starts one worker per GPU.

set -o errexit
set -o nounset
set -o pipefail

script_path=$(realpath "$0")
repository_dir=${FORWARD_REPO_DIR:-$(dirname "${script_path}")}

if [[ ${1:-} == "--node-worker" ]]; then
    prediction_name=$2
    solve_h5=$3
    prediction_output=$4
    master_address=$5
    master_port=$6
    node_count=$7
    gpu_count=$8
    rendezvous_id=$9
    torchrun_path=${FORWARD_TORCHRUN:-/cluster/home/harshm/nvidiaenv/bin/torchrun}
    rendezvous_join_timeout=${FORWARD_RDZV_JOIN_TIMEOUT:-120}
    [[ ${rendezvous_join_timeout} =~ ^[1-9][0-9]*$ ]] || {
        echo "FORWARD_RDZV_JOIN_TIMEOUT must be a positive integer, got: ${rendezvous_join_timeout}" >&2
        exit 2
    }

    cd "${repository_dir}"
    echo "torchrun node worker: host=$(hostname) node_rank=${SLURM_PROCID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    exec "${torchrun_path}" \
        --nnodes="${node_count}" \
        --nproc_per_node="${gpu_count}" \
        --node_rank="${SLURM_PROCID}" \
        --rdzv_id="${rendezvous_id}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${master_address}:${master_port}" \
        --rdzv_conf="join_timeout=${rendezvous_join_timeout}" \
        pipeline.py --fsdppredict --predname "${prediction_name}" \
        --solve-h5 "${solve_h5}" \
        --prediction-output "${prediction_output}"
fi

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 PREDICTION_NAME SOLVE_H5 PREDICTION_OUTPUT" >&2
    exit 2
fi

prediction_name=$1
solve_h5=$(realpath "$2")
prediction_output=$(realpath -m "$3")
: "${SLURM_JOB_ID:?forward_fsdppredict.sh must run inside a Slurm allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is not set}"
: "${SLURM_NNODES:?SLURM_NNODES is not set}"
: "${FORWARD_GPUS_PER_NODE:?FORWARD_GPUS_PER_NODE is not set}"
[[ ${FORWARD_GPUS_PER_NODE} =~ ^[1-9][0-9]*$ ]] || {
    echo "FORWARD_GPUS_PER_NODE must be a positive integer, got: ${FORWARD_GPUS_PER_NODE}" >&2
    exit 2
}

master_address=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
master_port=${FORWARD_MASTER_PORT:-29501}
rendezvous_id="${SLURM_JOB_ID}-${prediction_name}-$$"
launcher_cpus_per_node=${FORWARD_FSDP_LAUNCHER_CPUS_PER_NODE:-64}
[[ ${launcher_cpus_per_node} =~ ^[1-9][0-9]*$ ]] || {
    echo "FORWARD_FSDP_LAUNCHER_CPUS_PER_NODE must be a positive integer, got: ${launcher_cpus_per_node}" >&2
    exit 2
}

echo "Launching coordinated FFNoML prediction '${prediction_name}' on ${SLURM_NNODES} nodes"
echo "torchrun rendezvous: ${master_address}:${master_port}"
echo "torchrun launcher CPUs/node: ${launcher_cpus_per_node}"

# These tasks are torchrun launchers, not MPI ranks.  Explicitly disable the
# inherited Slurm MPI plugin so the nested step does not wait on a PMIx fence.
# Do not ask Slurm to configure a second VNI for this overlapping launcher
# step. forward_mpi_gpu.sh allocates a shared job VNI; the torch/NCCL children
# use that existing authorization for their Slingshot traffic.
# Also discard the outer Julia rank's all-core CPU mask. Explicitly request the
# smaller CPU slice used successfully by the old four-ranks-per-node layout;
# otherwise srun inherits SLURM_CPUS_PER_TASK=256 and each nested launcher asks
# for the entire node. --exact keeps the overlapping step to these CPUs plus
# its GPUs while the Julia ranks wait at their collective.
srun --overlap \
    --exact \
    --kill-on-bad-exit=1 \
    --mpi=none \
    --network=no_vni \
    --cpu-bind=none \
    --cpus-per-task="${launcher_cpus_per_node}" \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${FORWARD_GPUS_PER_NODE}" \
    "${script_path}" --node-worker \
    "${prediction_name}" \
    "${solve_h5}" \
    "${prediction_output}" \
    "${master_address}" \
    "${master_port}" \
    "${SLURM_NNODES}" \
    "${FORWARD_GPUS_PER_NODE}" \
    "${rendezvous_id}"
