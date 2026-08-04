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

    cd "${repository_dir}"
    exec "${torchrun_path}" \
        --nnodes="${node_count}" \
        --nproc_per_node="${gpu_count}" \
        --node_rank="${SLURM_PROCID}" \
        --rdzv_id="${rendezvous_id}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${master_address}:${master_port}" \
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
: "${SLURM_GPUS_ON_NODE:?SLURM_GPUS_ON_NODE is not set}"

master_address=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
master_port=${FORWARD_MASTER_PORT:-29501}
rendezvous_id="${SLURM_JOB_ID}-${prediction_name}-$$"

echo "Launching coordinated FFNoML prediction '${prediction_name}' on ${SLURM_NNODES} nodes"
echo "torchrun rendezvous: ${master_address}:${master_port}"

srun --overlap \
    --kill-on-bad-exit=1 \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${SLURM_GPUS_ON_NODE}" \
    "${script_path}" --node-worker \
    "${prediction_name}" \
    "${solve_h5}" \
    "${prediction_output}" \
    "${master_address}" \
    "${master_port}" \
    "${SLURM_NNODES}" \
    "${SLURM_GPUS_ON_NODE}" \
    "${rendezvous_id}"
