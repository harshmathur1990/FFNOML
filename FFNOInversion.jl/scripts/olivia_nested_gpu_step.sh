#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

mode=${1:?usage: olivia_nested_gpu_step.sh success|failure|stall}
case "${mode}" in
    success) port_offset=1 ;;
    failure) port_offset=2 ;;
    stall) port_offset=3 ;;
    *) echo "unknown GPU probe mode: ${mode}" >&2; exit 2 ;;
esac

: "${SLURM_JOB_ID:?must run inside an Olivia Slurm allocation}"
: "${SLURM_NNODES:?SLURM_NNODES is not set}"
: "${OLIVIA_REPO_DIR:?OLIVIA_REPO_DIR is not set}"
: "${OLIVIA_CASE_DIAGNOSTICS:?OLIVIA_CASE_DIAGNOSTICS is not set}"

gpus_per_node=${OLIVIA_GPUS_PER_NODE:-4}
gpu_cpus_per_node=${OLIVIA_GPU_CPUS_PER_NODE:-8}
python_executable=${OLIVIA_PYTHON:-python3}
master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
master_port=$((22000 + SLURM_JOB_ID % 20000 + port_offset))
probe=${OLIVIA_REPO_DIR}/test/olivia_gpu_collective_probe.py

export MASTER_ADDR=${master_addr}
export MASTER_PORT=${master_port}
export OLIVIA_GPU_PROBE_MODE=${mode}
export OLIVIA_GPU_PROBE_PATH=${probe}
export OLIVIA_PYTHON=${python_executable}
export OLIVIA_GPUS_PER_NODE=${gpus_per_node}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

echo "nested_gpu_step mode=${mode} nodes=${SLURM_NNODES} gpus_per_node=${gpus_per_node} master=${MASTER_ADDR}:${MASTER_PORT}"

srun --overlap --exact --kill-on-bad-exit=1 --mpi=none --network=no_vni --cpu-bind=none \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${gpus_per_node}" \
    --cpus-per-task="${gpu_cpus_per_node}" \
    bash -c '
        if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == "-1" ]]; then
            echo "nested GPU step did not receive a Slurm GPU visibility mapping on $(hostname)" >&2
            exit 2
        fi
        echo "torchrun node worker: host=$(hostname) node_rank=${SLURM_PROCID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        exec "${OLIVIA_PYTHON}" -m torch.distributed.run \
        --nnodes="${SLURM_NNODES}" \
        --nproc_per_node="${OLIVIA_GPUS_PER_NODE}" \
        --node_rank="${SLURM_PROCID}" \
        --rdzv_id="${SLURM_JOB_ID}-${OLIVIA_GPU_PROBE_MODE}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        "${OLIVIA_GPU_PROBE_PATH}" "${OLIVIA_GPU_PROBE_MODE}"'
