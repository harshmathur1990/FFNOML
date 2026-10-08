#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

: "${SLURM_JOB_ID:?must run inside an Olivia Slurm allocation}"
: "${SLURM_NNODES:?SLURM_NNODES is not set}"
: "${OLIVIA_REPO_DIR:?OLIVIA_REPO_DIR is not set}"

python_executable=${OLIVIA_PYTHON:-python3}
gpus_per_node=${OLIVIA_GLOO_GPUS_PER_NODE:-1}
gpu_cpus_per_node=${OLIVIA_GLOO_CPUS_PER_NODE:-2}
master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
master_port=$((24000 + SLURM_JOB_ID % 20000))
probe=${OLIVIA_REPO_DIR}/test/olivia_gloo_control_probe.py

[[ -f "${probe}" ]] || { echo "Missing Gloo probe: ${probe}" >&2; exit 2; }

export MASTER_ADDR=${master_addr}
export MASTER_PORT=${master_port}
export OLIVIA_GLOO_PROBE_PATH=${probe}
export OLIVIA_PYTHON=${python_executable}
export OLIVIA_GLOO_GPUS_PER_NODE=${gpus_per_node}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

echo "Gloo control smoke test: nodes=${SLURM_NNODES} ranks_per_node=${gpus_per_node} master=${MASTER_ADDR}:${MASTER_PORT}"

srun --overlap --exact --kill-on-bad-exit=1 --mpi=none --network=no_vni --cpu-bind=none \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${gpus_per_node}" \
    --cpus-per-task="${gpu_cpus_per_node}" \
    bash -c '
        if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == "-1" ]]; then
            echo "Gloo smoke step did not receive a Slurm GPU mapping on $(hostname)" >&2
            exit 2
        fi
        exec "${OLIVIA_PYTHON}" -m torch.distributed.run \
            --nnodes="${SLURM_NNODES}" \
            --nproc_per_node="${OLIVIA_GLOO_GPUS_PER_NODE}" \
            --node_rank="${SLURM_PROCID}" \
            --rdzv_id="${SLURM_JOB_ID}-gloo-control" \
            --rdzv_backend=c10d \
            --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
            "${OLIVIA_GLOO_PROBE_PATH}"'
