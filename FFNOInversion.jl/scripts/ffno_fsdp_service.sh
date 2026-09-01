#!/bin/bash
# Persistent multi-node torchrun/FSDP service launched only by Julia MPI rank 0.

set -o errexit
set -o nounset
set -o pipefail

manifest=${1:?usage: ffno_fsdp_service.sh MANIFEST.toml}
: "${SLURM_JOB_ID:?must run inside an Olivia Slurm allocation}"
: "${SLURM_NNODES:?SLURM_NNODES is not set}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is not set}"
: "${OLIVIA_REPO_DIR:?OLIVIA_REPO_DIR is not set}"

gpus_per_node=${FFNO_FSDP_GPUS_PER_NODE:-${OLIVIA_GPUS_PER_NODE:-4}}
cpus_per_node=${FFNO_FSDP_CPUS_PER_NODE:-${OLIVIA_GPU_CPUS_PER_NODE:-8}}
python_executable=${OLIVIA_PYTHON:-python3}
master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
master_port=$((26000 + SLURM_JOB_ID % 20000))
rendezvous_id="${SLURM_JOB_ID}-ffno-fsdp-service"
service_script=$(realpath "${OLIVIA_REPO_DIR}/../ffno_fsdp_service.py")
manifest=$(realpath "${manifest}")

echo "Launching persistent FFNO FSDP service nodes=${SLURM_NNODES} gpus_per_node=${gpus_per_node}"
exec srun --overlap --exact --kill-on-bad-exit=1 --mpi=none --cpu-bind=none \
    --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    --gpus-per-node="${gpus_per_node}" --cpus-per-task="${cpus_per_node}" \
    env FFNO_FSDP_SERVICE_SCRIPT="${service_script}" \
        FFNO_FSDP_SERVICE_MANIFEST="${manifest}" \
        FFNO_FSDP_MASTER_ADDR="${master_addr}" \
        FFNO_FSDP_MASTER_PORT="${master_port}" \
        FFNO_FSDP_RENDEZVOUS_ID="${rendezvous_id}" \
        FFNO_FSDP_NPROC_PER_NODE="${gpus_per_node}" \
    bash -c '
        exec "${OLIVIA_PYTHON}" -m torch.distributed.run \
        --nnodes="${SLURM_NNODES}" \
        --nproc_per_node="${FFNO_FSDP_NPROC_PER_NODE}" \
        --node_rank="${SLURM_PROCID}" \
        --rdzv_id="${FFNO_FSDP_RENDEZVOUS_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${FFNO_FSDP_MASTER_ADDR}:${FFNO_FSDP_MASTER_PORT}" \
        "${FFNO_FSDP_SERVICE_SCRIPT}" "${FFNO_FSDP_SERVICE_MANIFEST}"'
