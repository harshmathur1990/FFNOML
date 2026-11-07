#!/bin/bash
#SBATCH --job-name=FNOML_lc_train
#SBATCH --partition=accel
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --account=nn2834k
#SBATCH --exclude=gpu-1-111

# One learning-curve experiment, from scratch: build its training and validation sets, then train as train_gpu.sh does.
# Submitted by scripts/submit_learning_curve.py, which writes ${FFNOML_RUN_DIR}/experiment.env (splits, atoms, repo).

set -o errexit
set -o nounset
set -o pipefail

module -q restore
source /cluster/home/harshm/loadnvidiampi.sh   # the accelerator-node environment (nvidiaenv, NCCL and CXI settings)

: "${FFNOML_RUN_DIR:?set by submit_learning_curve.py}"
source "${FFNOML_RUN_DIR}/experiment.env"
export FFNOML_REPO_DIR FFNOML_RUN_DIR FFNOML_DATA_DIR FFNOML_PACKED_DATA FFNOML_TRAIN_SPLIT FFNOML_VAL_SPLIT FFNOML_ACTIVE_ATOMS FFNOML_NUM_EPOCHS

mkdir -p "${FFNOML_RUN_DIR}/IO" "${FFNOML_RUN_DIR}/training_FFNO3D_zscale_expand_lognlte"
cd "${FFNOML_REPO_DIR}"
PYTHON=/cluster/home/harshm/nvidiaenv/bin/python
TORCHRUN=/cluster/home/harshm/nvidiaenv/bin/torchrun

echo "Experiment: ${FFNOML_EXPERIMENT}"
echo "Train split: ${FFNOML_TRAIN_SPLIT}"
echo "Val split: ${FFNOML_VAL_SPLIT}"
echo "Atoms: ${FFNOML_ACTIVE_ATOMS}  epochs: ${FFNOML_NUM_EPOCHS}"
echo "Repository: ${FFNOML_REPO_DIR}  run dir: ${FFNOML_RUN_DIR}  data: ${FFNOML_DATA_DIR}"
echo "Job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}, ${SLURM_NNODES} nodes x ${SLURM_GPUS_ON_NODE} GPUs. Start: $(date)"

# the datasets: one process, skipped when they exist (a resubmitted job only trains)
srun --nodes=1 --ntasks=1 "${PYTHON}" pipeline.py --build 2>&1 | tee "${FFNOML_RUN_DIR}/build-${SLURM_JOB_ID}.txt"

export OMP_NUM_THREADS=8
export NCCL_DEBUG=ERROR
export HSA_FORCE_FINE_GRAIN_PCIE=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_DEFAULT_TX_SIZE=2048
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export FI_CXI_RX_MATCH_MODE=hybrid
export NCCL_PROTO=^LL128

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=29501

# from scratch (no --expand), so every experiment starts the same way; --resume continues an interrupted one
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  "${TORCHRUN}" \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${SLURM_GPUS_ON_NODE}" \
    --node_rank="${SLURM_PROCID}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    pipeline.py --train ${FFNOML_TRAIN_FLAGS:-} \
  2>&1 | tee "${FFNOML_RUN_DIR}/output-${SLURM_JOB_ID}.txt"

echo "End: $(date)"
