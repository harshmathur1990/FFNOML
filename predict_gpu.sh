#!/bin/bash
#SBATCH --job-name=FNOML_train
#SBATCH --partition=accel
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --account=nn2834k
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --exclude=gpu-1-111

set -o errexit
set -o nounset
set -o pipefail

module -q restore
module list

source /cluster/home/harshm/loadnvidia.sh
#module load NRIS/GPU
#module load PyTorch/2.10.0

cd /cluster/work/projects/nn2834k/harshm/FFNOMLcopy

echo "Work dir: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Num nodes: ${SLURM_NNODES}"
echo "GPUs per node: ${SLURM_GPUS_ON_NODE}"
echo "Start: $(date)"

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

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

# Bind libfabric (adjust the path based on your host system)
# export LIBFABRIC_PATH="/opt/cray/libfabric/2.3.1/lib64"

# Explicitly specify the full path to torchrun
# export TORCHRUN_PATH="/usr/local/bin/torchrun"


srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  /cluster/home/harshm/nvidiaenv/bin/torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${SLURM_GPUS_ON_NODE}" \
    --node_rank="${SLURM_PROCID}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    pipeline.py --fsdppredict --predname ar098192_270000 \
  2>&1 | tee "/cluster/work/projects/nn2834k/harshm/FFNOMLcopy/output-${SLURM_JOB_ID}.txt"

echo "End: $(date)"