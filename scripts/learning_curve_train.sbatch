#!/bin/bash
#SBATCH --job-name=FNOML_lc_train
#SBATCH --partition=accel
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --account=nn2834k
#SBATCH --exclude=gpu-1-111

# One learning-curve experiment, GPU work only: train from scratch as train_gpu.sh does, then predict every scoring
# snapshot as predict_gpu.sh does. The datasets (${FFNOML_RUN_DIR}/IO) and the solving sets (${FFNOML_SOLVING_SETS})
# were built on eagle7 by scripts/learning_curve_eagle7.py prepare; the checkpoint and predictions go back there to
# be scored. Submitted by scripts/submit_learning_curve.py, which writes ${FFNOML_RUN_DIR}/experiment.env and
# eval_snapshots.txt. Resubmitted with --resume, training continues from its last epoch and predictions already
# written are kept (pipeline.py skips them).

set -o errexit
set -o nounset
set -o pipefail

module -q restore
source /cluster/home/harshm/loadnvidiampi.sh   # the accelerator-node environment (nvidiaenv, NCCL and CXI settings)

: "${FFNOML_RUN_DIR:?set by submit_learning_curve.py}"
source "${FFNOML_RUN_DIR}/experiment.env"
export FFNOML_REPO_DIR FFNOML_RUN_DIR FFNOML_TRAIN_SPLIT FFNOML_VAL_SPLIT FFNOML_ACTIVE_ATOMS FFNOML_NUM_EPOCHS
export FFNOML_OUTPUT_MODE="${FFNOML_OUTPUT_MODE:-log_population}" FFNOML_LAM_TOTAL="${FFNOML_LAM_TOTAL:-0}"

MODEL_DIR="${FFNOML_RUN_DIR}/training_FFNO3D_zscale_expand_lognlte"
mkdir -p "${MODEL_DIR}"
cd "${FFNOML_REPO_DIR}"
TORCHRUN=/cluster/home/harshm/nvidiaenv/bin/torchrun

echo "Experiment: ${FFNOML_EXPERIMENT}"
echo "Train split: ${FFNOML_TRAIN_SPLIT}"
echo "Val split: ${FFNOML_VAL_SPLIT}"
echo "Atoms: ${FFNOML_ACTIVE_ATOMS}  epochs: ${FFNOML_NUM_EPOCHS}  flags: ${FFNOML_TRAIN_FLAGS:-none}"
echo "Head: ${FFNOML_OUTPUT_MODE}  lam_total: ${FFNOML_LAM_TOTAL}"
echo "Repository: ${FFNOML_REPO_DIR} ($(git -C "${FFNOML_REPO_DIR}" rev-parse --short HEAD))  run dir: ${FFNOML_RUN_DIR}"
echo "Job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}, ${SLURM_NNODES} nodes x ${SLURM_GPUS_ON_NODE} GPUs. Start: $(date)"

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
port=29501

run_torch() {   # label, pipeline.py arguments: one torchrun over every node, its own rendezvous and port
  local label=$1; shift
  srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    "${TORCHRUN}" \
      --nnodes="${SLURM_NNODES}" \
      --nproc_per_node="${SLURM_GPUS_ON_NODE}" \
      --node_rank="${SLURM_PROCID}" \
      --rdzv_id="${SLURM_JOB_ID}_${label}" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="${MASTER_ADDR}:${port}" \
      pipeline.py "$@" \
    < /dev/null 2>&1 | tee -a "${FFNOML_RUN_DIR}/output-${SLURM_JOB_ID}.txt"
  port=$((port + 1))
}

# from scratch (no --expand), so every experiment starts the same way
run_torch train --train ${FFNOML_TRAIN_FLAGS:-}
echo "Training done: $(date)"

while IFS='=' read -r role snapshot <&3; do
  name=${snapshot/\//_}
  ln -sfn "${FFNOML_SOLVING_SETS}/3D_sim_predict_${name}.hdf5" "${MODEL_DIR}/3D_sim_predict_${name}.hdf5"
  echo "Predicting ${name} (${role}): $(date)"
  run_torch "predict_${name}" --fsdppredict --predname "${name}"
done 3< "${FFNOML_RUN_DIR}/eval_snapshots.txt"

echo "End: $(date)"
