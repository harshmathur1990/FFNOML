#!/bin/bash
# Full fine-tuning branch of the controlled experiment (claude-next-steps-uq step 4) on Olivia GH200.
# Not submitted automatically. Before submitting:
#   1. sync the uq-calibration branch to the Olivia checkout and copy the branch run dir
#      uq/retrain/<ATOM>_<BRANCH>/ (manifest, production checkpoint) plus uq/retrain/data/*.hdf5;
#   2. rerun scripts/uq_retrain_setup.py there so the HDF5 external links point at Olivia paths.
# Usage: sbatch --export=ALL,ATOM=H,BRANCH=B_nw1050,EPOCHS=10 uq/olivia_finetune.sbatch
# A_base gets round(1.2 * EPOCHS) epochs to match optimizer steps (805 vs 966 patches per epoch).
#SBATCH --job-name=FFNO_ft
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=32
#SBATCH --time=0-12:00:00
#SBATCH --account=nn2834k
#SBATCH --output=slurm-ft-%j.out
set -euo pipefail
source /cluster/home/harshm/loadnvidia.sh
cd "${SLURM_SUBMIT_DIR}"
ep=${EPOCHS}; [ "${BRANCH}" = A_base ] && ep=$(( (12 * EPOCHS + 5) / 10 ))
export FFNOML_ACTIVE_ATOMS=${ATOM} FFNOML_NUM_EPOCHS=${ep} FFNOML_RUN_DIR=$PWD/uq/retrain/${ATOM}_${BRANCH}
export FFNOML_TRAIN_ONLY= OMP_NUM_THREADS=8 NCCL_DEBUG=ERROR
/cluster/home/harshm/nvidiaenv/bin/torchrun --standalone --nproc_per_node=4 pipeline.py --train --resume --bestpath \
  2>&1 | tee "${FFNOML_RUN_DIR}/train_olivia_${SLURM_JOB_ID}.log"
