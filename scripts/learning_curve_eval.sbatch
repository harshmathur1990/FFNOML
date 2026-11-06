#!/bin/bash
#SBATCH --job-name=FNOML_lc_eval
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=60G
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --account=nn2834k
#SBATCH --exclude=gpu-1-111

# Population errors of one learning-curve experiment's checkpoint against MULTI3D, on the snapshots of
# ${FFNOML_RUN_DIR}/eval_snapshots.txt (its training, validation and held-out snapshots). Submitted after the training job;
# the solving sets are shared by the experiments in ../solving_sets, so the evaluations run one after another.

set -o errexit
set -o nounset
set -o pipefail

module -q restore
source /cluster/home/harshm/loadnvidia.sh

: "${FFNOML_RUN_DIR:?set by submit_learning_curve.sh}"
source "${FFNOML_RUN_DIR}/experiment.env"
export FFNOML_REPO_DIR FFNOML_DATA_DIR FFNOML_ACTIVE_ATOMS
export OMP_NUM_THREADS=8
cd "${FFNOML_REPO_DIR}"

/cluster/home/harshm/nvidiaenv/bin/python scripts/population_error_diagnostic.py \
  --run-dir "${FFNOML_RUN_DIR}/eval" \
  --solve-run-dir "$(dirname "${FFNOML_RUN_DIR}")/solving_sets" \
  --model-dir "${FFNOML_RUN_DIR}/training_FFNO3D_zscale_expand_lognlte" \
  --label "${FFNOML_EXPERIMENT}" \
  --data-dir "${FFNOML_DATA_DIR}" \
  --atoms "${FFNOML_ACTIVE_ATOMS}" \
  --snapshots-file "${FFNOML_RUN_DIR}/eval_snapshots.txt" \
  2>&1 | tee "${FFNOML_RUN_DIR}/eval-${SLURM_JOB_ID}.txt"
