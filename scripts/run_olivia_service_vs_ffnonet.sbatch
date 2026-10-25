#!/bin/bash
#SBATCH --job-name=ffno-vs-ffnonet
#SBATCH --partition=accel
#SBATCH --account=nn2834k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=100G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Compare the forward run's FFNO populations (FSDP service, column-HSE atmosphere from HSEMHS) with
# FFNONet.py's direct prediction on the same atmosphere. Submit from the run folder after the forward run:
#
#   cd "$run" && sbatch "$repo/scripts/run_olivia_service_vs_ffnonet.sbatch"
#
# Uses $run/olivia_runtime_environment.sh (OLIVIA_PYTHON, OLIVIA_ENV_SCRIPT), the checkpoints in
# $run/training_FFNO3D_zscale_expand_lognlte/ and $run/outputs/forward_atmosphere.h5 unless overridden.

set -o errexit
set -o nounset
set -o pipefail

: "${SLURM_JOB_ID:?submit this file with sbatch}"
FFNOML_RUN_DIR=$(cd -- "${FFNOML_RUN_DIR:-${SLURM_SUBMIT_DIR}}" && pwd)
export FFNOML_RUN_DIR
source "${FFNO_RUNTIME_ENV_FILE:-${FFNOML_RUN_DIR}/olivia_runtime_environment.sh}"
module --quiet reset
source "${OLIVIA_ENV_SCRIPT:-/cluster/home/harshm/loadnvidiampi.sh}"

repository=$(cd -- "$(dirname -- "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
# sbatch copies the script; prefer the checkout named by the runtime environment.
[[ -f "${repository}/FFNONet.py" ]] || repository=$(cd -- "${OLIVIA_REPO_DIR}/.." && pwd)
script="${repository}/scripts/compare_service_with_ffnonet.py"
atmosphere=${FFNO_COMPARE_ATMOSPHERE:-${FFNOML_RUN_DIR}/outputs/forward_atmosphere.h5}
checkpoints=${FFNO_COMPARE_CHECKPOINT_DIR:-${FFNOML_RUN_DIR}/training_FFNO3D_zscale_expand_lognlte}
out=${FFNO_COMPARE_DIR:-${FFNOML_RUN_DIR}/compare/slurm-${SLURM_JOB_ID}}
python=${OLIVIA_PYTHON:-python3}
mkdir -p "${out}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export FFNOML_RUN_DIR

echo "repository=${repository} atmosphere=${atmosphere} output=${out}"
git -C "${repository}" rev-parse HEAD || true
"${python}" "${script}" build --atmosphere "${atmosphere}" --out "${out}/solve.h5"
for species in H CA; do
    "${python}" -m torch.distributed.run --standalone --nproc_per_node=4 "${script}" predict \
        --solve "${out}/solve.h5" --species "${species}" \
        --checkpoint "${checkpoints}/3D_sim_train_${species}.pt" --out "${out}/direct_${species}.h5"
done
"${python}" "${script}" compare --atmosphere "${atmosphere}" \
    --direct "H=${out}/direct_H.h5" "CA=${out}/direct_CA.h5" --report "${out}/comparison.json"
