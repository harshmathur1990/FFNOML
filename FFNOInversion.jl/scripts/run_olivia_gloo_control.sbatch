#!/bin/bash
#SBATCH --job-name=ffno-gloo-test
#SBATCH --partition=accel
#SBATCH --account=nn2834k
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=4G
#SBATCH --network=job_vni
#SBATCH --time=00:05:00
#SBATCH --output=gloo-control-%j.out
#SBATCH --error=gloo-control-%j.err

set -o errexit
set -o nounset
set -o pipefail

module --quiet reset
environment_script=${OLIVIA_ENV_SCRIPT:-/cluster/home/harshm/loadnvidiampi.sh}
[[ -r "${environment_script}" ]] || {
    echo "Cannot read Olivia environment script: ${environment_script}" >&2
    exit 2
}
source "${environment_script}"

default_repository=/cluster/projects/nn2834k/harshm/FFNOML/FFNOInversion.jl
if [[ -n "${OLIVIA_REPO_DIR:-}" ]]; then
    repository=${OLIVIA_REPO_DIR}
elif [[ -f "${default_repository}/test/olivia_gloo_control_probe.py" ]]; then
    repository=${default_repository}
elif [[ -f "${SLURM_SUBMIT_DIR}/test/olivia_gloo_control_probe.py" ]]; then
    repository=${SLURM_SUBMIT_DIR}
elif [[ -f "${SLURM_SUBMIT_DIR}/FFNOInversion.jl/test/olivia_gloo_control_probe.py" ]]; then
    repository=${SLURM_SUBMIT_DIR}/FFNOInversion.jl
else
    echo "Cannot locate FFNOInversion.jl; set OLIVIA_REPO_DIR to its directory" >&2
    exit 2
fi
export OLIVIA_REPO_DIR
OLIVIA_REPO_DIR=$(cd -- "${repository}" && pwd)
export OLIVIA_PYTHON=${OLIVIA_PYTHON:-python3}
export OLIVIA_GLOO_GPUS_PER_NODE=1
export OLIVIA_GLOO_CPUS_PER_NODE=${SLURM_CPUS_PER_TASK}

echo "Starting mixed NCCL/Gloo smoke test on ${SLURM_JOB_NODELIST}"
bash "${OLIVIA_REPO_DIR}/scripts/run_olivia_gloo_control.sh"
