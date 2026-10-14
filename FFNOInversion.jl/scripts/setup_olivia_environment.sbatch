#!/bin/bash
#SBATCH --job-name=ffno-julia-setup
#SBATCH --partition=accel
#SBATCH --account=nn2834k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=ffno-julia-setup-%j.out
#SBATCH --error=ffno-julia-setup-%j.err

set -o errexit
set -o nounset
set -o pipefail

: "${SLURM_JOB_ID:?submit this file with sbatch}"

FFNOML_RUN_DIR=${FFNOML_RUN_DIR:-${SLURM_SUBMIT_DIR}}
FFNOML_RUN_DIR=$(cd -- "${FFNOML_RUN_DIR}" && pwd)
export FFNOML_RUN_DIR

runtime_environment=${FFNO_RUNTIME_ENV_FILE:-${FFNOML_RUN_DIR}/olivia_runtime_environment.sh}
[[ -r "${runtime_environment}" ]] || {
  echo "Cannot read accelerator runtime environment: ${runtime_environment}" >&2
  exit 2
}
source "${runtime_environment}"
: "${OLIVIA_JULIA_DEPOT:?olivia_runtime_environment.sh must set OLIVIA_JULIA_DEPOT}"

module --quiet reset
environment_script=${OLIVIA_ENV_SCRIPT:-/cluster/home/harshm/loadnvidiampi.sh}
[[ -r "${environment_script}" ]] || {
  echo "Cannot read Olivia environment script: ${environment_script}" >&2
  exit 2
}
source "${environment_script}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OLIVIA_REPO_DIR=${OLIVIA_REPO_DIR:-$(dirname "${script_dir}")}
OLIVIA_REPO_DIR=$(cd -- "${OLIVIA_REPO_DIR}" && pwd)
export OLIVIA_REPO_DIR

# Kept for the multi-job regression harness. Production inversion submission
# performs this initializer inside run_olivia_inversion.sbatch instead.
"${script_dir}/initialize_olivia_environment.sh"
