#!/bin/bash
# Column response functions of a finished forward run (scripts/column_response.jl). One node, one Julia process,
# no MPI and no FFNO service: populations and the HSE structure come from the forward run's atmosphere file.
#
# Submit from the forward run directory:
#   sbatch /cluster/projects/nn2834k/harshm/FFNOML/FFNOInversion.jl/scripts/run_olivia_column_response.sbatch
# Options for column_response.jl go in FFNO_RESPONSE_OPTIONS, e.g.
#   FFNO_RESPONSE_OPTIONS="--variables temperature,vz --stride 4" sbatch ...
#SBATCH --job-name=ffno-response
#SBATCH --partition=accel
#SBATCH --account=nn2834k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=256
#SBATCH --mem-per-gpu=40G
#SBATCH --time=0-08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -o errexit
set -o nounset
set -o pipefail

: "${SLURM_JOB_ID:?submit this file with sbatch}"

FFNOML_RUN_DIR=$(cd -- "${FFNOML_RUN_DIR:-${SLURM_SUBMIT_DIR}}" && pwd)
export FFNOML_RUN_DIR
source "${FFNO_RUNTIME_ENV_FILE:-${FFNOML_RUN_DIR}/olivia_runtime_environment.sh}"

module --quiet reset
source "${OLIVIA_ENV_SCRIPT:-/cluster/home/harshm/loadnvidiampi.sh}"
export JULIA_PROJECT=${OLIVIA_JULIA_PROJECT:-${FFNOML_RUN_DIR}/julia-environment}
export JULIA_DEPOT_PATH=${OLIVIA_JULIA_DEPOT:-${JULIA_DEPOT_PATH:-/cluster/projects/nn2834k/harshm/julia-depot-prod}}
"${OLIVIA_REPO_DIR}/scripts/initialize_olivia_environment.sh"

config_file=${FFNO_FORWARD_CONFIG:-${FFNOML_RUN_DIR}/forward.toml}
factory_file=${FFNO_INVERSION_FACTORY:-${FFNOML_RUN_DIR}/model_factory.jl}
atmosphere_file=${FFNO_FORWARD_ATMOSPHERE:-${FFNOML_RUN_DIR}/outputs/forward_atmosphere.h5}
output_file=${FFNO_RESPONSE_OUTPUT:-${FFNOML_RUN_DIR}/outputs/column_response.h5}
for file in "${config_file}" "${factory_file}" "${atmosphere_file}"; do
  [[ -r "${file}" ]] || { echo "Missing input: ${file}" >&2; exit 2; }
done

cd "${FFNOML_RUN_DIR}"
# shellcheck disable=SC2086  # options are split on purpose
"${OLIVIA_JULIA}" --startup-file=no --threads="${SLURM_CPUS_PER_TASK}" \
  "${OLIVIA_REPO_DIR}/scripts/column_response.jl" \
  "${config_file}" "${factory_file}" "${atmosphere_file}" "${output_file}" ${FFNO_RESPONSE_OPTIONS:-}
