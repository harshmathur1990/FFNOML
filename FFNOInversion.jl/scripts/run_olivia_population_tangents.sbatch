#!/bin/bash
# FFNO population temperature tangents of a finished forward run (scripts/ffno_population_tangents.jl): one Julia
# process on the first node launches the persistent FSDP service over every GPU of the allocation and asks it for
# one forward-mode (dual tensor) JVP per log tau500 point and species. Use as many nodes as the forward run: a JVP
# holds a tangent next to every activation.
#
# Submit from the forward run directory:
#   sbatch /cluster/projects/nn2834k/harshm/FFNOML/FFNOInversion.jl/scripts/run_olivia_population_tangents.sbatch
# FFNO_TANGENT_OPTIONS passes options, e.g. FFNO_TANGENT_OPTIONS="--depths 30:30" for a one-depth timing test.
# Then the CPU part:
#   FFNO_RESPONSE_OPTIONS="--variables temperature --stride 1 --population-tangents outputs/population_tangents.h5" \
#     sbatch .../run_olivia_column_response.sbatch
#SBATCH --job-name=ffno-tangents
#SBATCH --partition=accel
#SBATCH --account=nn2834k
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=256
#SBATCH --mem-per-gpu=40G
#SBATCH --network=job_vni
#SBATCH --time=0-12:00:00
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
OLIVIA_JULIA=$(command -v "${OLIVIA_JULIA:-julia}")
OLIVIA_PYTHON=$(command -v "${OLIVIA_PYTHON:-python3}")
export OLIVIA_JULIA OLIVIA_PYTHON
export JULIA_PROJECT=${OLIVIA_JULIA_PROJECT:-${FFNOML_RUN_DIR}/julia-environment}
export JULIA_DEPOT_PATH=${OLIVIA_JULIA_DEPOT:-${JULIA_DEPOT_PATH:-${HOME}/.julia}}
"${OLIVIA_REPO_DIR}/scripts/initialize_olivia_environment.sh"

export FFNO_GPU_DIAGNOSTICS_DIR=${FFNO_GPU_DIAGNOSTICS_DIR:-${FFNOML_RUN_DIR}/diagnostics/slurm-${SLURM_JOB_ID}}
export TMPDIR=${FFNO_INVERSION_TMPDIR:-${FFNOML_RUN_DIR}/tmp/slurm-${SLURM_JOB_ID}}
mkdir -p "${FFNOML_RUN_DIR}/outputs" "${FFNO_GPU_DIAGNOSTICS_DIR}" "${TMPDIR}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
# NCCL over Slingshot, as in run_olivia_inversion.sbatch.
export FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR:-userfaultfd}
export FI_CXI_DISABLE_HOST_REGISTER=${FI_CXI_DISABLE_HOST_REGISTER:-1}
export FI_CXI_DEFAULT_CQ_SIZE=${FI_CXI_DEFAULT_CQ_SIZE:-131072}
export FI_CXI_RDZV_PROTO=${FI_CXI_RDZV_PROTO:-alt_read}
export FI_CXI_RDZV_EAGER_SIZE=${FI_CXI_RDZV_EAGER_SIZE:-0}
export FI_CXI_RDZV_THRESHOLD=${FI_CXI_RDZV_THRESHOLD:-0}
export FI_CXI_RDZV_GET_MIN=${FI_CXI_RDZV_GET_MIN:-0}
export FI_CXI_DEFAULT_TX_SIZE=${FI_CXI_DEFAULT_TX_SIZE:-2048}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-PHB}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}
export FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE:-hybrid}
export NCCL_PROTO=${NCCL_PROTO:-^LL128}
export FI_PROVIDER=${FI_PROVIDER:-cxi}

config_file=${FFNO_FORWARD_CONFIG:-${FFNOML_RUN_DIR}/forward.toml}
factory_file=${FFNO_INVERSION_FACTORY:-${FFNOML_RUN_DIR}/model_factory.jl}
atmosphere_file=${FFNO_FORWARD_ATMOSPHERE:-${FFNOML_RUN_DIR}/outputs/forward_atmosphere.h5}
output_file=${FFNO_TANGENT_OUTPUT:-${FFNOML_RUN_DIR}/outputs/population_tangents.h5}
for file in "${config_file}" "${factory_file}" "${atmosphere_file}"; do
  [[ -r "${file}" ]] || { echo "Missing input: ${file}" >&2; exit 2; }
done

cd "${FFNOML_RUN_DIR}"
# One Julia task on the first node; the FSDP service it launches overlaps every node with its GPUs.
# shellcheck disable=SC2086  # options are split on purpose
srun --kill-on-bad-exit=1 --mpi=none --gres=none --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  env CUDA_VISIBLE_DEVICES=-1 \
  "${OLIVIA_JULIA}" --project="${JULIA_PROJECT}" --startup-file=no --threads="${SLURM_CPUS_PER_TASK}" \
    "${OLIVIA_REPO_DIR}/scripts/ffno_population_tangents.jl" \
    "${config_file}" "${factory_file}" "${atmosphere_file}" "${output_file}" ${FFNO_TANGENT_OPTIONS:-}
