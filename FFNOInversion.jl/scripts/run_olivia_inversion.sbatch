#!/bin/bash
#SBATCH --job-name=ffno-invert
#SBATCH --partition=accel
#SBATCH --account=nn2834k
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH --network=job_vni
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -o errexit
set -o nounset
set -o pipefail

: "${SLURM_JOB_ID:?submit this file with sbatch}"

module --quiet reset
environment_script=${OLIVIA_ENV_SCRIPT:-/cluster/home/harshm/loadnvidiampi.sh}
source "${environment_script}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OLIVIA_REPO_DIR=${OLIVIA_REPO_DIR:-$(dirname "${script_dir}")}
OLIVIA_REPO_DIR=$(cd -- "${OLIVIA_REPO_DIR}" && pwd)
export OLIVIA_REPO_DIR

FFNOML_RUN_DIR=${FFNOML_RUN_DIR:-${SLURM_SUBMIT_DIR}}
mkdir -p "${FFNOML_RUN_DIR}"
FFNOML_RUN_DIR=$(cd -- "${FFNOML_RUN_DIR}" && pwd)
export FFNOML_RUN_DIR

mkdir -p "${FFNOML_RUN_DIR}/outputs" "${FFNOML_RUN_DIR}/diagnostics"
export FFNO_GPU_DIAGNOSTICS_DIR=${FFNO_GPU_DIAGNOSTICS_DIR:-${FFNOML_RUN_DIR}/diagnostics/slurm-${SLURM_JOB_ID}}
export TMPDIR=${FFNO_INVERSION_TMPDIR:-${FFNOML_RUN_DIR}/tmp/slurm-${SLURM_JOB_ID}}
mkdir -p "${FFNO_GPU_DIAGNOSTICS_DIR}" "${TMPDIR}"

config_file=${FFNO_INVERSION_CONFIG:-${FFNOML_RUN_DIR}/inversion.toml}
factory_file=${FFNO_INVERSION_FACTORY:-${FFNOML_RUN_DIR}/model_factory.jl}
[[ -r "${config_file}" ]] || { echo "Missing inversion config: ${config_file}" >&2; exit 2; }
[[ -r "${factory_file}" ]] || { echo "Missing model factory: ${factory_file}" >&2; exit 2; }

julia_candidate=${OLIVIA_JULIA:-julia}
python_candidate=${OLIVIA_PYTHON:-python3}
OLIVIA_JULIA=$(command -v "${julia_candidate}") || {
  echo "Cannot resolve Julia executable: ${julia_candidate}" >&2
  exit 2
}
OLIVIA_PYTHON=$(command -v "${python_candidate}") || {
  echo "Cannot resolve Python executable: ${python_candidate}" >&2
  exit 2
}
export OLIVIA_JULIA OLIVIA_PYTHON
export JULIA_PROJECT=${OLIVIA_JULIA_PROJECT:-${OLIVIA_REPO_DIR}}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}

# Olivia/Slingshot settings used by the validated MPI plus nested-NCCL path.
export HSA_FORCE_FINE_GRAIN_PCIE=${HSA_FORCE_FINE_GRAIN_PCIE:-1}
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
export OMPI_MCA_pml=${OMPI_MCA_pml:-cm}
export OMPI_MCA_mtl=${OMPI_MCA_mtl:-ofi}
export OMPI_MCA_mtl_ofi_av=${OMPI_MCA_mtl_ofi_av:-table}

cd "${FFNOML_RUN_DIR}"
echo "Repository dir: ${OLIVIA_REPO_DIR}"
echo "Run dir: ${FFNOML_RUN_DIR}"
echo "Config: ${config_file}"
echo "Factory: ${factory_file}"

srun --kill-on-bad-exit=1 --mpi=pmix --gres=none \
  --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  --cpu-bind=cores --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  env CUDA_VISIBLE_DEVICES=-1 \
  "${OLIVIA_JULIA}" --project="${JULIA_PROJECT}" --startup-file=no \
    --threads="${SLURM_CPUS_PER_TASK}" \
    "${OLIVIA_REPO_DIR}/scripts/invert.jl" "${config_file}" "${factory_file}"
