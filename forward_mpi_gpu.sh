#!/bin/bash
#SBATCH --job-name=FNOML_forward
#SBATCH --partition=accel
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=120G
#SBATCH --time=0-02:00:00
# Ask Slurm to warn the batch shell two minutes before termination so it can
# leave a marker even when Julia cannot catch an external kill or node failure.
#SBATCH --signal=B:TERM@120
#SBATCH --account=nn2834k
#SBATCH --output=forward-%j.out
#SBATCH --error=forward-%j.err

set -o errexit
set -o nounset
set -o pipefail

module -q restore
module list
source /cluster/home/harshm/loadnvidia.sh

repository_dir=${FORWARD_REPO_DIR:-/cluster/work/projects/nn2834k/harshm/FFNOML}
cd "${repository_dir}"

# config.py and Forward.jl use these roots.  Defaults point at Olivia project
# storage; override them when the same checkout is submitted on another site.
project_storage_root=${FNOML_PROJECT_STORAGE_ROOT:-/cluster/work/projects/nn2834k/harshm}
export FNOML_PRED_DIR="${FNOML_PRED_DIR:-${project_storage_root}}"
export FNOML_ATOM_DIR="${FNOML_ATOM_DIR:-${project_storage_root}/multi3d/input/atoms}"

export FORWARD_DIAGNOSTICS_DIR=${FORWARD_DIAGNOSTICS_DIR:-${repository_dir}/forward-diagnostics-slurm-${SLURM_JOB_ID}}
mkdir -p "${FORWARD_DIAGNOSTICS_DIR}"

record_slurm_termination() {
    termination_log=${FORWARD_DIAGNOSTICS_DIR}/slurm-termination.log
    printf '%s Slurm sent TERM/INT to batch job %s; inspect sacct for the final state.\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SLURM_JOB_ID}" >> "${termination_log}"
    scontrol show job "${SLURM_JOB_ID}" >> "${termination_log}" 2>&1 || true
}
trap record_slurm_termination TERM INT

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1
export HYDROGEN_SE_WAVELENGTH_STRIDE=${HYDROGEN_SE_WAVELENGTH_STRIDE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-ERROR}
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

# MPI ranks perform CPU/SE/synthesis work. During each FFNoML call, MPI rank 0
# creates one overlapping Slurm GPU step via forward_fsdppredict.sh; all Julia
# ranks wait at an MPI collective until that one distributed torchrun finishes.
srun --ntasks="${SLURM_NTASKS}" \
    --cpu-bind=cores \
    julia --threads="${JULIA_NUM_THREADS}" Forward.jl \
        --mpi \
        --population-consistency-mode hydrogen-se-3d \
        --hydrogen-se-wavelength-stride "${HYDROGEN_SE_WAVELENGTH_STRIDE}" \
        --fsdp-launcher "${repository_dir}/forward_fsdppredict.sh" \
        "$@"
