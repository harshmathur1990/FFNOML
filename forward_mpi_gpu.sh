#!/bin/bash
#SBATCH --job-name=FNOML_forward
#SBATCH --partition=accel
#SBATCH --nodes=8
# One full-volume Hydrogen-SE rank per node. Wavelengths are distributed over
# nodes and its cell-local work is split into height slabs over all node CPUs.
# The overlapping torchrun step still launches four GPU workers per node.
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
# gpu-1-102 failed to configure the Slingshot interconnect for job 1751001.
# gpu-1-37 failed to contribute its local task to the PMIx startup fence for
# job 1792315, leaving every other node waiting for it. Remove these temporary
# exclusions after the nodes have been returned to service.
#SBATCH --exclude=gpu-1-37,gpu-1-102
#SBATCH --cpus-per-task=256
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

# Start from Olivia's default module environment instead of restoring a
# potentially stale user collection. loadnvidia.sh must load a mutually
# compatible CUDA/NCCL/Python stack; list the result after setup so compiler
# and library version conflicts are visible in the job log.
module --quiet reset
source /cluster/home/harshm/loadnvidia.sh
module list

repository_dir=${FORWARD_REPO_DIR:-/cluster/work/projects/nn2834k/harshm/FFNOMLcopy}
cd "${repository_dir}"

# Do not spend time entering a PMIx fence when a node already known to have a
# broken Slingshot/PMIx startup path slips into the allocation. The SBATCH
# exclusion above should prevent this; this guard makes a copied or overridden
# submission fail immediately and explain why.
while IFS= read -r allocated_node; do
    case "${allocated_node}" in
        gpu-1-37|gpu-1-102)
            echo "Refusing allocation containing quarantined node ${allocated_node}" >&2
            exit 1
            ;;
    esac
done < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")

# Use the environment prepared by make_gpu_env.sh instead of whichever global
# Julia environment happens to be active in the submitting shell.
export JULIA_PROJECT=${FNOML_JULIA_PROJECT:-/cluster/work/projects/nn2834k/harshm/julia-envs/fnoml-forward-julia-1.12.2}
export JULIA_DEPOT_PATH=${FNOML_JULIA_DEPOT:-/cluster/work/projects/nn2834k/harshm/julia-depot-1.12.2}

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
# Preserve the batch allocation's GPU count before the CPU-only Julia step is
# created with --gres=none.  Step-local SLURM_GPUS_ON_NODE may then be unset.
export FORWARD_GPUS_PER_NODE="${FORWARD_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE}}"
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

# Sigma2's multi-node OpenMPI configuration for Olivia. Force OpenMPI through
# libfabric's CXI provider so it uses Slingshot and cannot silently fall back
# to TCP if the fabric cannot be initialized on a rank.
export FI_PROVIDER=cxi
export OMPI_MCA_pml=cm
export OMPI_MCA_mtl=ofi
export OMPI_MCA_mtl_ofi_av=table
export PRTE_MCA_ras_base_launch_orted_on_hn=1

# MPI ranks perform charge-consistency and synthesis work. During each FFNoML
# call, MPI rank 0 creates one overlapping Slurm GPU step via
# forward_fsdppredict.sh; all Julia ranks wait until torchrun finishes.
srun --mpi=pmix \
    --ntasks="${SLURM_NTASKS}" \
    --cpu-bind=cores \
    --gres=none \
    julia --project="${JULIA_PROJECT}" --startup-file=no \
        --threads="${JULIA_NUM_THREADS}" Forward.jl \
        --mpi \
        --population-consistency-mode charge-only \
        --hydrogen-se-wavelength-stride "${HYDROGEN_SE_WAVELENGTH_STRIDE}" \
        --fsdp-launcher "${repository_dir}/forward_fsdppredict.sh" \
        "$@"
