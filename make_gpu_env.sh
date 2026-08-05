#!/bin/bash
#SBATCH --job-name=FNOML_train_julia
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=32
#SBATCH --time=0-01:00:00
#SBATCH --account=nn2834k
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --exclude=gpu-1-111

set -o errexit
set -o nounset
set -o pipefail

module -q restore
module list

source /cluster/home/harshm/loadnvidia.sh
module load Julia/1.12.2

repository_dir=${FNOML_REPO_DIR:-/cluster/work/projects/nn2834k/harshm/FFNOMLcopy}
cd "${repository_dir}"

# Keep the Julia environment and downloaded packages on persistent shared
# project storage. Both locations can be overridden when testing a new setup.
export JULIA_PROJECT=${FNOML_JULIA_PROJECT:-/cluster/work/projects/nn2834k/harshm/julia-envs/fnoml-forward-julia-1.12.2}
export JULIA_DEPOT_PATH=${FNOML_JULIA_DEPOT:-/cluster/work/projects/nn2834k/harshm/julia-depot-1.12.2}
mkdir -p "${JULIA_PROJECT}" "${JULIA_DEPOT_PATH}"

echo "Work dir: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Num nodes: ${SLURM_NNODES}"
echo "GPUs per node: ${SLURM_GPUS_ON_NODE}"
echo "Julia: $(command -v julia)"
echo "Julia project: ${JULIA_PROJECT}"
echo "Julia depot: ${JULIA_DEPOT_PATH}"
echo "Start: $(date)"

# This runs once in the batch shell before any srun workers are launched, so
# multiple ranks never write the Julia environment concurrently. Muspel is
# deliberately installed from the existing shared cluster checkout rather than
# cloned from Git. Base.Threads is part of Julia and needs no Pkg.add.
julia --project="${JULIA_PROJECT}" --startup-file=no -e '
using Pkg

project = ENV["JULIA_PROJECT"]
Pkg.activate(project)
Pkg.develop(path="/cluster/work/projects/nn2834k/harshm/Muspel.jl")
Pkg.add([
    "StaticArrays",
    "AtomicData",
    "HDF5",
    "ProgressMeter",
    "Interpolations",
    "Plots",
    "MPI",
    "MPIPreferences",
])
Pkg.instantiate()
using MPIPreferences
MPIPreferences.use_system_binary(mpiexec="srun", vendor="cray")
Pkg.precompile()
'

# Fail before the expensive training step if the environment cannot import
# everything required by Forward.jl. Base.Threads is checked here as well.
julia --project="${JULIA_PROJECT}" --startup-file=no -e '
using Muspel
using StaticArrays
using AtomicData
using HDF5
using ProgressMeter
using Base.Threads
using Interpolations
using Plots
using MPI
println("Julia forward environment import check passed")
'

echo "End: $(date)"
