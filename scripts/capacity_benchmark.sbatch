#!/bin/bash
#SBATCH --job-name=FNOML_capacity_bench
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --account=nn2834k
#SBATCH --exclude=gpu-1-111

# Phase 3 benchmark (docs/level-fractions-capacity-plan.md): memory and time of one training step per capacity
# configuration at the training patch, 40 x 40 x 496, on one Olivia GPU. One GPU is enough: under FSDP each GPU holds
# one patch's activations, which dominate, and a shard of the (small) parameters.
#     sbatch scripts/capacity_benchmark.sbatch          (from the repository, on an Olivia login node)

set -o errexit
set -o nounset
set -o pipefail

module -q restore
source /cluster/home/harshm/loadnvidiampi.sh   # the accelerator-node environment (nvidiaenv)

OUT=/cluster/work/projects/nn2834k/harshm/outputs/capacity_benchmark
mkdir -p "${OUT}"
cd "${SLURM_SUBMIT_DIR}"
echo "Repository: $(pwd) ($(git rev-parse --short HEAD)). Job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}. Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

/cluster/home/harshm/nvidiaenv/bin/python scripts/capacity_benchmark.py --patch 40 --depth 496 --steps 5 \
    --out "${OUT}/benchmark_${SLURM_JOB_ID}.jsonl"
echo "End: $(date)"
