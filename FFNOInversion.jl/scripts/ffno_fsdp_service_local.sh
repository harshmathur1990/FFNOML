#!/bin/bash
# Persistent FFNO FSDP service on the GPUs of this host, without Slurm (e.g. eagle7): one torchrun over
# FFNO_FSDP_NPROC GPUs (default: all). Pass it to launch_fsdp_service as `launcher`; the Julia client sets
# FFNO_FSDP_LAUNCHER for ffno_population_tangents.jl. Set FFNO_FSDP_PYTHON to a Python with CUDA torch.
set -o errexit
set -o nounset
set -o pipefail

manifest=$(realpath "${1:?usage: ffno_fsdp_service_local.sh MANIFEST.toml}")
repository=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)        # ARUN: ffno_fsdp_service.py, models/
python=${FFNO_FSDP_PYTHON:-python3}
nproc=${FFNO_FSDP_NPROC:-$(nvidia-smi --list-gpus | wc -l)}
port=${FFNO_FSDP_MASTER_PORT:-$((29500 + $$ % 1000))}
echo "Launching local FFNO FSDP service on $(hostname) with ${nproc} GPUs"
cd "${repository}"
exec "${python}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${nproc}" \
    --master_port="${port}" "${repository}/ffno/ffno_fsdp_service.py" "${manifest}"
