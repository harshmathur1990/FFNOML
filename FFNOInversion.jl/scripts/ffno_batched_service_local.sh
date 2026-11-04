#!/bin/bash
# Batched forward-only FFNO service on the GPUs of this host, without Slurm (e.g. eagle7): pass it as FFNO_FSDP_LAUNCHER.
# One process and no torchrun. CUDA_VISIBLE_DEVICES picks the GPUs and FFNO_FSDP_PYTHON a Python with CUDA torch; see
# ../../ffno/ffno_batched_service.py for the options.
set -o errexit
set -o nounset
set -o pipefail

manifest=$(realpath "${1:?usage: ffno_batched_service_local.sh MANIFEST.toml}")
repository=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python=${FFNO_FSDP_PYTHON:-python3}
echo "Launching batched FFNO service on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
cd "${repository}"
exec "${python}" "${repository}/ffno/ffno_batched_service.py" "${manifest}"
