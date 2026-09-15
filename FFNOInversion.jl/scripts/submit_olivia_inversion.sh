#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
batch_script=${script_dir}/run_olivia_inversion.sbatch
run_dir=${FFNOML_RUN_DIR:-${PWD}}

command -v sbatch >/dev/null 2>&1 || {
  echo "sbatch is not available; run this helper on Olivia" >&2
  exit 2
}
[[ -r "${batch_script}" ]] || {
  echo "Cannot read batch script: ${batch_script}" >&2
  exit 2
}

mkdir -p "${run_dir}"
run_dir=$(cd -- "${run_dir}" && pwd)
[[ -r "${run_dir}/inversion.toml" ]] || {
  echo "Missing ${run_dir}/inversion.toml" >&2
  exit 2
}
[[ -r "${run_dir}/model_factory.jl" ]] || {
  echo "Missing ${run_dir}/model_factory.jl" >&2
  exit 2
}

sbatch --chdir="${run_dir}" \
  --output="${run_dir}/slurm-%j.out" \
  --error="${run_dir}/slurm-%j.err" \
  --export="ALL,FFNOML_RUN_DIR=${run_dir}" \
  "$@" "${batch_script}"
