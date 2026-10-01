#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
batch_script=${script_dir}/run_olivia_inversion.sbatch
preflight_script=${script_dir}/preflight_olivia_inversion.jl
run_dir=${FFNOML_RUN_DIR:-${PWD}}

command -v sbatch >/dev/null 2>&1 || {
  echo "sbatch is not available; run this helper on Olivia" >&2
  exit 2
}
[[ -r "${batch_script}" ]] || {
  echo "Cannot read batch script: ${batch_script}" >&2
  exit 2
}
[[ -r "${preflight_script}" ]] || {
  echo "Cannot read preflight script: ${preflight_script}" >&2
  exit 2
}

mkdir -p "${run_dir}"
run_dir=$(cd -- "${run_dir}" && pwd)
config_file=${FFNO_INVERSION_CONFIG:-${run_dir}/inversion.toml}
[[ "${config_file}" = /* ]] || config_file=${run_dir}/${config_file}
[[ -r "${config_file}" ]] || {
  echo "Missing ${config_file}" >&2
  exit 2
}
factory_file=${FFNO_INVERSION_FACTORY:-${run_dir}/model_factory.jl}
[[ "${factory_file}" = /* ]] || factory_file=${run_dir}/${factory_file}
[[ -r "${factory_file}" ]] || {
  echo "Missing ${factory_file}" >&2
  exit 2
}

julia_project=${OLIVIA_JULIA_PROJECT:-${run_dir}/julia-environment}
[[ "${julia_project}" = /* ]] || julia_project=${run_dir}/${julia_project}
[[ -r "${julia_project}/Project.toml" ]] || {
  echo "Missing Julia environment: ${julia_project}/Project.toml" >&2
  echo "Run setup_olivia_environment.sbatch successfully before submitting." >&2
  exit 2
}

julia_candidate=${OLIVIA_JULIA:-julia}
if ! julia_executable=$(command -v "${julia_candidate}"); then
  environment_script=${OLIVIA_ENV_SCRIPT:-/cluster/home/harshm/loadnvidiampi.sh}
  [[ -r "${environment_script}" ]] || {
    echo "Cannot resolve Julia executable '${julia_candidate}' and cannot read ${environment_script}" >&2
    exit 2
  }
  source "${environment_script}"
  julia_executable=$(command -v "${julia_candidate}") || {
    echo "Cannot resolve Julia executable after loading ${environment_script}: ${julia_candidate}" >&2
    exit 2
  }
fi

python_candidate=${OLIVIA_PYTHON:-python3}
python_executable=$(command -v "${python_candidate}") || {
  echo "Cannot resolve Python executable required by the FSDP service: ${python_candidate}" >&2
  exit 2
}
echo "Runtime executables: Julia=${julia_executable} Python=${python_executable}"

echo "Checking submission inputs before requesting an allocation..."
FFNOML_RUN_DIR="${run_dir}" "${julia_executable}" --project="${julia_project}" --startup-file=no \
  "${preflight_script}" "${config_file}" "${factory_file}" "${run_dir}" "${batch_script}"

if (($#)); then
  printf 'Additional sbatch arguments:'
  printf ' %q' "$@"
  printf '\n'
fi
echo "Submitting validated ${config_file}..."

sbatch --chdir="${run_dir}" \
  --output="${run_dir}/slurm-%j.out" \
  --error="${run_dir}/slurm-%j.err" \
  --export="ALL,FFNOML_RUN_DIR=${run_dir},FFNO_INVERSION_CONFIG=${config_file},FFNO_INVERSION_FACTORY=${factory_file}" \
  "$@" "${batch_script}"
