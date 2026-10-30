#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
batch_script=${script_dir}/run_olivia_inversion.sbatch
preflight_script=${script_dir}/preflight_olivia_inversion.jl
preflight_runner=${script_dir}/run_olivia_login_preflight.sh
initializer=${script_dir}/initialize_olivia_environment.sh
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
[[ -x "${preflight_runner}" ]] || {
  echo "Cannot execute login-node preflight runner: ${preflight_runner}" >&2
  exit 2
}
[[ -x "${initializer}" ]] || {
  echo "Cannot execute accelerator runtime initializer: ${initializer}" >&2
  exit 2
}
package_dir=$(cd -- "$(dirname -- "${script_dir}")" && pwd)
repository_root=$(dirname -- "${package_dir}")
for initializer_input in \
    "${script_dir}/build_wittmann_backend.jl" \
    "${script_dir}/prepare_olivia_environment.jl" \
    "${repository_root}/scripts/witt_eos_cpp.cpp"; do
  [[ -r "${initializer_input}" ]] || {
    echo "Missing accelerator initializer input: ${initializer_input}" >&2
    exit 2
  }
done

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

runtime_environment=${FFNO_RUNTIME_ENV_FILE:-${run_dir}/olivia_runtime_environment.sh}
[[ "${runtime_environment}" = /* ]] || runtime_environment=${run_dir}/${runtime_environment}
[[ -r "${runtime_environment}" ]] || {
  echo "Missing accelerator runtime environment: ${runtime_environment}" >&2
  exit 2
}
echo "Checking submission inputs before requesting an allocation..."
FFNOML_RUN_DIR="${run_dir}" "${preflight_runner}" \
  "${config_file}" "${factory_file}" "${run_dir}" "${batch_script}"

if (($#)); then
  printf 'Additional sbatch arguments:'
  printf ' %q' "$@"
  printf '\n'
fi
echo "Submitting validated ${config_file}..."

sbatch --chdir="${run_dir}" \
  --output="${run_dir}/slurm-%j.out" \
  --error="${run_dir}/slurm-%j.err" \
  --export="ALL,FFNO_SUBMITTED_REPO_DIR=${package_dir},FFNOML_RUN_DIR=${run_dir},FFNO_RUNTIME_ENV_FILE=${runtime_environment},FFNO_INVERSION_CONFIG=${config_file},FFNO_INVERSION_FACTORY=${factory_file}" \
  "$@" "${batch_script}"
