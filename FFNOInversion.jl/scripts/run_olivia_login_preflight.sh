#!/bin/bash
# Run the submission preflight with x86-64 software on an Olivia login node.
# Accelerator-node executables and environment files must not be loaded here.

set -o errexit
set -o nounset
set -o pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
package_dir=$(dirname "${script_dir}")
preflight_script=${script_dir}/preflight_olivia_inversion.jl

# Never inherit accelerator-node Julia paths or package selections from the
# login shell. A dedicated CPU preflight depot may be selected explicitly.
preflight_depot=${FFNO_PREFLIGHT_JULIA_DEPOT:-}
unset JULIA_PROJECT JULIA_DEPOT_PATH JULIA_LOAD_PATH
unset OLIVIA_ENV_SCRIPT OLIVIA_JULIA OLIVIA_PYTHON OLIVIA_JULIA_DEPOT OLIVIA_JULIA_PROJECT
[[ -z "${preflight_depot}" ]] || export JULIA_DEPOT_PATH="${preflight_depot}"

module --quiet reset
module load "${FFNO_PREFLIGHT_STACK_MODULE:-NRIS/CPU}"
module load "${FFNO_PREFLIGHT_JULIA_MODULE:-Julia/1.12.2}"

julia_candidate=${FFNO_PREFLIGHT_JULIA:-julia}
julia_executable=$(command -v "${julia_candidate}") || {
  echo "Cannot resolve CPU/login-node Julia executable: ${julia_candidate}" >&2
  exit 2
}
preflight_project=${FFNO_PREFLIGHT_JULIA_PROJECT:-${package_dir}}

echo "Preflight environment: stack=${FFNO_PREFLIGHT_STACK_MODULE:-NRIS/CPU} Julia=${julia_executable}"
exec "${julia_executable}" --project="${preflight_project}" --startup-file=no \
  "${preflight_script}" "$@"
