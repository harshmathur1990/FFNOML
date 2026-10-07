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
# login shell. Keep login-node packages and compiled caches in their own depot.
preflight_depot=${FFNO_PREFLIGHT_JULIA_DEPOT:-${HOME}/julia-depot-ffno-login-1.12.2}
case "${preflight_depot}" in
  /*) ;;
  *)
    echo "FFNO_PREFLIGHT_JULIA_DEPOT must be one absolute directory, not '${preflight_depot}'" >&2
    echo "Unset it to use ${HOME}/julia-depot-ffno-login-1.12.2" >&2
    exit 2
    ;;
esac
unset JULIA_PROJECT JULIA_DEPOT_PATH JULIA_LOAD_PATH
unset OLIVIA_ENV_SCRIPT OLIVIA_JULIA OLIVIA_PYTHON OLIVIA_JULIA_DEPOT OLIVIA_JULIA_PROJECT
mkdir -p "${preflight_depot}"
export JULIA_DEPOT_PATH="${preflight_depot}"
export JULIA_LOAD_PATH="@:@stdlib"
export JULIA_PKG_PRECOMPILE_AUTO=0

module --quiet reset
module load "${FFNO_PREFLIGHT_STACK_MODULE:-NRIS/Login}"
module load "${FFNO_PREFLIGHT_JULIA_MODULE:-Julia/1.12.2}"

julia_candidate=${FFNO_PREFLIGHT_JULIA:-julia}
julia_executable=$(command -v "${julia_candidate}") || {
  echo "Cannot resolve CPU/login-node Julia executable: ${julia_candidate}" >&2
  exit 2
}
preflight_project=${FFNO_PREFLIGHT_JULIA_PROJECT:-${package_dir}}

echo "Preflight environment: stack=${FFNO_PREFLIGHT_STACK_MODULE:-NRIS/Login} Julia=${julia_executable}"
echo "Preflight Julia depot: ${JULIA_DEPOT_PATH}"
echo "Ensuring login-node preflight dependencies are installed..."
"${julia_executable}" --project="${preflight_project}" --startup-file=no -e '
using Pkg
Pkg.instantiate(; allow_autoprecomp=false)
'
exec "${julia_executable}" --project="${preflight_project}" --startup-file=no \
  "${preflight_script}" "$@"
