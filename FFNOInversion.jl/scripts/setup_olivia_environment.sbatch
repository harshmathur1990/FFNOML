#!/bin/bash
#SBATCH --job-name=ffno-julia-setup
#SBATCH --partition=accel
#SBATCH --account=nn2834k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=ffno-julia-setup-%j.out
#SBATCH --error=ffno-julia-setup-%j.err

set -o errexit
set -o nounset
set -o pipefail

: "${SLURM_JOB_ID:?submit this file with sbatch}"
: "${OLIVIA_JULIA_DEPOT:?set OLIVIA_JULIA_DEPOT to the permanent Julia depot path}"

module --quiet reset
environment_script=${OLIVIA_ENV_SCRIPT:-/cluster/home/harshm/loadnvidiampi.sh}
[[ -r "${environment_script}" ]] || {
  echo "Cannot read Olivia environment script: ${environment_script}" >&2
  exit 2
}
source "${environment_script}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
package_dir=${OLIVIA_REPO_DIR:-$(dirname "${script_dir}")}
package_dir=$(cd -- "${package_dir}" && pwd)
julia_project=${OLIVIA_JULIA_PROJECT:-${package_dir}}
[[ -r "${julia_project}/Project.toml" && -r "${julia_project}/Manifest.toml" ]] || {
  echo "Cannot read Julia project/manifest under: ${julia_project}" >&2
  exit 2
}
local_muspel_dir=${OLIVIA_LOCAL_MUSPEL_DIR:-}
if [[ -n "${local_muspel_dir}" && ! -r "${local_muspel_dir}/Project.toml" ]]; then
  echo "Cannot read local Muspel checkout: ${local_muspel_dir}" >&2
  exit 2
fi

julia_candidate=${OLIVIA_JULIA:-julia}
julia_executable=$(command -v "${julia_candidate}") || {
  echo "Cannot resolve Julia executable: ${julia_candidate}" >&2
  exit 2
}

mkdir -p "${OLIVIA_JULIA_DEPOT}"
export JULIA_DEPOT_PATH=${OLIVIA_JULIA_DEPOT}

echo "Package dir: ${package_dir}"
echo "Julia project: ${julia_project}"
echo "Julia depot: ${JULIA_DEPOT_PATH}"
echo "Julia: ${julia_executable}"

"${julia_executable}" --project="${julia_project}" --startup-file=no -e '
using Pkg
Pkg.instantiate()
Pkg.precompile()
println("FFNO_OLIVIA_JULIA_ENVIRONMENT_READY")
'
