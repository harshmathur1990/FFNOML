#!/bin/bash
# Run inside an allocated Olivia accelerator job after the GPU environment has
# been loaded. This prepares the shared Julia project and native EOS library
# before any distributed application ranks are started.

set -o errexit
set -o nounset
set -o pipefail

: "${SLURM_JOB_ID:?run this initializer inside a Slurm allocation}"
: "${FFNOML_RUN_DIR:?FFNOML_RUN_DIR must identify the run directory}"
: "${OLIVIA_REPO_DIR:?OLIVIA_REPO_DIR must identify FFNOInversion.jl}"

julia_candidate=${OLIVIA_JULIA:-julia}
julia_executable=$(command -v "${julia_candidate}") || {
  echo "Cannot resolve accelerator Julia executable: ${julia_candidate}" >&2
  exit 2
}
julia_project=${OLIVIA_JULIA_PROJECT:-${FFNOML_RUN_DIR}/julia-environment}
julia_depot=${OLIVIA_JULIA_DEPOT:-${JULIA_DEPOT_PATH:-}}
[[ -n "${julia_depot}" ]] || {
  echo "OLIVIA_JULIA_DEPOT must be set in the accelerator runtime environment" >&2
  exit 2
}

mkdir -p "${julia_depot}"
export JULIA_PROJECT=${julia_project}
export JULIA_DEPOT_PATH=${julia_depot}

echo "Initializing accelerator runtime before distributed work..."
echo "Package dir: ${OLIVIA_REPO_DIR}"
echo "Julia project: ${JULIA_PROJECT}"
echo "Julia depot: ${JULIA_DEPOT_PATH}"
echo "Julia: ${julia_executable}"

# Build to a temporary path, prove accelerator Julia can load the result, and
# only then replace the run-local library. This also safely replaces an old
# login-node library or symlink without following the symlink target.
wittmann_library=${FFNO_WITTMANN_LIBRARY:-${FFNOML_RUN_DIR}/inputs/wittmann/libwitt_ffno.so}
wittmann_directory=$(dirname -- "${wittmann_library}")
mkdir -p "${wittmann_directory}"
wittmann_staging=$(mktemp "${wittmann_library}.slurm-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -f -- "${wittmann_staging}"' EXIT

compiler_candidate=${OLIVIA_CXX:-c++}
compiler_executable=$(command -v "${compiler_candidate}") || {
  echo "Cannot resolve accelerator C++ compiler: ${compiler_candidate}" >&2
  exit 2
}
echo "C++ compiler: ${compiler_executable}"
echo "Building accelerator Wittmann EOS library: ${wittmann_library}"
CXX="${compiler_executable}" "${julia_executable}" --startup-file=no \
  "${OLIVIA_REPO_DIR}/scripts/build_wittmann_backend.jl" "${wittmann_staging}"
"${julia_executable}" --startup-file=no -e '
using Libdl
library = only(ARGS)
handle = Libdl.dlopen(library)
Libdl.dlclose(handle)
println("Wittmann EOS load check OK: ", library)
' "${wittmann_staging}"
mv -f -- "${wittmann_staging}" "${wittmann_library}"
trap - EXIT
file "${wittmann_library}"

# Recreate the disposable project from permanent sources on every job. Pkg
# operations are complete before the multi-node srun begins.
"${julia_executable}" --startup-file=no \
  "${OLIVIA_REPO_DIR}/scripts/prepare_olivia_environment.jl" \
  "${OLIVIA_REPO_DIR}" "${julia_project}"

if [[ ! -r "${julia_project}/Manifest.toml" ]]; then
  "${julia_executable}" --project="${JULIA_PROJECT}" --startup-file=no -e '
using Pkg
Pkg.instantiate(; allow_autoprecomp=false)
'
fi

"${julia_executable}" --project="${JULIA_PROJECT}" --startup-file=no -e '
using MPIPreferences
MPIPreferences.use_system_binary(
    mpiexec=`srun --mpi=pmix --exact --cpus-per-task=2 --cpu-bind=cores`,
    abi="OpenMPI",
)
'

# MPIPreferences changes MPI-dependent artifact selection, so use a fresh
# process for the final installation and precompilation pass.
"${julia_executable}" --project="${JULIA_PROJECT}" --startup-file=no -e '
using Pkg
Pkg.instantiate()
Pkg.precompile()
println("FFNO_OLIVIA_JULIA_ENVIRONMENT_READY")
'
