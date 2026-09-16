#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

usage() {
    cat <<'EOF'
Usage:
  bootstrap_olivia_regression.sh \
    --run-dir DIR \
    --model-assets DIR \
    --atmosphere-dir DIR \
    --atom-dir DIR \
    --muspel-dir DIR \
    --julia-depot DIR \
    [-- SBATCH_ARGUMENT ...]

Creates a disposable Olivia regression run directory, links all required
scientific assets, submits Julia Pkg.instantiate/Pkg.precompile on a compute
node, and submits the five-job regression chain after that setup succeeds.

The six paths may instead be supplied with OLIVIA_TEST_RUN_DIR,
FFNO_MODEL_ASSET_SOURCE_DIR, FFNO_REFERENCE_ATMOSPHERE_SOURCE_DIR,
FFNO_ATOM_SOURCE_DIR, OLIVIA_LOCAL_MUSPEL_DIR, and OLIVIA_JULIA_DEPOT.

If --muspel-dir does not exist, this helper clones the pinned Muspel revision
there from the Olivia login node. Override MUSPEL_GIT_URL only when needed.
EOF
}

run_dir=${OLIVIA_TEST_RUN_DIR:-}
model_assets=${FFNO_MODEL_ASSET_SOURCE_DIR:-}
atmosphere_dir=${FFNO_REFERENCE_ATMOSPHERE_SOURCE_DIR:-}
atom_dir=${FFNO_ATOM_SOURCE_DIR:-}
muspel_dir=${OLIVIA_LOCAL_MUSPEL_DIR:-}
julia_depot=${OLIVIA_JULIA_DEPOT:-}
sbatch_arguments=()

while (($#)); do
    case "$1" in
        --run-dir|--model-assets|--atmosphere-dir|--atom-dir|--muspel-dir|--julia-depot)
            (($# >= 2)) || { echo "Missing value for $1" >&2; usage >&2; exit 2; }
            option=$1
            value=$2
            shift 2
            case "${option}" in
                --run-dir) run_dir=${value} ;;
                --model-assets) model_assets=${value} ;;
                --atmosphere-dir) atmosphere_dir=${value} ;;
                --atom-dir) atom_dir=${value} ;;
                --muspel-dir) muspel_dir=${value} ;;
                --julia-depot) julia_depot=${value} ;;
            esac
            ;;
        --)
            shift
            sbatch_arguments=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for variable_name in run_dir model_assets atmosphere_dir atom_dir muspel_dir julia_depot; do
    [[ -n "${!variable_name}" ]] || {
        echo "Missing required option: ${variable_name//_/-}" >&2
        usage >&2
        exit 2
    }
done

for required_command in git sbatch realpath; do
    command -v "${required_command}" >/dev/null 2>&1 || {
        echo "Missing required command on Olivia: ${required_command}" >&2
        exit 2
    }
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
package_dir=${OLIVIA_REPO_DIR:-$(dirname "${script_dir}")}
package_dir=$(cd -- "${package_dir}" && pwd)
setup_script=${script_dir}/setup_olivia_environment.sbatch
submit_script=${script_dir}/submit_olivia_regression.sh

for required_file in "${package_dir}/Project.toml" "${package_dir}/Manifest.toml" \
        "${setup_script}" "${submit_script}"; do
    [[ -r "${required_file}" ]] || {
        echo "Missing required repository file: ${required_file}" >&2
        exit 2
    }
done

canonical_directory() {
    local directory=$1
    [[ -d "${directory}" ]] || {
        echo "Directory does not exist: ${directory}" >&2
        return 1
    }
    realpath -- "${directory}"
}

canonical_file() {
    local file=$1
    [[ -f "${file}" ]] || {
        echo "File does not exist: ${file}" >&2
        return 1
    }
    realpath -- "${file}"
}

mkdir -p -- "${run_dir}"
run_dir=$(canonical_directory "${run_dir}")
model_assets=$(canonical_directory "${model_assets}")
atmosphere_dir=$(canonical_directory "${atmosphere_dir}")
atom_dir=$(canonical_directory "${atom_dir}")

muspel_revision=01ec68da389be75c9ce31494a910095d3590499a
muspel_url=${MUSPEL_GIT_URL:-git@github.com:harshmathur1990/Muspel.jl.git}
if [[ ! -e "${muspel_dir}" ]]; then
    mkdir -p -- "$(dirname -- "${muspel_dir}")"
    echo "Cloning Muspel on the login node into permanent storage: ${muspel_dir}"
    git clone --no-checkout -- "${muspel_url}" "${muspel_dir}"
    git -C "${muspel_dir}" checkout --detach "${muspel_revision}"
fi
muspel_dir=$(canonical_directory "${muspel_dir}")
[[ -d "${muspel_dir}/.git" ]] || {
    echo "Muspel source is not a Git checkout: ${muspel_dir}" >&2
    exit 2
}
muspel_head=$(git -C "${muspel_dir}" rev-parse HEAD)
[[ "${muspel_head}" == "${muspel_revision}" ]] || {
    echo "Muspel checkout must be at pinned revision ${muspel_revision}." >&2
    echo "Current ${muspel_dir} revision: ${muspel_head}" >&2
    echo "Use a separate pinned checkout or update it explicitly before retrying." >&2
    exit 2
}

case "${julia_depot}" in
    /*) ;;
    *) julia_depot=${PWD}/${julia_depot} ;;
esac
julia_depot_parent=$(dirname -- "${julia_depot}")
[[ -d "${julia_depot_parent}" ]] || {
    echo "Julia depot parent does not exist: ${julia_depot_parent}" >&2
    echo "Choose a depot below permanent storage whose parent already exists." >&2
    exit 2
}
julia_depot_parent=$(canonical_directory "${julia_depot_parent}")
julia_depot=${julia_depot_parent}/$(basename -- "${julia_depot}")

for exported_path in "${run_dir}" "${package_dir}" "${julia_depot}" "${muspel_dir}"; do
    [[ "${exported_path}" != *','* && "${exported_path}" != *$'\n'* ]] || {
        echo "Slurm-exported paths cannot contain commas or newlines: ${exported_path}" >&2
        exit 2
    }
done
[[ "${muspel_dir}" != *'"'* && "${muspel_dir}" != *'\'* ]] || {
    echo "Muspel path cannot contain a quote or backslash because it is stored in Manifest.toml: ${muspel_dir}" >&2
    exit 2
}

require_file() {
    [[ -f "$1" ]] || { echo "Missing required asset: $1" >&2; exit 2; }
}

require_file "${atmosphere_dir}/mesh"
require_file "${atmosphere_dir}/atm3d"
require_file "${atom_dir}/atom.h6_tiago2.yaml"
require_file "${atom_dir}/atom.ca2.yaml"

model_names=(
    3D_sim_train_H.pt
    output_3D_sim_s5_en024048_hion_385_FFNO3D_H.hdf5
    output_3D_sim_s5_en024048_hion_385_FFNO3D_CA.hdf5
    intensity_ml_en024048_hion_385_FFNO3D_H.h5
    intensity_ml_en024048_hion_385_FFNO3D_CA.h5
)
for name in "${model_names[@]}"; do
    require_file "${model_assets}/${name}"
done

link_asset() {
    local source=$1
    local destination=$2
    local source_resolved destination_resolved
    if [[ -d "${source}" ]]; then
        source_resolved=$(canonical_directory "${source}")
    else
        source_resolved=$(canonical_file "${source}")
    fi
    if [[ -e "${destination}" || -L "${destination}" ]]; then
        if [[ -d "${destination}" ]]; then
            destination_resolved=$(canonical_directory "${destination}")
        elif [[ -f "${destination}" ]]; then
            destination_resolved=$(canonical_file "${destination}")
        else
            echo "Broken or unsupported existing path: ${destination}" >&2
            exit 2
        fi
        [[ "${source_resolved}" == "${destination_resolved}" ]] || {
            echo "Refusing to replace existing path: ${destination}" >&2
            echo "  existing target: ${destination_resolved}" >&2
            echo "  requested target: ${source_resolved}" >&2
            exit 2
        }
        echo "Using existing asset: ${destination}"
    else
        ln -s -- "${source_resolved}" "${destination}"
        echo "Linked: ${destination} -> ${source_resolved}"
    fi
}

model_run_dir=${run_dir}/training_FFNO3D_zscale_expand_lognlte
reference_dir=${run_dir}/reference
julia_project=${run_dir}/julia-environment
mkdir -p -- "${model_run_dir}" "${reference_dir}" "${run_dir}/tmp" "${julia_project}"
for name in "${model_names[@]}"; do
    link_asset "${model_assets}/${name}" "${model_run_dir}/${name}"
done
link_asset "${atmosphere_dir}" "${reference_dir}/atmosphere"
link_asset "${atom_dir}" "${reference_dir}/atoms"

# Keep package resolution and LocalPreferences.toml in the disposable run
# directory. The source and extension code remain in the permanent checkout.
cp -- "${package_dir}/Project.toml" "${julia_project}/Project.toml"
cp -- "${package_dir}/Manifest.toml" "${julia_project}/Manifest.toml"
grep -q '^\[\[deps\.Muspel\]\]$' "${julia_project}/Manifest.toml" || {
    echo "Run-local Manifest.toml has no Muspel entry" >&2
    exit 2
}
manifest_tmp=$(mktemp "${julia_project}/Manifest.toml.XXXXXX")
awk -v muspel_path="${muspel_dir}" '
    /^\[\[deps\.Muspel\]\]$/ {
        in_muspel = 1
        path_written = 0
        print
        next
    }
    in_muspel && /^\[\[/ {
        if (!path_written) {
            print "path = \"" muspel_path "\""
            path_written = 1
        }
        in_muspel = 0
    }
    in_muspel && /^(git-tree-sha1|repo-rev|repo-url) =/ { next }
    { print }
    END {
        if (in_muspel && !path_written) {
            print "path = \"" muspel_path "\""
        }
    }
' "${julia_project}/Manifest.toml" > "${manifest_tmp}"
mv -- "${manifest_tmp}" "${julia_project}/Manifest.toml"
link_asset "${package_dir}/src" "${julia_project}/src"
if [[ -d "${package_dir}/ext" ]]; then
    link_asset "${package_dir}/ext" "${julia_project}/ext"
fi
link_asset "${package_dir}/scripts" "${julia_project}/scripts"

export OLIVIA_TEST_RUN_DIR=${run_dir}
export FFNOML_RUN_DIR=${run_dir}
export FFNO_REFERENCE_MODEL_DIR=${model_run_dir}
export FFNO_REFERENCE_ATMOSPHERE_DIR=${reference_dir}/atmosphere
export FFNO_ATOM_DIR=${reference_dir}/atoms
export OLIVIA_JULIA_DEPOT=${julia_depot}
export OLIVIA_JULIA_PROJECT=${julia_project}
export OLIVIA_LOCAL_MUSPEL_DIR=${muspel_dir}
export OLIVIA_REPO_DIR=${package_dir}

setup_response=$(sbatch --parsable \
    --chdir="${run_dir}" \
    --output="${run_dir}/ffno-julia-setup-%j.out" \
    --error="${run_dir}/ffno-julia-setup-%j.err" \
    --export="ALL,OLIVIA_TEST_RUN_DIR=${run_dir},FFNOML_RUN_DIR=${run_dir},OLIVIA_JULIA_DEPOT=${julia_depot},OLIVIA_JULIA_PROJECT=${julia_project},OLIVIA_LOCAL_MUSPEL_DIR=${muspel_dir},OLIVIA_REPO_DIR=${package_dir}" \
    "${sbatch_arguments[@]}" "${setup_script}")
setup_job=${setup_response%%;*}
[[ "${setup_job}" =~ ^[0-9]+$ ]] || {
    echo "Could not parse Julia setup job id: ${setup_response}" >&2
    exit 2
}

export OLIVIA_INITIAL_DEPENDENCY=${setup_job}
echo "Submitted Julia environment setup job ${setup_job}."
echo "Submitting regression chain with afterok:${setup_job}."
bash "${submit_script}" "${sbatch_arguments[@]}"

echo
echo "Run directory prepared at: ${run_dir}"
echo "Julia setup log: ${run_dir}/ffno-julia-setup-${setup_job}.out"
echo "If setup fails, Slurm leaves the dependent regression jobs pending with DependencyNeverSatisfied."
