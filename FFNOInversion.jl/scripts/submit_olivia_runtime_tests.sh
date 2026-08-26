#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
batch_script=${script_dir}/run_olivia_runtime_tests.sbatch

command -v sbatch >/dev/null 2>&1 || {
    echo "sbatch is not available; run this submission helper on Olivia" >&2
    exit 2
}
[[ -r "${batch_script}" ]] || {
    echo "Cannot read batch script: ${batch_script}" >&2
    exit 2
}

submit_group() {
    local group=$1
    local dependency=$2
    shift 2
    local arguments=(--parsable --job-name="ffno-${group}"
        --export="ALL,OLIVIA_TEST_GROUP=${group}")
    if [[ -n "${dependency}" ]]; then
        arguments+=(--dependency="afterany:${dependency}")
    fi

    local response
    response=$(sbatch "$@" "${arguments[@]}" "${batch_script}")
    response=${response%%;*}
    [[ "${response}" =~ ^[0-9]+$ ]] || {
        echo "Could not parse sbatch job id for ${group}: ${response}" >&2
        exit 2
    }
    printf '%s' "${response}"
}

safe_job=$(submit_group safe "" "$@")
phase6_job=$(submit_group phase6 "${safe_job}" "$@")
internal_job=$(submit_group internal_timeout "${phase6_job}" "$@")
external_job=$(submit_group external_timeout "${internal_job}" "$@")
recovery_job=$(submit_group recovery "${external_job}" "$@")

echo "Submitted Olivia runtime validation chain:"
echo "  safe/recoverable cases:         ${safe_job}"
echo "  Phase 6 solver/GPU prerequisites: ${phase6_job} (after ${safe_job})"
echo "  internal-timeout containment:   ${internal_job} (after ${phase6_job})"
echo "  post-internal + external stall: ${external_job} (after ${internal_job})"
echo "  post-external recovery:         ${recovery_job} (after ${external_job})"
echo
echo "Final job to monitor: ${recovery_job}"
echo "Each allocation writes olivia-runtime-JOBID.out and olivia-runtime-evidence-JOBID.tar.gz."
