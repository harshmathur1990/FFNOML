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
    local label=$2
    local dependency=$3
    shift 3
    local arguments=(--parsable --job-name="ffno-${label}"
        --export="ALL,OLIVIA_TEST_GROUP=${group}")
    if [[ -n "${dependency}" ]]; then
        arguments+=(--dependency="afterok:${dependency}")
    fi
    local response
    response=$(sbatch "$@" "${arguments[@]}" "${batch_script}")
    response=${response%%;*}
    [[ "${response}" =~ ^[0-9]+$ ]] || {
        echo "Could not parse sbatch job id for ${label}: ${response}" >&2
        exit 2
    }
    printf '%s' "${response}"
}

regression_job=$(submit_group regression regression "" "$@")
internal_job=$(submit_group internal_timeout internal-timeout "${regression_job}" "$@")
internal_recovery_job=$(submit_group recovery internal-recovery "${internal_job}" "$@")
external_job=$(submit_group external_timeout external-timeout "${internal_recovery_job}" "$@")
external_recovery_job=$(submit_group recovery external-recovery "${external_job}" "$@")

echo "Submitted the cumulative Olivia Phase 1-6 regression chain:"
echo "  full old+new regression:       ${regression_job}"
echo "  internal-timeout containment:  ${internal_job} (afterok ${regression_job})"
echo "  recovery after internal stall: ${internal_recovery_job} (afterok ${internal_job})"
echo "  external-watchdog containment: ${external_job} (afterok ${internal_recovery_job})"
echo "  recovery after external stall: ${external_recovery_job} (afterok ${external_job})"
echo
echo "All five jobs must exit normally for the regression to pass."
echo "Final job to monitor: ${external_recovery_job}"
echo "Each allocation writes olivia-runtime-JOBID.out and olivia-runtime-evidence-JOBID.tar.gz."
