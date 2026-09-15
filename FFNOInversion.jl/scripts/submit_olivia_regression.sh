#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
batch_script=${script_dir}/run_olivia_runtime_tests.sbatch
test_run_dir=${OLIVIA_TEST_RUN_DIR:-${PWD}}
mkdir -p "${test_run_dir}"
test_run_dir=$(cd -- "${test_run_dir}" && pwd)

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
        --chdir="${test_run_dir}"
        --output="${test_run_dir}/olivia-runtime-%j.out"
        --error="${test_run_dir}/olivia-runtime-%j.err"
        --export="ALL,OLIVIA_TEST_GROUP=${group},OLIVIA_TEST_RUN_DIR=${test_run_dir}")
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

initial_dependency=${OLIVIA_INITIAL_DEPENDENCY:-}
if [[ -n "${initial_dependency}" && ! "${initial_dependency}" =~ ^[0-9]+$ ]]; then
    echo "OLIVIA_INITIAL_DEPENDENCY must be a numeric Slurm job id: ${initial_dependency}" >&2
    exit 2
fi

regression_job=$(submit_group regression regression "${initial_dependency}" "$@")
internal_job=$(submit_group internal_timeout internal-timeout "${regression_job}" "$@")
internal_recovery_job=$(submit_group recovery internal-recovery "${internal_job}" "$@")
external_job=$(submit_group external_timeout external-timeout "${internal_recovery_job}" "$@")
external_recovery_job=$(submit_group recovery external-recovery "${external_job}" "$@")

echo "Submitted the cumulative Olivia Phase 1-6 regression chain:"
if [[ -n "${initial_dependency}" ]]; then
    echo "  Julia environment setup:       ${initial_dependency}"
    echo "  full old+new regression:       ${regression_job} (afterok ${initial_dependency})"
else
    echo "  full old+new regression:       ${regression_job}"
fi
echo "  internal-timeout containment:  ${internal_job} (afterok ${regression_job})"
echo "  recovery after internal stall: ${internal_recovery_job} (afterok ${internal_job})"
echo "  external-watchdog containment: ${external_job} (afterok ${internal_recovery_job})"
echo "  recovery after external stall: ${external_recovery_job} (afterok ${external_job})"
echo
echo "All five jobs must exit normally for the regression to pass."
echo "Final job to monitor: ${external_recovery_job}"
echo "Run directory: ${test_run_dir}"
echo "Each allocation writes olivia-runtime-JOBID.out and olivia-runtime-evidence-JOBID.tar.gz there."
