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

response=$(sbatch "$@" --parsable --job-name=ffno-phase6 \
    --export=ALL,OLIVIA_TEST_GROUP=phase6 "${batch_script}")
job_id=${response%%;*}
[[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "Could not parse Phase 6 job id: ${response}" >&2
    exit 2
}

echo "Submitted Olivia Phase 6 validation job: ${job_id}"
echo "Expected outputs:"
echo "  olivia-runtime-${job_id}.out"
echo "  olivia-runtime-${job_id}.err"
echo "  olivia-runtime-evidence-${job_id}.tar.gz"
