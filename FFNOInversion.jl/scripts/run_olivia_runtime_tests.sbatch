#!/bin/bash
#SBATCH --job-name=ffno-runtime-test
#SBATCH --partition=accel
#SBATCH --account=nn2834k
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=8G
#SBATCH --network=job_vni
#SBATCH --time=00:20:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=olivia-runtime-%j.out
#SBATCH --error=olivia-runtime-%j.err

set -o nounset
set -o pipefail
set -o errexit

: "${SLURM_JOB_ID:?submit this file with sbatch}"

module --quiet reset
environment_script=${OLIVIA_ENV_SCRIPT:-/cluster/home/harshm/loadnvidiampi.sh}
if [[ ! -r "${environment_script}" ]]; then
    echo "Cannot read Olivia environment script: ${environment_script}" >&2
    exit 2
fi
source "${environment_script}"

export OLIVIA_REPO_DIR=${OLIVIA_REPO_DIR:-${SLURM_SUBMIT_DIR}}
cd "${OLIVIA_REPO_DIR}" || exit 2

export JULIA_PROJECT=${OLIVIA_JULIA_PROJECT:-${OLIVIA_REPO_DIR}}
if [[ -n "${OLIVIA_JULIA_DEPOT:-}" ]]; then
    export JULIA_DEPOT_PATH=${OLIVIA_JULIA_DEPOT}
fi
export OLIVIA_PYTHON=${OLIVIA_PYTHON:-python3}
export OLIVIA_GPUS_PER_NODE=${OLIVIA_GPUS_PER_NODE:-4}
export OLIVIA_JULIA_THREADS=${OLIVIA_JULIA_THREADS:-4}
export OLIVIA_GPU_CPUS_PER_NODE=${OLIVIA_GPU_CPUS_PER_NODE:-8}
export FFNO_GPU_CONTROL_HOST=${FFNO_GPU_CONTROL_HOST:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Olivia/Slingshot defaults used by the existing charge-branch production job.
export HSA_FORCE_FINE_GRAIN_PCIE=${HSA_FORCE_FINE_GRAIN_PCIE:-1}
export FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR:-userfaultfd}
export FI_CXI_DISABLE_HOST_REGISTER=${FI_CXI_DISABLE_HOST_REGISTER:-1}
export FI_CXI_DEFAULT_CQ_SIZE=${FI_CXI_DEFAULT_CQ_SIZE:-131072}
export FI_CXI_RDZV_PROTO=${FI_CXI_RDZV_PROTO:-alt_read}
export FI_CXI_RDZV_EAGER_SIZE=${FI_CXI_RDZV_EAGER_SIZE:-0}
export FI_CXI_RDZV_THRESHOLD=${FI_CXI_RDZV_THRESHOLD:-0}
export FI_CXI_RDZV_GET_MIN=${FI_CXI_RDZV_GET_MIN:-0}
export FI_CXI_DEFAULT_TX_SIZE=${FI_CXI_DEFAULT_TX_SIZE:-2048}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-PHB}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}
export FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE:-hybrid}
export NCCL_PROTO=${NCCL_PROTO:-^LL128}
export FI_PROVIDER=${FI_PROVIDER:-cxi}
export OMPI_MCA_pml=${OMPI_MCA_pml:-cm}
export OMPI_MCA_mtl=${OMPI_MCA_mtl:-ofi}
export OMPI_MCA_mtl_ofi_av=${OMPI_MCA_mtl_ofi_av:-table}
export PRTE_MCA_ras_base_launch_orted_on_hn=${PRTE_MCA_ras_base_launch_orted_on_hn:-1}

diagnostics_root=${OLIVIA_DIAGNOSTICS_DIR:-${OLIVIA_REPO_DIR}/olivia-runtime-evidence-${SLURM_JOB_ID}}
mkdir -p "${diagnostics_root}"
summary=${diagnostics_root}/summary.tsv
printf 'test\texpectation\tresult\texit_code\ttimeout_seconds\trequired_marker\n' > "${summary}"

failures=0

snapshot_scheduler() {
    local destination=$1
    {
        date -u +%Y-%m-%dT%H:%M:%SZ
        scontrol show job "${SLURM_JOB_ID}" || true
        squeue --steps --jobs="${SLURM_JOB_ID}" --noheader --format='%i %T %M %N' || true
        timeout 8s sstat --allsteps --jobs="${SLURM_JOB_ID}" \
            --format='JobID,State,AveCPU,MaxRSS,MaxVMSize,NTasks,NodeList' || true
        timeout 8s sacct --jobs="${SLURM_JOB_ID}" \
            --format='JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList' || true
    } >> "${destination}" 2>&1
}

cleanup_numeric_steps() {
    local destination=$1
    local step
    mapfile -t steps < <(squeue --steps --jobs="${SLURM_JOB_ID}" --noheader --format='%i' 2>/dev/null || true)
    for step in "${steps[@]}"; do
        if [[ "${step}" =~ ^${SLURM_JOB_ID}\.[0-9]+$ ]]; then
            echo "killing orphaned step ${step}" >> "${destination}"
            scancel --signal=KILL "${step}" >> "${destination}" 2>&1 || true
        fi
    done
    for _ in {1..20}; do
        local remaining=0
        mapfile -t steps < <(squeue --steps --jobs="${SLURM_JOB_ID}" --noheader --format='%i' 2>/dev/null || true)
        for step in "${steps[@]}"; do
            [[ "${step}" =~ ^${SLURM_JOB_ID}\.[0-9]+$ ]] && remaining=$((remaining+1))
        done
        [[ ${remaining} -eq 0 ]] && return 0
        sleep 1
    done
    echo "numeric Slurm steps still visible 20 seconds after cleanup" >> "${destination}"
}

capture_node_health() {
    local destination=$1
    timeout --signal=TERM --kill-after=5s 20s \
        srun --overlap --exact --kill-on-bad-exit=1 --mpi=none --network=no_vni \
        --cpu-bind=none --cpus-per-task=1 --nodes="${SLURM_NNODES}" \
        --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
        --gpus-per-node="${OLIVIA_GPUS_PER_NODE}" \
        bash -c 'hostname; nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader' \
        >> "${destination}" 2>&1 || echo "post-case nvidia-smi probe failed or timed out" >> "${destination}"
}

sample_case() {
    local process_id=$1
    local destination=$2
    while kill -0 "${process_id}" 2>/dev/null; do
        snapshot_scheduler "${destination}"
        sleep 5
    done
}

run_case() {
    local label=$1
    local expectation=$2
    local timeout_seconds=$3
    local required_marker=$4
    shift 4
    local case_dir=${diagnostics_root}/${label}
    mkdir -p "${case_dir}"
    export OLIVIA_CASE_DIAGNOSTICS=${case_dir}
    printf '%q ' "$@" > "${case_dir}/command.txt"
    printf '\n' >> "${case_dir}/command.txt"
    snapshot_scheduler "${case_dir}/scheduler-before.txt"

    echo "===== ${label}: expectation=${expectation} timeout=${timeout_seconds}s ====="
    set +o errexit
    timeout --signal=TERM --kill-after=15s "${timeout_seconds}s" "$@" \
        > "${case_dir}/stdout.log" 2> "${case_dir}/stderr.log" &
    local command_pid=$!
    sample_case "${command_pid}" "${case_dir}/scheduler-samples.txt" &
    local sampler_pid=$!
    wait "${command_pid}"
    local status=$?
    wait "${sampler_pid}" 2>/dev/null || true
    set -o errexit

    snapshot_scheduler "${case_dir}/scheduler-after-command.txt"
    if [[ ${status} -ne 0 ]]; then
        cleanup_numeric_steps "${case_dir}/cleanup.log"
    fi
    snapshot_scheduler "${case_dir}/scheduler-after-cleanup.txt"
    capture_node_health "${case_dir}/node-health-after.txt"

    local result=FAIL
    if [[ "${expectation}" == success && ${status} -eq 0 ]]; then
        result=PASS
    elif [[ "${expectation}" == timeout && ( ${status} -eq 124 || ${status} -eq 137 ) ]]; then
        if [[ "${required_marker}" != - ]] && grep --recursive --quiet --fixed-strings \
                "${required_marker}" "${case_dir}"; then
            result=PASS
        else
            echo "timeout occurred before required injected-stall marker: ${required_marker}" \
                >> "${case_dir}/validation-error.txt"
        fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${label}" "${expectation}" "${result}" \
        "${status}" "${timeout_seconds}" "${required_marker}" >> "${summary}"
    echo "${label}: ${result} (exit=${status})"
    if [[ "${result}" != PASS ]]; then
        failures=$((failures+1))
    fi
}

record_batch_termination() {
    snapshot_scheduler "${diagnostics_root}/batch-termination.txt"
    cleanup_numeric_steps "${diagnostics_root}/batch-termination.txt"
}
trap 'record_batch_termination; exit 143' TERM INT

for required in timeout srun scontrol squeue scancel sstat sacct julia "${OLIVIA_PYTHON}"; do
    command -v "${required}" >> "${diagnostics_root}/preflight.txt" 2>&1 || {
        echo "Missing required command: ${required}" | tee -a "${diagnostics_root}/preflight.txt"
        exit 2
    }
done

{
    date -u +%Y-%m-%dT%H:%M:%SZ
    echo "repository=${OLIVIA_REPO_DIR}"
    echo "job=${SLURM_JOB_ID} nodes=${SLURM_NNODES} nodelist=${SLURM_JOB_NODELIST}"
    echo "julia_project=${JULIA_PROJECT} julia_threads=${OLIVIA_JULIA_THREADS}"
    echo "python=${OLIVIA_PYTHON} gpus_per_node=${OLIVIA_GPUS_PER_NODE}"
    module list 2>&1
    julia --version
    "${OLIVIA_PYTHON}" -c 'import torch; print("torch",torch.__version__,"cuda",torch.version.cuda,"available",torch.cuda.is_available())'
    scontrol show job "${SLURM_JOB_ID}"
} > "${diagnostics_root}/environment.txt" 2>&1

mpi_command=(srun --kill-on-bad-exit=1 --mpi=pmix --gres=none
    --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1
    --cpu-bind=cores --cpus-per-task="${SLURM_CPUS_PER_TASK}"
    julia --project="${JULIA_PROJECT}" --startup-file=no --threads="${OLIVIA_JULIA_THREADS}")

run_case cpu_mpi_success success 90 - \
    "${mpi_command[@]}" test/olivia_mpi_probe.jl success
run_case cpu_mpi_intentional_stall timeout 30 intentional_cpu_stall_enter \
    "${mpi_command[@]}" test/olivia_mpi_probe.jl stall
run_case cpu_mpi_post_timeout_success success 90 - \
    "${mpi_command[@]}" test/olivia_mpi_probe.jl success

run_case hybrid_gpu_success success 240 - \
    "${mpi_command[@]}" test/olivia_hybrid_gpu_probe.jl success
run_case hybrid_gpu_failure_recovery success 240 - \
    "${mpi_command[@]}" test/olivia_hybrid_gpu_probe.jl failure
run_case hybrid_gpu_intentional_stall timeout 60 intentional_gpu_stall_enter \
    "${mpi_command[@]}" test/olivia_hybrid_gpu_probe.jl stall
run_case hybrid_gpu_post_timeout_success success 240 - \
    "${mpi_command[@]}" test/olivia_hybrid_gpu_probe.jl success

snapshot_scheduler "${diagnostics_root}/scheduler-final.txt"
tarball=${diagnostics_root}.tar.gz
tar -czf "${tarball}" -C "$(dirname "${diagnostics_root}")" "$(basename "${diagnostics_root}")"

echo
echo "===== Olivia runtime test summary ====="
column -t -s $'\t' "${summary}" || cat "${summary}"
echo "Diagnostics: ${diagnostics_root}"
echo "Archive: ${tarball}"

if [[ ${failures} -ne 0 ]]; then
    echo "OLIVIA_RUNTIME_TESTS_FAILED count=${failures}"
    exit 1
fi
echo "OLIVIA_RUNTIME_TESTS_OK"
