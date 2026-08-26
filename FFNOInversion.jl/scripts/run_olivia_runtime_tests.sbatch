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

# Resolve module-provided executables once on the batch host. Some Olivia
# nodes have presented a different task PATH even though the module setup was
# successful in the batch shell; using absolute paths prevents nine repeated
# exit-127 failures and lets the all-node preflight identify a missing mount.
julia_candidate=${OLIVIA_JULIA:-julia}
python_candidate=${OLIVIA_PYTHON:-python3}
OLIVIA_JULIA=$(command -v "${julia_candidate}") || {
    echo "Cannot resolve Julia executable: ${julia_candidate}" >&2
    exit 2
}
OLIVIA_PYTHON=$(command -v "${python_candidate}") || {
    echo "Cannot resolve Python executable: ${python_candidate}" >&2
    exit 2
}
export OLIVIA_JULIA OLIVIA_PYTHON

# Resolve the package root independently of the directory from which sbatch was
# invoked. The established Olivia checkout is the default; OLIVIA_REPO_DIR is
# only needed when deliberately testing another checkout.
default_olivia_repo=/cluster/work/projects/nn2834k/harshm/FNOML/FFNOInversion.jl
if [[ -n "${OLIVIA_REPO_DIR:-}" ]]; then
    repository_candidate=${OLIVIA_REPO_DIR}
elif [[ -f "${default_olivia_repo}/Project.toml" &&
        -f "${default_olivia_repo}/src/FFNOInversion.jl" ]]; then
    repository_candidate=${default_olivia_repo}
elif [[ -f "${SLURM_SUBMIT_DIR}/Project.toml" &&
        -f "${SLURM_SUBMIT_DIR}/src/FFNOInversion.jl" ]]; then
    repository_candidate=${SLURM_SUBMIT_DIR}
elif [[ -f "$(dirname "${SLURM_SUBMIT_DIR}")/Project.toml" &&
        -f "$(dirname "${SLURM_SUBMIT_DIR}")/src/FFNOInversion.jl" ]]; then
    repository_candidate=$(dirname "${SLURM_SUBMIT_DIR}")
else
    echo "Cannot locate the default Olivia checkout at ${default_olivia_repo}." >&2
    echo "Also checked SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR} and its parent." >&2
    echo "Set OLIVIA_REPO_DIR only when using a different checkout." >&2
    exit 2
fi
export OLIVIA_REPO_DIR
OLIVIA_REPO_DIR=$(cd "${repository_candidate}" && pwd)
export OLIVIA_REPO_DIR
cd "${OLIVIA_REPO_DIR}" || exit 2

export JULIA_PROJECT=${OLIVIA_JULIA_PROJECT:-${OLIVIA_REPO_DIR}}
if [[ -n "${OLIVIA_JULIA_DEPOT:-}" ]]; then
    export JULIA_DEPOT_PATH=${OLIVIA_JULIA_DEPOT}
fi
export OLIVIA_GPUS_PER_NODE=${OLIVIA_GPUS_PER_NODE:-4}
export OLIVIA_JULIA_THREADS=${OLIVIA_JULIA_THREADS:-4}
export OLIVIA_GPU_CPUS_PER_NODE=${OLIVIA_GPU_CPUS_PER_NODE:-8}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export FFNO_GPU_CONTROL_HOST=${FFNO_GPU_CONTROL_HOST:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export FFNO_GPU_CONNECT_TIMEOUT=${FFNO_GPU_CONNECT_TIMEOUT:-60}
export FFNO_GPU_STATUS_TIMEOUT=${FFNO_GPU_STATUS_TIMEOUT:-180}
export FFNO_GPU_DIAGNOSTIC_INTERVAL=${FFNO_GPU_DIAGNOSTIC_INTERVAL:-5}
export PHASE6_MEMORY_NX=${PHASE6_MEMORY_NX:-800}
export PHASE6_MEMORY_NY=${PHASE6_MEMORY_NY:-800}
export PHASE6_MEMORY_NZ=${PHASE6_MEMORY_NZ:-64}
export PHASE6_MEMORY_NLAMBDA=${PHASE6_MEMORY_NLAMBDA:-134}
export PHASE6_MEMORY_LEVELS=${PHASE6_MEMORY_LEVELS:-10}
export PHASE6_MEMORY_LIMIT_GIB=${PHASE6_MEMORY_LIMIT_GIB:-24}

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
test_group=${OLIVIA_TEST_GROUP:-safe}
case "${test_group}" in
    safe|phase6|internal_timeout|external_timeout|recovery) ;;
    *)
        echo "Unknown OLIVIA_TEST_GROUP=${test_group}; expected safe, phase6, internal_timeout, external_timeout, or recovery" >&2
        exit 2
        ;;
esac

failures=0

snapshot_scheduler() {
    local destination=$1
    {
        date -u +%Y-%m-%dT%H:%M:%SZ
        scontrol show job "${SLURM_JOB_ID}" || true
        squeue --steps --jobs="${SLURM_JOB_ID}" --noheader --format='%i %t %M %N' || true
        timeout 8s sstat --allsteps --jobs="${SLURM_JOB_ID}" \
            --format='JobID,AveCPU,MaxRSS,MaxVMSize,NTasks,NodeList' || true
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
    return 1
}

wait_for_interconnect_recovery() {
    local destination=$1
    local deadline=$((SECONDS+90))
    local attempt=0
    local status

    while (( SECONDS < deadline )); do
        attempt=$((attempt+1))
        printf '%s interconnect readiness attempt=%d\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${attempt}" >> "${destination}"
        if timeout --signal=TERM --kill-after=5s 15s \
                srun --kill-on-bad-exit=1 --mpi=pmix --gres=none \
                --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" \
                --ntasks-per-node=1 --cpu-bind=none --cpus-per-task=1 \
                env CUDA_VISIBLE_DEVICES=-1 /bin/true \
                >> "${destination}" 2>&1; then
            printf '%s interconnect ready attempt=%d\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${attempt}" >> "${destination}"
            return 0
        else
            status=$?
            printf '%s interconnect not ready attempt=%d exit=%d\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${attempt}" "${status}" \
                >> "${destination}"
        fi
        cleanup_numeric_steps "${destination}" || return 1
        sleep 5
    done

    printf '%s interconnect did not recover within 90 seconds\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${destination}"
    return 1
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
    local recovery_policy=$5
    shift 5
    [[ "${recovery_policy}" == reusable || "${recovery_policy}" == terminal ]] || {
        echo "unknown recovery policy for ${label}: ${recovery_policy}" >&2
        exit 2
    }
    local case_dir=${diagnostics_root}/${label}
    local cleanup_ready=1
    mkdir -p "${case_dir}"
    export OLIVIA_CASE_DIAGNOSTICS=${case_dir}
    export FFNO_GPU_DIAGNOSTICS_DIR=${case_dir}/gpu-control
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
        if ! cleanup_numeric_steps "${case_dir}/cleanup.log"; then
            cleanup_ready=0
        elif [[ "${recovery_policy}" == reusable ]] && \
                ! wait_for_interconnect_recovery "${case_dir}/interconnect-recovery.log"; then
            cleanup_ready=0
        fi
    fi
    snapshot_scheduler "${case_dir}/scheduler-after-cleanup.txt"
    capture_node_health "${case_dir}/node-health-after.txt"

    local result=FAIL
    if [[ ${cleanup_ready} -eq 1 && "${expectation}" == success && ${status} -eq 0 ]]; then
        if [[ "${required_marker}" == - ]] || grep --recursive --quiet --fixed-strings \
                "${required_marker}" "${case_dir}"; then
            result=PASS
        else
            echo "successful command did not contain required marker: ${required_marker}" \
                >> "${case_dir}/validation-error.txt"
        fi
    elif [[ ${cleanup_ready} -eq 1 && "${expectation}" == controlled_failure && ${status} -ne 0 &&
            ${status} -ne 124 && ${status} -ne 137 ]]; then
        if [[ "${required_marker}" != - ]] && grep --recursive --quiet --fixed-strings \
                "${required_marker}" "${case_dir}"; then
            result=PASS
        else
            echo "controlled failure did not contain required marker: ${required_marker}" \
                >> "${case_dir}/validation-error.txt"
        fi
    elif [[ ${cleanup_ready} -eq 1 && "${expectation}" == timeout && ( ${status} -eq 124 || ${status} -eq 137 ) ]]; then
        if [[ "${required_marker}" != - ]] && grep --recursive --quiet --fixed-strings \
                "${required_marker}" "${case_dir}"; then
            result=PASS
        else
            echo "timeout occurred before required injected-stall marker: ${required_marker}" \
                >> "${case_dir}/validation-error.txt"
        fi
    fi
    if [[ ${cleanup_ready} -ne 1 ]]; then
        echo "Slingshot/PMIx did not become ready after forced step cleanup" \
            >> "${case_dir}/validation-error.txt"
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
    cleanup_numeric_steps "${diagnostics_root}/batch-termination.txt" || true
}
trap 'record_batch_termination; exit 143' TERM INT

for required in timeout srun scontrol squeue scancel sstat sacct; do
    command -v "${required}" >> "${diagnostics_root}/preflight.txt" 2>&1 || {
        echo "Missing required command: ${required}" | tee -a "${diagnostics_root}/preflight.txt"
        exit 2
    }
done

for required_executable in "${OLIVIA_JULIA}" "${OLIVIA_PYTHON}"; do
    if [[ ! -x "${required_executable}" ]]; then
        echo "Resolved executable is not executable: ${required_executable}" \
            | tee -a "${diagnostics_root}/preflight.txt"
        exit 2
    fi
done

for required_file in Project.toml src/FFNOInversion.jl \
        test/olivia_mpi_probe.jl test/olivia_hybrid_gpu_probe.jl \
        test/olivia_phase6_solver_probe.jl test/olivia_phase6_memory_probe.jl \
        test/olivia_gpu_collective_probe.py scripts/olivia_nested_gpu_step.sh; do
    if [[ ! -f "${OLIVIA_REPO_DIR}/${required_file}" ]]; then
        echo "Missing required repository file: ${OLIVIA_REPO_DIR}/${required_file}" \
            | tee -a "${diagnostics_root}/preflight.txt"
        exit 2
    fi
done

# Julia MPI defaults to MPICH_jll, but Olivia's `srun --mpi=pmix` and the
# loadnvidiampi.sh module stack use system OpenMPI. Persist a checkout-local MPI
# preference before importing FFNOInversion/MPI in a fresh Julia process. This
# is the same selection used by the established charge-branch environment.
set +o errexit
timeout --signal=TERM --kill-after=15s 180s \
    "${OLIVIA_JULIA}" --project="${JULIA_PROJECT}" --startup-file=no -e \
    'using MPIPreferences; MPIPreferences.use_system_binary(mpiexec="srun", abi="OpenMPI")' \
    >> "${diagnostics_root}/preflight.txt" 2>&1
mpi_preference_status=$?
set -o errexit
if [[ ${mpi_preference_status} -ne 0 ]]; then
    echo "Julia system-OpenMPI preference setup failed (exit=${mpi_preference_status})." \
        | tee -a "${diagnostics_root}/preflight.txt"
    exit 2
fi

# Detect a missing/uninstantiated Julia environment or system-MPI loading
# problem once. Without this gate, Slurm's --kill-on-bad-exit can report 143
# for every case after one rank exits with the same immediate load error.
set +o errexit
timeout --signal=TERM --kill-after=15s 180s \
    "${OLIVIA_JULIA}" --project="${JULIA_PROJECT}" --startup-file=no -e \
    'using FFNOInversion; import MPI; MPI.MPIPreferences.binary == "system" || error("MPI binary is not system"); MPI.MPIPreferences.abi == "OpenMPI" || error("MPI ABI is not OpenMPI"); println("OLIVIA_JULIA_PREFLIGHT_OK binary=", MPI.MPIPreferences.binary, " abi=", MPI.MPIPreferences.abi, " threads=", Threads.nthreads())' \
    >> "${diagnostics_root}/preflight.txt" 2>&1
julia_preflight_status=$?
set -o errexit
if [[ ${julia_preflight_status} -ne 0 ]]; then
    echo "Julia project/system-MPI preflight failed (exit=${julia_preflight_status}); no runtime cases were launched." \
        | tee -a "${diagnostics_root}/preflight.txt"
    exit 2
fi

# Verify that both module-provided executable paths are mounted and executable
# on every allocated node. This is deliberately non-MPI and requests no GPUs,
# so environment failures are reported once before the runtime sequence.
set +o errexit
timeout --signal=TERM --kill-after=5s 60s \
    srun --exact --kill-on-bad-exit=1 --mpi=none --network=no_vni --gres=none \
    --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    --cpu-bind=none --cpus-per-task=1 \
    env OLIVIA_JULIA="${OLIVIA_JULIA}" OLIVIA_PYTHON="${OLIVIA_PYTHON}" \
    /bin/bash -c '
        echo "host=$(hostname) julia=${OLIVIA_JULIA} python=${OLIVIA_PYTHON}"
        [[ -x "${OLIVIA_JULIA}" ]] || exit 127
        [[ -x "${OLIVIA_PYTHON}" ]] || exit 127
        "${OLIVIA_JULIA}" --version
        "${OLIVIA_PYTHON}" --version
    ' >> "${diagnostics_root}/preflight.txt" 2>&1
node_executable_status=$?
set -o errexit
if [[ ${node_executable_status} -ne 0 ]]; then
    echo "Julia/Python all-node executable preflight failed (exit=${node_executable_status}); no runtime cases were launched." \
        | tee -a "${diagnostics_root}/preflight.txt"
    exit 2
fi

{
    date -u +%Y-%m-%dT%H:%M:%SZ
    echo "submit_dir=${SLURM_SUBMIT_DIR}"
    echo "repository=${OLIVIA_REPO_DIR}"
    echo "job=${SLURM_JOB_ID} nodes=${SLURM_NNODES} nodelist=${SLURM_JOB_NODELIST}"
    echo "test_group=${test_group}"
    echo "julia_project=${JULIA_PROJECT} julia_threads=${OLIVIA_JULIA_THREADS}"
    echo "python=${OLIVIA_PYTHON} gpus_per_node=${OLIVIA_GPUS_PER_NODE}"
    module list 2>&1
    "${OLIVIA_JULIA}" --version
    "${OLIVIA_PYTHON}" -c 'import torch; print("torch",torch.__version__,"cuda",torch.version.cuda,"available",torch.cuda.is_available())'
    scontrol show job "${SLURM_JOB_ID}"
} > "${diagnostics_root}/environment.txt" 2>&1

# Keep the long-lived Julia/OpenMPI ranks CUDA-blind. Olivia's CUDA-aware
# OpenMPI/OFI stack may otherwise initialize the default device on every node,
# which prevents nested torchrun local-rank 0 from acquiring that GPU when the
# devices use exclusive-process mode. The nested GPU srun step receives a new,
# step-local CUDA_VISIBLE_DEVICES value from Slurm's GRES plugin.
mpi_command=(srun --kill-on-bad-exit=1 --mpi=pmix --gres=none
    --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1
    --cpu-bind=cores --cpus-per-task="${SLURM_CPUS_PER_TASK}"
    env CUDA_VISIBLE_DEVICES=-1
    "${OLIVIA_JULIA}" --project="${JULIA_PROJECT}" --startup-file=no --threads="${OLIVIA_JULIA_THREADS}")

case "${test_group}" in
    safe)
        run_case cpu_mpi_success success 90 OLIVIA_MPI_PROBE_OK reusable \
            "${mpi_command[@]}" "${OLIVIA_REPO_DIR}/test/olivia_mpi_probe.jl" success
        run_case cpu_mpi_intentional_stall timeout 30 intentional_cpu_stall_enter reusable \
            "${mpi_command[@]}" "${OLIVIA_REPO_DIR}/test/olivia_mpi_probe.jl" stall
        run_case cpu_mpi_post_timeout_success success 90 OLIVIA_MPI_PROBE_OK reusable \
            "${mpi_command[@]}" "${OLIVIA_REPO_DIR}/test/olivia_mpi_probe.jl" success
        run_case hybrid_gpu_success success 240 OLIVIA_GPU_PROBE_OK reusable \
            "${mpi_command[@]}" "${OLIVIA_REPO_DIR}/test/olivia_hybrid_gpu_probe.jl" success
        run_case hybrid_gpu_failure_recovery success 240 intentional_gpu_process_failure reusable \
            "${mpi_command[@]}" "${OLIVIA_REPO_DIR}/test/olivia_hybrid_gpu_probe.jl" failure
        ;;
    phase6)
        run_case phase6_solver_mpi success 180 OLIVIA_PHASE6_SOLVER_OK reusable \
            "${mpi_command[@]}" "${OLIVIA_REPO_DIR}/test/olivia_phase6_solver_probe.jl" success
        run_case phase6_rejected_trial_recovery success 120 OLIVIA_PHASE6_REJECTION_RECOVERY_OK reusable \
            "${mpi_command[@]}" "${OLIVIA_REPO_DIR}/test/olivia_phase6_solver_probe.jl" rejection
        run_case phase6_full_layout_memory success 300 OLIVIA_PHASE6_MEMORY_LAYOUT_OK reusable \
            "${mpi_command[@]}" "${OLIVIA_REPO_DIR}/test/olivia_phase6_memory_probe.jl"
        run_case phase6_rank0_gpu_autograd_vjp success 240 OLIVIA_GPU_VJP_PROBE_OK reusable \
            "${mpi_command[@]}" "${OLIVIA_REPO_DIR}/test/olivia_hybrid_gpu_probe.jl" vjp
        ;;
    internal_timeout)
        run_case hybrid_gpu_pre_internal_timeout_success success 240 OLIVIA_GPU_PROBE_OK reusable \
            "${mpi_command[@]}" "${OLIVIA_REPO_DIR}/test/olivia_hybrid_gpu_probe.jl" success
        run_case hybrid_gpu_internal_status_timeout controlled_failure 90 gpu_status_timeout terminal \
            env FFNO_GPU_STATUS_TIMEOUT=20 "${mpi_command[@]}" \
            "${OLIVIA_REPO_DIR}/test/olivia_hybrid_gpu_probe.jl" stall
        ;;
    external_timeout)
        run_case hybrid_gpu_post_internal_timeout_success success 240 OLIVIA_GPU_PROBE_OK reusable \
            "${mpi_command[@]}" "${OLIVIA_REPO_DIR}/test/olivia_hybrid_gpu_probe.jl" success
        run_case hybrid_gpu_external_watchdog_stall timeout 60 intentional_gpu_stall_enter terminal \
            env FFNO_GPU_STATUS_TIMEOUT=0 "${mpi_command[@]}" \
            "${OLIVIA_REPO_DIR}/test/olivia_hybrid_gpu_probe.jl" stall
        ;;
    recovery)
        run_case hybrid_gpu_post_timeout_success success 240 OLIVIA_GPU_PROBE_OK reusable \
            "${mpi_command[@]}" "${OLIVIA_REPO_DIR}/test/olivia_hybrid_gpu_probe.jl" success
        ;;
esac

snapshot_scheduler "${diagnostics_root}/scheduler-final.txt"
tarball=${diagnostics_root}.tar.gz
tar -czf "${tarball}" -C "$(dirname "${diagnostics_root}")" "$(basename "${diagnostics_root}")"

echo
echo "===== Olivia runtime test summary ====="
echo "Test group: ${test_group}"
column -t -s $'\t' "${summary}" || cat "${summary}"
echo "Diagnostics: ${diagnostics_root}"
echo "Archive: ${tarball}"

if [[ ${failures} -ne 0 ]]; then
    echo "OLIVIA_RUNTIME_TESTS_FAILED group=${test_group} count=${failures}"
    exit 1
fi
echo "OLIVIA_RUNTIME_TESTS_OK group=${test_group}"
