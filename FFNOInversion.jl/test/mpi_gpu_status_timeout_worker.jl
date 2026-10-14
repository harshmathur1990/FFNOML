using FFNOInversion

directory=get(ENV,"GPU_TIMEOUT_DIAGNOSTICS_DIR","")
isempty(directory) && error("GPU_TIMEOUT_DIAGNOSTICS_DIR must be set")
context=initialize_parallel(options=ParallelOptions(enabled=true,
    threads_per_rank=Threads.nthreads(),gpu_connect_timeout_seconds=5.0,
    gpu_status_timeout_seconds=0.3,gpu_diagnostic_interval_seconds=0.05,
    gpu_diagnostics_directory=directory))
try
    caught=false; timed_out=false
    try
        launch_gpu!(RootGPUCoordinator(() -> (sleep(1.0); :late)),context)
    catch exception
        message=sprint(showerror,exception)
        caught=true
        timed_out=occursin("timed out after 0.3 s waiting",message)
    end
    allreduce_sum(caught ? 1 : 0,context)==context.size ||
        error("status-timeout failure did not reach every rank")
    allreduce_sum(timed_out ? 1 : 0,context)==context.size-1 ||
        error("non-root ranks did not report the internal status timeout")
    barrier(context)
    recovered=launch_gpu!(RootGPUCoordinator(() -> 42),context)
    isroot(context) && recovered!=42 && error("GPU control did not recover after status timeout")
    barrier(context)
    label=lpad(context.rank,4,'0')
    events=read(joinpath(directory,"gpu-control-events-rank-$label.log"),String)
    local_ok=occursin("gpu_diagnostics_initialized",events) &&
        occursin("gpu_diagnostics_stopped",events) &&
        (isroot(context) || occursin("gpu_status_timeout",events))
    allreduce_sum(local_ok ? 1 : 0,context)==context.size ||
        error("periodic GPU-control diagnostics are incomplete")
    isroot(context) && println("MPI_GPU_STATUS_TIMEOUT_RECOVERY_OK ranks=$(context.size)")
finally
    finalize_parallel!(context)
end
