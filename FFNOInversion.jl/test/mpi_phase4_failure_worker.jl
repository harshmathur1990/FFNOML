using FFNOInversion

context=initialize_parallel(options=ParallelOptions(enabled=true,threads_per_rank=Threads.nthreads()))
try
    caught=false
    try
        launch_gpu!(RootGPUCoordinator(() -> error("injected Phase 4 GPU-service failure")),context)
    catch exception
        caught=occursin("injected Phase 4 GPU-service failure",sprint(showerror,exception))
    end
    allreduce_sum(caught ? 1 : 0,context)==context.size || error("not every rank received the injected failure")
    barrier(context)
    recovered=launch_gpu!(RootGPUCoordinator(() -> 42),context)
    isroot(context) && recovered!=42 && error("rank 0 did not recover after injected failure")
    allreduce_sum(isroot(context) && recovered==42 ? 1 : 0,context)==1 || error("GPU control recovery was inconsistent")
    isroot(context) && println("MPI_PHASE4_FAILURE_RECOVERY_OK ranks=$(context.size)")
finally
    finalize_parallel!(context)
end

