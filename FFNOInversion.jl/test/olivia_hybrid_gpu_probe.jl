using FFNOInversion
using Sockets

mode=length(ARGS)==1 ? Symbol(ARGS[1]) : error(
    "usage: olivia_hybrid_gpu_probe.jl success|failure|stall|vjp|ffno_vjp")
mode in (:success,:failure,:stall,:vjp,:ffno_vjp) || error("unknown hybrid probe mode: $mode")

context=initialize_parallel(options=ParallelOptions(enabled=true,threads_per_rank=Threads.nthreads()))
diagnostics_root=get(ENV,"OLIVIA_CASE_DIAGNOSTICS",pwd())
mkpath(diagnostics_root)
log_path=joinpath(diagnostics_root,"hybrid-rank-$(context.rank).log")

function record(event;details="")
    open(log_path,"a") do io
        println(io,"$(time()) rank=$(context.rank) pid=$(getpid()) host=$(gethostname()) event=$event $details")
        flush(io)
    end
end

stop_watchdog=Threads.Atomic{Bool}(false)
watchdog=Threads.@spawn begin
    while !stop_watchdog[]
        record("watchdog_alive";details="thread=$(Threads.threadid())")
        sleep(5)
    end
end

probe_failed=Ref(false)
try
    launcher_path=get(ENV,"OLIVIA_GPU_PROBE_LAUNCHER",
        joinpath(dirname(@__DIR__),"scripts","olivia_nested_gpu_step.sh"))
    gpu_mode=String(mode)
    coordinator=RootGPUCoordinator(() -> begin
        record("rank0_nested_gpu_launch_enter";details="mode=$gpu_mode launcher=$launcher_path")
        run(Cmd([launcher_path,gpu_mode]))
        record("rank0_nested_gpu_launch_complete";details="mode=$gpu_mode")
        true
    end;launcher_rank=context.root)

    record("hybrid_launch_enter";details="mode=$mode outer_cuda_visible_devices=$(get(ENV,"CUDA_VISIBLE_DEVICES","<unset>"))")
    if mode==:failure
        caught=false
        try
            launch_gpu!(coordinator,context)
        catch exception
            caught=occursin("rank-0 GPU launcher failed",sprint(showerror,exception))
            record("expected_failure_caught";details="matched=$caught")
        end
        allreduce_sum(caught ? 1 : 0,context)==context.size ||
            error("GPU-process failure was not propagated to every MPI rank")
        barrier(context)
        recovery=launch_gpu!(RootGPUCoordinator(() -> 42),context)
        isroot(context) && recovery!=42 && error("post-failure control recovery returned $recovery")
        barrier(context)
        isroot(context) && println("OLIVIA_HYBRID_GPU_FAILURE_RECOVERY_OK ranks=$(context.size)")
    else
        result=launch_gpu!(coordinator,context)
        isroot(context) && result!==true && error("GPU success launcher returned $result")
        barrier(context)
        isroot(context) && println("OLIVIA_HYBRID_GPU_PROBE_OK mode=$mode ranks=$(context.size)")
    end
catch exception
    probe_failed[]=true
    record("hybrid_probe_exception";
        details="exception=$(repr(sprint(showerror,exception,catch_backtrace())))")
    rethrow()
finally
    stop_watchdog[]=true
    wait(watchdog)
    if probe_failed[]
        # MPI_Finalize after one rank fails can mask the original exception
        # with a libfabric/OFI teardown abort.  Let srun terminate the peers;
        # the failing rank's durable log above remains the primary evidence.
        record("finalize_skipped_after_exception")
    else
        barrier(context)
        record("finalizing")
        finalize_parallel!(context)
    end
end
