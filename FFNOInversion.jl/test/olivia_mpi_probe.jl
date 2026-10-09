using FFNOInversion
using Sockets

mode=length(ARGS)==1 ? Symbol(ARGS[1]) : error("usage: olivia_mpi_probe.jl success|stall")
mode in (:success,:stall) || error("unknown MPI probe mode: $mode")

context=initialize_parallel(options=ParallelOptions(enabled=true,threads_per_rank=Threads.nthreads()))
diagnostics_root=get(ENV,"OLIVIA_CASE_DIAGNOSTICS",pwd())
mkpath(diagnostics_root)
log_path=joinpath(diagnostics_root,"mpi-rank-$(context.rank).log")

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

try
    record("mpi_initialized";details="size=$(context.size) threads=$(Threads.nthreads())")
    barrier(context)
    record("initial_barrier_complete")
    if mode==:stall
        if context.rank==0
            record("intentional_cpu_stall_enter")
            while true
                sleep(60)
            end
        else
            record("intentional_cpu_stall_barrier_enter")
            barrier(context)
        end
    end
    reduced=allreduce_sum(context.rank+1,context)
    expected=context.size*(context.size+1)÷2
    reduced==expected || error("MPI allreduce returned $reduced, expected $expected")
    record("allreduce_complete";details="value=$reduced")
    barrier(context)
    isroot(context) && println("OLIVIA_MPI_PROBE_OK ranks=$(context.size) threads=$(Threads.nthreads()) sum=$reduced")
finally
    stop_watchdog[]=true
    wait(watchdog)
    record("finalizing")
    finalize_parallel!(context)
end
