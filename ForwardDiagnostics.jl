using Dates
using Sockets


mutable struct ForwardDiagnostics
    directory::String
    event_path::String
    resource_path::String
    failure_path::String
    rank::Int
    host::String
    start_time::Float64
    interval::Float64
    phase::Base.RefValue{String}
    dataset::Base.RefValue{String}
    iteration::Base.RefValue{Int}
    stop_requested::Base.RefValue{Bool}
    monitor_task::Any
    lock::ReentrantLock
    last_cpu_ticks::Base.RefValue{Float64}
    last_cpu_wall::Base.RefValue{Float64}
end


diagnostic_event!(::Nothing, event::AbstractString; kwargs...) = nothing
set_diagnostic_context!(::Nothing; kwargs...) = nothing
write_resource_snapshot!(::Nothing) = nothing
diagnostic_checkpoint!(::Nothing, event::AbstractString; kwargs...) = nothing
record_diagnostic_failure!(::Nothing, exception, backtrace) = nothing


function diagnostic_timestamp()
    return Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ")
end


function default_diagnostics_directory()
    configured = get(ENV, "FORWARD_DIAGNOSTICS_DIR", "")
    isempty(configured) || return abspath(configured)
    job_id = get(ENV, "SLURM_JOB_ID", "")
    identifier = isempty(job_id) ?
                 "local-" * Dates.format(now(UTC), dateformat"yyyymmddTHHMMSS") :
                 "slurm-$(job_id)"
    return abspath("forward-diagnostics-$(identifier)")
end


function linux_status_values()
    result = Dict{String,String}()
    Sys.islinux() || return result
    for line in eachline("/proc/self/status")
        parts = split(line, ':'; limit=2)
        length(parts) == 2 || continue
        result[strip(parts[1])] = strip(parts[2])
    end
    return result
end


function status_kib(status, key)
    value = get(status, key, "")
    isempty(value) && return NaN
    fields = split(value)
    isempty(fields) && return NaN
    return parse(Float64, fields[1])
end


function linux_memory_values()
    result = Dict{String,Float64}()
    Sys.islinux() || return result
    for line in eachline("/proc/meminfo")
        parts = split(line, ':'; limit=2)
        length(parts) == 2 || continue
        fields = split(strip(parts[2]))
        isempty(fields) && continue
        result[strip(parts[1])] = parse(Float64, fields[1])
    end
    return result
end


function linux_process_cpu_ticks()
    Sys.islinux() || return NaN
    line = read("/proc/self/stat", String)
    closing_parenthesis = findlast(')', line)
    closing_parenthesis === nothing && return NaN
    # Fields following the executable name begin with process-state (field 3).
    fields = split(strip(line[nextind(line, closing_parenthesis):end]))
    length(fields) >= 13 || return NaN
    user_ticks = parse(Float64, fields[12])  # /proc field 14
    system_ticks = parse(Float64, fields[13]) # /proc field 15
    return user_ticks + system_ticks
end


function linux_clock_ticks_per_second()
    Sys.islinux() || return NaN
    # Linux _SC_CLK_TCK is 2. This is used only on Linux with /proc/self/stat.
    return Float64(ccall(:sysconf, Clong, (Cint,), 2))
end


function resource_snapshot_unlocked(diagnostics::ForwardDiagnostics)
    wall_time = time()
    cpu_ticks = linux_process_cpu_ticks()
    ticks_per_second = linux_clock_ticks_per_second()
    cpu_percent = NaN
    if isfinite(cpu_ticks) && isfinite(diagnostics.last_cpu_ticks[]) &&
       wall_time > diagnostics.last_cpu_wall[] && ticks_per_second > 0
        cpu_percent = 100 * (cpu_ticks - diagnostics.last_cpu_ticks[]) /
                      ticks_per_second / (wall_time - diagnostics.last_cpu_wall[])
    end
    diagnostics.last_cpu_ticks[] = cpu_ticks
    diagnostics.last_cpu_wall[] = wall_time

    status = linux_status_values()
    memory = linux_memory_values()
    rss_kib = status_kib(status, "VmRSS")
    hwm_kib = status_kib(status, "VmHWM")
    if !isfinite(rss_kib)
        rss_kib = Sys.maxrss() / 1024
    end
    if !isfinite(hwm_kib)
        hwm_kib = Sys.maxrss() / 1024
    end

    return (
        timestamp=diagnostic_timestamp(),
        elapsed_s=wall_time - diagnostics.start_time,
        rank=diagnostics.rank,
        host=diagnostics.host,
        dataset=diagnostics.dataset[],
        iteration=diagnostics.iteration[],
        phase=diagnostics.phase[],
        rss_mib=rss_kib / 1024,
        hwm_mib=hwm_kib / 1024,
        gc_live_mib=Base.gc_live_bytes() / 1024^2,
        cpu_percent=cpu_percent,
        load1=Sys.loadavg()[1],
        node_mem_available_gib=get(memory, "MemAvailable", NaN) / 1024^2,
        node_mem_total_gib=get(memory, "MemTotal", NaN) / 1024^2,
        julia_threads=Threads.nthreads(),
        slurm_cpus_per_task=get(ENV, "SLURM_CPUS_PER_TASK", ""),
        cpus_allowed=get(status, "Cpus_allowed_list", ""),
    )
end


function csv_value(value)
    value isa AbstractString && return replace(value, ',' => ';', '\n' => ' ')
    value isa AbstractFloat && return isfinite(value) ? string(round(value; digits=4)) : ""
    return string(value)
end


function write_resource_snapshot!(diagnostics::ForwardDiagnostics)
    lock(diagnostics.lock) do
        snapshot = resource_snapshot_unlocked(diagnostics)
        open(diagnostics.resource_path, "a") do io
            println(io, join(csv_value.(values(snapshot)), ','))
            flush(io)
        end
        return snapshot
    end
end


function diagnostic_event!(
    diagnostics::ForwardDiagnostics,
    event::AbstractString;
    level::AbstractString="INFO",
    values...,
)
    lock(diagnostics.lock) do
        details = join(["$(key)=$(repr(value))" for (key, value) in pairs(values)], " ")
        open(diagnostics.event_path, "a") do io
            print(
                io,
                diagnostic_timestamp(),
                " level=", level,
                " rank=", diagnostics.rank,
                " host=", diagnostics.host,
                " dataset=", repr(diagnostics.dataset[]),
                " iteration=", diagnostics.iteration[],
                " phase=", repr(diagnostics.phase[]),
                " event=", repr(event),
            )
            isempty(details) || print(io, " ", details)
            println(io)
            flush(io)
        end
    end
    return nothing
end


"""Write a durable event and an accompanying resource sample at a phase boundary."""
function diagnostic_checkpoint!(
    diagnostics::ForwardDiagnostics,
    event::AbstractString;
    values...,
)
    diagnostic_event!(diagnostics, event; values...)
    write_resource_snapshot!(diagnostics)
    return nothing
end


function set_diagnostic_context!(
    diagnostics::ForwardDiagnostics;
    dataset=nothing,
    iteration=nothing,
    phase=nothing,
)
    dataset === nothing || (diagnostics.dataset[] = String(dataset))
    iteration === nothing || (diagnostics.iteration[] = Int(iteration))
    phase === nothing || (diagnostics.phase[] = String(phase))
    return nothing
end


function record_diagnostic_failure!(diagnostics::ForwardDiagnostics, exception, backtrace)
    snapshot = write_resource_snapshot!(diagnostics)
    lock(diagnostics.lock) do
        open(diagnostics.failure_path, "a") do io
            println(io, "="^80)
            println(io, "timestamp: ", diagnostic_timestamp())
            println(io, "rank: ", diagnostics.rank, " host: ", diagnostics.host)
            println(io, "dataset: ", diagnostics.dataset[])
            println(io, "iteration: ", diagnostics.iteration[])
            println(io, "phase: ", diagnostics.phase[])
            println(io, "resource_snapshot: ", snapshot)
            println(io, "exception:")
            println(io, sprint(showerror, exception, backtrace))
            flush(io)
        end
    end
    diagnostic_event!(diagnostics, "failure"; level="ERROR", exception=string(exception))
    return nothing
end


function initialize_diagnostics(
    context::ForwardParallelContext;
    directory::Union{Nothing,String}=nothing,
    interval::Real=30.0,
)
    interval >= 0 || error("Resource-monitor interval cannot be negative")
    selected_directory = parallel_bcast(
        parallel_isroot(context) ?
        (directory === nothing ? default_diagnostics_directory() : abspath(directory)) : nothing,
        context,
    )
    parallel_root_call(context) do
        mkpath(selected_directory)
        probe = joinpath(selected_directory, ".write-probe")
        open(probe, "w") do io
            write(io, "Forward.jl diagnostics write test\n")
        end
        rm(probe)
    end
    parallel_barrier(context)

    rank_label = lpad(context.rank, 4, '0')
    diagnostics = ForwardDiagnostics(
        selected_directory,
        joinpath(selected_directory, "events-rank-$(rank_label).log"),
        joinpath(selected_directory, "resources-rank-$(rank_label).csv"),
        joinpath(selected_directory, "failure-rank-$(rank_label).log"),
        context.rank,
        gethostname(),
        time(),
        Float64(interval),
        Ref("initialization"),
        Ref(""),
        Ref(0),
        Ref(false),
        nothing,
        ReentrantLock(),
        Ref(NaN),
        Ref(time()),
    )
    open(diagnostics.resource_path, "w") do io
        println(
            io,
            "timestamp,elapsed_s,rank,host,dataset,iteration,phase,rss_mib,hwm_mib," *
            "gc_live_mib,cpu_percent,load1,node_mem_available_gib,node_mem_total_gib," *
            "julia_threads,slurm_cpus_per_task,cpus_allowed",
        )
    end
    diagnostic_event!(
        diagnostics,
        "diagnostics_initialized";
        directory=selected_directory,
        interval_s=interval,
        mpi_size=context.size,
        julia_version=VERSION,
        julia_threads=Threads.nthreads(),
        command=join(Base.julia_cmd().exec, " ") * " " * join(ARGS, " "),
        slurm_job_id=get(ENV, "SLURM_JOB_ID", ""),
        slurm_node_id=get(ENV, "SLURM_NODEID", ""),
        slurm_local_id=get(ENV, "SLURM_LOCALID", ""),
        slurm_ntasks=get(ENV, "SLURM_NTASKS", ""),
        slurm_cpus_per_task=get(ENV, "SLURM_CPUS_PER_TASK", ""),
    )
    write_resource_snapshot!(diagnostics)

    if interval > 0
        diagnostics.monitor_task = @async begin
            try
                while !diagnostics.stop_requested[]
                    remaining = diagnostics.interval
                    while remaining > 0 && !diagnostics.stop_requested[]
                        duration = min(1.0, remaining)
                        sleep(duration)
                        remaining -= duration
                    end
                    diagnostics.stop_requested[] || write_resource_snapshot!(diagnostics)
                end
            catch exception
                # Monitoring must never terminate the science calculation. The
                # event log records its own failure for post-mortem diagnosis.
                try
                    diagnostic_event!(
                        diagnostics,
                        "resource_monitor_failure";
                        level="ERROR",
                        exception=sprint(showerror, exception, catch_backtrace()),
                    )
                catch
                end
            end
        end
    end
    return diagnostics
end


function stop_diagnostics!(diagnostics::ForwardDiagnostics)
    diagnostics.stop_requested[] = true
    diagnostics.monitor_task === nothing || wait(diagnostics.monitor_task)
    set_diagnostic_context!(diagnostics; phase="shutdown")
    write_resource_snapshot!(diagnostics)
    diagnostic_event!(diagnostics, "diagnostics_stopped")
    return nothing
end
