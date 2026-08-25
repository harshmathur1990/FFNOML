"""Durable per-rank diagnostics for the MPI/rank-0 GPU control plane."""
mutable struct GPUControlDiagnostics
    directory::String
    event_path::String
    resource_path::String
    failure_path::String
    rank::Int
    host::String
    start_time::Float64
    interval::Float64
    phase::Base.RefValue{String}
    last_event::Base.RefValue{String}
    stop_requested::Threads.Atomic{Bool}
    monitor_task::Any
    lock::ReentrantLock
end

diagnostic_event!(::Nothing,event::AbstractString;kwargs...)=nothing
diagnostic_checkpoint!(::Nothing,event::AbstractString;kwargs...)=nothing
set_diagnostic_context!(::Nothing;kwargs...)=nothing
record_diagnostic_failure!(::Nothing,exception,backtrace)=nothing
stop_diagnostics!(::Nothing)=nothing

_diagnostic_timestamp()=Dates.format(now(UTC),dateformat"yyyy-mm-ddTHH:MM:SS.sssZ")

function _linux_status()
    result=Dict{String,String}()
    Sys.islinux() || return result
    for line in eachline("/proc/self/status")
        parts=split(line,':';limit=2); length(parts)==2 || continue
        result[strip(parts[1])]=strip(parts[2])
    end
    result
end

function _status_mib(status,key)
    fields=split(get(status,key,"")); isempty(fields) && return NaN
    try
        parse(Float64,fields[1])/1024
    catch
        NaN
    end
end

function _resource_snapshot(diagnostics::GPUControlDiagnostics)
    status=_linux_status(); rss=_status_mib(status,"VmRSS"); hwm=_status_mib(status,"VmHWM")
    isfinite(rss) || (rss=Sys.maxrss()/1024^2)
    isfinite(hwm) || (hwm=Sys.maxrss()/1024^2)
    (timestamp=_diagnostic_timestamp(),elapsed_s=time()-diagnostics.start_time,
        rank=diagnostics.rank,host=diagnostics.host,pid=getpid(),phase=diagnostics.phase[],
        last_event=diagnostics.last_event[],rss_mib=rss,hwm_mib=hwm,
        gc_live_mib=Base.gc_live_bytes()/1024^2,load1=Sys.loadavg()[1],
        julia_threads=Threads.nthreads(),thread=Threads.threadid(),
        slurm_job_id=get(ENV,"SLURM_JOB_ID",""),slurm_step_id=get(ENV,"SLURM_STEP_ID",""))
end

_csv(value)=value isa AbstractString ? replace(value,','=>';','\n'=>' ') : string(value)

function write_resource_snapshot!(diagnostics::GPUControlDiagnostics)
    lock(diagnostics.lock) do
        snapshot=_resource_snapshot(diagnostics)
        open(diagnostics.resource_path,"a") do io
            println(io,join(_csv.(values(snapshot)),',')); flush(io)
        end
        snapshot
    end
end

function diagnostic_event!(diagnostics::GPUControlDiagnostics,event::AbstractString;
        level::AbstractString="INFO",values...)
    diagnostics.last_event[]=String(event)
    lock(diagnostics.lock) do
        details=join(["$(key)=$(repr(value))" for (key,value) in pairs(values)]," ")
        open(diagnostics.event_path,"a") do io
            print(io,_diagnostic_timestamp()," level=",level," rank=",diagnostics.rank,
                " host=",diagnostics.host," pid=",getpid()," thread=",Threads.threadid(),
                " phase=",repr(diagnostics.phase[])," event=",repr(event))
            isempty(details) || print(io," ",details)
            println(io); flush(io)
        end
    end
    nothing
end

function diagnostic_checkpoint!(diagnostics::GPUControlDiagnostics,event::AbstractString;values...)
    diagnostic_event!(diagnostics,event;values...)
    write_resource_snapshot!(diagnostics)
    nothing
end

function set_diagnostic_context!(diagnostics::GPUControlDiagnostics;phase=nothing)
    phase===nothing || (diagnostics.phase[]=String(phase)); nothing
end

function record_diagnostic_failure!(diagnostics::GPUControlDiagnostics,exception,backtrace)
    snapshot=write_resource_snapshot!(diagnostics)
    lock(diagnostics.lock) do
        open(diagnostics.failure_path,"a") do io
            println(io,"="^80,"\ntimestamp: ",_diagnostic_timestamp(),"\nrank: ",diagnostics.rank,
                " host: ",diagnostics.host," pid: ",getpid(),"\nphase: ",diagnostics.phase[],
                "\nlast_event: ",diagnostics.last_event[],"\nresource_snapshot: ",snapshot,
                "\nexception:\n",sprint(showerror,exception,backtrace)); flush(io)
        end
    end
    diagnostic_event!(diagnostics,"gpu_control_failure";level="ERROR",exception=sprint(showerror,exception))
    nothing
end

function initialize_gpu_control_diagnostics(context::ParallelContext;
        directory::AbstractString,interval::Real=30.0)
    interval>=0 || throw(ArgumentError("diagnostic interval must be non-negative"))
    selected=mpi_broadcast(isroot(context) ? abspath(directory) : nothing,context)
    isroot(context) && mkpath(selected)
    barrier(context)
    label=lpad(context.rank,4,'0')
    events=joinpath(selected,"gpu-control-events-rank-$label.log")
    resources=joinpath(selected,"gpu-control-resources-rank-$label.csv")
    failures=joinpath(selected,"gpu-control-failure-rank-$label.log")
    if !isfile(resources) || filesize(resources)==0
        open(resources,"w") do io
            println(io,"timestamp,elapsed_s,rank,host,pid,phase,last_event,rss_mib,hwm_mib,gc_live_mib,load1,julia_threads,thread,slurm_job_id,slurm_step_id")
        end
    end
    diagnostics=GPUControlDiagnostics(selected,events,resources,failures,context.rank,
        gethostname(),time(),Float64(interval),Ref("setup"),Ref("initializing"),
        Threads.Atomic{Bool}(false),nothing,ReentrantLock())
    diagnostic_checkpoint!(diagnostics,"gpu_diagnostics_initialized";
        mpi_size=context.size,interval_s=interval,julia_version=VERSION,
        slurm_node_id=get(ENV,"SLURM_NODEID",""),slurm_local_id=get(ENV,"SLURM_LOCALID",""))
    if interval>0
        diagnostics.monitor_task=Threads.@spawn begin
            try
                while !diagnostics.stop_requested[]
                    remaining=diagnostics.interval
                    while remaining>0 && !diagnostics.stop_requested[]
                        duration=min(1.0,remaining); sleep(duration); remaining-=duration
                    end
                    diagnostics.stop_requested[] || write_resource_snapshot!(diagnostics)
                end
            catch exception
                try
                    diagnostic_event!(diagnostics,"gpu_resource_monitor_failure";
                        level="ERROR",exception=sprint(showerror,exception,catch_backtrace()))
                catch
                end
            end
        end
    end
    diagnostics
end

function stop_diagnostics!(diagnostics::GPUControlDiagnostics)
    diagnostics.stop_requested[]=true
    diagnostics.monitor_task===nothing || wait(diagnostics.monitor_task)
    set_diagnostic_context!(diagnostics;phase="stopped")
    diagnostic_checkpoint!(diagnostics,"gpu_diagnostics_stopped")
    nothing
end

diagnostic_location(::Nothing)=""
diagnostic_location(diagnostics::GPUControlDiagnostics)=
    " diagnostics=$(diagnostics.directory) rank=$(diagnostics.rank) phase=$(diagnostics.phase[]) last_event=$(diagnostics.last_event[])"
