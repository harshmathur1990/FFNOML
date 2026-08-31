using FFNOInversion
using Sockets

context=initialize_parallel(options=ParallelOptions(enabled=true,
    threads_per_rank=Threads.nthreads()))
diagnostics_root=get(ENV,"OLIVIA_CASE_DIAGNOSTICS",pwd()); mkpath(diagnostics_root)
log_path=joinpath(diagnostics_root,"fsdp-service-rank-$(context.rank).log")

function record(event;details="")
    open(log_path,"a") do stream
        println(stream,"$(time()) rank=$(context.rank) pid=$(getpid()) host=$(gethostname()) event=$event $details")
        flush(stream)
    end
end

stop_watchdog=Threads.Atomic{Bool}(false)
watchdog=Threads.@spawn begin
    while !stop_watchdog[]
        record("watchdog_alive";details="thread=$(Threads.threadid())")
        sleep(5)
    end
end

backend=nothing; probe_failed=Ref(false)
try
    repository=dirname(@__DIR__)
    checkpoint=normpath(joinpath(repository,"..","training_FFNO3D_zscale_expand_lognlte",
        "3D_sim_train_H.pt"))
    metadata=PopulationMetadata(FFNO_INPUT_CHANNELS,Tuple("H level $index" for index in 1:6),
        "901dcd28a6ee651c12a26a60effdd28c7ea211b596a30b87654435e87803c755")
    spec=FSDPModelSpec(:H,checkpoint,metadata)
    record("fsdp_service_launch_enter")
    backend=launch_fsdp_population_models([spec],context;timeout_seconds=300,
        diagnostics_directory=diagnostics_root)
    record("fsdp_service_ready")

    nx,ny,nz=8,8,4
    grid=Grid3D(collect(range(-5.0,-1.0,length=nz)),collect(0.0:48e3:(nx-1)*48e3),
        collect(0.0:48e3:(ny-1)*48e3))
    root_atmosphere=if isroot(context)
        shape=(nz,nx,ny); zero3=zeros(shape)
        Atmosphere3D(grid,fill(5500.0,shape),copy(zero3),copy(zero3),copy(zero3),copy(zero3);
            rho=fill(1e-7,shape),ne=fill(1e17,shape),
            z=repeat(reshape(collect(range(-3e5,0.0,length=nz)),nz,1,1),1,nx,ny))
    else
        nothing
    end
    distributed=distribute_atmosphere(Float64,root_atmosphere,context)
    local_shape=size(distributed.local_atmosphere.temperature)
    populations=Dict(:H=>zeros(Float64,local_shape...,6))
    predict_distributed_populations!(populations,backend,distributed,context)
    local_valid=all(isfinite,populations[:H])&&all(>(0),populations[:H])
    allreduce_sum(local_valid ? 1 : 0,context)==context.size || error(
        "persistent FSDP service returned invalid populations")

    population_bar=Dict(:H=>fill(1e-20,local_shape...,6))
    atmosphere_bar=AtmosphereCotangent(distributed.local_atmosphere)
    predict_distributed_populations_vjp!(atmosphere_bar,backend,distributed,population_bar,context)
    local_gradient_valid=all(isfinite,atmosphere_bar.temperature)&&all(isfinite,atmosphere_bar.z)
    allreduce_sum(local_gradient_valid ? 1 : 0,context)==context.size || error(
        "persistent FSDP service returned invalid population VJP")
    gradient_norm=allreduce_sum(sum(abs,atmosphere_bar.temperature)+sum(abs,atmosphere_bar.z),context)
    gradient_norm>0 || error("persistent FSDP service returned a zero population VJP")
    calls=isroot(context) ? backend.models[:H].root_model.calls : 0
    calls=mpi_broadcast(isroot(context) ? calls : nothing,context)
    calls==2 || error("persistent FSDP service was not reused across predict and VJP")
    record("fsdp_service_science_complete";details="calls=$calls gradient_norm=$gradient_norm")
    isroot(context) && println("OLIVIA_FSDP_SERVICE_INTEGRATION_OK ranks=$(context.size) calls=$calls gradient_norm=$gradient_norm")
catch exception
    probe_failed[]=true
    record("fsdp_service_probe_exception";
        details="exception=$(repr(sprint(showerror,exception,catch_backtrace())))")
    rethrow()
finally
    if backend!==nothing
        try
            close_distributed_population_model!(backend,context)
            record("fsdp_service_closed")
        catch exception
            record("fsdp_service_close_exception";
                details="exception=$(repr(sprint(showerror,exception,catch_backtrace())))")
            probe_failed[]=true
            rethrow()
        end
    end
    stop_watchdog[]=true; wait(watchdog)
    if probe_failed[]
        record("finalize_skipped_after_exception")
    else
        barrier(context); record("finalizing"); finalize_parallel!(context)
    end
end
