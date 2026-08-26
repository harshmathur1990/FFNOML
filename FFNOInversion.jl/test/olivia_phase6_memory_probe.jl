using FFNOInversion
using Sockets

parse_positive(name,default)=begin
    value=parse(Int,get(ENV,name,string(default)))
    value>0 || error("$name must be positive")
    value
end

nx=parse_positive("PHASE6_MEMORY_NX",800)
ny=parse_positive("PHASE6_MEMORY_NY",800)
nz=parse_positive("PHASE6_MEMORY_NZ",64)
nlambda=parse_positive("PHASE6_MEMORY_NLAMBDA",134)
levels=parse_positive("PHASE6_MEMORY_LEVELS",10)
limit_gib=parse(Float64,get(ENV,"PHASE6_MEMORY_LIMIT_GIB","24"))
limit_gib>0 || error("PHASE6_MEMORY_LIMIT_GIB must be positive")

context=initialize_parallel(options=ParallelOptions(enabled=true,threads_per_rank=Threads.nthreads()))
diagnostics_root=get(ENV,"OLIVIA_CASE_DIAGNOSTICS",pwd()); mkpath(diagnostics_root)
log_path=joinpath(diagnostics_root,"phase6-memory-rank-$(context.rank).log")

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
    record("allocation_enter";details="global_shape=($nz,$nx,$ny) nlambda=$nlambda levels=$levels")
    logtau=collect(range(-7.0,1.0,length=nz)); x=collect(0.0:48e3:(nx-1)*48e3)
    y=collect(0.0:48e3:(ny-1)*48e3); global_grid=Grid3D(logtau,x,y)
    tile=local_tile(context,nx,ny)
    local_grid=Grid3D(logtau,copy(x[tile.xrange]),copy(y[tile.yrange]))
    shape=(nz,length(tile.xrange),length(tile.yrange)); zero3=zeros(Float64,shape)
    atmosphere=Atmosphere3D(local_grid,fill(5500.0,shape),copy(zero3),copy(zero3),copy(zero3),copy(zero3);
        pgas=fill(0.1,shape),rho=fill(1e-8,shape),ne=fill(1e16,shape),z=copy(zero3))
    distributed=DistributedAtmosphere(atmosphere,global_grid,tile)
    wavelength=collect(range(630.0e-9,630.3e-9,length=nlambda))
    workspace=HybridForwardWorkspace(Float64,distributed,wavelength,StokesSet(:I),levels)
    report=distributed_memory_report(distributed,workspace,context)
    GC.gc(); rss_bytes=Sys.maxrss()
    maximum_rss=allreduce_max(rss_bytes,context)
    limit_bytes=round(Int,limit_gib*1024^3)
    if isroot(context)
        maximum_owned=report["maximum_owned_bytes"]
        maximum_owned<=limit_bytes || error(
            "reported rank-owned bytes $maximum_owned exceed limit $limit_bytes")
        maximum_rss<=limit_bytes || error("rank peak RSS $maximum_rss exceeds limit $limit_bytes")
        open(joinpath(diagnostics_root,"phase6-memory-summary.toml"),"w") do io
            println(io,"global_nx = $nx\nglobal_ny = $ny\nnz = $nz\nnlambda = $nlambda\nlevels = $levels")
            println(io,"rank_count = $(context.size)\nmaximum_owned_bytes = $maximum_owned")
            println(io,"maximum_rss_bytes = $maximum_rss\nlimit_bytes = $limit_bytes")
        end
        println("OLIVIA_PHASE6_MEMORY_LAYOUT_OK ranks=$(context.size) global_shape=($nz,$nx,$ny) maximum_owned_bytes=$maximum_owned maximum_rss_bytes=$maximum_rss limit_gib=$limit_gib")
    end
    record("allocation_complete";details="rss_bytes=$rss_bytes")
    barrier(context)
finally
    stop_watchdog[]=true
    wait(watchdog)
    record("finalizing")
    finalize_parallel!(context)
end
