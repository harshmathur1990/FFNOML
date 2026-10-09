"""Description of one checkpoint hosted by the persistent FSDP service."""
struct FSDPModelSpec
    name::Symbol
    checkpoint_path::String
    factory_module::String
    factory_name::String
    metadata::PopulationMetadata
    factory_kwargs::Dict{String,Any}
end

function FSDPModelSpec(name::Symbol,checkpoint_path::AbstractString,
        metadata::PopulationMetadata;factory_module::AbstractString="ffno_model_factory",
        factory_name::AbstractString="create_ffno3d",factory_kwargs=Dict{String,Any}())
    isfile(checkpoint_path) || throw(ArgumentError("FSDP checkpoint does not exist: $checkpoint_path"))
    FSDPModelSpec(name,abspath(checkpoint_path),String(factory_module),String(factory_name),metadata,
        Dict{String,Any}(string(key)=>value for (key,value) in factory_kwargs))
end

mutable struct FSDPServiceClient
    socket::TCPSocket
    process::Any
    log_stream::Any
    host::String
    port::Int
    token::String
    world_size::Int
    models::Dict{Symbol,PopulationMetadata}
    lock::ReentrantLock
    closed::Bool
end

mutable struct FSDPFFNOModel <: AbstractPopulationModel
    service::FSDPServiceClient
    name::Symbol
    metadata::PopulationMetadata
    calls::Int
    function FSDPFFNOModel(service::FSDPServiceClient,name::Symbol,
            metadata::PopulationMetadata,calls::Integer=0)
        service.world_size>=2 || throw(ArgumentError(
            "FFNO inversion requires an FSDP service with at least two GPU ranks"))
        haskey(service.models,name) || throw(ArgumentError(
            "FSDP service does not host model $name"))
        new(service,name,metadata,Int(calls))
    end
end

function _write_float32_array(stream,value)
    array=Float32.(value)
    write(stream,reinterpret(UInt8,vec(array)))
end

function _read_float32_array(stream,shape)
    bytes=Vector{UInt8}(undef,4*prod(shape)); read!(stream,bytes)
    copy(reshape(reinterpret(Float32,bytes),shape))
end

function _service_response(stream,operation,shape)
    fields=split(readline(stream))
    length(fields)>=2 || error("truncated FSDP service response")
    fields[1]=="OK" || error("FSDP service returned: $(join(fields,' '))")
    fields[2]==operation || error("FSDP service response operation differs from request")
    returned=Tuple(parse.(Int,fields[3:end]))
    returned==shape || throw(DimensionMismatch(
        "FSDP service returned shape $returned; expected $shape"))
end

function predict_populations!(out::AbstractArray{T,4},model::FSDPFFNOModel,
        atmosphere::Atmosphere3D,cache=nothing) where T
    service=model.service; service.closed && error("FSDP service is closed")
    size(out,4)==length(model.metadata.level_names) || throw(DimensionMismatch(
        "population level count differs from FSDP metadata"))
    size(out)[1:3]==size(atmosphere.temperature) || throw(DimensionMismatch(
        "population shape differs from atmosphere"))
    features=population_features(atmosphere); z=Float32.(atmosphere.z)
    dx=_spacing(atmosphere.grid.x,:x); dy=_spacing(atmosphere.grid.y,:y)
    nz,nx,ny=size(z); levels=size(out,4)
    lock(service.lock) do
        println(service.socket,"PREDICT $(model.name) $nz $nx $ny $levels $dx $dy")
        _write_float32_array(service.socket,features)
        _write_float32_array(service.socket,z)
        flush(service.socket)
        _service_response(service.socket,"PREDICT",(nz,nx,ny,levels))
        predicted=_read_float32_array(service.socket,(nz,nx,ny,levels))
        all(isfinite,predicted)&&all(>(0),predicted) || error(
            "FSDP FFNO returned non-positive or non-finite populations")
        out.=T.(predicted)
    end
    model.calls+=1; out
end

function population_vjp!(feature_bar::AbstractArray{T,4},z_bar::AbstractArray{T,3},
        model::FSDPFFNOModel,atmosphere::Atmosphere3D,
        population_bar::AbstractArray{T,4}) where T
    service=model.service; service.closed && error("FSDP service is closed")
    size(feature_bar)==(6,size(atmosphere.temperature)...) || throw(DimensionMismatch(
        "FFNO feature cotangent shape differs from atmosphere"))
    size(z_bar)==size(atmosphere.temperature) || throw(DimensionMismatch(
        "FFNO z cotangent shape differs from atmosphere"))
    size(population_bar)[1:3]==size(atmosphere.temperature) || throw(DimensionMismatch(
        "FFNO population cotangent grid differs from atmosphere"))
    size(population_bar,4)==length(model.metadata.level_names) || throw(DimensionMismatch(
        "FFNO population cotangent level count differs from FSDP metadata"))
    features=population_features(atmosphere); z=Float32.(atmosphere.z)
    dx=_spacing(atmosphere.grid.x,:x); dy=_spacing(atmosphere.grid.y,:y)
    nz,nx,ny=size(z); levels=size(population_bar,4)
    lock(service.lock) do
        println(service.socket,"VJP $(model.name) $nz $nx $ny $levels $dx $dy")
        _write_float32_array(service.socket,features)
        _write_float32_array(service.socket,z)
        _write_float32_array(service.socket,population_bar)
        flush(service.socket)
        _service_response(service.socket,"VJP",(nz,nx,ny,levels))
        feature_bar.=T.(_read_float32_array(service.socket,(6,nz,nx,ny)))
        z_bar.=T.(_read_float32_array(service.socket,(nz,nx,ny)))
    end
    all(isfinite,feature_bar)&&all(isfinite,z_bar) || error(
        "FSDP FFNO VJP returned NaN or Inf")
    model.calls+=1; feature_bar,z_bar
end

function _available_service_port()
    server=listen(ip"0.0.0.0",0)
    port=Int(getsockname(server)[2]); close(server); port
end

function _write_fsdp_manifest(path,specs,host,port,token)
    document=Dict{String,Any}(
        "service"=>Dict("bind_host"=>"0.0.0.0","advertised_host"=>host,
            "port"=>port,"token"=>token),
        "models"=>[Dict("name"=>string(spec.name),
            "checkpoint_path"=>spec.checkpoint_path,
            "factory_module"=>spec.factory_module,"factory_name"=>spec.factory_name,
            "level_names"=>collect(spec.metadata.level_names),
            "factory_kwargs"=>spec.factory_kwargs) for spec in specs])
    open(path,"w") do stream; TOML.print(stream,document;sorted=true); end
    path
end

function _connect_fsdp_service(host,port,token,process,timeout_seconds)
    deadline=time()+timeout_seconds; last_error=nothing
    while time()<deadline
        if process!==nothing && !process_running(process)
            error("FSDP launcher exited before the service became ready")
        end
        try
            socket=connect(host,port)
            println(socket,"HELLO $token"); flush(socket)
            fields=split(readline(socket))
            length(fields)==5 && fields[1]=="READY" || error(
                "invalid FSDP service handshake: $(join(fields,' '))")
            parse(Int,fields[2])==2 || error("unsupported FSDP service protocol")
            world=parse(Int,fields[3]); world>=2 || error(
                "FSDP service started with fewer than two GPU processes")
            fields[5]=="FULL_SHARD_H_SLAB" || error(
                "FSDP service did not advertise the required FULL_SHARD H-slab backend")
            return socket,world
        catch exception
            last_error=exception; sleep(0.2)
        end
    end
    error("timed out after $timeout_seconds s connecting to FSDP service at $host:$port: " *
        sprint(showerror,last_error))
end

"""Launch one persistent torchrun/FSDP service from MPI rank 0.

The TOML manifest is startup metadata only. Every population and VJP request is
transferred in memory over the persistent socket; no per-evaluation HDF5 files
or Python process launches occur.
"""
function launch_fsdp_service(specs::AbstractVector{FSDPModelSpec};
        launcher::AbstractString=joinpath(dirname(@__DIR__),"scripts","ffno_fsdp_service.sh"),
        host::AbstractString=get(ENV,"FFNO_GPU_CONTROL_HOST",gethostname()),
        port::Int=let raw=get(ENV,"FFNO_FSDP_SERVICE_PORT",""); isempty(raw) ?
            _available_service_port() : parse(Int,raw) end,
        timeout_seconds::Real=180,
        diagnostics_directory::AbstractString=get(ENV,"FFNO_GPU_DIAGNOSTICS_DIR",pwd()))
    isempty(specs) && throw(ArgumentError("at least one FSDP model specification is required"))
    length(unique(spec.name for spec in specs))==length(specs) || throw(ArgumentError(
        "FSDP model names must be unique"))
    isfile(launcher)&&isexecutable(launcher) || throw(ArgumentError(
        "FSDP launcher is missing or not executable: $launcher"))
    timeout_seconds>0 || throw(ArgumentError("FSDP service timeout must be positive"))
    directory=abspath(diagnostics_directory); mkpath(directory)
    token="$(getpid())-$(time_ns())"
    manifest=joinpath(directory,"ffno-fsdp-service-$(getpid()).toml")
    _write_fsdp_manifest(manifest,specs,String(host),port,token)
    log_path=joinpath(directory,"ffno-fsdp-service-$(getpid()).log")
    log_stream=open(log_path,"w")
    process=run(pipeline(`$(abspath(launcher)) $manifest`,stdout=log_stream,stderr=log_stream);wait=false)
    socket,world=try
        _connect_fsdp_service(String(host),port,token,process,Float64(timeout_seconds))
    catch
        process_running(process)&&kill(process)
        close(log_stream)
        rethrow()
    end
    metadata=Dict(spec.name=>spec.metadata for spec in specs)
    FSDPServiceClient(socket,process,log_stream,String(host),port,token,world,metadata,
        ReentrantLock(),false)
end

function close_fsdp_service!(service::FSDPServiceClient;timeout_seconds::Real=60)
    service.closed && return nothing
    service.closed=true
    try
        println(service.socket,"SHUTDOWN"); flush(service.socket)
        strip(readline(service.socket))=="BYE" || error("FSDP service did not acknowledge shutdown")
    finally
        isopen(service.socket)&&close(service.socket)
    end
    deadline=time()+timeout_seconds
    while service.process!==nothing && process_running(service.process) && time()<deadline
        sleep(0.1)
    end
    if service.process!==nothing && process_running(service.process)
        kill(service.process); error("FSDP service did not stop within $timeout_seconds s")
    end
    service.log_stream===nothing || close(service.log_stream)
    nothing
end

"""Collectively start the rank-0-controlled FSDP service and return population backends."""
function launch_fsdp_population_models(specs::AbstractVector{FSDPModelSpec},context::ParallelContext;kwargs...)
    client=launch_gpu!(RootGPUCoordinator(() -> launch_fsdp_service(specs;kwargs...);
        launcher_rank=context.root),context)
    models=Dict{Symbol,Any}()
    for spec in specs
        root_model=isroot(context) ? FSDPFFNOModel(client,spec.name,spec.metadata,0) : nothing
        models[spec.name]=RootDistributedPopulationModel(root_model,length(spec.metadata.level_names))
    end
    CompositeDistributedPopulationModel(models)
end

close_distributed_population_model!(model,context)=nothing
function _close_fsdp_services_collectively!(services,context)
    count=mpi_broadcast(isroot(context) ? length(services) : nothing,context)
    for index in 1:count
        coordinator=RootGPUCoordinator(() -> begin
            close_fsdp_service!(services[index]); true
        end;launcher_rank=context.root)
        launch_gpu!(coordinator,context)
    end
    nothing
end

function close_distributed_population_model!(model::RootDistributedPopulationModel,context)
    services=isroot(context) && model.root_model isa FSDPFFNOModel ?
        FSDPServiceClient[model.root_model.service] : FSDPServiceClient[]
    _close_fsdp_services_collectively!(services,context)
end
function close_distributed_population_model!(model::CompositeDistributedPopulationModel,context)
    services=IdSet{FSDPServiceClient}()
    if isroot(context)
        for backend in values(model.models)
            root_model=backend isa RootDistributedPopulationModel ? backend.root_model : nothing
            root_model isa FSDPFFNOModel && push!(services,root_model.service)
        end
    end
    _close_fsdp_services_collectively!(collect(services),context)
end
