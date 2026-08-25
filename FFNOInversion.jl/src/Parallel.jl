"""Hybrid MPI/Julia-thread execution support.

MPI calls are made only by the thread that initialized the context.  Julia
threads are used inside rank-local kernels and never call MPI directly.
"""
Base.@kwdef struct ParallelOptions
    enabled::Bool = false
    decomposition::Symbol = :cartesian_2d
    threads_per_rank::Int = Threads.nthreads()
    gpu_launcher_rank::Int = 0
end

struct Tile2D
    global_nx::Int
    global_ny::Int
    process_grid::NTuple{2,Int}
    coordinates::NTuple{2,Int}
    xrange::UnitRange{Int}
    yrange::UnitRange{Int}
end

struct ParallelContext{C}
    enabled::Bool
    rank::Int
    size::Int
    root::Int
    comm::C
    owns_mpi::Bool
    initializing_thread::Int
    options::ParallelOptions
end

serial_context(;options=ParallelOptions()) = ParallelContext(false,0,1,options.gpu_launcher_rank,
    nothing,false,Threads.threadid(),options)
isroot(context::ParallelContext) = context.rank == context.root

function initialize_parallel(;options=ParallelOptions())
    options.threads_per_rank > 0 || throw(ArgumentError("threads_per_rank must be positive"))
    options.threads_per_rank == Threads.nthreads() || throw(ArgumentError(
        "configured threads_per_rank=$(options.threads_per_rank), but Julia started with $(Threads.nthreads()) threads"))
    options.decomposition == :cartesian_2d || throw(ArgumentError("only cartesian_2d decomposition is supported"))
    options.enabled || return serial_context(options=options)
    owns = !MPI.Initialized()
    owns && MPI.Init()
    comm = MPI.COMM_WORLD
    size = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    0 <= options.gpu_launcher_rank < size || throw(ArgumentError("gpu_launcher_rank is outside COMM_WORLD"))
    ParallelContext(true,rank,size,options.gpu_launcher_rank,comm,owns,Threads.threadid(),options)
end

function finalize_parallel!(context::ParallelContext)
    context.enabled && context.owns_mpi && !MPI.Finalized() && MPI.Finalize()
    nothing
end

function _assert_mpi_thread(context::ParallelContext)
    Threads.threadid() == context.initializing_thread || throw(ArgumentError(
        "MPI operation attempted from a Julia worker thread; MPI is restricted to the initializing thread"))
    nothing
end

barrier(context::ParallelContext) = context.enabled ? (_assert_mpi_thread(context); MPI.Barrier(context.comm); nothing) : nothing
mpi_broadcast(value,context::ParallelContext) = context.enabled ? (_assert_mpi_thread(context); MPI.bcast(value,context.comm;root=context.root)) : value
allreduce_sum(value,context::ParallelContext) = context.enabled ? (_assert_mpi_thread(context); MPI.Allreduce(value,MPI.SUM,context.comm)) : value
allreduce_max(value,context::ParallelContext) = context.enabled ? (_assert_mpi_thread(context); MPI.Allreduce(value,MPI.MAX,context.comm)) : value

"""Collect reproducibility metadata for the active MPI/thread topology.

Only rank 0 receives the returned dictionary.  The payload contains metadata,
not atmospheric arrays, and is therefore safe to gather as Julia objects.
"""
function parallel_provenance(context::ParallelContext,tile::Tile2D;
        configuration_hash::AbstractString="",model_hash::AbstractString="",
        source_revision::AbstractString="",capabilities=String[])
    local_entry=Dict{String,Any}(
        "rank"=>context.rank,
        "hostname"=>gethostname(),
        "threads"=>Threads.nthreads(),
        "coordinates"=>collect(tile.coordinates),
        "xrange"=>[first(tile.xrange),last(tile.xrange)],
        "yrange"=>[first(tile.yrange),last(tile.yrange)])
    ranks=context.enabled ? (_assert_mpi_thread(context); MPI.gather(local_entry,context.comm;root=context.root)) : [local_entry]
    isroot(context) || return nothing
    Dict{String,Any}(
        "schema_version"=>1,
        "julia_version"=>string(VERSION),
        "mpi_ranks"=>context.size,
        "threads_per_rank"=>context.options.threads_per_rank,
        "decomposition"=>string(context.options.decomposition),
        "process_grid"=>collect(tile.process_grid),
        "gpu_launcher_rank"=>context.root,
        "global_spatial_shape"=>[tile.global_nx,tile.global_ny],
        "configuration_hash"=>String(configuration_hash),
        "model_hash"=>String(model_hash),
        "source_revision"=>String(source_revision),
        "capabilities"=>String.(capabilities),
        "rank_layout"=>ranks)
end

"""Write a rank-0 provenance dictionary in stable, human-readable TOML."""
function write_parallel_provenance(path::AbstractString,provenance)
    provenance===nothing && return nothing
    open(path,"w") do io
        TOML.print(io,provenance;sorted=true)
    end
    path
end

function _axis_partition(n::Int,index::Int,parts::Int)
    n >= parts || throw(ArgumentError("cannot split axis of length $n over $parts non-empty tiles"))
    q,r = divrem(n,parts)
    first = index*q + min(index,r) + 1
    count = q + (index < r)
    first:first+count-1
end

"""Choose a near-square process grid that fits the spatial domain."""
function process_grid(nranks::Int,nx::Int,ny::Int)
    nranks > 0 || throw(ArgumentError("nranks must be positive"))
    nx*ny >= nranks || throw(ArgumentError("more MPI ranks than spatial pixels"))
    candidates = NTuple{2,Int}[]
    for px in 1:nranks
        nranks % px == 0 || continue
        py = nranks ÷ px
        px <= nx && py <= ny && push!(candidates,(px,py))
    end
    isempty(candidates) && throw(ArgumentError("no non-empty 2D decomposition for ($nx,$ny) over $nranks ranks"))
    argmin(g -> abs(log((nx/g[1])/(ny/g[2]))),candidates)
end

function tile_for_rank(nx::Int,ny::Int,rank::Int,nranks::Int;grid=process_grid(nranks,nx,ny))
    0 <= rank < nranks || throw(ArgumentError("rank is outside communicator"))
    prod(grid) == nranks || throw(ArgumentError("process-grid product must equal rank count"))
    cx = rank % grid[1]; cy = rank ÷ grid[1]
    Tile2D(nx,ny,grid,(cx,cy),_axis_partition(nx,cx,grid[1]),_axis_partition(ny,cy,grid[2]))
end
local_tile(context::ParallelContext,nx::Int,ny::Int) = tile_for_rank(nx,ny,context.rank,context.size)

"""Rank-owned array plus global shape and tile metadata; arrays exclude halos."""
struct DistributedField{T,N,A<:AbstractArray{T,N}}
    values::A
    global_shape::NTuple{N,Int}
    tile::Tile2D
    function DistributedField(values::A,global_shape::NTuple{N,Int},tile::Tile2D) where {T,N,A<:AbstractArray{T,N}}
        N >= 2 || throw(ArgumentError("distributed fields require x and y dimensions"))
        size(values,N-1) == length(tile.xrange) || throw(DimensionMismatch("local x extent differs from tile"))
        size(values,N) == length(tile.yrange) || throw(DimensionMismatch("local y extent differs from tile"))
        new{T,N,A}(values,global_shape,tile)
    end
end

function _tile_array(values::AbstractArray,tile::Tile2D)
    n = ndims(values)
    n >= 2 || throw(ArgumentError("tile arrays require x and y trailing dimensions"))
    inds = (ntuple(_ -> Colon(),n-2)...,tile.xrange,tile.yrange)
    copy(@view values[inds...])
end

"""Scatter a numeric array whose final two axes are x and y.

Only `context.root` supplies `root_values`; every rank receives one contiguous,
typed tile. Metadata may be broadcast separately, but atmospheric payloads are
never serialized as Julia objects.
"""
function distribute_field(::Type{T},root_values,global_shape::NTuple{N,Int},
                          context::ParallelContext;tag::Int=110) where {T<:Number,N}
    _assert_mpi_thread(context)
    tile=local_tile(context,global_shape[N-1],global_shape[N])
    local_shape=(global_shape[1:N-2]...,length(tile.xrange),length(tile.yrange))
    if !context.enabled
        root_values === nothing && throw(ArgumentError("serial distribution requires root_values"))
        return DistributedField(_tile_array(root_values,tile),global_shape,tile)
    end
    values=Array{T}(undef,local_shape)
    if isroot(context)
        root_values === nothing && throw(ArgumentError("MPI root must supply root_values"))
        size(root_values)==global_shape || throw(DimensionMismatch("root field shape differs from global shape"))
        for rank in 0:context.size-1
            target=tile_for_rank(global_shape[N-1],global_shape[N],rank,context.size;grid=tile.process_grid)
            packed=_tile_array(root_values,target)
            if rank==context.root
                values.=packed
            else
                MPI.Send(vec(packed),context.comm;dest=rank,tag=tag)
            end
        end
    else
        MPI.Recv!(vec(values),context.comm;source=context.root,tag=tag)
    end
    DistributedField(values,global_shape,tile)
end

"""Gather a typed distributed field. Only root receives a global array."""
function gather_field(field::DistributedField,context::ParallelContext;tag::Int=111)
    _assert_mpi_thread(context)
    context.enabled || return copy(field.values)
    if !isroot(context)
        MPI.Send(vec(field.values),context.comm;dest=context.root,tag=tag)
        return nothing
    end
    result=Array{eltype(field.values)}(undef,field.global_shape)
    n=ndims(result)
    for rank in 0:context.size-1
        tile=tile_for_rank(field.tile.global_nx,field.tile.global_ny,rank,context.size;grid=field.tile.process_grid)
        inds=(ntuple(_->Colon(),n-2)...,tile.xrange,tile.yrange)
        if rank==context.root
            @views result[inds...].=field.values
        else
            shape=(field.global_shape[1:n-2]...,length(tile.xrange),length(tile.yrange))
            packed=Array{eltype(field.values)}(undef,shape)
            MPI.Recv!(vec(packed),context.comm;source=rank,tag=tag)
            @views result[inds...].=packed
        end
    end
    result
end

_rank_at(tile::Tile2D,cx::Int,cy::Int) =
    0 <= cx < tile.process_grid[1] && 0 <= cy < tile.process_grid[2] ? cy*tile.process_grid[1]+cx : nothing

"""Return a halo-padded copy of a distributed field.

The final two axes are spatial. X boundaries are exchanged first, then Y
boundaries including the new X halos, which also transfers corner values.
Physical domain boundaries use edge replication.
"""
function exchange_halos(field::DistributedField,context::ParallelContext,width::Int)
    width >= 0 || throw(ArgumentError("halo width must be non-negative"))
    width==0 && return copy(field.values)
    lx,ly=size(field.values,ndims(field.values)-1),size(field.values,ndims(field.values))
    width<=min(lx,ly) || throw(ArgumentError("halo width exceeds local tile extent"))
    lead=size(field.values)[1:end-2]
    padded=Array{eltype(field.values)}(undef,(lead...,lx+2width,ly+2width))
    center=(ntuple(_->Colon(),length(lead))...,width+1:width+lx,width+1:width+ly)
    @views padded[center...].=field.values
    cx,cy=field.tile.coordinates
    left=_rank_at(field.tile,cx-1,cy); right=_rank_at(field.tile,cx+1,cy)
    prefix=ntuple(_->Colon(),length(lead))

    left_recv=Array{eltype(field.values)}(undef,(lead...,width,ly))
    right_recv=similar(left_recv)
    if context.enabled
        _assert_mpi_thread(context)
        send_right=copy(@view field.values[prefix...,lx-width+1:lx,:])
        send_left=copy(@view field.values[prefix...,1:width,:])
        left===nothing && (@views left_recv.=field.values[prefix...,1:1,:])
        right===nothing && (@views right_recv.=field.values[prefix...,lx:lx,:])
        MPI.Sendrecv!(send_right,left_recv,context.comm;dest=something(right,MPI.PROC_NULL),sendtag=201,
            source=something(left,MPI.PROC_NULL),recvtag=201)
        MPI.Sendrecv!(send_left,right_recv,context.comm;dest=something(left,MPI.PROC_NULL),sendtag=202,
            source=something(right,MPI.PROC_NULL),recvtag=202)
    else
        @views left_recv.=field.values[prefix...,1:1,:]
        @views right_recv.=field.values[prefix...,lx:lx,:]
    end
    @views padded[prefix...,1:width,width+1:width+ly].=left_recv
    @views padded[prefix...,width+lx+1:width+lx+width,width+1:width+ly].=right_recv

    down=_rank_at(field.tile,cx,cy-1); up=_rank_at(field.tile,cx,cy+1)
    lower_recv=Array{eltype(field.values)}(undef,(lead...,lx+2width,width))
    upper_recv=similar(lower_recv)
    if context.enabled
        send_up=copy(@view padded[prefix...,:,width+ly-width+1:width+ly])
        send_down=copy(@view padded[prefix...,:,width+1:width+width])
        down===nothing && (@views lower_recv.=padded[prefix...,:,width+1:width+1])
        up===nothing && (@views upper_recv.=padded[prefix...,:,width+ly:width+ly])
        MPI.Sendrecv!(send_up,lower_recv,context.comm;dest=something(up,MPI.PROC_NULL),sendtag=203,
            source=something(down,MPI.PROC_NULL),recvtag=203)
        MPI.Sendrecv!(send_down,upper_recv,context.comm;dest=something(down,MPI.PROC_NULL),sendtag=204,
            source=something(up,MPI.PROC_NULL),recvtag=204)
    else
        @views lower_recv.=padded[prefix...,:,width+1:width+1]
        @views upper_recv.=padded[prefix...,:,width+ly:width+ly]
    end
    @views padded[prefix...,:,1:width].=lower_recv
    @views padded[prefix...,:,width+ly+1:width+ly+width].=upper_recv
    padded
end

"""Copy one rank-local atmosphere. The returned object owns no global arrays."""
function local_atmosphere(atmosphere::Atmosphere3D,tile::Tile2D)
    g = atmosphere.grid
    grid = Grid3D(copy(g.log_tau500),copy(g.x[tile.xrange]),copy(g.y[tile.yrange]))
    cut(value) = value === nothing ? nothing : _tile_array(value,tile)
    B = atmosphere.magnetic_field
    local_B = B === nothing ? nothing : MagneticField3D(cut(B.Bx),cut(B.By),cut(B.Bz))
    Atmosphere3D(grid,cut(atmosphere.temperature),cut(atmosphere.vx),cut(atmosphere.vy),
        cut(atmosphere.vz),cut(atmosphere.vturb);magnetic_field=local_B,pgas=cut(atmosphere.pgas),
        rho=cut(atmosphere.rho),ne=cut(atmosphere.ne),z=cut(atmosphere.z))
end

abstract type AbstractGPUCoordinator end
struct RootGPUCoordinator{F} <: AbstractGPUCoordinator
    launcher::F
    launcher_rank::Int
end
RootGPUCoordinator(launcher;launcher_rank=0) = RootGPUCoordinator(launcher,launcher_rank)

"""Execute a GPU control-plane operation on exactly one MPI rank.

The launcher should manage a persistent worker. Its result remains on the
launcher rank; subsequent distributed staging/scatter is a separate operation.
"""
function launch_gpu!(coordinator::RootGPUCoordinator,context::ParallelContext,args...;kwargs...)
    coordinator.launcher_rank == context.root || throw(ArgumentError("GPU coordinator rank differs from ParallelContext root"))
    context.enabled || return coordinator.launcher(args...;kwargs...)
    _assert_mpi_thread(context)

    # Establish the non-MPI control path before the nested GPU operation. This
    # is the charge-branch pattern that avoids holding an outer MPI collective
    # open while Slurm/NCCL launches overlapping work.
    server=isroot(context) ? listen(ip"0.0.0.0",0) : nothing
    endpoint=mpi_broadcast(isroot(context) ?
        (get(ENV,"FFNO_GPU_CONTROL_HOST",string(getipaddr())),Int(getsockname(server)[2])) : nothing,context)
    peers=TCPSocket[]
    control=nothing
    if isroot(context)
        ranks=Set{Int}()
        for _ in 1:context.size-1
            socket=accept(server); rank=Int(read(socket,Int32))
            rank in ranks && error("duplicate GPU-control connection from MPI rank $rank")
            push!(ranks,rank); push!(peers,socket)
        end
        close(server)
    else
        control=nothing
        for attempt in 1:100
            try
                control=connect(endpoint[1],endpoint[2]); break
            catch exception
                attempt==100 && rethrow(exception)
                sleep(0.05)
            end
        end
        write(control,Int32(context.rank)); flush(control)
    end
    barrier(context)
    result=nothing; success=true; message=""
    if isroot(context)
        try
            result=coordinator.launcher(args...;kwargs...)
        catch exception
            success=false; message=sprint(showerror,exception,catch_backtrace())
        end
        for socket in peers
            serialize(socket,(success=success,message=message)); flush(socket); close(socket)
        end
    else
        status=try
            deserialize(control)
        catch exception
            close(control)
            error("lost rank-0 GPU-control connection: $(sprint(showerror,exception))")
        end
        close(control); success=status.success; message=status.message
    end
    barrier(context)
    success || error("rank-0 GPU launcher failed:\n$message")
    result
end
