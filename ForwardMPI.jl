"""Optional MPI support for Forward.jl.

Serial execution does not require MPI.jl. Passing `--mpi` imports MPI.jl,
initializes `COMM_WORLD`, and partitions the horizontal x dimension across
ranks. Julia threads remain available inside every rank.
"""

const FORWARD_MPI_AVAILABLE = Base.find_package("MPI") !== nothing
if FORWARD_MPI_AVAILABLE
    @eval import MPI
end


struct ForwardParallelContext
    enabled::Bool
    rank::Int
    size::Int
    root::Int
    comm::Any
    owns_mpi::Bool
end


serial_parallel_context() = ForwardParallelContext(false, 0, 1, 0, nothing, false)


function initialize_parallel_context(enable_mpi::Bool)
    enable_mpi || return serial_parallel_context()
    FORWARD_MPI_AVAILABLE || error(
        "--mpi requires MPI.jl. Install it with: import Pkg; Pkg.add(\"MPI\")"
    )

    owns_mpi = !MPI.Initialized()
    owns_mpi && MPI.Init()
    comm = MPI.COMM_WORLD
    return ForwardParallelContext(
        true,
        MPI.Comm_rank(comm),
        MPI.Comm_size(comm),
        0,
        comm,
        owns_mpi,
    )
end


function finalize_parallel_context(context::ForwardParallelContext)
    if context.enabled && context.owns_mpi && !MPI.Finalized()
        MPI.Finalize()
    end
    return nothing
end


function abort_parallel_context(context::ForwardParallelContext, code::Int=1)
    context.enabled && !MPI.Finalized() && MPI.Abort(context.comm, code)
    return nothing
end


parallel_isroot(context::ForwardParallelContext) = context.rank == context.root


function parallel_println(context::ForwardParallelContext, values...)
    parallel_isroot(context) && println(values...)
    return nothing
end


function parallel_barrier(context::ForwardParallelContext)
    context.enabled && MPI.Barrier(context.comm)
    return nothing
end


function parallel_bcast(value, context::ForwardParallelContext)
    context.enabled || return value
    return MPI.bcast(value, context.comm; root=context.root)
end


function parallel_root_call(operation, context::ForwardParallelContext)
    context.enabled || return operation()
    value = nothing
    status = (success=true, message="")
    if parallel_isroot(context)
        try
            value = operation()
        catch exception
            status = (
                success=false,
                message=sprint(showerror, exception, catch_backtrace()),
            )
        end
    end
    status = parallel_bcast(status, context)
    status.success || error("MPI rank-0 operation failed:\n$(status.message)")
    return value
end


function parallel_allreduce_max(value::Real, context::ForwardParallelContext)
    context.enabled || return value
    return MPI.Allreduce(Float64(value), MPI.MAX, context.comm)
end


function parallel_allreduce_min(value::Real, context::ForwardParallelContext)
    context.enabled || return value
    return MPI.Allreduce(Float64(value), MPI.MIN, context.comm)
end


function parallel_partition(total::Int, rank::Int, nranks::Int)
    total >= nranks || error(
        "Cannot partition $(total) x columns over $(nranks) MPI ranks; " *
        "use at most one MPI rank per x column."
    )
    base, remainder = divrem(total, nranks)
    first = rank * base + min(rank, remainder) + 1
    count = base + (rank < remainder ? 1 : 0)
    return first:first + count - 1
end


parallel_local_xrange(total::Int, context::ForwardParallelContext) =
    parallel_partition(total, context.rank, context.size)


function parallel_gather_x(local_values, context::ForwardParallelContext; dimension::Int=3)
    context.enabled || return local_values

    if dimension == ndims(local_values)
        local_array = Array(local_values)
        x_counts = MPI.gather(size(local_array, dimension), context.comm; root=context.root)
        leading_count = prod(size(local_array)[1:end-1])
        global_values = parallel_isroot(context) ? Array{eltype(local_array)}(
            undef,
            size(local_array)[1:end-1]...,
            sum(x_counts),
        ) : nothing
        receive_buffer = parallel_isroot(context) ?
                         MPI.VBuffer(vec(global_values), leading_count .* x_counts) : nothing
        MPI.Gatherv!(vec(local_array), receive_buffer, context.comm; root=context.root)
        return global_values
    end

    # Corrected population arrays have levels after the x dimension and are
    # therefore not contiguous x slabs in Julia memory. They are gathered as
    # objects only once, for the final serial prediction-HDF5 write.
    chunks = MPI.gather(Array(local_values), context.comm; root=context.root)
    parallel_isroot(context) || return nothing
    return cat(chunks...; dims=dimension)
end


function parallel_scatter_objects(objects, context::ForwardParallelContext)
    context.enabled || return only(objects)
    return MPI.scatter(objects, context.comm; root=context.root)
end


function parallel_scatterv_vector(
    global_values,
    counts::AbstractVector{<:Integer},
    ::Type{T},
    context::ForwardParallelContext,
) where {T}
    context.enabled || return Vector{T}(global_values)
    local_values = Vector{T}(undef, counts[context.rank + 1])
    send_buffer = parallel_isroot(context) ?
                  MPI.VBuffer(vec(global_values), Int.(counts)) : nothing
    MPI.Scatterv!(send_buffer, local_values, context.comm; root=context.root)
    return local_values
end
