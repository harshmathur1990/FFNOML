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
    if owns_mpi
        forward_startup_log("entering MPI.Init")
        MPI.Init()
        forward_startup_log("MPI.Init complete")
    end
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
    if parallel_isroot(context)
        println(values...)
        flush(stdout)
    end
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

    # Use a fixed-width collective for control flow. Broadcasting the
    # serialized status object immediately before another serialized broadcast
    # allowed a delayed rank to receive the following payload (for example the
    # atmosphere-shape tuple) as the status value. Allreduce provides an
    # unambiguous, same-type synchronization point on every rank.
    local_success = parallel_isroot(context) && !status.success ? Int32(0) : Int32(1)
    operation_succeeded = MPI.Allreduce(local_success, MPI.MIN, context.comm) == Int32(1)
    if !operation_succeeded
        message = parallel_bcast(
            parallel_isroot(context) ? status.message : "",
            context,
        )
        error("MPI rank-0 operation failed:\n$(message)")
    end
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


function parallel_gather_x(
    local_values,
    context::ForwardParallelContext;
    dimension::Int=3,
    diagnostics=nothing,
    label::AbstractString="x_values",
)
    context.enabled || return local_values
    1 <= dimension <= ndims(local_values) || throw(ArgumentError(
        "gather dimension $(dimension) is outside a $(ndims(local_values))-dimensional array"
    ))

    # MPI.gather(::Array) serializes Julia objects.  Apart from being expensive
    # for the population volumes, that path is not reliable across all Julia
    # and MPI.jl combinations (Julia 1.12 can fail while deserializing a type).
    # Move the distributed dimension to the end so each rank contributes one
    # contiguous, typed buffer to Gatherv!, then restore the original order.
    permutation = (
        (axis for axis in 1:ndims(local_values) if axis != dimension)...,
        dimension,
    )
    diagnostic_checkpoint!(
        diagnostics,
        "mpi_gather_x_local_copy_start";
        label=label,
        local_size=size(local_values),
        dimension=dimension,
    )
    local_copy_start = time()
    local_array = Array(PermutedDimsArray(local_values, permutation))
    diagnostic_checkpoint!(
        diagnostics,
        "mpi_gather_x_local_copy_complete";
        label=label,
        seconds=time() - local_copy_start,
        local_size=size(local_array),
        local_mib=sizeof(eltype(local_array)) * length(local_array) / 2.0^20,
    )

    counts_start = time()
    diagnostic_checkpoint!(
        diagnostics,
        "mpi_gather_x_counts_start";
        label=label,
        local_x_count=size(local_array, ndims(local_array)),
    )
    x_counts = MPI.gather(
        size(local_array, ndims(local_array)),
        context.comm;
        root=context.root,
    )
    diagnostic_checkpoint!(
        diagnostics,
        "mpi_gather_x_counts_complete";
        label=label,
        seconds=time() - counts_start,
        local_x_count=size(local_array, ndims(local_array)),
    )

    leading_count = prod(size(local_array)[1:end-1])
    global_permuted = if parallel_isroot(context)
        global_size = (size(local_array)[1:end-1]..., sum(x_counts))
        diagnostic_checkpoint!(
            diagnostics,
            "mpi_gather_x_root_allocation_start";
            label=label,
            global_size=global_size,
            global_mib=sizeof(eltype(local_array)) * prod(global_size) / 2.0^20,
        )
        allocation_start = time()
        values = Array{eltype(local_array)}(undef, global_size)
        diagnostic_checkpoint!(
            diagnostics,
            "mpi_gather_x_root_allocation_complete";
            label=label,
            seconds=time() - allocation_start,
            global_size=size(values),
        )
        values
    else
        nothing
    end
    receive_buffer = parallel_isroot(context) ?
                     MPI.VBuffer(vec(global_permuted), leading_count .* x_counts) : nothing

    payload_start = time()
    diagnostic_checkpoint!(
        diagnostics,
        "mpi_gather_x_payload_start";
        label=label,
        local_elements=length(local_array),
    )
    MPI.Gatherv!(vec(local_array), receive_buffer, context.comm; root=context.root)
    diagnostic_checkpoint!(
        diagnostics,
        "mpi_gather_x_payload_complete";
        label=label,
        seconds=time() - payload_start,
        local_elements=length(local_array),
    )
    parallel_isroot(context) || return nothing
    dimension == ndims(local_values) && return global_permuted

    restore_start = time()
    diagnostic_checkpoint!(
        diagnostics,
        "mpi_gather_x_restore_dimensions_start";
        label=label,
        permutation=permutation,
    )
    result = permutedims(global_permuted, invperm(collect(permutation)))
    diagnostic_checkpoint!(
        diagnostics,
        "mpi_gather_x_restore_dimensions_complete";
        label=label,
        seconds=time() - restore_start,
        global_size=size(result),
    )
    return result
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
