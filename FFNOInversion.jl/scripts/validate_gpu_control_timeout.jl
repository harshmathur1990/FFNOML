using MPI
using FFNOInversion

mktempdir() do directory
    command=`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=$(dirname(@__DIR__)) --threads=2 $(joinpath(dirname(@__DIR__),"test","mpi_gpu_status_timeout_worker.jl"))`
    output=read(addenv(command,"GPU_TIMEOUT_DIAGNOSTICS_DIR"=>directory),String)
    occursin("MPI_GPU_STATUS_TIMEOUT_RECOVERY_OK ranks=4",output) ||
        error("GPU status-timeout worker did not report recovery:\n$output")
    event_files=filter(name->startswith(name,"gpu-control-events-rank-"),readdir(directory))
    resource_files=filter(name->startswith(name,"gpu-control-resources-rank-"),readdir(directory))
    length(event_files)==4 || error("expected four per-rank event logs")
    length(resource_files)==4 || error("expected four per-rank resource logs")
    print(output)
    println("GPU_CONTROL_PERIODIC_DIAGNOSTICS_OK event_files=4 resource_files=4")
end
