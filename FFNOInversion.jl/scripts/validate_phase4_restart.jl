using MPI

worker=joinpath(dirname(@__DIR__),"test","mpi_phase4_restart_worker.jl")
function run_worker(ranks,mode,path)
    command=`$(MPI.mpiexec()) -n $ranks $(Base.julia_cmd()) --project=$(dirname(@__DIR__)) --threads=2 $worker`
    read(addenv(command,"PHASE4_RESTART_MODE"=>mode,"PHASE4_RESTART_PATH"=>path),String)
end

mktempdir() do directory
    first_path=joinpath(directory,"one_to_four.bin")
    print(run_worker(1,"write",first_path)); print(run_worker(4,"read",first_path))
    second_path=joinpath(directory,"four_to_one.bin")
    print(run_worker(4,"write",second_path)); print(run_worker(1,"read",second_path))
end
println("PHASE4_RESTART_TOPOLOGY_PARITY_OK")

