using MPI
using Serialization
using FFNOInversion

root=dirname(@__DIR__); worker=joinpath(root,"test","mpi_phase5_worker.jl")
directory=mktempdir(); checkpoint=joinpath(directory,"phase5.checkpoint")

function launch(ranks,result;iterations=8,restart=false,checkpoint_path="")
    command=`$(MPI.mpiexec()) -n $ranks $(Base.julia_cmd()) --project=$root --threads=2 $worker`
    environment=Dict("PHASE5_RESULT_PATH"=>result,"PHASE5_MAX_ITERATIONS"=>string(iterations),
        "PHASE5_RESTART"=>(restart ? "1" : "0"),"PHASE5_CHECKPOINT_PATH"=>checkpoint_path)
    run(addenv(command,environment))
    open(deserialize,result)
end

one_rank=launch(1,joinpath(directory,"one-rank.bin"))
four_rank=launch(4,joinpath(directory,"four-rank.bin"))
one_rank.parameters==four_rank.parameters || error("Phase 5 parameters depend on MPI topology")
isapprox(one_rank.objective,four_rank.objective;rtol=1e-13,atol=1e-13) ||
    error("Phase 5 objective depends on MPI topology")

function equivalent_history(a,b)
    length(a)==length(b) || return false
    all(zip(a,b)) do (x,y)
        x.iteration==y.iteration && x.evaluations==y.evaluations && x.accepted==y.accepted &&
        x.accepted_coordinate==y.accepted_coordinate && x.accepted_direction==y.accepted_direction &&
        isapprox(x.total,y.total;rtol=1e-13,atol=1e-13) &&
        isapprox(x.data,y.data;rtol=1e-13,atol=1e-13) &&
        isapprox(x.regularization,y.regularization;rtol=1e-13,atol=1e-13)
    end
end
equivalent_history(one_rank.history,four_rank.history) || error("Phase 5 history depends on MPI topology")

launch(1,joinpath(directory,"partial.bin");iterations=3,checkpoint_path=checkpoint)
restarted=launch(4,joinpath(directory,"restarted.bin");iterations=8,restart=true,checkpoint_path=checkpoint)
restarted.parameters==one_rank.parameters || error("cross-topology restart parameters differ")
isapprox(restarted.objective,one_rank.objective;rtol=1e-13,atol=1e-13) ||
    error("cross-topology restart objective differs")
equivalent_history(restarted.history,one_rank.history) || error("cross-topology restart history differs")
restarted.evaluations==one_rank.evaluations || error("cross-topology restart evaluation count differs")
println("PHASE5_MPI_TOPOLOGY_RESTART_OK one_rank=1 restarted_ranks=4 iterations=$(restarted.iteration)")
