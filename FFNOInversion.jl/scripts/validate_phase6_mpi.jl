using MPI
using Serialization
using FFNOInversion

root=dirname(@__DIR__); worker=joinpath(root,"test","mpi_phase6_worker.jl")
directory=mktempdir(); checkpoint=joinpath(directory,"phase6.checkpoint")

function launch(ranks,result;iterations=8,restart=false,checkpoint_path="")
    command=`$(MPI.mpiexec()) -n $ranks $(Base.julia_cmd()) --project=$root --threads=2 $worker`
    environment=Dict("PHASE6_RESULT_PATH"=>result,"PHASE6_MAX_ITERATIONS"=>string(iterations),
        "PHASE6_RESTART"=>(restart ? "1" : "0"),"PHASE6_CHECKPOINT_PATH"=>checkpoint_path)
    run(addenv(command,environment)); open(deserialize,result)
end

one_rank=launch(1,joinpath(directory,"one-rank.bin"))
four_rank=launch(4,joinpath(directory,"four-rank.bin"))
isapprox(one_rank.parameters,four_rank.parameters;rtol=1e-12,atol=1e-10) ||
    error("Phase 6 parameters depend materially on MPI topology")
isapprox(one_rank.gradient,four_rank.gradient;rtol=1e-12,atol=1e-12) ||
    error("Phase 6 gradient depends materially on MPI topology")
isapprox(one_rank.objective,four_rank.objective;rtol=1e-13,atol=1e-13) ||
    error("Phase 6 objective depends on MPI topology")

function equivalent_history(a,b)
    length(a)==length(b) || return false
    for (x,y) in zip(a,b)
        structural=x.iteration==y.iteration && x.forward_evaluations==y.forward_evaluations &&
            x.line_search_trials==y.line_search_trials && x.rejected_trials==y.rejected_trials &&
            x.accepted==y.accepted
        numeric=isapprox(x.total,y.total;rtol=1e-10,atol=1e-12) &&
            isapprox(x.data,y.data;rtol=1e-10,atol=1e-12) &&
            isapprox(x.regularization,y.regularization;rtol=1e-10,atol=1e-12) &&
            isapprox(x.projected_gradient_norm,y.projected_gradient_norm;rtol=1e-10,atol=1e-12) &&
            isapprox(x.accepted_step,y.accepted_step;rtol=1e-10,atol=1e-12)
        structural && numeric || return false
    end
    true
end
if !equivalent_history(one_rank.history,four_rank.history)
    for (index,(a,b)) in enumerate(zip(one_rank.history,four_rank.history))
        println(stderr,"history[$index] one_rank=$a four_rank=$b")
    end
    error("Phase 6 history depends on MPI topology")
end

launch(1,joinpath(directory,"partial.bin");iterations=1,checkpoint_path=checkpoint)
restarted=launch(4,joinpath(directory,"restarted.bin");iterations=8,restart=true,checkpoint_path=checkpoint)
isapprox(restarted.parameters,one_rank.parameters;rtol=1e-12,atol=1e-10) ||
    error("Phase 6 cross-topology restart parameters differ materially")
isapprox(restarted.gradient,one_rank.gradient;rtol=1e-12,atol=1e-12) ||
    error("Phase 6 cross-topology restart gradient differs materially")
isapprox(restarted.objective,one_rank.objective;rtol=1e-13,atol=1e-13) ||
    error("Phase 6 cross-topology restart objective differs")
equivalent_history(restarted.history,one_rank.history) || error("Phase 6 cross-topology restart history differs")
restarted.forward_evaluations==one_rank.forward_evaluations ||
    error("Phase 6 cross-topology restart evaluation count differs")
println("PHASE6_MPI_TOPOLOGY_RESTART_OK one_rank=1 restarted_ranks=4 iterations=$(restarted.iteration)")
