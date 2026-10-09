using MPI
using Serialization
using FFNOInversion

function run_layout(ranks,threads,path)
    command=`$(MPI.mpiexec()) -n $ranks $(Base.julia_cmd()) --project=$(dirname(@__DIR__)) --threads=$threads $(joinpath(dirname(@__DIR__),"test","mpi_phase4_worker.jl"))`
    output=read(addenv(command,"PHASE4_RESULT_PATH"=>path),String)
    result=open(deserialize,path)
    (ranks=ranks,threads=threads,seconds=result.forward_seconds,timings=result.timings,
        checksum=sum(result.spectrum),memory=result.memory["maximum_owned_bytes"])
end

mktempdir() do directory
    layouts=((1,4),(2,2),(4,1))
    results=[run_layout(ranks,threads,joinpath(directory,"$(ranks)x$(threads).bin"))
        for (ranks,threads) in layouts]
    reference=results[1].checksum
    all(isapprox(result.checksum,reference;rtol=1e-12,atol=1e-14) for result in results) ||
        error("rank/thread layouts produced different spectra")
    println("ranks,threads,cold_wall_s,instrumented_forward_s,force_s,populations_s,synthesis_s,observation_s,max_owned_bytes")
    for result in results
        timing=result.timings
        println(join((result.ranks,result.threads,result.seconds,timing.total_seconds,timing.force_balance_seconds,
            timing.populations_seconds,timing.synthesis_seconds,timing.observation_seconds,result.memory),","))
    end
    println("PHASE4_LOCAL_LAYOUT_BENCHMARK_OK layouts=$(length(results)) checksum=$reference")
end
