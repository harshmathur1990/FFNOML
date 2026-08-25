using MPI
using Serialization
using FFNOInversion

function run_topology(ranks,threads,path)
    command=`$(MPI.mpiexec()) -n $ranks $(Base.julia_cmd()) --project=$(dirname(@__DIR__)) --threads=$threads $(joinpath(dirname(@__DIR__),"test","mpi_phase4_worker.jl"))`
    output=read(addenv(command,"PHASE4_RESULT_PATH"=>path),String)
    matched=match(r"MPI_PHASE4_OK ranks=(\d+) threads=(\d+) checksum=([^ ]+) reg=([^ ]+) seconds=([^\n]+)",output)
    matched===nothing && error("Phase 4 worker did not report success:\n$output")
    (output=output,result=open(deserialize,path),seconds=parse(Float64,matched.captures[5]))
end

mktempdir() do directory
    one=run_topology(1,2,joinpath(directory,"one.bin")); four=run_topology(4,2,joinpath(directory,"four.bin"))
    fields=(:spectrum,:populations,:temperature,:pgas,:rho,:ne,:z,:Bx,:By,:Bz)
    for field in fields
        a=getfield(one.result,field); b=getfield(four.result,field)
        size(a)==size(b) || error("$field shape differs between topologies")
        isapprox(a,b;rtol=1e-12,atol=1e-14) || error("$field values differ between topologies")
    end
    isapprox(one.result.regularization.total,four.result.regularization.total;rtol=1e-13,atol=0) ||
        error("one-rank/four-rank regularization differs")
    one.result.force_balance.mode==four.result.force_balance.mode==:MHS || error("force-balance modes differ")
    for result in (one.result,four.result)
        provenance=result.provenance; memory=result.memory
        length(provenance["rank_layout"])==result.ranks || error("provenance omitted an MPI rank")
        sort([entry["rank"] for entry in provenance["rank_layout"]])==collect(0:result.ranks-1) ||
            error("provenance rank identities are incomplete")
        memory["rank_count"]==result.ranks || error("memory audit rank count differs")
        length(memory["rank_owned"])==result.ranks || error("memory audit omitted a rank")
        if result.ranks>1
            all(prod(entry["tile_shape"])<11*7 for entry in memory["rank_owned"]) ||
                error("a rank owns the complete spatial domain")
        end
    end
    print(one.output); print(four.output)
    max_owned_bytes=four.result.memory["maximum_owned_bytes"]
    println("PHASE4_FULL_FIELD_PARITY_OK fields=$(length(fields)) nx=11 ny=7 max_owned_bytes=$max_owned_bytes")
end
