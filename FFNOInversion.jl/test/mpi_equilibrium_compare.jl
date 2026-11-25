# Compares the outputs of mpi_equilibrium_worker.jl on 1, 2 and 3 ranks: julia --project test/mpi_equilibrium_compare.jl OUT
using Test, Serialization
out=ARGS[1]
r=Dict(n=>open(deserialize,"$(out)_n$(n).jls") for n in (1,2,3))
relmax(a,b)=maximum(abs.(a.-b))/max(maximum(abs,b),1e-300)
@testset "equilibrium and its adjoint on 1, 2, 3 ranks" begin
    for key in keys(r[1]), n in (2,3)
        for f in propertynames(r[1][key])
            e=relmax(getproperty(r[n][key],f),getproperty(r[1][key],f))
            e<1e-10 || @info "rank difference" key f n e
            @test e<1e-10
        end
    end
end
