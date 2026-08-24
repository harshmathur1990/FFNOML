using Test
using PythonCall
using FFNOInversion

include("helpers.jl")

@testset "persistent PythonCall population backend (CPU fixture)" begin
    a=test_atmosphere(); shape=size(a.temperature)
    a.pgas=fill(10.0,shape); a.rho=fill(1e-7,shape); a.ne=fill(1e17,shape)
    a.z=repeat(reshape([-3e5,-2e5,-1e5,0.0],4,1,1),1,2,3)
    metadata=PopulationMetadata(FFNO_INPUT_CHANNELS,("level 1","level 2"),"fixture-checkpoint")
    model=load_python_ffno_model("python_ffno_fixture","create_backend",metadata;
        python_path=joinpath(@__DIR__,"fixtures"),factory_kwargs=Dict("scale"=>3.0))
    out=zeros(Float64,4,2,3,2)
    predict_populations!(out,model,a); first=copy(out)
    predict_populations!(out,model,a)
    module_object=pyimport("python_ffno_fixture")
    @test pyconvert(Int,module_object.factory_calls)==1
    @test model.calls==2 && out==first
    @test all(@view(out[:,:,:,2]).==3 .* a.temperature)
end
