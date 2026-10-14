include("helpers.jl")

function populated_test_atmosphere()
    a=test_atmosphere(); shape=size(a.temperature)
    a.pgas=fill(10.0,shape); a.rho=fill(1e-7,shape); a.ne=fill(1e17,shape)
    a.z=repeat(reshape([-3e5,-2e5,-1e5,0.0],4,1,1),1,2,3)
    a
end

@testset "Phase 2 population metadata and record/replay" begin
    a=populated_test_atmosphere(); features=population_features(a)
    @test size(features)==(6,4,2,3)
    @test FFNO_INPUT_CHANNELS==(:temperature,:vx,:vy,:vz,:log10_ne,:log10_rho)
    metadata=PopulationMetadata(FFNO_INPUT_CHANNELS,("H I 1","H I 2"),"recorded-sha256")
    populations=Array{Float32}(undef,4,2,3,2)
    @views populations[:,:,:,1].=Float32.(a.temperature)
    @views populations[:,:,:,2].=2f0.*Float32.(a.temperature)
    model=RecordedPopulationModel(features,Float32.(a.z),populations,metadata)
    out=zeros(Float64,size(populations)); predict_populations!(out,model,a)
    @test out≈populations
    mktemp() do path,io
        close(io); save_population_record(path,model)
        replay=load_population_record(path); fill!(out,0); predict_populations!(out,replay,a)
        @test out≈populations
    end
    changed=populated_test_atmosphere(); changed.temperature[1,1,1]+=1
    @test_throws ArgumentError predict_populations!(out,model,changed)
    missing=test_atmosphere(); @test_throws ArgumentError population_features(missing)
    @test_throws ArgumentError PopulationMetadata((:temperature,),("level",),"hash")
    @test_throws DimensionMismatch RecordedPopulationModel(features,Float32.(a.z),populations[:,:,:,1:1],metadata)
end
