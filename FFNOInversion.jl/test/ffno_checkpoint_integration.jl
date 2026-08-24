using Test
using PythonCall
using FFNOInversion

repo_root=normpath(joinpath(@__DIR__,"..",".."))

function real_atmosphere()
    grid=Grid3D([-5.,-4.,-3.,-2.],collect(0.:48000.:7*48000),collect(0.:48000.:7*48000))
    shape=(4,8,8); zero3=zeros(shape)
    Atmosphere3D(grid,fill(5500.,shape),copy(zero3),copy(zero3),copy(zero3),copy(zero3);
        rho=fill(1e-7,shape),ne=fill(1e17,shape),
        z=repeat(reshape([-3e5,-2e5,-1e5,0.],4,1,1),1,8,8))
end

@testset "real FFNO checkpoints" begin
    cases=(
        ("H","901dcd28a6ee651c12a26a60effdd28c7ea211b596a30b87654435e87803c755"),
        ("CA","7a9365d26543dcb9b94da29dc574ae1e18b879f0818909e9d579a7aa4af9b760"),
    )
    atmosphere=real_atmosphere()
    for (atom,hash) in cases
        checkpoint=joinpath(repo_root,"training_FFNO3D_zscale_expand_lognite","3D_sim_train_$(atom).pt")
        isfile(checkpoint) || error("missing integration checkpoint: $checkpoint")
        names=Tuple("$atom level $i" for i in 1:6)
        metadata=PopulationMetadata(FFNO_INPUT_CHANNELS,names,hash)
        kwargs=Dict{String,Any}("checkpoint_path"=>checkpoint,"factory_module"=>"ffno_model_factory",
            "factory_name"=>"create_ffno3d","level_names"=>collect(names),"device"=>"cpu")
        model=load_python_ffno_model("ffno_runtime","create_persistent_backend",metadata;
            python_path=repo_root,factory_kwargs=kwargs)
        out=zeros(Float32,4,8,8,6); predict_populations!(out,model,atmosphere); first=copy(out)
        predict_populations!(out,model,atmosphere)
        @test out==first
        @test all(isfinite,out) && all(out.>0)
        @test model.calls==2 && pyconvert(Int,model.backend.load_count)==1
    end
end
