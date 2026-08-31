using Test

@testset "inversion has exactly one production FFNO backend" begin
    package_root=normpath(joinpath(@__DIR__,".."))
    repository_root=normpath(joinpath(package_root,".."))
    project=read(joinpath(package_root,"Project.toml"),String)
    @test !occursin("PythonCall",project)
    @test !isfile(joinpath(package_root,"ext","FFNOInversionPythonCallExt.jl"))
    @test !isfile(joinpath(repository_root,"ffno_runtime.py"))

    population_source=read(joinpath(package_root,"src","PopulationModels.jl"),String)
    execution_source=read(joinpath(package_root,"src","Execution.jl"),String)
    fsdp_source=read(joinpath(repository_root,"ffno_fsdp_runtime.py"),String)
    service_source=read(joinpath(repository_root,"ffno_fsdp_service.py"),String)
    @test !occursin("PythonFFNOModel",population_source)
    @test !occursin("load_python_ffno_model",population_source)
    @test occursin("_require_production_population_backend",execution_source)
    @test occursin("FSDPFFNOModel",execution_source)
    @test occursin("ShardingStrategy.FULL_SHARD",fsdp_source)
    @test occursin("sync_module_states=True",fsdp_source)
    @test occursin("require_multi_gpu",fsdp_source)
    @test occursin("FULL_SHARD_H_SLAB",service_source)
    @test occursin("PROTOCOL_VERSION = 2",service_source)
end
