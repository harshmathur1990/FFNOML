using HDF5
include(joinpath(@__DIR__,"..","scripts","preflight_olivia_inversion.jl"))

@testset "Olivia submission preflight" begin
    mktempdir() do run_dir
        config_path=joinpath(run_dir,"forward.toml")
        factory_path=joinpath(run_dir,"model_factory.jl")
        atmosphere_path=joinpath(run_dir,"atmosphere.h5")
        line_list_path=joinpath(run_dir,"lines.list")
        batch_path=joinpath(@__DIR__,"..","scripts","run_olivia_inversion.sbatch")
        write(config_path,"""
        mode = "forward"
        [inputs]
        initial_atmosphere_file = "atmosphere.h5"
        [outputs]
        synthesis_file = "outputs/synthesis.h5"
        atmosphere_file = "outputs/atmosphere.h5"
        [atmosphere]
        pressure_top_pa = 0.1
        storage_order = "python"
        [atmosphere.datasets]
        logtau500 = "logtau_500"
        temperature = "temperature"
        vx = "vx"
        vy = "vy"
        vz = "vz"
        [grid]
        dx_m = 1.0
        dy_m = 1.0
        [[regions]]
        start_angstrom = 6301.0
        step_angstrom = 0.1
        count = 3
        normalization = 1.0
        psf_type = "none"
        [[regions.sources]]
        mode = "ffno"
        species = "CA"
        line = "ca_ii_8542"
        [[regions.sources]]
        mode = "kurucz_lte"
        linelist_file = "lines.list"
        [physics]
        redistribution = "non_prd"
        [observation]
        stokes = ["I"]
        """)
        write(factory_path,"""
        using FFNOInversion
        FFNO_INVERSION_ASSETS = Dict("fixture line list"=>joinpath(ENV["FFNOML_RUN_DIR"],"lines.list"))
        FFNO_INVERSION_GENERATED_ASSETS = Dict(
            "generated fixture"=>joinpath(ENV["FFNOML_RUN_DIR"],"generated","missing.so"))
        FFNO_INVERSION_FACTORY = InversionModelFactory(Dict(:CA=>6), (args...)->nothing)
        (;factory=FFNO_INVERSION_FACTORY,assets=FFNO_INVERSION_ASSETS,
          generated_assets=FFNO_INVERSION_GENERATED_ASSETS)
        """)
        write(line_list_path,"test line list\n")

        @test_throws ArgumentError run_submission_preflight(
            config_path,factory_path,run_dir,batch_path;io=devnull)

        values=reshape(collect(1.0:24.0),2,3,4)
        python_stored_values=permutedims(values,(3,2,1))
        h5open(atmosphere_path,"w") do file
            file["logtau_500"]=[-1.0,0.0]
            for name in ("temperature","vx","vy","vz")
                file[name]=python_stored_values
            end
        end
        output=IOBuffer()
        config=run_submission_preflight(config_path,factory_path,run_dir,batch_path;io=output)
        report=String(take!(output))
        @test config.mode===:forward
        @test config.atmosphere.storage_order===:python
        @test occursin("Atmosphere storage order: PYTHON",report)
        @test occursin("shape=(2, 3, 4) (z,y,x)",report)
        @test occursin("Mode:           FORWARD",report)
        @test occursin("FFNO CA/ca_ii_8542",report)
        @test occursin("Kurucz LTE",report)
        @test occursin("fixture line list",report)
        @test occursin("Generated inside allocated job:",report)
        @test occursin("generated fixture",report)
        @test occursin("nodes=8 ranks=16 ranks/node=2 CPUs/rank=128 GPUs/node=4",report)
        @test occursin("Sanity check OK",report)
        @test isdir(joinpath(run_dir,"outputs"))

        observation_path=joinpath(run_dir,"observations.h5")
        h5open(observation_path,"w") do file
            file["intensity"]=ones(1,3,3,4)
            file["sigma"]=ones(1,3,3,4)
            file["wavelength_weights"]=ones(3,1)
            file["spatial_weights"]=ones(3,4)
        end
        inversion=replace(read(config_path,String),
            "mode = \"forward\""=>"mode = \"inversion\"",
            "initial_atmosphere_file = \"atmosphere.h5\""=>
                "initial_atmosphere_file = \"atmosphere.h5\"\nobservation_file = \"observations.h5\"")
        inversion *= """
        [observation.datasets]
        intensity = "intensity"
        sigma = "sigma"
        wavelength_weights = "wavelength_weights"
        spatial_weights = "spatial_weights"
        [inversion]
        [[inversion.controls]]
        variable = "temperature"
        log_tau_nodes = [-1.0, 0.0]
        lower = 3000.0
        upper = 10000.0
        scale = 1000.0
        [solver]
        max_iterations = 1
        """
        write(config_path,inversion)
        inversion_output=IOBuffer()
        inversion_config=run_submission_preflight(
            config_path,factory_path,run_dir,batch_path;io=inversion_output)
        @test inversion_config.mode===:inversion
        @test occursin("Mode:           INVERSION",String(take!(inversion_output)))
    end
end
