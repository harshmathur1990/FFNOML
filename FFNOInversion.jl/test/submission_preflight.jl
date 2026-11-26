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
        [equilibrium]
        backend = "HSEMHS"
        model = "column_hse"
        eos_table_file = "eos_table.h5"
        [atmosphere]
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
        start_angstrom = 8541.9
        step_angstrom = 0.1
        count = 3
        normalization = 1.0
        psf_type = "none"
        [[regions.sources]]
        mode = "ffno"
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
        ATOM_FILES = Dict(:CA=>joinpath(ENV["FFNOML_RUN_DIR"],"ca.yaml"))
        FFNO_INVERSION_FACTORY = InversionModelFactory(Dict(:CA=>6), (args...)->nothing)
        (;factory=FFNO_INVERSION_FACTORY,assets=FFNO_INVERSION_ASSETS,
          generated_assets=FFNO_INVERSION_GENERATED_ASSETS)
        """)
        write(joinpath(run_dir,"ca.yaml"),"""
        element: {symbol: Ca}
        atomic_levels:
          lev1: {energy: {value: 0.0, unit: cm^-1}, g: 2, stage: 2, label: a}
          lev2: {energy: {value: 13650.248, unit: cm^-1}, g: 4, stage: 2, label: b}
          lev3: {energy: {value: 13710.9, unit: cm^-1}, g: 6, stage: 2, label: c}
          lev4: {energy: {value: 25191.535, unit: cm^-1}, g: 2, stage: 2, label: d}
          lev5: {energy: {value: 25414.465, unit: cm^-1}, g: 4, stage: 2, label: e}
          ion: {energy: {value: 95751.87, unit: cm^-1}, g: 1, stage: 3, label: f}
        radiative_bound_bound:
          - {transition: [lev5, lev1], f_value: 0.6}
          - {transition: [lev5, lev3], f_value: 0.06}
        """)
        write(line_list_path,"test line list\n")
        write(joinpath(run_dir,"eos_table.h5"),"placeholder HSEMHS table (preflight checks existence only)\n")

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
        @test occursin("FFNO CA levels 3-5 8542.09 Å",report)
        @test occursin("Kurucz LTE",report)
        @test occursin("fixture line list",report)
        @test occursin("Generated inside allocated job:",report)
        @test occursin("generated fixture",report)
        @test occursin("nodes=8 ranks=16 ranks/node=2 CPUs/rank=128 GPUs/node=4",report)
        @test occursin("Sanity check OK",report)
        @test isdir(joinpath(run_dir,"outputs"))
        # the configuration as the run will use it, and the file itself
        @test occursin("  Code:           FFNOInversion ",report)
        @test occursin("EOS:            LTE (Wittmann)",report)
        @test occursin("EOS table:      "*joinpath(run_dir,"eos_table.h5")*" (0.0 MiB, modified ",report)
        @test occursin("mode=column_hse_ltau",report)
        @test occursin("  Population mesh:\n    adaptive: ",report)
        @test occursin("  Physics:        redistribution=non_prd stokes=I",report)
        @test occursin("  View:           mu=1.0",report)
        @test occursin("  Configuration file (verbatim):\n    | mode = \"forward\"\n",report)
        @test occursin("    | eos_table_file = \"eos_table.h5\"\n",report)
        @test findfirst("Configuration file (verbatim)",report)[1]<findfirst("Sanity check OK",report)[1]
        @test _eos_description("nlte")=="NLTE (HSEMHS eos.model = nlte: hydrogen ionization of Bifrost hion runs)"
        @test _short(collect(1.0:20.0))=="[1.0, 2.0, 3.0, … 19.0, 20.0] (20)"
        @test _short(nothing)=="none"

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
        inversion_report=String(take!(inversion_output))
        @test occursin("Mode:           INVERSION",inversion_report)
        @test occursin("  Controls:\n    temperature: nodes [-1.0, 0.0] (log10tau500)",inversion_report)
        @test occursin("  Solver:         ",inversion_report)
        @test occursin("  Observed data:  file=",inversion_report)
    end
end

@testset "Olivia submission keeps the preflight report in the run directory" begin
    Sys.which("bash")!==nothing && mktempdir() do root
        # the real submit helper beside a stand-in preflight runner, with a fake sbatch on PATH
        scripts=joinpath(root,"pkg","scripts"); mkpath(scripts)
        source=joinpath(@__DIR__,"..","scripts")
        for name in ("submit_olivia_inversion.sh","run_olivia_inversion.sbatch","preflight_olivia_inversion.jl",
                "initialize_olivia_environment.sh","build_wittmann_backend.jl","prepare_olivia_environment.jl")
            cp(joinpath(source,name),joinpath(scripts,name))
        end
        write(joinpath(scripts,"run_olivia_login_preflight.sh"),"#!/bin/bash\necho \"stand-in report for \$1\"\necho \"Sanity check OK\"\n")
        chmod(joinpath(scripts,"run_olivia_login_preflight.sh"),0o755); chmod(joinpath(scripts,"initialize_olivia_environment.sh"),0o755)
        bin=joinpath(root,"bin"); mkpath(bin)
        write(joinpath(bin,"sbatch"),"#!/bin/bash\necho \"Submitted batch job 4242\"\n"); chmod(joinpath(bin,"sbatch"),0o755)
        run_dir=joinpath(root,"run"); mkpath(run_dir)
        for name in ("inversion.toml","model_factory.jl","olivia_runtime_environment.sh")
            write(joinpath(run_dir,name),"# fixture\n")
        end
        output=withenv("PATH"=>bin*":"*ENV["PATH"],"FFNOML_RUN_DIR"=>run_dir) do
            read(`bash $(joinpath(scripts,"submit_olivia_inversion.sh")) --time=00:10:00`,String)
        end
        @test occursin("Submitted batch job 4242",output)
        logs=readdir(joinpath(run_dir,"preflight"))
        @test "preflight-4242.log" in logs
        @test count(name->occursin(r"^preflight-\d{8}-\d{6}\.log$",name),logs)==1
        saved=read(joinpath(run_dir,"preflight","preflight-4242.log"),String)
        @test occursin("stand-in report for "*joinpath(run_dir,"inversion.toml"),saved)
        @test occursin("Sanity check OK",saved)
        @test occursin("Additional sbatch arguments: --time=00:10:00",saved)
        @test occursin("Submitted batch job 4242",saved)

        # a failed preflight stops before sbatch and keeps its report
        write(joinpath(scripts,"run_olivia_login_preflight.sh"),"#!/bin/bash\necho \"Sanity check FAILED: fixture\" >&2\nexit 2\n")
        failed=withenv("PATH"=>bin*":"*ENV["PATH"],"FFNOML_RUN_DIR"=>run_dir) do
            sleep(1.1)                    # a new timestamp for the second report
            success(`bash $(joinpath(scripts,"submit_olivia_inversion.sh"))`)
        end
        @test !failed
        reports=filter(name->occursin(r"^preflight-\d{8}-\d{6}\.log$",name),readdir(joinpath(run_dir,"preflight")))
        @test length(reports)==2
        newest=read(joinpath(run_dir,"preflight",sort(reports)[end]),String)
        @test occursin("Sanity check FAILED: fixture",newest)
        @test !occursin("Submitted batch job",newest)
        @test !isfile(joinpath(run_dir,"preflight","preflight-.log"))
    end
end
