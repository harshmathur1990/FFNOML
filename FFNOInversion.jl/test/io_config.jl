@testset "configuration and checkpoints" begin
    path = joinpath(@__DIR__,"..","configs","example_intensity_nonprd.toml")
    cfg = load_config(path)
    @test cfg.stokes.components == (:I,)
    summary = dry_run_summary(cfg)
    @test occursin("force_balance=HE3D",summary)
    @test occursin("full_grid_psf=true",summary)
    @test occursin("zero_weight_exclusion=true",summary)
    @test cfg.atmosphere.logtau500_dataset == "logtau_500"
    @test cfg.atmosphere.temperature_dataset == "temperature"
    @test cfg.atmosphere.pressure_top == 0.1
    @test cfg.observed.file == "inputs/observations.h5"
    @test cfg.observed.intensity_dataset == "intensity"
    @test cfg.weights.wavelength_dataset == "wavelength_weights"
    @test cfg.outputs.synthesis_file == "outputs/synthesis.h5"
    @test cfg.outputs.atmosphere_file == "outputs/inverted_atmosphere.h5"
    @test cfg.synthesis.dx_m == 48000.0
    @test length(cfg.regions) == 2
    @test cfg.regions[1].count == 88
    @test cfg.regions[1].psf_type == :fpi
    @test cfg.regions[1].psf_file == "8542.nc"
    @test length(cfg.regions[1].sources) == 2
    @test cfg.regions[1].sources[1].mode == :ffno
    @test cfg.regions[1].sources[2].mode == :kurucz_lte
    @test length(cfg.synthesis.wavelength_m) == 134
    @test cfg.synthesis.wavelength_m[1] ≈ 8540.23102e-10
    @test cfg.synthesis.wavelength_m[89] ≈ 6301.24942e-10
    @test occursin("spectral_regions=2",summary)
    @test cfg.weights.spatial_dataset == "spatial_weights"
    @test cfg.regularization.vertical.types == (4,1,3,0,0,0,1)
    @test cfg.regularization.vertical.regularize == 1.0
    @test cfg.regularization.vertical.weights[4] == 0.1
    @test cfg.parallel.decomposition == :cartesian_2d
    @test cfg.parallel.gpu_launcher_rank == 0
    @test cfg.parallel.gpu_connect_timeout_seconds == 30.0
    @test cfg.parallel.gpu_status_timeout_seconds == 0.0
    @test cfg.parallel.gpu_diagnostic_interval_seconds == 30.0
    mktemp() do path, io
        write(io,"[grid]\nnz=4\nnx=2\nny=2\n[observation]\nnlambda=3\nstokes=[\"I\",\"Q\"]\n[physics]\nredistribution=\"non_prd\"\n")
        close(io)
        @test_throws ArgumentError load_config(path)
    end
    mktempdir() do dir
        checkpoint = joinpath(dir,"restart.bin")
        state = (iteration=3,parameters=[1.0,2.0],objective=0.25)
        checkpoint!(checkpoint,state)
        restored = restore_checkpoint(checkpoint;expected=CapabilityManifest())
        @test restored.state == state
        @test_throws ArgumentError restore_checkpoint(checkpoint;expected=CapabilityManifest(prd=true))
    end
end
