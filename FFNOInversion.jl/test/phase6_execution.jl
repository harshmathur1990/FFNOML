using HDF5

@testset "Phase 6 two-input/two-output executable route" begin
    mktempdir() do directory
        atmosphere_path=joinpath(directory,"initial.h5")
        observation_path=joinpath(directory,"observation.h5")
        synthesis_path=joinpath(directory,"synthesis.h5")
        output_atmosphere_path=joinpath(directory,"atmosphere.h5")
        config_path=joinpath(directory,"run.toml")
        logtau=[-5.0,-3.0,-1.0]; nx=2; ny=2; shape=(3,nx,ny)
        temperature=fill(5500.0,shape)
        zeros3=zeros(shape); grid=Grid3D(logtau,[0.0,40e3],[0.0,40e3])
        atmosphere=Atmosphere3D(grid,temperature,copy(zeros3),copy(zeros3),copy(zeros3),copy(zeros3))
        h5open(atmosphere_path,"w") do file
            file["logtau_500"]=logtau
            file["temperature"]=FFNOInversion._with_time_zyx(temperature)
            file["vx"]=FFNOInversion._with_time_zyx(zeros3)
            file["vy"]=FFNOInversion._with_time_zyx(zeros3)
            file["vz"]=FFNOInversion._with_time_zyx(zeros3)
        end
        wavelength=collect(range(656.24e-9,656.32e-9,length=4))
        context=serial_context(); distributed=distribute_atmosphere(Float64,atmosphere,context)
        force=ForceBalanceOptions(max_iterations=4,relative_tolerance=5.0,force_tolerance=5.0,
            height_tolerance_m=1e9,relaxation=0.5,pressure_sweeps=4)
        model=HybridForwardModel(LocalDistributedPopulationModel(MockPopulationModel(1e10),1),
            NonPRD(),MockIntensitySynthesizer(),IdentityObservation(),IdealGasEOS(),
            ReferenceOpacity500(kappa_m2_kg=0.02),HE3DBoundaryState(fill(1e-10,nx,ny),fill(1.0,nx,ny),:top),
            force,CapabilityManifest())
        workspace=HybridForwardWorkspace(Float64,distributed,wavelength,StokesSet(:I),1)
        truth=forward!(workspace,model,distributed,context).spectrum
        h5open(observation_path,"w") do file
            file["intensity"]=FFNOInversion._with_time_slyx(truth.data)
            file["sigma"]=FFNOInversion._with_time_slyx(fill(1e8,size(truth.data)))
            file["wavelength_weights"]=ones(4,1)
            file["spatial_weights"]=ones(ny,nx)
        end
        open(config_path,"w") do io
            write(io,"""
[inputs]
observation_file = \"$observation_path\"
initial_atmosphere_file = \"$atmosphere_path\"
time_index = 1
[outputs]
synthesis_file = \"$synthesis_path\"
atmosphere_file = \"$output_atmosphere_path\"
[atmosphere]
pressure_top_pa = 1.0
[atmosphere.datasets]
logtau500 = \"logtau_500\"
temperature = \"temperature\"
vx = \"vx\"
vy = \"vy\"
vz = \"vz\"
[grid]
dx_m = 40000.0
dy_m = 40000.0
[[regions]]
start_angstrom = 6562.4
step_angstrom = 0.2666666666667
count = 4
normalization = 1.0
psf_type = \"none\"
[physics]
redistribution = \"non_prd\"
[observation]
stokes = [\"I\"]
[observation.datasets]
intensity = \"intensity\"
sigma = \"sigma\"
wavelength_weights = \"wavelength_weights\"
spatial_weights = \"spatial_weights\"
[regularization]
[regularization.vertical]
types = [0,0,0,0,0,0,0]
regularize = 0.0
weights = [1,1,1,1,1,1,1]
[inversion]
[[inversion.controls]]
variable = \"temperature\"
log_tau_nodes = [-5.0,-1.0]
control_nx = 1
control_ny = 1
lower = 3000.0
upper = 8000.0
scale = 1000.0
[solver]
method = \"lbfgs\"
max_iterations = 0
history_length = 3
checkpoint_path = \"\"
[parallel]
enabled = false
threads_per_rank = $(Threads.nthreads())
""")
        end
        config=load_config(config_path)
        inputs=read_inversion_inputs(config)
        @test inputs.atmosphere.temperature==temperature
        @test inputs.observation.spectrum.data==truth.data
        factory=InversionModelFactory(1,(cfg,dist,ws,pressure,ctx)->HybridForwardModel(
            LocalDistributedPopulationModel(MockPopulationModel(1e10),1),NonPRD(),
            MockIntensitySynthesizer(),IdentityObservation(),IdealGasEOS(),
            ReferenceOpacity500(kappa_m2_kg=0.02),
            HE3DBoundaryState(fill(1e-10,size(pressure)),pressure,:top),force,CapabilityManifest()))
        result=run_inversion_files!(config_path,factory)
        @test isfile(synthesis_path) && isfile(output_atmosphere_path)
        @test result.objective.components.total<1e-20
        h5open(synthesis_path) do file
            @test size(read(file["intensity"]))==(1,1,4,ny,nx)
        end
        h5open(output_atmosphere_path) do file
            @test size(read(file["temperature"]))==(1,3,ny,nx)
            @test haskey(file,"populations")
        end
    end
end
