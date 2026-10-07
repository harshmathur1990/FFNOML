using HDF5
using Sockets
using TOML

struct ForbiddenForwardGradient <: AbstractObjectiveGradient end
FFNOInversion.objective_gradient!(::ForbiddenForwardGradient,args...)=error("forward run attempted a gradient")

mutable struct StartingAtmosphereGradient <: AbstractObjectiveGradient
    temperature::Array{Float64,3}
    calls::Int
end
function FFNOInversion.objective_gradient!(backend::StartingAtmosphereGradient,problem,layout,parameters,context)
    evaluation=evaluate_objective!(problem,layout,parameters,context)
    @test problem.distributed.local_atmosphere.temperature==backend.temperature
    backend.calls+=1
    ObjectiveGradientEvaluation(evaluation,zeros(length(parameters)),1)
end


function fake_fsdp_population_backend(;levels=1,value=1.0f10,temperature_scaled=false)
    listener=listen(ip"127.0.0.1",0); port=Int(getsockname(listener)[2])
    server=@async begin
        peer=accept(listener)
        try
            while true
                fields=split(readline(peer)); operation=fields[1]
                if operation=="PREDICT"
                    nz,nx,ny,nlevels=parse.(Int,fields[3:6])
                    @test nlevels==levels
                    features=FFNOInversion._read_float32_array(peer,(6,nz,nx,ny))
                    FFNOInversion._read_float32_array(peer,(nz,nx,ny))
                    println(peer,"OK PREDICT $nz $nx $ny $nlevels")
                    base=temperature_scaled ? value.*(@view(features[1,:,:,:]))./5000f0 :
                        fill(value,nz,nx,ny)
                    predicted=repeat(reshape(base,nz,nx,ny,1),1,1,1,nlevels)
                    FFNOInversion._write_float32_array(peer,predicted); flush(peer)
                elseif operation=="VJP"
                    nz,nx,ny,nlevels=parse.(Int,fields[3:6])
                    FFNOInversion._read_float32_array(peer,(6,nz,nx,ny))
                    FFNOInversion._read_float32_array(peer,(nz,nx,ny))
                    FFNOInversion._read_float32_array(peer,(nz,nx,ny,nlevels))
                    println(peer,"OK VJP $nz $nx $ny $nlevels")
                    FFNOInversion._write_float32_array(peer,zeros(Float32,6,nz,nx,ny))
                    FFNOInversion._write_float32_array(peer,zeros(Float32,nz,nx,ny)); flush(peer)
                elseif operation=="SHUTDOWN"
                    println(peer,"BYE"); flush(peer); break
                else
                    error("unexpected fake FSDP operation: $operation")
                end
            end
        finally
            close(peer); close(listener)
        end
    end
    socket=connect(ip"127.0.0.1",port)
    metadata=PopulationMetadata(FFNO_INPUT_CHANNELS,Tuple("level $index" for index in 1:levels),
        "fake-fsdp-checkpoint")
    service=FSDPServiceClient(socket,nothing,nothing,"127.0.0.1",port,"test-token",2,
        Dict(:H=>metadata),ReentrantLock(),false)
    RootDistributedPopulationModel(FSDPFFNOModel(service,:H,metadata,0),levels),server
end

@testset "production inversion accepts only multi-GPU FSDP" begin
    context=serial_context()
    @test_throws ArgumentError FFNOInversion._require_production_population_backend(
        LocalDistributedPopulationModel(MockPopulationModel(),1),context)
    @test_throws ArgumentError FFNOInversion._require_production_population_backend(
        RootDistributedPopulationModel(MockPopulationModel(),1),context)
    composite=CompositeDistributedPopulationModel(Dict(
        :not_fsdp=>RootDistributedPopulationModel(MockPopulationModel(),1)))
    @test_throws ArgumentError FFNOInversion._require_production_population_backend(
        composite,context)
    backend,server=fake_fsdp_population_backend()
    @test FFNOInversion._require_production_population_backend(backend,context)===nothing
    close_distributed_population_model!(backend,context); wait(server)
end

@testset "Phase 6 two-input/two-output executable route" begin
    mktempdir() do directory
        atmosphere_path=joinpath(directory,"initial.h5")
        observation_path=joinpath(directory,"observation.h5")
        synthesis_path=joinpath(directory,"synthesis.h5")
        output_atmosphere_path=joinpath(directory,"atmosphere.h5")
        config_path=joinpath(directory,"run.toml")
        logtau=[-5.0,-3.0,-1.0]; nx=2; ny=2; shape=(3,nx,ny)
        temperature=reshape(collect(5100.0:100.0:6200.0),shape)
        zeros3=zeros(shape); grid=Grid3D(logtau,[0.0,40e3],[0.0,40e3])
        atmosphere=Atmosphere3D(grid,temperature,copy(zeros3),copy(zeros3),copy(zeros3),copy(zeros3))
        python_storage(value)=permutedims(value,reverse(1:ndims(value)))
        h5open(atmosphere_path,"w") do file
            file["logtau_500"]=logtau
            file["temperature"]=python_storage(FFNOInversion._with_time_zyx(temperature))
            file["vturb"]=python_storage(FFNOInversion._with_time_zyx(fill(800.0,shape)))
            file["vx"]=python_storage(FFNOInversion._with_time_zyx(zeros3))
            file["vy"]=python_storage(FFNOInversion._with_time_zyx(zeros3))
            file["vz"]=python_storage(FFNOInversion._with_time_zyx(zeros3))
        end
        wavelength=collect(range(656.24e-9,656.32e-9,length=4))
        context=serial_context(); distributed=distribute_atmosphere(Float64,atmosphere,context)
        force=ForceBalanceOptions(max_iterations=4,relative_tolerance=5.0,force_tolerance=5.0,
            height_tolerance_m=1e9,relaxation=0.5,pressure_sweeps=4)
        truth_backend,truth_server=fake_fsdp_population_backend(value=1.0f10,
            temperature_scaled=true)
        model=HybridForwardModel(truth_backend,
            NonPRD(),MockIntensitySynthesizer(),IdentityObservation(),IdealGasEOS(),
            ReferenceOpacity500(kappa_m2_kg=0.02),HE3DBoundaryState(fill(1e-10,nx,ny),fill(1.0,nx,ny),:top),
            force,CapabilityManifest())
        workspace=HybridForwardWorkspace(Float64,distributed,wavelength,StokesSet(:I),1)
        truth=forward!(workspace,model,distributed,context).spectrum
        close_distributed_population_model!(truth_backend,context); wait(truth_server)
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
storage_order = "python"
[atmosphere.datasets]
logtau500 = \"logtau_500\"
temperature = \"temperature\"
vturb = \"vturb\"
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
        @test all(inputs.atmosphere.vturb.==800)
        @test inputs.observation===nothing
        server_ref=Ref{Any}()
        factory=InversionModelFactory(1,(cfg,dist,ws,pressure,ctx)->begin
            backend,server=fake_fsdp_population_backend(value=1.0f10,
                temperature_scaled=true); server_ref[]=server
            HybridForwardModel(backend,NonPRD(),MockIntensitySynthesizer(),IdentityObservation(),
                IdealGasEOS(),ReferenceOpacity500(kappa_m2_kg=0.02),
                HE3DBoundaryState(fill(1e-10,size(pressure)),pressure,:top),force,CapabilityManifest())
        end)
        result=run_inversion_files!(config_path,factory;gradient_backend=ForbiddenForwardGradient())
        wait(server_ref[])
        @test isfile(synthesis_path) && isfile(output_atmosphere_path)
        @test result.objective===nothing && result.solver===nothing
        @test result.atmosphere.temperature==temperature
        @test result.synthesis.data≈truth.data
        @test result.atmosphere.pgas!==nothing && result.atmosphere.ne!==nothing
        h5open(synthesis_path) do file
            @test size(read(file["intensity"]))==(1,1,4,ny,nx)
        end
        h5open(output_atmosphere_path) do file
            @test size(read(file["temperature"]))==(1,3,ny,nx)
            @test haskey(file,"populations")
            @test !haskey(attributes(file),"solver_termination")
        end
        # Explicit mode needs no observation or controls; malformed ignored
        # optimizer/node settings must not affect forward-only synthesis.
        inversion_document=TOML.parsefile(config_path)
        inversion_document["solver"]["max_iterations"]=1
        open(config_path,"w") do io; TOML.print(io,inversion_document); end
        initial_gradient=StartingAtmosphereGradient(copy(temperature),0)
        inverted=run_inversion_files!(config_path,factory;gradient_backend=initial_gradient)
        wait(server_ref[])
        @test initial_gradient.calls==1
        @test inverted.atmosphere.temperature==temperature
        @test inverted.synthesis.data==result.synthesis.data
        document=TOML.parsefile(config_path)
        document["mode"]="forward"
        delete!(document["inputs"],"observation_file")
        delete!(document,"observation")
        document["inversion"]=Dict("controls"=>"ignored")
        document["solver"]=Dict("method"=>"ignored")
        open(config_path,"w") do io; TOML.print(io,document); end
        explicit=run_inversion_files!(config_path,factory;gradient_backend=ForbiddenForwardGradient())
        wait(server_ref[])
        @test explicit.synthesis.data==result.synthesis.data
        @test explicit.populations==result.populations
        @test explicit.atmosphere.temperature==temperature
        @test explicit.atmosphere.z==result.atmosphere.z
        h5open(synthesis_path) do file
            @test !haskey(file,"objective_total")
            @test read(attributes(file)["execution_mode"])=="forward"
        end
        # Zero iterations has exactly the same semantics without mode=forward.
        delete!(document,"mode")
        document["solver"]=Dict("max_iterations"=>0)
        open(config_path,"w") do io; TOML.print(io,document); end
        @test load_config(config_path).mode===:forward
    end
end
