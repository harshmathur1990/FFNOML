# [[inversion.cycles]]: the cycles run in order, each a fresh L-BFGS start with its own control layout (variables, nodes,
# map size) built from the atmosphere the previous cycle ended with, and its own checkpoint (<checkpoint>.cycle<k>).
# Resuming replays the finished cycles from their checkpoints and resumes the first unfinished one; --finish writes the
# outputs of the last cycle with a checkpoint. Uses fake_fsdp_population_backend and ForbiddenForwardGradient
# (phase6_execution.jl) and _captured_stdout (soft_stop_resume.jl).

@testset "Inversion cycles: configuration" begin
    controls=[ControlMapConfig(:temperature,[-5.0,-1.0],:log10tau500,1,1,3000.0,8000.0,1000.0),
        ControlMapConfig(:vz,[-5.0,-1.0],:log10tau500,2,2,-5000.0,5000.0,1000.0)]
    parse(cycles)=FFNOInversion._cycles(Dict{String,Any}("cycles"=>cycles),controls,:log10tau500)
    cycles=parse([Dict{String,Any}("max_iterations"=>2,"variables"=>["temperature"]),
        Dict{String,Any}("max_iterations"=>3,"log_tau_nodes"=>Dict{String,Any}("temperature"=>[-5.0,-3.0,-1.0]),
            "control_nx"=>4,"control_ny"=>3)])
    @test length(cycles)==2 && cycles[1].max_iterations==2 && cycles[2].max_iterations==3
    first=cycle_controls(controls,cycles[1])
    @test getfield.(first,:variable)==[:temperature] && first[1].nodes==[-5.0,-1.0] && (first[1].control_nx,first[1].control_ny)==(1,1)
    second=cycle_controls(controls,cycles[2])
    @test getfield.(second,:variable)==[:temperature,:vz]
    @test second[1].nodes==[-5.0,-3.0,-1.0] && second[2].nodes==[-5.0,-1.0]
    @test all(c->(c.control_nx,c.control_ny)==(4,3),second)
    @test second[2].lower==-5000.0 && second[2].scale==1000.0                  # bounds and scales are the controls' own
    @test isempty(parse(Any[]))
    @test_throws ArgumentError parse([Dict{String,Any}("variables"=>["temperature"])])                    # no max_iterations
    @test_throws ArgumentError parse([Dict{String,Any}("max_iterations"=>0)])
    @test_throws ArgumentError parse([Dict{String,Any}("max_iterations"=>2,"variables"=>["vturb"])])     # not a control
    @test_throws ArgumentError parse([Dict{String,Any}("max_iterations"=>2,"variables"=>["vz"],
        "log_tau_nodes"=>Dict{String,Any}("temperature"=>[-5.0,-1.0]))])                                  # not in the cycle
    @test_throws ArgumentError parse([Dict{String,Any}("max_iterations"=>2,"log_tau_nodes"=>Dict{String,Any}("vz"=>[-1.0,-1.0]))])
    @test_throws ArgumentError parse([Dict{String,Any}("max_iterations"=>2,"z_nodes_m"=>Dict{String,Any}("vz"=>[0.0,1.0e5]))])
    @test_throws ArgumentError parse([Dict{String,Any}("max_iterations"=>2,"nodes"=>Dict{String,Any}())])
end

@testset "Inversion cycles: run, replay, resume and finish" begin
    mktempdir() do directory
        atmosphere_path=joinpath(directory,"initial.h5"); observation_path=joinpath(directory,"observation.h5")
        synthesis_path=joinpath(directory,"synthesis.h5"); output_atmosphere_path=joinpath(directory,"atmosphere.h5")
        checkpoint=joinpath(directory,"run.checkpoint"); config_path=joinpath(directory,"run.toml")
        logtau=[-5.0,-3.0,-1.0]; nx=2; ny=2; shape=(3,nx,ny)
        temperature=reshape(collect(5100.0:100.0:6200.0),shape); zeros3=zeros(shape)
        python_storage(value)=permutedims(value,reverse(1:ndims(value)))
        h5open(atmosphere_path,"w") do file
            file["logtau_500"]=logtau
            file["temperature"]=python_storage(FFNOInversion._with_time_zyx(temperature))
            file["vturb"]=python_storage(FFNOInversion._with_time_zyx(fill(800.0,shape)))
            for name in ("vx","vy","vz"); file[name]=python_storage(FFNOInversion._with_time_zyx(zeros3)); end
        end
        h5open(observation_path,"w") do file
            file["intensity"]=FFNOInversion._with_time_slyx(fill(2.0e10,4,1,nx,ny))
            file["sigma"]=FFNOInversion._with_time_slyx(fill(1e8,4,1,nx,ny))
            file["wavelength_weights"]=ones(4,1); file["spatial_weights"]=ones(ny,nx)
        end
        write_config(;second=3,solver="")=open(config_path,"w") do io
            write(io,"""
[inputs]
observation_file = "$observation_path"
initial_atmosphere_file = "$atmosphere_path"
time_index = 1
[outputs]
synthesis_file = "$synthesis_path"
atmosphere_file = "$output_atmosphere_path"
[equilibrium]
backend = "HSEMHS"
model = "column_hse"
eos_table_file = "eos_table.h5"
[atmosphere]
storage_order = "python"
[atmosphere.datasets]
logtau500 = "logtau_500"
temperature = "temperature"
vturb = "vturb"
vx = "vx"
vy = "vy"
vz = "vz"
[grid]
dx_m = 40000.0
dy_m = 40000.0
[[regions]]
start_angstrom = 6562.4
step_angstrom = 0.2666666666667
count = 4
normalization = 1.0
psf_type = "none"
[physics]
redistribution = "non_prd"
[observation]
stokes = ["I"]
[observation.datasets]
intensity = "intensity"
sigma = "sigma"
wavelength_weights = "wavelength_weights"
spatial_weights = "spatial_weights"
[regularization]
[regularization.vertical]
types = [0,0,0,0,0,0,0]
regularize = 0.0
weights = [1,1,1,1,1,1,1]
[inversion]
[[inversion.controls]]
variable = "temperature"
log_tau_nodes = [-5.0,-1.0]
control_nx = 1
control_ny = 1
lower = 3000.0
upper = 8000.0
scale = 1000.0
[[inversion.controls]]
variable = "vturb"
log_tau_nodes = [-5.0,-1.0]
control_nx = 1
control_ny = 1
lower = 0.0
upper = 5000.0
scale = 1000.0
# the first cycle: temperature only, two nodes, one map
[[inversion.cycles]]
max_iterations = 2
variables = ["temperature"]
# the second: three temperature nodes on 2 x 2 maps, and vturb
[[inversion.cycles]]
max_iterations = $second
control_nx = 2
control_ny = 2
[inversion.cycles.log_tau_nodes]
temperature = [-5.0,-3.0,-1.0]
[solver]
method = "lbfgs"
max_iterations = 50
history_length = 3
gradient_tolerance = 1.0e-30
objective_tolerance = 0.0
checkpoint_path = "$checkpoint"
$solver
[parallel]
enabled = false
threads_per_rank = $(Threads.nthreads())
""")
        end
        server_ref=Ref{Any}()
        factory=InversionModelFactory(1,(cfg,dist,ws,ctx)->begin
            backend,server=fake_fsdp_population_backend(value=1.0f10,temperature_scaled=true); server_ref[]=server
            HybridForwardModel(backend,NonPRD(),MockIntensitySynthesizer(),IdentityObservation(),
                TestHydrostaticEquilibrium(),CapabilityManifest())
        end)
        gradient=FiniteDifferenceObjectiveGradient(step=1e-3)
        run(;kwargs...)=_captured_stdout(()->(result=run_inversion_files!(config_path,factory;kwargs...); wait(server_ref[]); result))
        state(path)=restore_checkpoint(path).state.state
        totals(log)=[parse(Float64,m.captures[1]) for m in eachmatch(r"LBFGS iteration=0 [^\n]*? total=(\S+) ",log)]

        write_config()
        full,log=run(gradient_backend=gradient)
        @test occursin(r"^Cycle 1 of 2: temperature nodes=\[-5\.0, -1\.0\] maps=1x1 max_iterations=2$"m,log)
        @test occursin(r"^Cycle 2 of 2: temperature nodes=\[-5\.0, -3\.0, -1\.0\] maps=2x2; vturb nodes=\[-5\.0, -1\.0\] maps=2x2 max_iterations=3$"m,log)
        @test isfile(checkpoint*".cycle1") && isfile(checkpoint*".cycle2") && !isfile(checkpoint)
        first=state(checkpoint*".cycle1"); second=state(checkpoint*".cycle2")
        @test first.iteration==2 && length(first.parameters)==2
        @test second.iteration==3 && length(second.parameters)==3*4+2*4
        @test full.solver.state.parameters==second.parameters
        # the second cycle starts from the first one's last accepted atmosphere and improves on it
        zero=totals(log)
        @test length(zero)==2
        @test zero[2]≈first.objective rtol=1e-12
        @test second.objective<first.objective<zero[1]
        @test full.objective.components.total≈second.objective rtol=1e-12
        @test isfile(synthesis_path) && isfile(output_atmosphere_path)
        # 2 x 2 temperature maps: the second cycle may change each column, the first could not
        @test length(unique(round.(second.parameters[1:12];digits=8)))>3

        # resume: both cycles are finished, so they are only restored (no gradient), and the outputs are the same
        write_config(solver="resume = \"if_present\"")
        replayed,replay_log=run(gradient_backend=ForbiddenForwardGradient())
        @test replayed.synthesis.data==full.synthesis.data
        @test replayed.solver.state.parameters==second.parameters
        @test occursin("Cycle 1 of 2: ",replay_log) && occursin("(finished checkpoint)",replay_log)

        # an unfinished second cycle (1 of 3 iterations) is resumed and ends as the uninterrupted run
        rm(checkpoint*".cycle1"); rm(checkpoint*".cycle2")
        write_config(second=1); run(gradient_backend=gradient)
        @test state(checkpoint*".cycle2").iteration==1
        write_config(solver="resume = \"if_present\"")
        resumed,resume_log=run(gradient_backend=gradient)
        @test occursin("Cycle 2 of 2: ",resume_log) && occursin("(unfinished checkpoint)",resume_log)
        @test resumed.solver.state.iteration==3
        @test resumed.solver.state.parameters==second.parameters
        @test resumed.synthesis.data==full.synthesis.data

        # finish: without the second cycle's checkpoint, the outputs are those of the first cycle, with no gradient
        rm(checkpoint*".cycle2")
        write_config()
        finished,finish_log=run(gradient_backend=ForbiddenForwardGradient(),finish=true)
        @test !occursin("Cycle 2 of 2",finish_log)
        @test finished.solver.state.parameters==first.parameters
        @test finished.objective.components.total≈first.objective rtol=1e-12
        rm(checkpoint*".cycle1")
        @test_throws ArgumentError run(gradient_backend=ForbiddenForwardGradient(),finish=true)

        # solver.resume = true needs the first cycle's checkpoint
        write_config(solver="resume = true")
        @test_throws ArgumentError run(gradient_backend=gradient)
    end
end
