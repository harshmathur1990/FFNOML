# Soft stops (stop file, job deadline), resuming from the checkpoint, and finishing a stopped run.
# Uses Phase6RecoveryVJP/phase6_observation (phase6_gradients_solver.jl) and the fake FSDP service
# and test equilibrium (phase6_execution.jl).

"""Phase6RecoveryVJP that creates `path` at its `after`-th call (the setup gradient is call 1)."""
mutable struct StopFileAfterGradient <: AbstractObjectiveGradient
    path::String
    after::Int
    calls::Int
end
function FFNOInversion.objective_gradient!(backend::StopFileAfterGradient,problem,layout,parameters,context)
    backend.calls+=1; backend.calls==backend.after && touch(backend.path)
    objective_gradient!(Phase6RecoveryVJP(),problem,layout,parameters,context)
end

function _captured_stdout(f)
    mktemp() do path,io
        value=redirect_stdout(f,io); flush(io)
        (value,read(path,String))
    end
end

@testset "L-BFGS soft stops, resume and finish" begin
    observation=phase6_observation()
    solver_case()=(fixture=phase5_fixture(); layout=phase5_layout(fill(4500.0,4),[0.0,0.0]);
        (fixture=fixture,layout=layout,problem=phase5_problem(fixture,observation)))
    base=(history_length=4,gradient_tolerance=1e-12,objective_tolerance=0.0)
    continuous_case=solver_case()
    continuous=lbfgs_invert!(continuous_case.problem,continuous_case.layout,Phase6RecoveryVJP(),
        continuous_case.fixture.context;options=LBFGSSolverOptions(;max_iterations=6,base...))

    mktempdir() do directory
        checkpoint=joinpath(directory,"run.checkpoint"); stop_file=joinpath(directory,"STOP")
        # the file appears during iteration 2 (gradient call 3): the run ends before iteration 3 with its checkpoint
        stopped_case=solver_case(); backend=StopFileAfterGradient(stop_file,3,0)
        options=LBFGSSolverOptions(;max_iterations=6,base...,checkpoint_path=checkpoint,stop_file=stop_file)
        stopped,log=_captured_stdout(()->lbfgs_invert!(stopped_case.problem,stopped_case.layout,backend,
            stopped_case.fixture.context;options=options))
        @test stopped.state.termination===:stop_requested
        @test !stopped.state.converged
        @test stopped.state.iteration==2 && length(stopped.state.history)==2
        @test stopped.state.history==continuous.state.history[1:2]
        @test occursin("LBFGS iteration=1 forward_evaluations=",log)
        @test occursin("LBFGS iteration=2 ",log) && !occursin("LBFGS iteration=3 ",log)
        @test occursin("LBFGS stop reason=stop_requested before iteration=3",log)
        @test occursin("LBFGS done iteration=2 termination=stop_requested",log)
        saved=restore_checkpoint(checkpoint).state.state
        @test saved.iteration==2 && saved.termination===:stop_requested

        # finishing restores the checkpoint without iterating
        finish_case=solver_case()
        finished=lbfgs_invert!(finish_case.problem,finish_case.layout,ForbiddenForwardGradient(),
            finish_case.fixture.context;restart=true,iterate=false,options=options)
        @test finished.state.iteration==2 && finished.state.parameters==stopped.state.parameters
        @test finished.state.termination===:stop_requested
        @test finished.evaluation.components.total≈stopped.state.objective rtol=1e-12
        @test_throws ArgumentError lbfgs_invert!(finish_case.problem,finish_case.layout,Phase6RecoveryVJP(),
            finish_case.fixture.context;iterate=false,options=options)

        # with the file removed, resuming continues exactly where the uninterrupted run went
        rm(stop_file); resumed_case=solver_case()
        resumed=lbfgs_invert!(resumed_case.problem,resumed_case.layout,Phase6RecoveryVJP(),
            resumed_case.fixture.context;restart=true,options=options)
        @test resumed.state.termination===:maximum_iterations
        @test resumed.state.parameters==continuous.state.parameters
        @test resumed.state.history==continuous.state.history
        @test resumed.state.forward_evaluations==continuous.state.forward_evaluations

        # a checkpoint stopped mid-run is incomplete when finished
        mid_case=solver_case()
        lbfgs_invert!(mid_case.problem,mid_case.layout,Phase6RecoveryVJP(),mid_case.fixture.context;
            options=LBFGSSolverOptions(;max_iterations=1,base...,checkpoint_path=checkpoint))
        payload=restore_checkpoint(checkpoint); payload.state.state.termination=:running   # as a run killed mid-way
        FFNOInversion.checkpoint!(checkpoint,payload.state;manifest=payload.manifest)
        incomplete=lbfgs_invert!(mid_case.problem,mid_case.layout,ForbiddenForwardGradient(),mid_case.fixture.context;
            restart=true,iterate=false,options=LBFGSSolverOptions(;max_iterations=6,base...,checkpoint_path=checkpoint))
        @test incomplete.state.termination===:incomplete_checkpoint
    end

    mktempdir() do directory
        checkpoint=joinpath(directory,"run.checkpoint")
        # a deadline already past: no iteration starts, and the checkpoint holds the starting point
        late_case=solver_case()
        late=lbfgs_invert!(late_case.problem,late_case.layout,Phase6RecoveryVJP(),late_case.fixture.context;
            options=LBFGSSolverOptions(;max_iterations=6,base...,checkpoint_path=checkpoint,deadline=time()-1))
        @test late.state.termination===:walltime && late.state.iteration==0
        @test restore_checkpoint(checkpoint).state.state.termination===:walltime
        # the reserve alone exceeds the time left
        reserve_case=solver_case()
        reserved=lbfgs_invert!(reserve_case.problem,reserve_case.layout,Phase6RecoveryVJP(),reserve_case.fixture.context;
            options=LBFGSSolverOptions(;max_iterations=6,base...,deadline=time()+60,walltime_reserve_seconds=120.0))
        @test reserved.state.termination===:walltime
        # a distant deadline changes nothing
        far_case=solver_case()
        far=lbfgs_invert!(far_case.problem,far_case.layout,Phase6RecoveryVJP(),far_case.fixture.context;
            options=LBFGSSolverOptions(;max_iterations=6,base...,deadline=time()+1e6))
        @test far.state.history==continuous.state.history
    end

    mktempdir() do directory
        # resuming a converged checkpoint must not iterate again (here converged on the objective tolerance)
        checkpoint=joinpath(directory,"converged.checkpoint")
        options=LBFGSSolverOptions(max_iterations=20,history_length=4,gradient_tolerance=1e-12,
            objective_tolerance=1e-2,checkpoint_path=checkpoint)
        converged_case=solver_case()
        converged=lbfgs_invert!(converged_case.problem,converged_case.layout,Phase6RecoveryVJP(),
            converged_case.fixture.context;options=options)
        @test converged.state.converged && converged.state.termination===:objective_tolerance
        again_case=solver_case()
        again=lbfgs_invert!(again_case.problem,again_case.layout,Phase6RecoveryVJP(),again_case.fixture.context;
            restart=true,options=options)
        @test again.state.history==converged.state.history
        @test again.state.termination===:objective_tolerance
    end

    mktempdir() do directory
        checkpoint=joinpath(directory,"run.checkpoint")
        context=serial_context()
        ifp=LBFGSSolverOptions(checkpoint_path=checkpoint,resume=:if_present)
        @test !FFNOInversion._resume_requested(ifp,context)
        touch(checkpoint); @test FFNOInversion._resume_requested(ifp,context)
        @test FFNOInversion._resume_requested(LBFGSSolverOptions(checkpoint_path=checkpoint,resume=:always),context)
        @test !FFNOInversion._resume_requested(LBFGSSolverOptions(checkpoint_path=checkpoint),context)
    end
    @test_throws ArgumentError FFNOInversion._validate(LBFGSSolverOptions(resume=:always))
    @test_throws ArgumentError FFNOInversion._validate(LBFGSSolverOptions(resume=:sometimes,checkpoint_path="x"))
    @test_throws ArgumentError FFNOInversion._validate(LBFGSSolverOptions(walltime_margin=0.0))
    @test_throws ArgumentError FFNOInversion._validate(LBFGSSolverOptions(walltime_reserve_seconds=-1.0))
    @test_throws ArgumentError FFNOInversion._validate(LBFGSSolverOptions(deadline=NaN))
    changed=FFNOInversion._with_options(LBFGSSolverOptions(max_iterations=7);deadline=5.0)
    @test changed.deadline==5.0 && changed.max_iterations==7

    # [solver] keys
    parsed=FFNOInversion._lbfgs_solver(Dict{String,Any}("checkpoint_path"=>"c","stop_file"=>"STOP",
        "resume"=>"if_present","deadline"=>12.0,"walltime_margin"=>2.0,"walltime_reserve_seconds"=>30.0))
    @test parsed.stop_file=="STOP" && parsed.resume===:if_present && parsed.deadline==12.0
    @test parsed.walltime_margin==2.0 && parsed.walltime_reserve_seconds==30.0
    defaults=FFNOInversion._lbfgs_solver(Dict{String,Any}())
    @test defaults.stop_file=="" && defaults.resume===:never && isinf(defaults.deadline)
    @test FFNOInversion._lbfgs_solver(Dict{String,Any}("checkpoint_path"=>"c","resume"=>true)).resume===:always
    @test FFNOInversion._lbfgs_solver(Dict{String,Any}("resume"=>false)).resume===:never
    @test_throws ArgumentError FFNOInversion._lbfgs_solver(Dict{String,Any}("checkpoint_path"=>"c","resume"=>"later"))
end

@testset "Inversion run: stop file, job deadline, resume and finish write the outputs" begin
    mktempdir() do directory
        atmosphere_path=joinpath(directory,"initial.h5"); observation_path=joinpath(directory,"observation.h5")
        synthesis_path=joinpath(directory,"synthesis.h5"); output_atmosphere_path=joinpath(directory,"atmosphere.h5")
        checkpoint=joinpath(directory,"run.checkpoint"); stop_file=joinpath(directory,"STOP")
        config_path=joinpath(directory,"run.toml")
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
            file["intensity"]=FFNOInversion._with_time_slyx(fill(1.0e10,4,1,nx,ny))
            file["sigma"]=FFNOInversion._with_time_slyx(fill(1e8,4,1,nx,ny))
            file["wavelength_weights"]=ones(4,1); file["spatial_weights"]=ones(ny,nx)
        end
        write_config(solver)=open(config_path,"w") do io
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
[solver]
method = "lbfgs"
max_iterations = 3
history_length = 3
checkpoint_path = "$checkpoint"
stop_file = "$stop_file"
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
        termination(path)=h5open(file->read(attributes(file)["solver_termination"]),path)
        run(;kwargs...)=(result=run_inversion_files!(config_path,factory;kwargs...); wait(server_ref[]); result)

        # stop file present: the setup gradient runs, no iteration, and the outputs are still written
        write_config(""); touch(stop_file)
        gradient=StartingAtmosphereGradient(copy(temperature),0)
        stopped=run(gradient_backend=gradient)
        @test stopped.solver.state.termination===:stop_requested && stopped.solver.state.iteration==0
        @test gradient.calls==1
        @test isfile(synthesis_path) && isfile(output_atmosphere_path) && isfile(checkpoint)
        @test termination(output_atmosphere_path)=="stop_requested"
        rm(synthesis_path); rm(output_atmosphere_path); rm(stop_file)

        # resume = "if_present" picks the checkpoint up; no new setup gradient is taken
        write_config("resume = \"if_present\""); resumed_gradient=StartingAtmosphereGradient(copy(temperature),0)
        resumed=run(gradient_backend=resumed_gradient)
        @test resumed_gradient.calls==0                         # zero gradient: converged before stepping
        @test resumed.solver.state.termination===:gradient_tolerance
        @test termination(output_atmosphere_path)=="gradient_tolerance"

        # finish: no gradient at all, outputs of the checkpoint's parameters
        rm(synthesis_path); rm(output_atmosphere_path)
        write_config("")
        finished=run(gradient_backend=ForbiddenForwardGradient(),finish=true)
        @test finished.solver.state.termination===:gradient_tolerance
        @test isfile(synthesis_path) && isfile(output_atmosphere_path)
        @test finished.synthesis.data==resumed.synthesis.data

        # the job's end time from the batch script stops the run before its first iteration
        rm(checkpoint)
        late=withenv("FFNO_JOB_END_EPOCH"=>string(round(Int,time())+60)) do
            run(gradient_backend=StartingAtmosphereGradient(copy(temperature),0))
        end
        @test late.solver.state.termination===:walltime
        @test termination(output_atmosphere_path)=="walltime"
        # a deadline in the configuration wins over the job's
        write_config("deadline = 1.0e12")
        config=withenv(()->FFNOInversion._with_job_deadline(load_config(config_path)),"FFNO_JOB_END_EPOCH"=>"100")
        @test config.solver.deadline==1.0e12
        write_config("")
        @test FFNOInversion._with_job_deadline(load_config(config_path),Dict("FFNO_JOB_END_EPOCH"=>"100")).solver.deadline==100.0
        @test isinf(FFNOInversion._with_job_deadline(load_config(config_path),Dict{String,String}()).solver.deadline)
        @test_throws ArgumentError FFNOInversion._with_job_deadline(load_config(config_path),Dict("FFNO_JOB_END_EPOCH"=>"soon"))
    end
end

@testset "Olivia batch script passes the job end and --finish" begin
    batch=read(joinpath(@__DIR__,"..","scripts","run_olivia_inversion.sbatch"),String)
    @test occursin("squeue --noheader --jobs=\"\${SLURM_JOB_ID}\" --format=%e",batch)
    @test occursin("export FFNO_JOB_END_EPOCH",batch)
    @test occursin("invert_arguments+=(--finish)",batch)
    @test occursin("\"\${invert_arguments[@]}\" \"\${config_file}\" \"\${factory_file}\"",batch)
    @test first(findfirst("export FFNO_JOB_END_EPOCH",batch))<first(findfirst("srun --kill-on-bad-exit",batch))
    invert=read(joinpath(@__DIR__,"..","scripts","invert.jl"),String)
    @test occursin("finish=\"--finish\" in ARGS",invert)
    @test occursin("run_inversion_files!(arguments[1],factory;finish=finish)",invert)
end
