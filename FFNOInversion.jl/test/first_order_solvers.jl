# First-order solvers (solver.method = sgd, adam, adamw, rmsprop, adagrad) with a learning rate and schedule: the
# schedules, one update of each method by hand, recovery of the exact phase-5 model, bounds, a step into an infeasible
# atmosphere, keep_best, checkpoint and resume, the stop file, the [solver] keys, and a run through the TOML route.
# Uses phase5_fixture/phase6_observation/Phase6RecoveryVJP (phase 5/6 tests), fake_fsdp_population_backend
# (phase6_execution.jl) and _captured_stdout (soft_stop_resume.jl).

# fails the gradient of every `fail_on`-th call as an atmosphere without a valid equilibrium would
mutable struct InfeasibleEveryNth <: AbstractObjectiveGradient
    fail_on::Int
    calls::Int
end
function FFNOInversion.objective_gradient!(backend::InfeasibleEveryNth,problem,layout,parameters,context)
    backend.calls+=1
    backend.calls%backend.fail_on==0 && throw(ArgumentError("cannot differentiate an infeasible atmosphere"))
    objective_gradient!(Phase6RecoveryVJP(),problem,layout,parameters,context)
end
# the starting point is feasible, every later trial is not
mutable struct InfeasibleAfterStart <: AbstractObjectiveGradient
    calls::Int
end
function FFNOInversion.objective_gradient!(backend::InfeasibleAfterStart,problem,layout,parameters,context)
    (backend.calls+=1)>1 && throw(ArgumentError("cannot differentiate an infeasible atmosphere"))
    objective_gradient!(Phase6RecoveryVJP(),problem,layout,parameters,context)
end

@testset "First-order solvers: learning-rate schedules" begin
    rate(t;kwargs...)=learning_rate(FirstOrderSolverOptions(;max_iterations=101,learning_rate=0.1,kwargs...),t)
    @test all(t->rate(t)==0.1,(1,50,101))
    @test rate(1;schedule=:cosine)≈0.1 && rate(51;schedule=:cosine)≈0.05 && isapprox(rate(101;schedule=:cosine),0.0;atol=1e-15)
    @test rate(101;schedule=:cosine,min_learning_rate=0.01)≈0.01
    @test rate(1;schedule=:step,step_size=10,gamma=0.5)==0.1 && rate(11;schedule=:step,step_size=10,gamma=0.5)==0.05
    @test rate(21;schedule=:step,step_size=10,gamma=0.5)==0.025
    @test rate(1;schedule=:exponential,gamma=0.9)==0.1 && rate(3;schedule=:exponential,gamma=0.9)≈0.081
    # warm-up: linear to the full rate, then the schedule starts at its beginning
    @test rate(1;warmup_iterations=4)≈0.02 && rate(4;warmup_iterations=4)≈0.08 && rate(5;warmup_iterations=4)==0.1
    @test rate(5;warmup_iterations=4,schedule=:cosine)≈0.1 && isapprox(rate(101;warmup_iterations=4,schedule=:cosine),0.0;atol=1e-15)
end

@testset "First-order solvers: one update of each method" begin
    state(n)=FirstOrderSolverState(v"1.0.0",:adam,0,0,zeros(n),zeros(n),0.0,0.0,0.0,[1.0,1.0],zeros(n),zeros(n),
        zeros(n),0.0,0.0,0.0,0,LBFGSIterationRecord{Float64}[],false,:running)
    x=[2.0,-1.0]; g=[0.5,-2.0]
    function update(t=1;kwargs...)
        s=state(2)
        (FFNOInversion._first_order_update!(s,FirstOrderSolverOptions(;learning_rate=0.1,kwargs...),x,g,t),s)
    end
    @test update(method=:sgd)[1]≈x.-0.1 .*g
    new,s=update(method=:sgd,momentum=0.9); @test new≈x.-0.1 .*g && s.first_moment≈g
    new,_=update(method=:sgd,momentum=0.9,nesterov=true); @test new≈x.-0.1 .*(g.+0.9 .*g)
    # Adam's first step is lr * sign(g) (bias-corrected moments)
    new,s=update(method=:adam); @test isapprox(new,x.-0.1 .*sign.(g);rtol=1e-6)
    @test s.first_moment≈0.1 .*g && s.second_moment≈0.001 .*g.^2
    new,_=update(method=:rmsprop); @test new≈x.-0.1 .*g./(sqrt.(0.01 .*g.^2).+1e-8)
    new,_=update(method=:adagrad); @test isapprox(new,x.-0.1 .*sign.(g);rtol=1e-6)
    # weight decay pulls toward the starting controls [1, 1]: in the gradient for adam, decoupled for adamw
    new,_=update(method=:sgd,weight_decay=0.5); @test new≈x.-0.1 .*(g.+0.5 .*(x.-1.0))
    new,_=update(method=:adamw,weight_decay=0.5); @test isapprox(new,x.-0.1 .*sign.(g).-0.1*0.5 .*(x.-1.0);rtol=1e-6)
end

@testset "First-order solvers: recovery, bounds, infeasible steps, keep_best" begin
    observation=phase6_observation(); truth=initial_parameters(phase5_layout([5000,6000,5500,6500],[-1000,1000]))
    function case()
        fixture=phase5_fixture(); layout=phase5_layout(fill(4500.0,4),[0.0,0.0])
        (fixture=fixture,layout=layout,problem=phase5_problem(fixture,observation))
    end
    run(c,backend=Phase6RecoveryVJP();kwargs...)=first(_captured_stdout(()->first_order_invert!(c.problem,c.layout,backend,
        c.fixture.context;options=FirstOrderSolverOptions(;kwargs...))))
    for (method,rate,extra) in ((:adam,0.1,(;)),(:adamw,0.1,(;)),(:sgd,0.02,(;momentum=0.9)),(:rmsprop,0.02,(;)),
            (:adagrad,0.5,(;))), schedule in (:constant,:cosine)
        c=case(); result=run(c;method=method,max_iterations=300,learning_rate=rate,schedule=schedule,extra...)
        @test result.state.termination===:maximum_iterations && result.state.iteration==300
        # from an objective of about 100; RMSprop with the cosine schedule is the slowest (about 5e-5, within 2 K)
        @test result.state.objective<1e-4
        @test maximum(abs.(result.state.parameters.-truth))<(method===:rmsprop && schedule===:cosine ? 3.0 : 0.01)   # K, m/s
        @test length(result.state.history)==300 && all(r->r.accepted,result.state.history)
        @test result.state.history[end].accepted_step≈learning_rate(FirstOrderSolverOptions(;method=method,
            max_iterations=300,learning_rate=rate,schedule=schedule,extra...),300)
    end

    # an upper temperature bound below the truth: every step is projected, the controls end on the bound
    c=case(); spec=c.layout.specs[1]
    bounded_layout=ControlMapLayout([ControlMapSpec(spec.variable,spec.initial,spec.nodes,spec.kind,spec.lower,5200.0,spec.scale),
        c.layout.specs[2:end]...])
    bounded=first(_captured_stdout(()->first_order_invert!(c.problem,bounded_layout,Phase6RecoveryVJP(),c.fixture.context;
        options=FirstOrderSolverOptions(method=:adam,max_iterations=100,learning_rate=0.1))))
    temperatures=bounded.state.parameters[bounded_layout.ranges[1]]
    @test all(<=(5200.0),temperatures) && any(==(5200.0),temperatures)

    # a step into an atmosphere without a valid equilibrium is halved and retried
    c=case(); backend=InfeasibleEveryNth(3,0)
    halved,log=_captured_stdout(()->first_order_invert!(c.problem,c.layout,backend,c.fixture.context;
        options=FirstOrderSolverOptions(method=:adam,max_iterations=4,learning_rate=0.1)))
    @test halved.state.termination===:maximum_iterations
    @test occursin("ADAM iteration=2 trial=1 has no valid equilibrium; halving the step",log)
    @test halved.state.history[2].line_search_trials==2 && halved.state.history[2].accepted_step≈0.05
    c=case()
    @test first(_captured_stdout(()->first_order_invert!(c.problem,c.layout,InfeasibleAfterStart(0),c.fixture.context;
        options=FirstOrderSolverOptions(method=:adam,max_iterations=4)))).state.termination===:infeasible_step

    # keep_best: a learning rate far too large makes the last iterate worse than the best one
    c=case(); wild=run(c;method=:sgd,max_iterations=12,learning_rate=3.0)
    @test wild.state.objective<=minimum(r.total for r in wild.state.history)
    @test wild.state.best_iteration<12 && wild.state.objective<wild.state.history[end].total
    @test wild.state.history[end].iteration==12                                   # the history keeps the last iterate
    @test wild.evaluation.components.total==wild.state.objective
    c=case(); last_iterate=run(c;method=:sgd,max_iterations=12,learning_rate=3.0,keep_best=false)
    @test last_iterate.state.objective==last_iterate.state.history[end].total
end

@testset "First-order solvers: checkpoint, resume and the stop file" begin
    observation=phase6_observation()
    function case()
        fixture=phase5_fixture(); layout=phase5_layout(fill(4500.0,4),[0.0,0.0])
        (fixture=fixture,layout=layout,problem=phase5_problem(fixture,observation))
    end
    # the step schedule depends on the iteration only (cosine spans max_iterations, so a shorter first run would change it)
    base=(method=:adam,learning_rate=0.1,schedule=:step,step_size=4,gamma=0.5,warmup_iterations=2)
    mktempdir() do directory
        checkpoint=joinpath(directory,"adam.checkpoint"); stop_file=joinpath(directory,"STOP")
        c=case(); continuous=first(_captured_stdout(()->first_order_invert!(c.problem,c.layout,Phase6RecoveryVJP(),
            c.fixture.context;options=FirstOrderSolverOptions(;base...,max_iterations=12))))
        c=case(); _captured_stdout(()->first_order_invert!(c.problem,c.layout,Phase6RecoveryVJP(),c.fixture.context;
            options=FirstOrderSolverOptions(;base...,max_iterations=5,checkpoint_path=checkpoint)))
        # resuming continues the same run: moments, schedule and history
        c=case(); resumed=first(_captured_stdout(()->first_order_invert!(c.problem,c.layout,Phase6RecoveryVJP(),
            c.fixture.context;restart=true,options=FirstOrderSolverOptions(;base...,max_iterations=12,checkpoint_path=checkpoint))))
        @test resumed.state.iteration==12
        @test resumed.state.parameters==continuous.state.parameters
        @test resumed.state.history==continuous.state.history
        @test resumed.state.second_moment==continuous.state.second_moment
        # another method cannot resume this checkpoint
        c=case()
        @test_throws ArgumentError first_order_invert!(c.problem,c.layout,Phase6RecoveryVJP(),c.fixture.context;restart=true,
            options=FirstOrderSolverOptions(;base...,method=:sgd,max_iterations=12,checkpoint_path=checkpoint))
        # finish: the checkpoint's parameters without a gradient
        c=case(); finished=first(_captured_stdout(()->first_order_invert!(c.problem,c.layout,ForbiddenForwardGradient(),
            c.fixture.context;restart=true,iterate=false,options=FirstOrderSolverOptions(;base...,max_iterations=12,checkpoint_path=checkpoint))))
        @test finished.state.parameters==continuous.state.parameters
        # the stop file ends the run before the next iteration, with the checkpoint saved
        touch(stop_file); rm(checkpoint)
        c=case(); stopped,log=_captured_stdout(()->first_order_invert!(c.problem,c.layout,Phase6RecoveryVJP(),c.fixture.context;
            options=FirstOrderSolverOptions(;base...,max_iterations=12,checkpoint_path=checkpoint,stop_file=stop_file)))
        @test stopped.state.termination===:stop_requested && stopped.state.iteration==0 && isfile(checkpoint)
        @test occursin("ADAM stop reason=stop_requested before iteration=1",log)
    end
end

@testset "First-order solvers: [solver] keys and the TOML route" begin
    solver(section)=FFNOInversion._solver(Dict{String,Any}(section))
    adam=solver(Dict("method"=>"adam","learning_rate"=>0.05,"schedule"=>"cosine","warmup_iterations"=>5,
        "min_learning_rate"=>0.001,"max_iterations"=>80,"weight_decay"=>0.01,"keep_best"=>false))
    @test adam isa FirstOrderSolverOptions && adam.method===:adam && adam.schedule===:cosine && adam.learning_rate==0.05
    @test adam.warmup_iterations==5 && adam.min_learning_rate==0.001 && adam.max_iterations==80 && !adam.keep_best
    @test solver(Dict("method"=>"SGD","momentum"=>0.9,"nesterov"=>true)).nesterov
    @test solver(Dict("method"=>"lbfgs")) isa LBFGSSolverOptions
    @test_throws ArgumentError solver(Dict("method"=>"adam","history_length"=>10))            # an L-BFGS key
    @test_throws ArgumentError solver(Dict("method"=>"adam","schedule"=>"triangle"))
    @test_throws ArgumentError solver(Dict("method"=>"adam","learning_rate"=>0.0))
    @test_throws ArgumentError solver(Dict("method"=>"sgd","nesterov"=>true))                  # no momentum
    @test_throws ArgumentError solver(Dict("method"=>"adam","min_learning_rate"=>1.0))
    @test_throws ArgumentError solver(Dict("method"=>"lion"))

    mktempdir() do directory
        atmosphere_path=joinpath(directory,"initial.h5"); observation_path=joinpath(directory,"observation.h5")
        synthesis_path=joinpath(directory,"synthesis.h5"); output_atmosphere_path=joinpath(directory,"atmosphere.h5")
        config_path=joinpath(directory,"run.toml"); shape=(3,2,2)
        python_storage(value)=permutedims(value,reverse(1:ndims(value)))
        h5open(atmosphere_path,"w") do file
            file["logtau_500"]=[-5.0,-3.0,-1.0]
            file["temperature"]=python_storage(FFNOInversion._with_time_zyx(reshape(collect(5100.0:100.0:6200.0),shape)))
            file["vturb"]=python_storage(FFNOInversion._with_time_zyx(fill(800.0,shape)))
            for name in ("vx","vy","vz"); file[name]=python_storage(FFNOInversion._with_time_zyx(zeros(shape))); end
        end
        h5open(observation_path,"w") do file
            file["intensity"]=FFNOInversion._with_time_slyx(fill(2.0e10,4,1,2,2))
            file["sigma"]=FFNOInversion._with_time_slyx(fill(1e8,4,1,2,2))
            file["wavelength_weights"]=ones(4,1); file["spatial_weights"]=ones(2,2)
        end
        write(config_path,"""
[inputs]
observation_file = "$observation_path"
initial_atmosphere_file = "$atmosphere_path"
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
method = "adam"
learning_rate = 0.05
schedule = "cosine"
max_iterations = 4
[parallel]
enabled = false
threads_per_rank = $(Threads.nthreads())
""")
        server_ref=Ref{Any}()
        factory=InversionModelFactory(1,(cfg,dist,ws,ctx)->begin
            backend,server=fake_fsdp_population_backend(value=1.0f10,temperature_scaled=true); server_ref[]=server
            HybridForwardModel(backend,NonPRD(),MockIntensitySynthesizer(),IdentityObservation(),
                TestHydrostaticEquilibrium(),CapabilityManifest())
        end)
        result,log=_captured_stdout(()->(r=run_inversion_files!(config_path,factory;gradient_backend=FiniteDifferenceObjectiveGradient(step=1e-3));
            wait(server_ref[]); r))
        @test result.solver isa FirstOrderInversionResult && result.solver.state.iteration==4
        @test result.solver.state.objective<result.solver.state.history[1].total || result.solver.state.best_iteration==0
        @test occursin("ADAM iteration=4 ",log) && occursin("learning_rate=",log)
        @test h5open(file->read(attributes(file)["solver_termination"]),output_atmosphere_path)=="maximum_iterations"
        @test isfile(synthesis_path)
    end
end
