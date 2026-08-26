using FFNOInversion
using Sockets

mode=length(ARGS)==1 ? Symbol(ARGS[1]) : error(
    "usage: olivia_phase6_solver_probe.jl success|rejection")
mode in (:success,:rejection) || error("unknown Phase 6 solver probe mode: $mode")

struct OliviaPhase6Synthesizer <: AbstractSynthesizer end
function FFNOInversion.synthesize!(cube::SpectralCube{T},::OliviaPhase6Synthesizer,::NonPRD,
        atmosphere::Atmosphere3D,populations,cache=nothing) where T
    nz,nx,ny=size(atmosphere.temperature)
    for i in 1:nx,j in 1:ny
        cube.data[1,1,i,j]=atmosphere.temperature[1,i,j]/T(5000)
        cube.data[2,1,i,j]=atmosphere.temperature[nz,i,j]/T(5000)
        cube.data[3,1,i,j]=one(T)+atmosphere.vz[1,i,j]/T(5000)
        cube.data[4,1,i,j]=one(T)+atmosphere.vz[nz,i,j]/T(5000)
    end
    cube
end

struct OliviaPhase6VJP <: AbstractObjectiveGradient end
function FFNOInversion.objective_gradient!(::OliviaPhase6VJP,
        problem::DistributedInversionProblem,layout::ControlMapLayout{T},
        parameters::AbstractVector,context::ParallelContext) where T
    evaluation=evaluate_objective!(problem,layout,parameters,context)
    synthetic=problem.workspace.output.data; observation=problem.observation
    spectrum_bar=similar(synthetic)
    @. spectrum_bar=(T(2)/problem.active_residual_count)*observation.inversion_weights^2*
        (synthetic-observation.spectrum.data)/observation.sigma^2
    atmosphere=problem.distributed.local_atmosphere; nz=size(atmosphere.temperature,1)
    temperature_bar=zeros(T,size(atmosphere.temperature)); vz_bar=zeros(T,size(atmosphere.vz))
    for i in axes(temperature_bar,2),j in axes(temperature_bar,3)
        temperature_bar[1,i,j]+=spectrum_bar[1,1,i,j]/T(5000)
        temperature_bar[nz,i,j]+=spectrum_bar[2,1,i,j]/T(5000)
        vz_bar[1,i,j]+=spectrum_bar[3,1,i,j]/T(5000)
        vz_bar[nz,i,j]+=spectrum_bar[4,1,i,j]/T(5000)
    end
    gradient=zeros(T,layout.parameter_count)
    accumulate_control_vjp!(gradient,layout,:temperature,temperature_bar,problem.distributed,context)
    accumulate_control_vjp!(gradient,layout,:vz,vz_bar,problem.distributed,context)
    ObjectiveGradientEvaluation(evaluation,gradient,1)
end

struct OliviaNegatedPhase6VJP <: AbstractObjectiveGradient end
function FFNOInversion.objective_gradient!(::OliviaNegatedPhase6VJP,
        problem,layout,parameters,context)
    result=objective_gradient!(OliviaPhase6VJP(),problem,layout,parameters,context)
    ObjectiveGradientEvaluation(result.evaluation,-result.gradient,result.forward_evaluations)
end

function control_layout(temperature,vz)
    temp=NodeField(reshape(Float64.(temperature),2,2,1),[-5.0,-1.0])
    velocity=NodeField(reshape(Float64.(vz),2,1,1),[-5.0,-1.0])
    ControlMapLayout(ControlMapSpec(:temperature,temp;lower=3000.0,upper=8000.0,scale=1000.0),
        ControlMapSpec(:vz,velocity;lower=-3000.0,upper=3000.0,scale=1000.0))
end

context=initialize_parallel(options=ParallelOptions(enabled=true,threads_per_rank=Threads.nthreads()))
diagnostics_root=get(ENV,"OLIVIA_CASE_DIAGNOSTICS",pwd()); mkpath(diagnostics_root)
log_path=joinpath(diagnostics_root,"phase6-solver-rank-$(context.rank).log")

function record(event;details="")
    open(log_path,"a") do io
        println(io,"$(time()) rank=$(context.rank) pid=$(getpid()) host=$(gethostname()) event=$event $details")
        flush(io)
    end
end

stop_watchdog=Threads.Atomic{Bool}(false)
watchdog=Threads.@spawn begin
    while !stop_watchdog[]
        record("watchdog_alive";details="thread=$(Threads.threadid())")
        sleep(5)
    end
end

try
    nx,ny=5,4
    grid=Grid3D([-5.0,-3.0,-1.0],collect(0.0:50e3:(nx-1)*50e3),collect(0.0:50e3:(ny-1)*50e3))
    root_atmosphere=if isroot(context)
        shape=(3,nx,ny); zero3=zeros(shape)
        Atmosphere3D(grid,fill(5500.0,shape),copy(zero3),copy(zero3),copy(zero3),copy(zero3))
    else
        nothing
    end
    distributed=distribute_atmosphere(Float64,root_atmosphere,context); root_atmosphere=nothing
    wavelength=[500.0e-9,500.1e-9,500.2e-9,500.3e-9]
    force=ForceBalanceOptions(max_iterations=4,relative_tolerance=5.0,force_tolerance=5.0,
        height_tolerance_m=1e9,relaxation=0.5,pressure_sweeps=4)
    model=HybridForwardModel(LocalDistributedPopulationModel(MockPopulationModel(1e10),1),NonPRD(),
        OliviaPhase6Synthesizer(),IdentityObservation(),IdealGasEOS(),ReferenceOpacity500(kappa_m2_kg=0.02),
        HE3DBoundaryState(1e-10,1.0,:top),force,CapabilityManifest())
    workspace=HybridForwardWorkspace(Float64,distributed,wavelength,StokesSet(:I),1)
    truth_layout=control_layout([5000,6000,5500,6500],[-1000,1000]); truth=initial_parameters(truth_layout)
    apply_control_maps!(distributed,truth_layout,truth)
    truth_result=forward!(workspace,model,distributed,context)
    observation=ObservationCube(SpectralCube(copy(truth_result.spectrum.data),wavelength,StokesSet(:I)),
        fill(0.02,size(truth_result.spectrum.data)),ones(size(truth_result.spectrum.data)))
    regularization=RegularizationSpec(vertical=VerticalRegularizationSpec(ntuple(_->0,7),0.0,ntuple(_->1.0,7)),
        horizontal=Dict{Symbol,Float64}(),scales=Dict{Symbol,Float64}(),horizontal_order=1)
    problem=DistributedInversionProblem(model,workspace,distributed,observation,regularization,50e3,50e3,context)
    layout=control_layout(fill(4500.0,4),[0.0,0.0]); initial=initial_parameters(layout)
    record("solver_enter";details="mode=$mode ranks=$(context.size) controls=$(length(initial))")

    if mode==:success
        result=lbfgs_invert!(problem,layout,OliviaPhase6VJP(),context;
            options=LBFGSSolverOptions(max_iterations=20,history_length=5,
                gradient_tolerance=1e-8,objective_tolerance=1e-14))
        error_max=maximum(abs.(result.state.parameters.-truth))
        result.state.converged || error("L-BFGS did not converge")
        result.state.objective<1e-12 || error("L-BFGS objective is too large: $(result.state.objective)")
        error_max<1e-5 || error("L-BFGS control error is too large: $error_max")
        result.state.forward_evaluations<100 || error(
            "L-BFGS used too many forward evaluations: $(result.state.forward_evaluations)")

        checkpoint_path=joinpath(diagnostics_root,"phase6-solver.checkpoint")
        lbfgs_invert!(problem,layout,OliviaPhase6VJP(),context;initial=initial,
            options=LBFGSSolverOptions(max_iterations=1,history_length=5,
                gradient_tolerance=1e-8,objective_tolerance=1e-14,
                checkpoint_path=checkpoint_path))
        restarted=lbfgs_invert!(problem,layout,OliviaPhase6VJP(),context;restart=true,
            options=LBFGSSolverOptions(max_iterations=20,history_length=5,
                gradient_tolerance=1e-8,objective_tolerance=1e-14,
                checkpoint_path=checkpoint_path))
        restarted.state.parameters==result.state.parameters || error(
            "checkpoint restart controls differ from uninterrupted solver")
        restarted.state.gradient==result.state.gradient || error(
            "checkpoint restart gradient differs from uninterrupted solver")
        restarted.state.objective==result.state.objective || error(
            "checkpoint restart objective differs from uninterrupted solver")
        restarted.state.history==result.state.history || error(
            "checkpoint restart history differs from uninterrupted solver")
        restarted.state.forward_evaluations==result.state.forward_evaluations || error(
            "checkpoint restart forward-evaluation count differs")
        barrier(context)
        record("solver_success";details="objective=$(result.state.objective) error=$error_max forwards=$(result.state.forward_evaluations) restart=matched")
        isroot(context) && println("OLIVIA_PHASE6_SOLVER_OK ranks=$(context.size) objective=$(result.state.objective) max_error=$error_max forward_evaluations=$(result.state.forward_evaluations) restart=matched")
    else
        result=lbfgs_invert!(problem,layout,OliviaNegatedPhase6VJP(),context;
            options=LBFGSSolverOptions(max_iterations=1,maximum_line_search_trials=4,
                backtracking_factor=0.25))
        result.state.termination==:line_search_failed || error(
            "expected line_search_failed, got $(result.state.termination)")
        result.state.parameters==initial || error("rejected trials changed accepted controls")
        length(result.state.history)==1 && !result.state.history[1].accepted || error(
            "rejected iteration was not recorded")
        result.state.history[1].rejected_trials==4 || error("not all injected trials were rejected")
        expected=expand_nodes(parameter_nodefield(layout,initial,:temperature),distributed.global_grid,distributed.tile)
        distributed.local_atmosphere.temperature==expected || error(
            "atmosphere was not restored after rejected trials")
        barrier(context)
        record("rejection_recovery_success";details="rejected_trials=4")
        isroot(context) && println("OLIVIA_PHASE6_REJECTION_RECOVERY_OK ranks=$(context.size) rejected_trials=4")
    end
finally
    stop_watchdog[]=true
    wait(watchdog)
    record("finalizing")
    finalize_parallel!(context)
end
