Base.@kwdef struct LBFGSSolverOptions
    max_iterations::Int=50
    history_length::Int=10
    gradient_tolerance::Float64=1e-6
    step_tolerance::Float64=1e-10
    objective_tolerance::Float64=1e-12
    initial_step::Float64=1.0
    backtracking_factor::Float64=0.5
    armijo_coefficient::Float64=1e-4
    maximum_line_search_trials::Int=20
    maximum_forward_evaluations::Int=typemax(Int)
    curvature_tolerance::Float64=1e-10
    checkpoint_every::Int=1
    checkpoint_path::String=""
end

function _validate(options::LBFGSSolverOptions)
    options.max_iterations>=0 || throw(ArgumentError("max_iterations must be non-negative"))
    options.history_length>0 || throw(ArgumentError("history_length must be positive"))
    options.gradient_tolerance>=0 || throw(ArgumentError("gradient_tolerance must be non-negative"))
    options.step_tolerance>=0 || throw(ArgumentError("step_tolerance must be non-negative"))
    options.objective_tolerance>=0 || throw(ArgumentError("objective_tolerance must be non-negative"))
    options.initial_step>0 || throw(ArgumentError("initial_step must be positive"))
    0<options.backtracking_factor<1 || throw(ArgumentError("backtracking_factor must be between zero and one"))
    0<options.armijo_coefficient<1 || throw(ArgumentError("armijo_coefficient must be between zero and one"))
    options.maximum_line_search_trials>0 || throw(ArgumentError("maximum_line_search_trials must be positive"))
    options.maximum_forward_evaluations>0 || throw(ArgumentError("maximum_forward_evaluations must be positive"))
    options.curvature_tolerance>=0 || throw(ArgumentError("curvature_tolerance must be non-negative"))
    options.checkpoint_every>0 || throw(ArgumentError("checkpoint_every must be positive"))
    options
end

struct LBFGSIterationRecord{T<:AbstractFloat}
    iteration::Int
    forward_evaluations::Int
    total::T
    data::T
    regularization::T
    projected_gradient_norm::T
    accepted_step::T
    line_search_trials::Int
    rejected_trials::Int
    accepted::Bool
end

mutable struct LBFGSSolverState{T<:AbstractFloat}
    schema_version::VersionNumber
    iteration::Int
    forward_evaluations::Int
    parameters::Vector{T}
    gradient::Vector{T}
    objective::T
    data_term::T
    regularization_term::T
    s_history::Vector{Vector{T}}
    y_history::Vector{Vector{T}}
    history::Vector{LBFGSIterationRecord{T}}
    converged::Bool
    termination::Symbol
end

struct LBFGSInversionResult{T<:AbstractFloat,E}
    state::LBFGSSolverState{T}
    evaluation::E
end

function _save_lbfgs_checkpoint(path,layout,state,manifest,history_length,context)
    isempty(path) && return nothing
    if isroot(context)
        mkpath(dirname(abspath(path)))
        checkpoint!(path,(kind=:phase6_lbfgs,layout_signature=_layout_signature(layout),
            history_length=history_length,state=state);manifest=manifest)
    end
    barrier(context); path
end

function _restore_lbfgs_checkpoint(path,layout,manifest,history_length,context)
    isempty(path) && throw(ArgumentError("restart requires checkpoint_path"))
    payload=if isroot(context)
        restored=restore_checkpoint(path;expected=manifest); state_payload=restored.state
        state_payload.kind===:phase6_lbfgs || throw(ArgumentError("checkpoint is not a Phase 6 L-BFGS state"))
        state_payload.layout_signature==_layout_signature(layout) || throw(ArgumentError(
            "checkpoint control-map layout differs from requested layout"))
        state_payload.history_length==history_length || throw(ArgumentError(
            "checkpoint L-BFGS history length differs from requested solver"))
        state_payload.state
    else
        nothing
    end
    mpi_broadcast(payload,context)
end

function _projected_gradient(x,g,lower,upper)
    projected=copy(g)
    for i in eachindex(projected)
        tolerance=eps(eltype(x))*max(abs(x[i]),abs(lower[i]),abs(upper[i]),one(eltype(x)))*20
        if (x[i]<=lower[i]+tolerance && g[i]>0) || (x[i]>=upper[i]-tolerance && g[i]<0)
            projected[i]=zero(eltype(projected))
        end
    end
    projected
end

function _lbfgs_direction(gradient,s_history,y_history)
    q=copy(gradient); count=length(s_history); alpha=zeros(eltype(q),count); rho=similar(alpha)
    for i in count:-1:1
        curvature=dot(y_history[i],s_history[i])
        curvature>0 || throw(ErrorException("L-BFGS history has non-positive curvature"))
        rho[i]=inv(curvature); alpha[i]=rho[i]*dot(s_history[i],q)
        q.-=alpha[i].*y_history[i]
    end
    if count==0
        r=q
    else
        sy=dot(s_history[end],y_history[end]); yy=dot(y_history[end],y_history[end])
        r=(sy/yy).*q
    end
    for i in 1:count
        beta=rho[i]*dot(y_history[i],r)
        r.+=s_history[i].*(alpha[i]-beta)
    end
    -r
end

function _feasible_direction!(direction,x,lower,upper)
    for i in eachindex(direction)
        if (x[i]<=lower[i] && direction[i]<0) || (x[i]>=upper[i] && direction[i]>0)
            direction[i]=zero(eltype(direction))
        end
    end
    direction
end

function _append_history!(state,s,y,options)
    curvature=dot(s,y)
    threshold=options.curvature_tolerance*max(norm(s)*norm(y),eps(eltype(s)))
    curvature>threshold || return false
    push!(state.s_history,copy(s)); push!(state.y_history,copy(y))
    if length(state.s_history)>options.history_length
        popfirst!(state.s_history); popfirst!(state.y_history)
    end
    true
end

"""Run synchronized bounded limited-memory BFGS on the distributed objective.

Controls and solver history are the only replicated arrays.  Rank 0 forms the
search direction and accepts/rejects trials; all ranks execute every objective
and VJP through the existing hybrid runtime.  Every rejected line-search path
explicitly restores the last accepted atmosphere before returning.
"""
function lbfgs_invert!(problem::DistributedInversionProblem,layout::ControlMapLayout{T},
        gradient_backend::AbstractObjectiveGradient,context::ParallelContext;
        initial=initial_parameters(layout),options::LBFGSSolverOptions=LBFGSSolverOptions(),
        restart::Bool=false,manifest::CapabilityManifest=problem.model.capabilities) where T
    _validate(options); lower,upper=_scaled_bounds(layout)
    state,current=if restart
        restored=_restore_lbfgs_checkpoint(options.checkpoint_path,layout,manifest,
            options.history_length,context)
        restored.schema_version==v"1.0.0" || throw(ArgumentError(
            "unsupported L-BFGS checkpoint schema $(restored.schema_version)"))
        evaluation=evaluate_objective!(problem,layout,restored.parameters,context)
        isapprox(evaluation.components.total,restored.objective;rtol=1e-12,atol=1e-12) ||
            throw(ErrorException("restart objective differs from checkpointed objective"))
        (restored,evaluation)
    else
        parameters=mpi_broadcast(isroot(context) ? collect(initial) : nothing,context)
        _check_parameter_length(layout,parameters); project_parameters!(parameters,layout)
        result=objective_gradient!(gradient_backend,problem,layout,parameters,context)
        length(result.gradient)==layout.parameter_count || throw(DimensionMismatch(
            "objective gradient length differs from control layout"))
        created=LBFGSSolverState(v"1.0.0",0,result.forward_evaluations,parameters,
            T.(result.gradient),T(result.evaluation.components.total),T(result.evaluation.components.data),
            T(result.evaluation.components.regularization),Vector{T}[],Vector{T}[],
            LBFGSIterationRecord{T}[],false,:running)
        (created,result.evaluation)
    end
    _check_parameter_length(layout,state.parameters); _check_parameter_length(layout,state.gradient)
    state.iteration<=options.max_iterations || throw(ArgumentError(
        "checkpoint iteration $(state.iteration) exceeds requested max_iterations $(options.max_iterations)"))
    length(state.s_history)==length(state.y_history)<=options.history_length || throw(ArgumentError(
        "invalid L-BFGS checkpoint history"))
    if !state.converged && state.iteration<options.max_iterations
        state.termination=:running
    end

    for iteration in state.iteration+1:options.max_iterations
        if state.forward_evaluations>=options.maximum_forward_evaluations
            state.termination=:maximum_forward_evaluations; break
        end
        x=scaled_parameters(layout,state.parameters)
        projected=_projected_gradient(x,state.gradient,lower,upper)
        gradient_norm=norm(projected,Inf)
        if gradient_norm<=options.gradient_tolerance
            state.converged=true; state.termination=:gradient_tolerance; break
        end
        direction=mpi_broadcast(if isroot(context)
            proposed=_lbfgs_direction(projected,state.s_history,state.y_history)
            _feasible_direction!(proposed,x,lower,upper)
            dot(projected,proposed)<0 || (proposed.=-projected; _feasible_direction!(proposed,x,lower,upper))
            proposed
        else
            nothing
        end,context)
        if norm(direction,Inf)<=options.step_tolerance
            state.converged=true; state.termination=:step_tolerance; break
        end

        accepted=false; accepted_step=zero(T); trials=0; rejected=0
        trial_parameters=state.parameters; trial_evaluation=current; trial_x=x
        step=T(options.initial_step)
        for line_trial in 1:options.maximum_line_search_trials
            state.forward_evaluations>=options.maximum_forward_evaluations && break
            candidate_x=clamp.(x.+step.*direction,lower,upper); delta=candidate_x.-x
            norm(delta,Inf)>options.step_tolerance || break
            candidate_parameters=mpi_broadcast(isroot(context) ?
                _physical_from_scaled(layout,candidate_x) : nothing,context)
            evaluation=evaluate_objective!(problem,layout,candidate_parameters,context)
            state.forward_evaluations+=1; trials=line_trial
            decision=mpi_broadcast(isroot(context) ?
                evaluation.components.total<=state.objective+options.armijo_coefficient*dot(state.gradient,delta) :
                nothing,context)
            if decision
                accepted=true; accepted_step=step; trial_parameters=candidate_parameters
                trial_evaluation=evaluation; trial_x=candidate_x
                break
            end
            rejected+=1; step*=T(options.backtracking_factor)
        end

        if !accepted
            # A rejected trial leaves mutable forward workspaces at that trial.
            # Re-evaluate the accepted point before checkpointing or returning.
            current=evaluate_objective!(problem,layout,state.parameters,context)
            state.forward_evaluations+=1; state.iteration=iteration
            state.termination=state.forward_evaluations>=options.maximum_forward_evaluations ?
                :maximum_forward_evaluations : :line_search_failed
            push!(state.history,LBFGSIterationRecord(iteration,state.forward_evaluations,state.objective,
                state.data_term,state.regularization_term,T(gradient_norm),zero(T),trials,rejected,false))
            _save_lbfgs_checkpoint(options.checkpoint_path,layout,state,manifest,options.history_length,context)
            break
        end

        previous_objective=state.objective; previous_gradient=copy(state.gradient)
        gradient_result=objective_gradient!(gradient_backend,problem,layout,trial_parameters,context)
        state.forward_evaluations+=gradient_result.forward_evaluations
        length(gradient_result.gradient)==layout.parameter_count || throw(DimensionMismatch(
            "objective gradient length differs from control layout"))
        isapprox(gradient_result.evaluation.components.total,trial_evaluation.components.total;
            rtol=1e-10,atol=1e-12) || throw(ErrorException(
            "objective changed between accepted trial and gradient pullback"))
        new_gradient=T.(gradient_result.gradient)
        _append_history!(state,trial_x.-x,new_gradient.-previous_gradient,options)
        state.parameters=collect(trial_parameters); state.gradient=new_gradient
        current=gradient_result.evaluation; state.objective=T(current.components.total)
        state.data_term=T(current.components.data); state.regularization_term=T(current.components.regularization)
        state.iteration=iteration
        projected_new=_projected_gradient(trial_x,state.gradient,lower,upper)
        projected_norm=T(norm(projected_new,Inf))
        push!(state.history,LBFGSIterationRecord(iteration,state.forward_evaluations,state.objective,
            state.data_term,state.regularization_term,projected_norm,accepted_step,trials,rejected,true))

        parameter_step=norm(trial_x.-x,Inf)
        objective_change=abs(previous_objective-state.objective)
        if projected_norm<=options.gradient_tolerance
            state.converged=true; state.termination=:gradient_tolerance
        elseif parameter_step<=options.step_tolerance
            state.converged=true; state.termination=:step_tolerance
        elseif objective_change<=options.objective_tolerance*max(abs(previous_objective),one(T))
            state.converged=true; state.termination=:objective_tolerance
        end
        if iteration%options.checkpoint_every==0 || state.converged
            _save_lbfgs_checkpoint(options.checkpoint_path,layout,state,manifest,options.history_length,context)
        end
        state.converged && break
    end
    if !state.converged && state.termination===:running && state.iteration>=options.max_iterations
        state.termination=:maximum_iterations
    end
    isempty(options.checkpoint_path) ||
        _save_lbfgs_checkpoint(options.checkpoint_path,layout,state,manifest,options.history_length,context)
    LBFGSInversionResult(state,current)
end

"""Write rank-0 Phase 6 convergence diagnostics without gathering model cubes."""
function write_lbfgs_diagnostics(directory::AbstractString,result::LBFGSInversionResult,
        layout::ControlMapLayout,context::ParallelContext;metadata::AbstractDict=Dict())
    isroot(context) || return nothing
    root=abspath(directory); mkpath(root)
    summary_path=joinpath(root,"summary.toml"); history_path=joinpath(root,"objective_history.csv")
    controls=[Dict{String,Any}("variable"=>string(spec.variable),"shape"=>collect(size(spec.initial)),
        "log_tau_nodes"=>spec.log_tau_nodes,"lower"=>spec.lower,"upper"=>spec.upper,
        "scale"=>spec.scale,"final_values"=>collect(result.state.parameters[range]))
        for (spec,range) in zip(layout.specs,layout.ranges)]
    summary=Dict{String,Any}("schema_version"=>string(result.state.schema_version),
        "method"=>"bounded_lbfgs","iteration"=>result.state.iteration,
        "forward_evaluations"=>result.state.forward_evaluations,"converged"=>result.state.converged,
        "termination"=>string(result.state.termination),"objective"=>result.state.objective,
        "data_term"=>result.state.data_term,"regularization_term"=>result.state.regularization_term,
        "mpi_ranks"=>context.size,"threads_per_rank"=>Threads.nthreads(),"controls"=>controls,
        "metadata"=>Dict(string(k)=>v for (k,v) in metadata))
    open(summary_path,"w") do io; TOML.print(io,summary;sorted=true); end
    open(history_path,"w") do io
        println(io,"iteration,forward_evaluations,total,data,regularization,projected_gradient_norm,accepted_step,line_search_trials,rejected_trials,accepted")
        for record in result.state.history
            println(io,join((record.iteration,record.forward_evaluations,record.total,record.data,
                record.regularization,record.projected_gradient_norm,record.accepted_step,
                record.line_search_trials,record.rejected_trials,record.accepted),','))
        end
    end
    (summary=summary_path,history=history_path)
end
