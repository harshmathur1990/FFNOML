# First-order optimizers with a learning rate and a learning-rate schedule, as in machine learning (SGD with momentum,
# Adam, AdamW, RMSprop, Adagrad), for the same distributed objective as the bounded L-BFGS of ScalableSolvers.jl.
# They work in the dimensionless control coordinates (physical value / control scale), take one gradient per iteration
# and no line search, and project every step back inside the control bounds.

const FIRST_ORDER_METHODS=(:sgd,:adam,:adamw,:rmsprop,:adagrad)
const LEARNING_RATE_SCHEDULES=(:constant,:cosine,:step,:exponential)

Base.@kwdef struct FirstOrderSolverOptions
    method::Symbol=:adam                     # :sgd, :adam, :adamw, :rmsprop or :adagrad
    max_iterations::Int=200
    learning_rate::Float64=1e-2              # per iteration, in control-scale units (a step of 1 is one `scale`)
    momentum::Float64=0.0                    # sgd and rmsprop
    nesterov::Bool=false                     # sgd: Nesterov momentum
    beta1::Float64=0.9                       # adam, adamw
    beta2::Float64=0.999                     # adam, adamw
    epsilon::Float64=1e-8                    # adam, adamw, rmsprop, adagrad
    rmsprop_alpha::Float64=0.99              # rmsprop: decay of the squared-gradient average
    # weight decay pulls the controls toward the starting atmosphere's (not toward zero): an L2 term in the gradient for
    # sgd, adam, rmsprop and adagrad, decoupled from the gradient for adamw
    weight_decay::Float64=0.0
    schedule::Symbol=:constant               # :constant, :cosine, :step or :exponential
    warmup_iterations::Int=0                 # linear warm-up from learning_rate/(warmup+1), before any schedule
    min_learning_rate::Float64=0.0           # cosine: the learning rate at the last iteration
    step_size::Int=50                        # step: multiply by gamma every step_size iterations
    gamma::Float64=0.5                       # step and exponential: the decay factor
    gradient_tolerance::Float64=0.0          # stop when the projected gradient's largest component falls to this
    keep_best::Bool=true                     # the outputs are those of the lowest objective, not of the last iterate
    checkpoint_every::Int=1
    checkpoint_path::String=""
    stop_file::String=""                     # the soft stops of LBFGSSolverOptions
    deadline::Float64=Inf
    walltime_margin::Float64=1.5
    walltime_reserve_seconds::Float64=600.0
    resume::Symbol=:never
end

function _validate(options::FirstOrderSolverOptions)
    options.method in FIRST_ORDER_METHODS || throw(ArgumentError(
        "solver.method must be one of $(join(FIRST_ORDER_METHODS,", ")) for a first-order solver"))
    options.schedule in LEARNING_RATE_SCHEDULES || throw(ArgumentError(
        "solver.schedule must be one of $(join(LEARNING_RATE_SCHEDULES,", "))"))
    options.max_iterations>=0 || throw(ArgumentError("max_iterations must be non-negative"))
    options.learning_rate>0 && isfinite(options.learning_rate) || throw(ArgumentError("learning_rate must be positive"))
    0<=options.momentum<1 || throw(ArgumentError("momentum must be in [0, 1)"))
    !options.nesterov || options.momentum>0 || throw(ArgumentError("nesterov needs momentum > 0"))
    0<=options.beta1<1 && 0<=options.beta2<1 || throw(ArgumentError("beta1 and beta2 must be in [0, 1)"))
    options.epsilon>0 || throw(ArgumentError("epsilon must be positive"))
    0<options.rmsprop_alpha<1 || throw(ArgumentError("rmsprop_alpha must be in (0, 1)"))
    options.weight_decay>=0 || throw(ArgumentError("weight_decay must be non-negative"))
    options.warmup_iterations>=0 || throw(ArgumentError("warmup_iterations must be non-negative"))
    0<=options.min_learning_rate<=options.learning_rate || throw(ArgumentError(
        "min_learning_rate must be between zero and learning_rate"))
    options.step_size>0 || throw(ArgumentError("step_size must be positive"))
    0<options.gamma<=1 || throw(ArgumentError("gamma must be in (0, 1]"))
    options.gradient_tolerance>=0 || throw(ArgumentError("gradient_tolerance must be non-negative"))
    options.checkpoint_every>0 || throw(ArgumentError("checkpoint_every must be positive"))
    !isnan(options.deadline) || throw(ArgumentError("deadline must be a unix time or Inf"))
    options.walltime_margin>0 || throw(ArgumentError("walltime_margin must be positive"))
    options.walltime_reserve_seconds>=0 || throw(ArgumentError("walltime_reserve_seconds must be non-negative"))
    options.resume in (:never,:always,:if_present) || throw(ArgumentError("resume must be never, always or if_present"))
    options.resume===:never || !isempty(options.checkpoint_path) || throw(ArgumentError("resume requires checkpoint_path"))
    options
end

_with_options(options::FirstOrderSolverOptions;changes...)=
    FirstOrderSolverOptions(;(name=>getfield(options,name) for name in fieldnames(FirstOrderSolverOptions))...,changes...)

"""The learning rate of iteration `t` (1, 2, ...): a linear warm-up over `warmup_iterations`, then the schedule over
the remaining iterations. The cosine schedule spans up to `max_iterations`, so resuming with another `max_iterations`
changes it from there on; the other schedules depend on `t` only."""
function learning_rate(options::FirstOrderSolverOptions,t::Integer)
    base=options.learning_rate; warmup=options.warmup_iterations
    t<=warmup && return base*t/(warmup+1)
    k=t-warmup-1                                         # iterations into the schedule, from 0
    if options.schedule===:constant
        base
    elseif options.schedule===:cosine
        span=max(options.max_iterations-warmup-1,1)
        options.min_learning_rate+(base-options.min_learning_rate)*(1+cos(pi*min(k,span)/span))/2
    elseif options.schedule===:step
        base*options.gamma^(k÷options.step_size)
    else
        base*options.gamma^k
    end
end

mutable struct FirstOrderSolverState{T<:AbstractFloat}
    schema_version::VersionNumber
    method::Symbol
    iteration::Int
    forward_evaluations::Int
    parameters::Vector{T}
    gradient::Vector{T}
    objective::T
    data_term::T
    regularization_term::T
    start::Vector{T}                         # scaled controls of the starting atmosphere (weight decay pulls toward them)
    first_moment::Vector{T}                  # momentum buffer (sgd, rmsprop) or Adam's m
    second_moment::Vector{T}                 # Adam's v, RMSprop's average or Adagrad's sum of squared gradients
    best_parameters::Vector{T}
    best_objective::T
    best_data_term::T
    best_regularization_term::T
    best_iteration::Int
    history::Vector{LBFGSIterationRecord{T}} # accepted_step holds the learning rate
    converged::Bool
    termination::Symbol
end

struct FirstOrderInversionResult{T<:AbstractFloat,E}
    state::FirstOrderSolverState{T}
    evaluation::E
end

"""One update of the scaled controls `x` with gradient `g` at iteration `t` (moments updated in place); returns the
new, unprojected controls."""
function _first_order_update!(state::FirstOrderSolverState{T},options::FirstOrderSolverOptions,x,g,t) where T
    lr=T(learning_rate(options,t)); decay=T(options.weight_decay); m=state.first_moment; v=state.second_moment
    method=options.method
    grad=decay>0 && method!==:adamw ? g.+decay.*(x.-state.start) : g
    step=if method===:sgd
        if options.momentum>0
            mu=T(options.momentum); @. m=mu*m+grad
            options.nesterov ? grad.+mu.*m : copy(m)
        else
            copy(grad)
        end
    elseif method in (:adam,:adamw)
        b1=T(options.beta1); b2=T(options.beta2)
        @. m=b1*m+(1-b1)*grad; @. v=b2*v+(1-b2)*grad^2
        @. (m/(1-b1^t))/(sqrt(v/(1-b2^t))+T(options.epsilon))
    elseif method===:rmsprop
        alpha=T(options.rmsprop_alpha); @. v=alpha*v+(1-alpha)*grad^2
        scaled=@. grad/(sqrt(v)+T(options.epsilon))
        options.momentum>0 ? (@. m=T(options.momentum)*m+scaled; copy(m)) : scaled
    else
        @. v=v+grad^2
        @. grad/(sqrt(v)+T(options.epsilon))
    end
    new=x.-lr.*step
    method===:adamw && decay>0 && (new.-=lr*decay.*(x.-state.start))
    new
end

function _save_first_order_checkpoint(path,layout,state,manifest,context,problem)
    isempty(path) && return nothing
    if isroot(context)
        mkpath(dirname(abspath(path)))
        checkpoint!(path,(kind=:first_order,method=state.method,layout_signature=_layout_signature(layout),
            control_mapping=_checkpoint_mapping(problem),state=state);manifest=manifest)
    end
    barrier(context); path
end

function _restore_first_order_checkpoint(path,layout,manifest,method,context,problem)
    isempty(path) && throw(ArgumentError("restart requires checkpoint_path"))
    _checkpoint_read_collectively(context) do
        payload=restore_checkpoint(path;expected=manifest).state
        _check_checkpoint_mapping(payload,problem)
        payload.kind===:first_order || throw(ArgumentError("checkpoint is not a first-order solver state"))
        payload.method===method || throw(ArgumentError(
            "checkpoint is of solver method $(payload.method), not $method"))
        payload.layout_signature==_layout_signature(layout) || throw(ArgumentError(
            "checkpoint control-map layout differs from requested layout"))
        payload.state
    end
end

_first_order_progress(context,method,message)=_log_line(context,uppercase(string(method))*" "*message)

"""Whether a gradient evaluation failed only because the trial atmosphere has no valid equilibrium (the same on every
rank: the objective is reduced before the check)."""
_infeasible_trial(exception)=exception isa ArgumentError && occursin("infeasible atmosphere",exception.msg)

"""Run a first-order optimizer (`options.method`) on the distributed objective.

Rank 0 forms every step; all ranks evaluate every objective and gradient. A trial atmosphere without a valid
equilibrium is retried with half the step, up to 10 times (then the run ends with `:infeasible_step`). With
`keep_best` the returned parameters are those of the lowest objective seen. The soft stops, checkpoints, `restart`
and `iterate=false` are those of `lbfgs_invert!`.
"""
function first_order_invert!(problem::DistributedInversionProblem,layout::ControlMapLayout{T},
        gradient_backend::AbstractObjectiveGradient,context::ParallelContext;
        initial=initial_parameters(layout),options::FirstOrderSolverOptions=FirstOrderSolverOptions(),
        restart::Bool=false,iterate::Bool=true,manifest::CapabilityManifest=problem.model.capabilities) where T
    _validate(options); lower,upper=_scaled_bounds(layout); method=options.method
    iterate || restart || throw(ArgumentError("iterate=false needs restart=true: there is no checkpoint to finish"))
    started=time()
    _log_section(context,restart ? "Restart from the checkpoint" : "Iteration 0")
    state,current=if restart
        restored=_restore_first_order_checkpoint(options.checkpoint_path,layout,manifest,method,context,problem)
        restored.schema_version==v"1.0.0" || throw(ArgumentError(
            "unsupported first-order checkpoint schema $(restored.schema_version)"))
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
        c=result.evaluation.components; n=layout.parameter_count
        created=FirstOrderSolverState(v"1.0.0",method,0,result.forward_evaluations,parameters,T.(result.gradient),
            T(c.total),T(c.data),T(c.regularization),scaled_parameters(layout,parameters),zeros(T,n),zeros(T,n),
            copy(parameters),T(c.total),T(c.data),T(c.regularization),0,LBFGSIterationRecord{T}[],false,:running)
        (created,result.evaluation)
    end
    _check_parameter_length(layout,state.parameters); _check_parameter_length(layout,state.gradient)
    _first_order_progress(context,method,"iteration=$(state.iteration) forward_evaluations=$(state.forward_evaluations) "*
        "total=$(state.objective) data=$(state.data_term) regularization=$(state.regularization_term) "*
        "projected_gradient=$(norm(_projected_gradient(scaled_parameters(layout,state.parameters),state.gradient,lower,upper),Inf)) "*
        "seconds=$(round(time()-started;digits=1))")
    log_memory(context,problem.model.populations)
    restart || _save_first_order_checkpoint(options.checkpoint_path,layout,state,manifest,context,problem)
    if !iterate
        state.termination===:running && (state.termination=:incomplete_checkpoint)
        return _first_order_result(problem,layout,state,current,options,context)
    end
    !state.converged && state.iteration<options.max_iterations && (state.termination=:running)

    longest_iteration=time()-started
    for iteration in (state.converged ? (1:0) : (state.iteration+1:options.max_iterations))
        stop=_soft_stop(options,longest_iteration,context)
        if stop!==:none
            state.termination=stop
            _first_order_progress(context,method,"stop reason=$stop before iteration=$iteration "*
                "longest_iteration_seconds=$(round(longest_iteration;digits=1))")
            break
        end
        _log_section(context,"Iteration $iteration")
        iteration_started=time()
        x=scaled_parameters(layout,state.parameters)
        projected=_projected_gradient(x,state.gradient,lower,upper)
        gradient_norm=norm(projected,Inf)
        if gradient_norm<=options.gradient_tolerance
            state.converged=true; state.termination=:gradient_tolerance; break
        end
        # rank 0 forms the step and owns the moments (they advance once per iteration: a step into an atmosphere without
        # a valid equilibrium is halved, not re-derived)
        proposed,state.first_moment,state.second_moment=mpi_broadcast(isroot(context) ?
            (_first_order_update!(state,options,x,state.gradient,iteration),state.first_moment,state.second_moment) :
            nothing,context)
        result=nothing; fraction=one(T); trials=0; candidate=x
        for trial in 1:11
            trials=trial
            candidate=clamp.(x.+fraction.*(proposed.-x),lower,upper)
            try
                result=objective_gradient!(gradient_backend,problem,layout,_physical_from_scaled(layout,candidate),context)
                break
            catch exception
                _infeasible_trial(exception) || rethrow()
                state.forward_evaluations+=1
                _first_order_progress(context,method,"iteration=$iteration trial=$trial has no valid equilibrium; halving the step")
                fraction/=2
            end
        end
        if result===nothing
            state.termination=:infeasible_step
            current=evaluate_objective!(problem,layout,state.parameters,context); state.forward_evaluations+=1
            break
        end
        length(result.gradient)==layout.parameter_count || throw(DimensionMismatch(
            "objective gradient length differs from control layout"))
        c=result.evaluation.components; current=result.evaluation
        state.parameters=_physical_from_scaled(layout,candidate)
        state.gradient=T.(result.gradient); state.forward_evaluations+=result.forward_evaluations
        state.objective=T(c.total); state.data_term=T(c.data); state.regularization_term=T(c.regularization)
        state.iteration=iteration
        if state.objective<state.best_objective
            state.best_parameters=copy(state.parameters); state.best_objective=state.objective
            state.best_data_term=state.data_term; state.best_regularization_term=state.regularization_term
            state.best_iteration=iteration
        end
        rate=T(learning_rate(options,iteration))*fraction
        projected_norm=T(norm(_projected_gradient(scaled_parameters(layout,state.parameters),state.gradient,lower,upper),Inf))
        push!(state.history,LBFGSIterationRecord(iteration,state.forward_evaluations,state.objective,state.data_term,
            state.regularization_term,projected_norm,rate,trials,trials-1,true))
        seconds=time()-iteration_started; longest_iteration=max(longest_iteration,seconds)
        _first_order_progress(context,method,"iteration=$iteration forward_evaluations=$(state.forward_evaluations) "*
            "total=$(state.objective) data=$(state.data_term) regularization=$(state.regularization_term) "*
            "projected_gradient=$projected_norm learning_rate=$rate best_iteration=$(state.best_iteration) "*
            "seconds=$(round(seconds;digits=1))")
        log_memory(context,problem.model.populations)
        iteration%options.checkpoint_every==0 &&
            _save_first_order_checkpoint(options.checkpoint_path,layout,state,manifest,context,problem)
    end
    if !state.converged && state.termination===:running && state.iteration>=options.max_iterations
        state.termination=:maximum_iterations
    end
    _save_first_order_checkpoint(options.checkpoint_path,layout,state,manifest,context,problem)
    _end_log_section()
    _first_order_progress(context,method,"done iteration=$(state.iteration) termination=$(state.termination) "*
        "best_iteration=$(state.best_iteration) best_total=$(state.best_objective)")
    _first_order_result(problem,layout,state,current,options,context)
end

"""The result: with `keep_best`, the state's parameters and objective are those of the best iterate (the history and
checkpoint keep the last one), and the atmosphere is put back at the best parameters."""
function _first_order_result(problem,layout,state,current,options,context)
    options.keep_best && state.best_iteration!=state.iteration || return FirstOrderInversionResult(state,current)
    best=deepcopy(state)
    best.parameters=copy(state.best_parameters); best.objective=state.best_objective
    best.data_term=state.best_data_term; best.regularization_term=state.best_regularization_term
    evaluation=evaluate_objective!(problem,layout,best.parameters,context); best.forward_evaluations+=1
    FirstOrderInversionResult(best,evaluation)
end
