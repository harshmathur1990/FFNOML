const PROTOTYPE_CONTROL_VARIABLES = (:temperature,:vx,:vy,:vz,:vlos,:vturb,:Bx,:By,:Bz)

"""One bounded atmospheric control map.

`initial` has shape `(vertical-node, control-x, control-y)`. Bounds and `scale`
are physical values in the units of the associated atmospheric variable.
"""
struct ControlMapSpec{T<:AbstractFloat}
    variable::Symbol
    initial::Array{T,3}
    log_tau_nodes::Vector{T}
    lower::T
    upper::T
    scale::T
end

function ControlMapSpec(variable::Symbol,field::NodeField{T};lower::Real,upper::Real,
                        scale::Real) where T<:AbstractFloat
    variable in PROTOTYPE_CONTROL_VARIABLES || throw(ArgumentError(
        "unsupported Phase 5 control variable $variable"))
    length(field.log_tau_nodes)>=2 || throw(ArgumentError("control maps require at least two vertical nodes"))
    lo,hi,s=T(lower),T(upper),T(scale)
    isfinite(lo) && isfinite(hi) && lo<hi || throw(ArgumentError("control-map bounds must be finite and increasing"))
    isfinite(s) && s>0 || throw(ArgumentError("control-map scale must be finite and positive"))
    all(v->lo<=v<=hi,field.values) || throw(ArgumentError("initial $variable controls violate bounds [$lo,$hi]"))
    variable===:temperature && lo<=0 && throw(ArgumentError("temperature lower bound must be positive"))
    variable===:vturb && lo<0 && throw(ArgumentError("vturb lower bound must be non-negative"))
    ControlMapSpec{T}(variable,Array(field.values),copy(field.log_tau_nodes),lo,hi,s)
end

"""Deterministic packing layout for the small global control maps.

Only these coarse maps are replicated. Atmospheric, population, and spectral
payloads remain rank-owned distributed tiles.
"""
struct ControlMapLayout{T<:AbstractFloat}
    specs::Vector{ControlMapSpec{T}}
    ranges::Vector{UnitRange{Int}}
    parameter_count::Int
end

function ControlMapLayout(specs::AbstractVector{ControlMapSpec{T}}) where T<:AbstractFloat
    isempty(specs) && throw(ArgumentError("at least one control map is required"))
    variables=getfield.(specs,:variable)
    length(unique(variables))==length(variables) || throw(ArgumentError("control-map variables must be unique"))
    ranges=UnitRange{Int}[]; first_index=1
    for spec in specs
        last_index=first_index+length(spec.initial)-1
        push!(ranges,first_index:last_index); first_index=last_index+1
    end
    ControlMapLayout{T}(collect(specs),ranges,first_index-1)
end
ControlMapLayout(specs::ControlMapSpec{T}...) where T<:AbstractFloat=ControlMapLayout(collect(specs))

"""Pack the configured initial physical controls in layout order."""
function initial_parameters(layout::ControlMapLayout{T}) where T
    out=Vector{T}(undef,layout.parameter_count)
    for (spec,range) in zip(layout.specs,layout.ranges)
        out[range].=vec(spec.initial)
    end
    out
end

function _check_parameter_length(layout,parameters)
    length(parameters)==layout.parameter_count || throw(DimensionMismatch(
        "parameter vector has length $(length(parameters)); expected $(layout.parameter_count)"))
end

"""Project a physical parameter vector onto all declared block bounds."""
function project_parameters!(parameters::AbstractVector,layout::ControlMapLayout)
    _check_parameter_length(layout,parameters)
    for (spec,range) in zip(layout.specs,layout.ranges)
        @views clamp!(parameters[range],spec.lower,spec.upper)
    end
    parameters
end

"""Return dimensionless optimizer coordinates (`physical_value / block_scale`)."""
function scaled_parameters(layout::ControlMapLayout{T},parameters::AbstractVector) where T
    _check_parameter_length(layout,parameters); out=T.(parameters)
    for (spec,range) in zip(layout.specs,layout.ranges)
        @views out[range]./=spec.scale
    end
    out
end

function _physical_from_scaled(layout::ControlMapLayout{T},scaled::AbstractVector) where T
    _check_parameter_length(layout,scaled); out=T.(scaled)
    for (spec,range) in zip(layout.specs,layout.ranges)
        @views out[range].*=spec.scale
    end
    project_parameters!(out,layout)
end

"""View one parameter block as a `NodeField` in physical units."""
function parameter_nodefield(layout::ControlMapLayout{T},parameters::AbstractVector,
                             variable::Symbol) where T
    _check_parameter_length(layout,parameters)
    index=findfirst(spec->spec.variable===variable,layout.specs)
    index===nothing && throw(KeyError(variable))
    spec=layout.specs[index]; values=reshape(T.(parameters[layout.ranges[index]]),size(spec.initial))
    NodeField(values,spec.log_tau_nodes)
end

"""Interpolate a solved set of controls onto a finer control-map layout."""
function refine_control_maps(layout::ControlMapLayout{T},parameters::AbstractVector;
        shapes::AbstractDict{Symbol,<:Tuple}=Dict{Symbol,Tuple{Int,Int}}(),
        log_tau_nodes::AbstractDict{Symbol,<:AbstractVector}=Dict{Symbol,Vector{T}}()) where T
    _check_parameter_length(layout,parameters); refined=ControlMapSpec{T}[]
    for spec in layout.specs
        ncx,ncy=get(shapes,spec.variable,size(spec.initial)[2:3])
        ncx>0 && ncy>0 || throw(ArgumentError("refined control dimensions must be positive"))
        nodes=T.(get(log_tau_nodes,spec.variable,spec.log_tau_nodes))
        length(nodes)>=2 || throw(ArgumentError("refined controls require at least two vertical nodes"))
        x=ncx==1 ? T[0] : collect(range(zero(T),one(T),length=ncx))
        y=ncy==1 ? T[0] : collect(range(zero(T),one(T),length=ncy))
        target=Grid3D(nodes,x,y)
        values=expand_nodes(parameter_nodefield(layout,parameters,spec.variable),target)
        field=NodeField(values,nodes)
        push!(refined,ControlMapSpec(spec.variable,field;lower=spec.lower,upper=spec.upper,scale=spec.scale))
    end
    new_layout=ControlMapLayout(refined)
    (layout=new_layout,parameters=initial_parameters(new_layout))
end

function _control_destination(atmosphere::Atmosphere3D,variable::Symbol)
    variable===:temperature && return atmosphere.temperature
    variable===:vlos && return atmosphere.vz
    variable in (:vx,:vy,:vz,:vturb) && return getfield(atmosphere,variable)
    if variable in (:Bx,:By,:Bz)
        atmosphere.magnetic_field===nothing && throw(ArgumentError("$variable control requested but B is unavailable"))
        return getfield(atmosphere.magnetic_field,variable)
    end
    throw(ArgumentError("unsupported control variable $variable"))
end

"""Apply global coarse controls directly to the current rank-owned tile."""
function apply_control_maps!(distributed::DistributedAtmosphere,layout::ControlMapLayout,
                             parameters::AbstractVector)
    _check_parameter_length(layout,parameters)
    for spec in layout.specs
        field=parameter_nodefield(layout,parameters,spec.variable)
        expanded=expand_nodes(field,distributed.global_grid,distributed.tile)
        copyto!(_control_destination(distributed.local_atmosphere,spec.variable),expanded)
    end
    all(isfinite,distributed.local_atmosphere.temperature) || throw(ArgumentError("control expansion produced invalid temperature"))
    minimum(distributed.local_atmosphere.temperature)>0 || throw(ArgumentError("control expansion produced non-positive temperature"))
    distributed
end

struct _ThermodynamicTrialReference{A}
    pgas::A; rho::A; ne::A; z::A
end
_trial_copy(value)=value===nothing ? nothing : copy(value)

"""Distributed Phase 5 objective and reusable forward workspace."""
struct DistributedInversionProblem{M,W,D,O,R,T,A}
    model::M
    workspace::W
    distributed::D
    observation::O
    regularization::R
    dx_m::T
    dy_m::T
    thermodynamic_reference::A
    active_residual_count::Int
end

function DistributedInversionProblem(model,workspace,distributed::DistributedAtmosphere,
        observation::ObservationCube,regularization::RegularizationSpec,dx_m::Real,dy_m::Real,
        context::ParallelContext)
    size(workspace.output.data)==size(observation.spectrum.data) || throw(DimensionMismatch(
        "rank-local synthesis and observation shapes differ"))
    dx_m>0 && dy_m>0 || throw(ArgumentError("objective pixel spacings must be positive"))
    local_count=count(>(0),observation.inversion_weights)
    active=allreduce_sum(local_count,context)
    active>0 || throw(ArgumentError("at least one inversion weight must be positive"))
    a=distributed.local_atmosphere
    reference=_ThermodynamicTrialReference(_trial_copy(a.pgas),_trial_copy(a.rho),_trial_copy(a.ne),_trial_copy(a.z))
    T=eltype(a.temperature)
    DistributedInversionProblem{typeof(model),typeof(workspace),typeof(distributed),typeof(observation),
        typeof(regularization),T,typeof(reference)}(model,workspace,distributed,observation,regularization,
        T(dx_m),T(dy_m),reference,active)
end

function _restore_trial_reference!(problem::DistributedInversionProblem)
    atmosphere=problem.distributed.local_atmosphere; ref=problem.thermodynamic_reference
    for name in (:pgas,:rho,:ne,:z)
        source=getfield(ref,name)
        if source===nothing
            setfield!(atmosphere,name,nothing)
        else
            destination=getfield(atmosphere,name)
            destination===nothing ? setfield!(atmosphere,name,copy(source)) : copyto!(destination,source)
        end
    end
    atmosphere
end

struct ObjectiveComponents{T<:AbstractFloat}
    total::T
    data::T
    regularization::T
    regularization_terms::Dict{Symbol,T}
end

struct ObjectiveEvaluation{T<:AbstractFloat,F,G}
    components::ObjectiveComponents{T}
    force_balance::F
    timings::G
end

"""Evaluate normalized chi-square plus the configured 3D regularization.

All ranks call this function. Spectral and regularization terms are reduced
over MPI; the returned components are therefore identical on every rank.
"""
function evaluate_objective!(problem::DistributedInversionProblem,layout::ControlMapLayout,
        parameters::AbstractVector,context::ParallelContext)
    trial=collect(parameters); project_parameters!(trial,layout)
    trial==parameters || throw(ArgumentError("objective parameters violate declared bounds"))
    _restore_trial_reference!(problem)
    apply_control_maps!(problem.distributed,layout,trial)
    forward_result=forward!(problem.workspace,problem.model,problem.distributed,context)
    data=distributed_chi2(forward_result.spectrum,problem.observation,context)/problem.active_residual_count
    reg=distributed_regularization_penalty(problem.distributed,problem.regularization,
        problem.dx_m,problem.dy_m,context)
    T=promote_type(typeof(data),typeof(reg.total))
    terms=Dict{Symbol,T}(key=>T(value) for (key,value) in reg.terms)
    components=ObjectiveComponents(T(data+reg.total),T(data),T(reg.total),terms)
    all(isfinite,(components.total,components.data,components.regularization)) ||
        throw(ErrorException("objective contains NaN or Inf"))
    ObjectiveEvaluation(components,forward_result.force_balance,forward_result.timings)
end

Base.@kwdef struct PrototypeSolverOptions
    max_iterations::Int=20
    initial_step::Float64=0.25
    step_shrink::Float64=0.5
    minimum_step::Float64=1e-3
    improvement_tolerance::Float64=1e-8
    maximum_evaluations::Int=typemax(Int)
    checkpoint_every::Int=1
    checkpoint_path::String=""
end

function _validate(options::PrototypeSolverOptions)
    options.max_iterations>=0 || throw(ArgumentError("max_iterations must be non-negative"))
    options.initial_step>0 || throw(ArgumentError("initial_step must be positive"))
    0<options.step_shrink<1 || throw(ArgumentError("step_shrink must be between zero and one"))
    options.minimum_step>0 || throw(ArgumentError("minimum_step must be positive"))
    options.improvement_tolerance>=0 || throw(ArgumentError("improvement_tolerance must be non-negative"))
    options.maximum_evaluations>0 || throw(ArgumentError("maximum_evaluations must be positive"))
    options.checkpoint_every>0 || throw(ArgumentError("checkpoint_every must be positive"))
    options
end

struct PrototypeIterationRecord{T<:AbstractFloat}
    iteration::Int
    evaluations::Int
    total::T
    data::T
    regularization::T
    maximum_scaled_step::T
    accepted::Bool
    accepted_coordinate::Int
    accepted_direction::Int
end

mutable struct PrototypeSolverState{T<:AbstractFloat}
    schema_version::VersionNumber
    iteration::Int
    evaluations::Int
    parameters::Vector{T}
    scaled_steps::Vector{T}
    objective::T
    data_term::T
    regularization_term::T
    history::Vector{PrototypeIterationRecord{T}}
    converged::Bool
end

struct PrototypeInversionResult{T<:AbstractFloat,E}
    state::PrototypeSolverState{T}
    evaluation::E
end

"""Write the rank-0 Phase 5 convergence bundle as TOML plus CSV history."""
function write_prototype_diagnostics(directory::AbstractString,result::PrototypeInversionResult,
        layout::ControlMapLayout,context::ParallelContext;metadata::AbstractDict=Dict())
    isroot(context) || return nothing
    root=abspath(directory); mkpath(root)
    summary_path=joinpath(root,"summary.toml")
    history_path=joinpath(root,"objective_history.csv")
    controls=[Dict{String,Any}(
        "variable"=>string(spec.variable),"shape"=>collect(size(spec.initial)),
        "log_tau_nodes"=>spec.log_tau_nodes,"lower"=>spec.lower,"upper"=>spec.upper,
        "scale"=>spec.scale,"final_values"=>collect(result.state.parameters[range]))
        for (spec,range) in zip(layout.specs,layout.ranges)]
    summary=Dict{String,Any}(
        "schema_version"=>string(result.state.schema_version),
        "iteration"=>result.state.iteration,"evaluations"=>result.state.evaluations,
        "converged"=>result.state.converged,"objective"=>result.state.objective,
        "data_term"=>result.state.data_term,"regularization_term"=>result.state.regularization_term,
        "mpi_ranks"=>context.size,"threads_per_rank"=>Threads.nthreads(),
        "controls"=>controls,"metadata"=>Dict(string(k)=>v for (k,v) in metadata))
    open(summary_path,"w") do io; TOML.print(io,summary;sorted=true); end
    open(history_path,"w") do io
        println(io,"iteration,evaluations,total,data,regularization,maximum_scaled_step,accepted,coordinate,direction")
        for record in result.state.history
            println(io,join((record.iteration,record.evaluations,record.total,record.data,
                record.regularization,record.maximum_scaled_step,record.accepted,
                record.accepted_coordinate,record.accepted_direction),','))
        end
    end
    (summary=summary_path,history=history_path)
end

function _layout_signature(layout::ControlMapLayout)
    join((string(spec.variable,"|",size(spec.initial),"|",join(repr.(spec.log_tau_nodes),","),
        "|",repr(spec.lower),"|",repr(spec.upper),"|",repr(spec.scale)) for spec in layout.specs),";")
end

function _save_prototype_checkpoint(path,layout,state,manifest,context)
    isempty(path) && return nothing
    if isroot(context)
        mkpath(dirname(abspath(path)))
        checkpoint!(path,(kind=:phase5_prototype,layout_signature=_layout_signature(layout),state=state);
            manifest=manifest)
    end
    barrier(context); path
end

function _restore_prototype_checkpoint(path,layout,manifest,context)
    isempty(path) && throw(ArgumentError("restart requires checkpoint_path"))
    payload=if isroot(context)
        restored=restore_checkpoint(path;expected=manifest)
        state_payload=restored.state
        state_payload.kind===:phase5_prototype || throw(ArgumentError("checkpoint is not a Phase 5 prototype state"))
        state_payload.layout_signature==_layout_signature(layout) || throw(ArgumentError(
            "checkpoint control-map layout differs from requested layout"))
        state_payload.state
    else
        nothing
    end
    mpi_broadcast(payload,context)
end

function _candidate(layout,state,coordinate,direction)
    scaled=scaled_parameters(layout,state.parameters)
    scaled[coordinate]+=direction*state.scaled_steps[coordinate]
    _physical_from_scaled(layout,scaled)
end

"""Run the deterministic bounded Phase 5 pattern-search prototype.

Rank 0 proposes and accepts trials; every rank evaluates the same distributed
objective. `restart=true` resumes a completed-iteration checkpoint and can use
a different compatible MPI/thread topology.
"""
function prototype_invert!(problem::DistributedInversionProblem,layout::ControlMapLayout,
        context::ParallelContext;initial=initial_parameters(layout),
        options::PrototypeSolverOptions=PrototypeSolverOptions(),restart::Bool=false,
        manifest::CapabilityManifest=problem.model.capabilities)
    _validate(options)
    state,current=if restart
        restored=_restore_prototype_checkpoint(options.checkpoint_path,layout,manifest,context)
        restored_current=evaluate_objective!(problem,layout,restored.parameters,context)
        isapprox(restored_current.components.total,restored.objective;rtol=1e-12,atol=1e-12) ||
            throw(ErrorException("restart objective differs from checkpointed objective"))
        (restored,restored_current)
    else
        parameters=mpi_broadcast(isroot(context) ? collect(initial) : nothing,context)
        _check_parameter_length(layout,parameters); project_parameters!(parameters,layout)
        evaluation=evaluate_objective!(problem,layout,parameters,context)
        T=eltype(parameters)
        created=PrototypeSolverState(v"1.0.0",0,1,parameters,fill(T(options.initial_step),length(parameters)),
            T(evaluation.components.total),T(evaluation.components.data),T(evaluation.components.regularization),
            PrototypeIterationRecord{T}[],false)
        (created,evaluation)
    end
    _check_parameter_length(layout,state.parameters)
    state.schema_version==v"1.0.0" || throw(ArgumentError("unsupported prototype checkpoint schema $(state.schema_version)"))
    state.iteration<=options.max_iterations || throw(ArgumentError(
        "checkpoint iteration $(state.iteration) exceeds requested max_iterations $(options.max_iterations)"))
    for iteration in state.iteration+1:options.max_iterations
        state.evaluations>=options.maximum_evaluations && break
        best_total=state.objective; best_evaluation=current; best_parameters=state.parameters
        best_coordinate=0; best_direction=0
        for coordinate in eachindex(state.parameters),direction in (-1,1)
            state.evaluations>=options.maximum_evaluations && break
            candidate=mpi_broadcast(isroot(context) ? _candidate(layout,state,coordinate,direction) : nothing,context)
            candidate==state.parameters && continue
            evaluation=evaluate_objective!(problem,layout,candidate,context); state.evaluations+=1
            threshold=options.improvement_tolerance*max(abs(best_total),1.0)
            if evaluation.components.total<best_total-threshold
                best_total=evaluation.components.total; best_evaluation=evaluation
                best_parameters=candidate; best_coordinate=coordinate; best_direction=direction
            end
        end
        decision=mpi_broadcast(isroot(context) ? (parameters=best_parameters,coordinate=best_coordinate,
            direction=best_direction,total=best_total) : nothing,context)
        accepted=decision.coordinate!=0
        if accepted
            state.parameters=decision.parameters; current=best_evaluation
            state.objective=current.components.total; state.data_term=current.components.data
            state.regularization_term=current.components.regularization
        else
            state.scaled_steps.*=options.step_shrink
            current=evaluate_objective!(problem,layout,state.parameters,context); state.evaluations+=1
        end
        state.iteration=iteration
        push!(state.history,PrototypeIterationRecord(iteration,state.evaluations,state.objective,
            state.data_term,state.regularization_term,maximum(state.scaled_steps),accepted,
            decision.coordinate,decision.direction))
        state.converged=maximum(state.scaled_steps)<=options.minimum_step
        if iteration%options.checkpoint_every==0 || state.converged
            _save_prototype_checkpoint(options.checkpoint_path,layout,state,manifest,context)
        end
        state.converged && break
    end
    current=evaluate_objective!(problem,layout,state.parameters,context)
    isempty(options.checkpoint_path) || _save_prototype_checkpoint(options.checkpoint_path,layout,state,manifest,context)
    PrototypeInversionResult(state,current)
end

struct DirectionalDerivativeEstimate{T<:AbstractFloat}
    step::T
    fplus::T
    fminus::T
    derivative::T
end

struct DirectionalDerivativeReport{T<:AbstractFloat}
    estimates::Vector{DirectionalDerivativeEstimate{T}}
    relative_change::T
end

"""Centered finite-difference validation in scaled optimizer coordinates.

The direction is normalized internally. Every `x +/- h*d` point must remain
inside the declared bounds; clipping would invalidate the centered estimate.
The central atmosphere is restored before returning.
"""
function centered_directional_validation(problem::DistributedInversionProblem,
        layout::ControlMapLayout{T},parameters::AbstractVector,direction::AbstractVector,
        context::ParallelContext;steps=(1e-3,5e-4)) where T
    _check_parameter_length(layout,parameters); _check_parameter_length(layout,direction)
    all(>(0),steps) || throw(ArgumentError("finite-difference steps must be positive"))
    norm_direction=norm(direction); norm_direction>0 || throw(ArgumentError("direction cannot be zero"))
    d=T.(direction)./T(norm_direction); center=scaled_parameters(layout,parameters)
    estimates=DirectionalDerivativeEstimate{T}[]
    for raw_h in steps
        h=T(raw_h); plus=_physical_from_scaled(layout,center.+h.*d); minus=_physical_from_scaled(layout,center.-h.*d)
        isapprox(scaled_parameters(layout,plus),center.+h.*d;rtol=1e-12,atol=10eps(T)) ||
            throw(ArgumentError("positive directional trial crosses a bound"))
        isapprox(scaled_parameters(layout,minus),center.-h.*d;rtol=1e-12,atol=10eps(T)) ||
            throw(ArgumentError("negative directional trial crosses a bound"))
        fplus=evaluate_objective!(problem,layout,plus,context).components.total
        fminus=evaluate_objective!(problem,layout,minus,context).components.total
        push!(estimates,DirectionalDerivativeEstimate(h,T(fplus),T(fminus),T((fplus-fminus)/(2h))))
    end
    evaluate_objective!(problem,layout,parameters,context)
    relative=length(estimates)<2 ? zero(T) : abs(estimates[end].derivative-estimates[end-1].derivative)/
        max(abs(estimates[end].derivative),abs(estimates[end-1].derivative),eps(T))
    DirectionalDerivativeReport(estimates,T(relative))
end
