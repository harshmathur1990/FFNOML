"""A matrix-free linearization represented only by JVP and VJP actions.

The callbacks have signatures `jvp!(output, input_tangent)` and
`vjp!(input_cotangent, output_cotangent)`.  No dense Jacobian is formed.
"""
struct MatrixFreeLinearization{J,V}
    input_length::Int
    output_length::Int
    jvp!::J
    vjp!::V
    function MatrixFreeLinearization(input_length::Int,output_length::Int,jvp!::J,vjp!::V) where {J,V}
        input_length>0 || throw(ArgumentError("linearization input length must be positive"))
        output_length>0 || throw(ArgumentError("linearization output length must be positive"))
        new{J,V}(input_length,output_length,jvp!,vjp!)
    end
end

function apply_jvp(linearization::MatrixFreeLinearization,
                   tangent::AbstractVector{T}) where T<:AbstractFloat
    length(tangent)==linearization.input_length || throw(DimensionMismatch(
        "JVP tangent length differs from linearization input length"))
    output=zeros(T,linearization.output_length)
    linearization.jvp!(output,tangent)
    all(isfinite,output) || throw(ErrorException("JVP produced NaN or Inf"))
    output
end

function apply_vjp(linearization::MatrixFreeLinearization,
                   cotangent::AbstractVector{T}) where T<:AbstractFloat
    length(cotangent)==linearization.output_length || throw(DimensionMismatch(
        "VJP cotangent length differs from linearization output length"))
    output=zeros(T,linearization.input_length)
    linearization.vjp!(output,cotangent)
    all(isfinite,output) || throw(ErrorException("VJP produced NaN or Inf"))
    output
end

struct DotProductReport{T<:AbstractFloat}
    jvp_dot::T
    vjp_dot::T
    absolute_error::T
    relative_error::T
    passed::Bool
end

"""Check `<J*v,w> == <v,J'*w>` for a matrix-free module pullback."""
function dot_product_validation(linearization::MatrixFreeLinearization,
        tangent::AbstractVector{T},cotangent::AbstractVector{T};
        rtol::Real=1e-10,atol::Real=1e-12) where T<:AbstractFloat
    rtol>=0 && atol>=0 || throw(ArgumentError("dot-product tolerances must be non-negative"))
    jv=apply_jvp(linearization,tangent); jtw=apply_vjp(linearization,cotangent)
    left=dot(jv,cotangent); right=dot(tangent,jtw)
    absolute=abs(left-right); relative=absolute/max(abs(left),abs(right),eps(T))
    DotProductReport(T(left),T(right),T(absolute),T(relative),absolute<=atol+rtol*max(abs(left),abs(right)))
end

function _node_expansion_vjp_local!(node_cotangent::AbstractArray{T,3},
        atmosphere_cotangent::AbstractArray{T,3},spec::ControlMapSpec{T},
        grid::Grid3D{T},tile::Tile2D) where T<:AbstractFloat
    size(node_cotangent)==size(spec.initial) || throw(DimensionMismatch(
        "node cotangent shape differs from control map"))
    size(atmosphere_cotangent)==(length(grid.log_tau500),length(tile.xrange),length(tile.yrange)) ||
        throw(DimensionMismatch("atmosphere cotangent shape differs from rank-owned tile"))
    fill!(node_cotangent,zero(T))
    _,ncx,ncy=size(spec.initial)
    cx=ncx==1 ? T[zero(T)] : collect(range(zero(T),one(T),length=ncx))
    cy=ncy==1 ? T[zero(T)] : collect(range(zero(T),one(T),length=ncy))
    xn=length(grid.x)==1 ? T[zero(T)] : collect(range(zero(T),one(T),length=length(grid.x)))
    yn=length(grid.y)==1 ? T[zero(T)] : collect(range(zero(T),one(T),length=length(grid.y)))
    for iz in eachindex(grid.log_tau500),(ii,ix) in enumerate(tile.xrange),(jj,iy) in enumerate(tile.yrange)
        z0,z1,wz=_bracket(spec.log_tau_nodes,grid.log_tau500[iz])
        x0,x1,wx=ncx==1 ? (1,1,zero(T)) : _bracket(cx,xn[ix])
        y0,y1,wy=ncy==1 ? (1,1,zero(T)) : _bracket(cy,yn[iy])
        value=atmosphere_cotangent[iz,ii,jj]
        node_cotangent[z0,x0,y0]+=value*(1-wz)*(1-wx)*(1-wy)
        node_cotangent[z0,x0,y1]+=value*(1-wz)*(1-wx)*wy
        node_cotangent[z0,x1,y0]+=value*(1-wz)*wx*(1-wy)
        node_cotangent[z0,x1,y1]+=value*(1-wz)*wx*wy
        node_cotangent[z1,x0,y0]+=value*wz*(1-wx)*(1-wy)
        node_cotangent[z1,x0,y1]+=value*wz*(1-wx)*wy
        node_cotangent[z1,x1,y0]+=value*wz*wx*(1-wy)
        node_cotangent[z1,x1,y1]+=value*wz*wx*wy
    end
    node_cotangent
end

"""Adjoint of rank-local node expansion, reduced onto the replicated control map."""
function node_expansion_vjp(atmosphere_cotangent::AbstractArray{T,3},
        spec::ControlMapSpec{T},grid::Grid3D{T},tile::Tile2D,
        context::ParallelContext) where T<:AbstractFloat
    local_cotangent=zeros(T,size(spec.initial))
    _node_expansion_vjp_local!(local_cotangent,atmosphere_cotangent,spec,grid,tile)
    if context.enabled
        _assert_mpi_thread(context)
        MPI.Allreduce!(local_cotangent,MPI.SUM,context.comm)
    end
    local_cotangent
end

"""Accumulate one atmospheric pullback into dimensionless optimizer coordinates.

The physical control is `scale * scaled_control`, so this function includes
that final chain-rule factor after the MPI reduction.
"""
function accumulate_control_vjp!(scaled_gradient::AbstractVector{T},
        layout::ControlMapLayout{T},variable::Symbol,
        atmosphere_cotangent::AbstractArray{T,3},distributed::DistributedAtmosphere,
        context::ParallelContext) where T<:AbstractFloat
    _check_parameter_length(layout,scaled_gradient)
    index=findfirst(spec->spec.variable===variable,layout.specs)
    index===nothing && throw(KeyError(variable))
    spec=layout.specs[index]
    node_cotangent=node_expansion_vjp(atmosphere_cotangent,spec,distributed.global_grid,
        distributed.tile,context)
    @views scaled_gradient[layout.ranges[index]].+=vec(node_cotangent).*spec.scale
    scaled_gradient
end

"""Interface for a complete matrix-free objective pullback.

Implementations must return an `ObjectiveGradientEvaluation`.  Gradients use
dimensionless scaled control coordinates and must already contain all MPI
reductions.  This is the seam used by the rank-0 FFNO GPU VJP service.
"""
abstract type AbstractObjectiveGradient end
function objective_gradient! end

struct ObjectiveGradientEvaluation{T<:AbstractFloat,E}
    evaluation::E
    gradient::Vector{T}
    forward_evaluations::Int
    function ObjectiveGradientEvaluation(evaluation::E,gradient::AbstractVector{T},
            forward_evaluations::Int) where {T<:AbstractFloat,E}
        forward_evaluations>0 || throw(ArgumentError("gradient evaluation must use at least one forward evaluation"))
        all(isfinite,gradient) || throw(ErrorException("objective gradient contains NaN or Inf"))
        new{T,E}(evaluation,collect(gradient),forward_evaluations)
    end
end

"""Retained bounded finite-difference oracle; validation only, not production."""
struct FiniteDifferenceObjectiveGradient{T<:AbstractFloat} <: AbstractObjectiveGradient
    step::T
    function FiniteDifferenceObjectiveGradient(step::T) where T<:AbstractFloat
        isfinite(step) && step>0 || throw(ArgumentError("finite-difference gradient step must be positive"))
        new{T}(step)
    end
end
FiniteDifferenceObjectiveGradient(;step::Real=1e-4)=FiniteDifferenceObjectiveGradient(Float64(step))

function _scaled_bounds(layout::ControlMapLayout{T}) where T
    lower=Vector{T}(undef,layout.parameter_count); upper=similar(lower)
    for (spec,range) in zip(layout.specs,layout.ranges)
        @views lower[range].=spec.lower/spec.scale
        @views upper[range].=spec.upper/spec.scale
    end
    lower,upper
end

function objective_gradient!(backend::FiniteDifferenceObjectiveGradient,
        problem::DistributedInversionProblem,layout::ControlMapLayout{T},
        parameters::AbstractVector,context::ParallelContext) where T
    _check_parameter_length(layout,parameters)
    center=scaled_parameters(layout,parameters); lower,upper=_scaled_bounds(layout)
    central=evaluate_objective!(problem,layout,parameters,context); f0=T(central.components.total)
    gradient=zeros(T,length(center)); evaluations=1
    for coordinate in eachindex(center)
        hp=min(T(backend.step),upper[coordinate]-center[coordinate])
        hm=min(T(backend.step),center[coordinate]-lower[coordinate])
        tolerance=eps(T)*max(abs(center[coordinate]),one(T))*10
        if hp>tolerance && hm>tolerance
            plus=copy(center); plus[coordinate]+=hp
            minus=copy(center); minus[coordinate]-=hm
            fp=evaluate_objective!(problem,layout,_physical_from_scaled(layout,plus),context).components.total
            fm=evaluate_objective!(problem,layout,_physical_from_scaled(layout,minus),context).components.total
            gradient[coordinate]=(hm^2*T(fp)+(hp^2-hm^2)*f0-hp^2*T(fm))/(hp*hm*(hp+hm))
            evaluations+=2
        elseif hp>tolerance
            plus=copy(center); plus[coordinate]+=hp
            fp=evaluate_objective!(problem,layout,_physical_from_scaled(layout,plus),context).components.total
            gradient[coordinate]=(T(fp)-f0)/hp; evaluations+=1
        elseif hm>tolerance
            minus=copy(center); minus[coordinate]-=hm
            fm=evaluate_objective!(problem,layout,_physical_from_scaled(layout,minus),context).components.total
            gradient[coordinate]=(f0-T(fm))/hm; evaluations+=1
        else
            gradient[coordinate]=zero(T)
        end
    end
    # Every gradient call leaves the accepted atmosphere, spectrum and
    # thermodynamics at the central point, including after one-sided stencils.
    restored=evaluate_objective!(problem,layout,parameters,context); evaluations+=1
    ObjectiveGradientEvaluation(restored,gradient,evaluations)
end

struct TaylorRemainderSample{T<:AbstractFloat}
    step::T
    objective::T
    first_order_error::T
end

struct GradientTaylorReport{T<:AbstractFloat,G}
    gradient_evaluation::G
    directional_derivative::T
    samples::Vector{TaylorRemainderSample{T}}
    observed_orders::Vector{T}
end

"""First-order Taylor test for a complete objective pullback."""
function gradient_taylor_validation(backend::AbstractObjectiveGradient,
        problem::DistributedInversionProblem,layout::ControlMapLayout{T},
        parameters::AbstractVector,direction::AbstractVector,context::ParallelContext;
        steps=(1e-2,5e-3,2.5e-3)) where T
    _check_parameter_length(layout,parameters); _check_parameter_length(layout,direction)
    length(steps)>=2 && all(>(0),steps) || throw(ArgumentError("Taylor test needs at least two positive steps"))
    norm_direction=norm(direction); norm_direction>0 || throw(ArgumentError("Taylor direction cannot be zero"))
    d=T.(direction)./T(norm_direction); center=scaled_parameters(layout,parameters)
    lower,upper=_scaled_bounds(layout)
    gradient_evaluation=objective_gradient!(backend,problem,layout,parameters,context)
    length(gradient_evaluation.gradient)==length(parameters) || throw(DimensionMismatch(
        "objective gradient length differs from controls"))
    derivative=dot(gradient_evaluation.gradient,d); f0=T(gradient_evaluation.evaluation.components.total)
    samples=TaylorRemainderSample{T}[]
    for raw_step in steps
        h=T(raw_step); trial=center.+h.*d
        all(trial.>=lower) && all(trial.<=upper) || throw(ArgumentError("Taylor trial crosses a control bound"))
        evaluation=evaluate_objective!(problem,layout,_physical_from_scaled(layout,trial),context)
        objective=T(evaluation.components.total)
        push!(samples,TaylorRemainderSample(h,objective,abs(objective-f0-h*derivative)))
    end
    evaluate_objective!(problem,layout,parameters,context)
    orders=T[]
    for i in 1:length(samples)-1
        e1=max(samples[i].first_order_error,eps(T)); e2=max(samples[i+1].first_order_error,eps(T))
        push!(orders,log(e1/e2)/log(samples[i].step/samples[i+1].step))
    end
    GradientTaylorReport(gradient_evaluation,T(derivative),samples,orders)
end
