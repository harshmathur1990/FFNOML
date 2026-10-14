const VERTICAL_PARAMETER_ORDER = (:temperature,:vlos,:vturb,:B,:inc,:azi,:pgas_boundary)

"""STiC-style seven-slot vertical regularization.

Types: 0 none, 1 first derivative, 2 deviation from depth mean,
3 deviation from zero, 4 second derivative. `regularize` multiplies all seven
relative `weights`. Temperature types 2/3 are normalized to 0.
"""
struct VerticalRegularizationSpec{T<:AbstractFloat}
    types::NTuple{7,Int}
    regularize::T
    weights::NTuple{7,T}
    function VerticalRegularizationSpec(types,regularize::T,weights) where T<:AbstractFloat
        length(types) == 7 && length(weights) == 7 || throw(ArgumentError("vertical regularization needs seven types and seven weights"))
        normalized = ntuple(i -> begin
            typ = Int(types[i])
            0 <= typ <= 4 || throw(ArgumentError("vertical regularization type must be in 0:4"))
            i == 1 && typ in (2,3) ? 0 : typ
        end,7)
        w = ntuple(i -> T(weights[i]),7)
        regularize >= 0 && all(>=(zero(T)),w) || throw(ArgumentError("regularization strengths must be non-negative"))
        new{T}(normalized,regularize,w)
    end
end

Base.@kwdef struct RegularizationSpec{T<:AbstractFloat}
    vertical::VerticalRegularizationSpec{T} = VerticalRegularizationSpec(ntuple(_->0,7),zero(T),ntuple(_->one(T),7))
    horizontal::Dict{Symbol,T} = Dict{Symbol,T}()
    scales::Dict{Symbol,T} = Dict{Symbol,T}()
    horizontal_order::Int = 1
    function RegularizationSpec(vertical::VerticalRegularizationSpec{T},horizontal::Dict{Symbol,T},
                                scales::Dict{Symbol,T},horizontal_order::Int) where T<:AbstractFloat
        horizontal_order in (1,2) || throw(ArgumentError("horizontal_order must be 1 or 2"))
        all(>=(zero(T)),values(horizontal)) || throw(ArgumentError("horizontal regularization weights must be non-negative"))
        all(>(zero(T)),values(scales)) || throw(ArgumentError("regularization scales must be positive"))
        new{T}(vertical,horizontal,scales,horizontal_order)
    end
end

function _atmospheric_variable(atmosphere::Atmosphere3D, variable::Symbol)
    variable === :temperature && return atmosphere.temperature
    variable === :vlos && return atmosphere.vz
    variable in (:vx,:vy,:vz,:vturb) && return getfield(atmosphere,variable)
    if variable in (:pgas,:rho,:ne,:z)
        value = getfield(atmosphere,variable)
        isnothing(value) && throw(ArgumentError("$variable is unavailable in atmosphere"))
        return value
    end
    if variable === :pgas_boundary
        isnothing(atmosphere.pgas) && throw(ArgumentError("pgas_boundary requested but pgas is unavailable"))
        return @view atmosphere.pgas[1:1,:,:]
    end
    if variable in (:Bx,:By,:Bz,:B,:inc,:azi)
        field = atmosphere.magnetic_field
        isnothing(field) && throw(ArgumentError("$variable requested but B is unavailable"))
        variable in (:Bx,:By,:Bz) && return getfield(field,variable)
        strength = sqrt.(field.Bx.^2 .+ field.By.^2 .+ field.Bz.^2)
        variable === :B && return strength
        variable === :inc && return acos.(clamp.(field.Bz ./ max.(strength,eps(eltype(strength))),-1,1))
        return atan.(field.By,field.Bx)
    end
    throw(ArgumentError("unsupported regularized variable $variable"))
end

_mean_square(a) = isempty(a) ? zero(eltype(a)) : sum(abs2,a)/length(a)

function _vertical_penalty(a,logtau,typ)
    typ == 0 && return 0.0
    typ == 2 && return _mean_square(a .- sum(a,dims=1)./size(a,1))
    typ == 3 && return _mean_square(a)
    size(a,1) >= 2 || return 0.0
    spacing = reshape(diff(logtau),:,1,1)
    slope = diff(a,dims=1) ./ spacing
    typ == 1 && return _mean_square(slope)
    size(a,1) >= 3 || return 0.0
    centers = (logtau[1:end-1] .+ logtau[2:end]) ./ 2
    _mean_square(diff(slope,dims=1) ./ reshape(diff(centers),:,1,1))
end

function _horizontal_penalty(a,dx,dy,order)
    gx = diff(a,dims=2) ./ dx; gy = diff(a,dims=3) ./ dy
    order == 1 && return _mean_square(gx)+_mean_square(gy)
    _mean_square(diff(gx,dims=2)./dx)+_mean_square(diff(gy,dims=3)./dy)
end

"""Return total and per-parameter regularization on the recovered atmosphere."""
function regularization_penalty(atmosphere::Atmosphere3D,spec::RegularizationSpec,dx_m::Real,dy_m::Real)
    dx_m > 0 && dy_m > 0 || throw(ArgumentError("dx_m and dy_m must be positive"))
    terms = Dict{Symbol,Float64}()
    for (i,variable) in enumerate(VERTICAL_PARAMETER_ORDER)
        typ = spec.vertical.types[i]
        typ == 0 && continue
        a = _atmospheric_variable(atmosphere,variable)
        scale = get(spec.scales,variable,one(eltype(a)))
        penalty = if variable === :pgas_boundary && typ == 1
            _mean_square(a./scale .- 1) # STiC-style boundary enhancement relative to its reference scale
        else
            _vertical_penalty(a./scale,atmosphere.grid.log_tau500,typ)
        end
        terms[variable] = spec.vertical.regularize*spec.vertical.weights[i]*penalty
    end
    for (variable,weight) in spec.horizontal
        a = _atmospheric_variable(atmosphere,variable)
        scale = get(spec.scales,variable,one(eltype(a)))
        terms[variable] = get(terms,variable,0.0)+weight*_horizontal_penalty(a./scale,dx_m,dy_m,spec.horizontal_order)
    end
    (total=sum(values(terms);init=0.0),terms=terms)
end
