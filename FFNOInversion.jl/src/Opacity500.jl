abstract type AbstractOpacity500 end

"""Production 500 nm mass-opacity using Wittmann populations and native Julia continuum physics."""
struct WittmannOpacity500 <: AbstractOpacity500
    eos::WittmannEOS
end

function opacity500!(kappa::Array{Float64,3},model::WittmannOpacity500,
                     temperature::Array{Float64,3},pgas::Array{Float64,3},rho,ne)
    size(kappa)==size(temperature)==size(pgas)==size(rho)==size(ne) || throw(DimensionMismatch("opacity arrays differ"))
    state=continuum_state(model.eos,vec(temperature),vec(pgas))
    extinction=continuum_extinction_m(state,vec(temperature),5000.0)
    kappa .= reshape(extinction,size(kappa))./reshape(state.rho_kg_m3,size(kappa))
    all(isfinite,kappa)&&all(>(0),kappa) || throw(ErrorException("native continuum returned invalid 500 nm opacity"))
    kappa
end

"""Positive constant mass extinction at 500 nm in m^2 kg^-1.

This reference law validates manufactured reconstruction tests. Production runs
use `WittmannOpacity500`.
"""
struct ReferenceOpacity500{T<:AbstractFloat} <: AbstractOpacity500
    kappa_m2_kg::T
end
ReferenceOpacity500(;kappa_m2_kg=0.01)=ReferenceOpacity500(kappa_m2_kg)

function opacity500!(kappa,model::ReferenceOpacity500,temperature,pgas,rho,ne)
    model.kappa_m2_kg>0 || throw(ArgumentError("500 nm mass opacity must be positive"))
    size(kappa)==size(temperature)==size(pgas)==size(rho)==size(ne) || throw(DimensionMismatch("opacity arrays differ"))
    fill!(kappa,model.kappa_m2_kg)
end
