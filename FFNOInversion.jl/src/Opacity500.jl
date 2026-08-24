abstract type AbstractOpacity500 end

"""Production 500 nm mass-opacity adapter backed by Wittmann/STiC `cop`."""
struct WittmannOpacity500 <: AbstractOpacity500
    eos::WittmannEOS
end

function opacity500!(kappa::Array{Float64,3},model::WittmannOpacity500,
                     temperature::Array{Float64,3},pgas::Array{Float64,3},rho,ne)
    size(kappa)==size(temperature)==size(pgas)==size(rho)==size(ne) || throw(DimensionMismatch("opacity arrays differ"))
    handle=Libdl.dlopen(model.eos.library)
    function_pointer=Libdl.dlsym(handle,:witt_opacity500_mass_from_pgas)
    status=ccall(function_pointer,Cint,(Cstring,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Csize_t),
        model.eos.partition_functions,temperature,pgas,kappa,length(kappa))
    status==0 || throw(ErrorException("Wittmann/STiC 500 nm opacity failed for one or more cells"))
    all(isfinite,kappa)&&all(>(0),kappa) || throw(ErrorException("Wittmann/STiC returned invalid 500 nm opacity"))
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
