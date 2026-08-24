abstract type AbstractEOS end

"""Wittmann EOS shared-library adapter. All public inputs and outputs use SI units."""
struct WittmannEOS <: AbstractEOS
    library::String
    partition_functions::String
    function WittmannEOS(library::AbstractString,partition_functions::AbstractString)
        isfile(library) || throw(ArgumentError("Wittmann library not found: $library"))
        isfile(partition_functions) || throw(ArgumentError("Wittmann partition-function file not found: $partition_functions"))
        new(abspath(library),abspath(partition_functions))
    end
end

"""Deterministic fixture EOS for manufactured tests; not a production replacement for Wittmann."""
struct IdealGasEOS{T<:AbstractFloat} <: AbstractEOS
    mean_molecular_weight::T
    electron_fraction::T
end
IdealGasEOS(;mean_molecular_weight=1.3,electron_fraction=1e-4)=IdealGasEOS(promote(mean_molecular_weight,electron_fraction)...)

function thermodynamics!(rho::Array{Float64,3},ne::Array{Float64,3},eos::WittmannEOS,
                         temperature::Array{Float64,3},pgas::Array{Float64,3})
    size(rho)==size(ne)==size(temperature)==size(pgas) || throw(DimensionMismatch("EOS arrays differ"))
    handle=Libdl.dlopen(eos.library)
    function_pointer=Libdl.dlsym(handle,:witt_thermodynamics_from_pgas)
    status=ccall(function_pointer,Cint,
        (Cstring,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Csize_t),
        eos.partition_functions,temperature,pgas,rho,ne,length(rho))
    status==0 || throw(ErrorException("Wittmann EOS failed for one or more cells"))
    all(isfinite,rho)&&all(>(0),rho) || throw(ErrorException("Wittmann EOS returned invalid density"))
    all(isfinite,ne)&&all(>(0),ne) || throw(ErrorException("Wittmann EOS returned invalid electron density"))
    rho,ne
end

function thermodynamics!(rho,ne,eos::IdealGasEOS{T},temperature,pgas) where T
    size(rho)==size(ne)==size(temperature)==size(pgas) || throw(DimensionMismatch("EOS arrays differ"))
    kB,mH=T(1.380649e-23),T(1.6735575e-27)
    @. rho=pgas*eos.mean_molecular_weight*mH/(kB*temperature)
    @. ne=eos.electron_fraction*pgas/(kB*temperature)
    rho,ne
end
