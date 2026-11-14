abstract type AbstractEOS end

"""
Wittmann EOS shared-library adapter (WittBifrostEOS's libwitt, the library HSEMHS's tables and models use). All public inputs and outputs
use SI units. `library` is a libwitt built by `scripts/build_wittmann_backend.jl` (`WittBifrostEOS.build_libwitt`), e.g. inside a job
on the node type that runs it; WittBifrostEOS (tables, the NLTE model) uses the same file unless it already loaded one, every one a
build of the same source. Without it, WittBifrostEOS's own.
"""
struct WittmannEOS <: AbstractEOS
    library::String
    partition_functions::String
    function WittmannEOS(library::AbstractString,partition_functions::AbstractString)
        isfile(library) || throw(ArgumentError("Wittmann library not found: $library"))
        isfile(partition_functions) || throw(ArgumentError("Wittmann partition-function file not found: $partition_functions"))
        Libdl.dlsym(Libdl.dlopen(library),:witt_ctx_new;throw_error=false)===nothing && throw(ArgumentError(
            "$library is not WittBifrostEOS's libwitt (an older FFNOInversion build); rebuild it with scripts/build_wittmann_backend.jl"))
        WittBifrostEOS.use_libwitt!(library;strict=false)
        new(abspath(library),abspath(partition_functions))
    end
end
WittmannEOS(partition_functions::AbstractString=WittBifrostEOS.default_pf_file())=WittmannEOS(WittBifrostEOS.default_libwitt_path(),partition_functions)

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
    function_pointer=Libdl.dlsym(handle,:witt_thermodynamics_from_pgas_parallel)
    status=ccall(function_pointer,Cint,
        (Cstring,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Csize_t,Cint),
        eos.partition_functions,temperature,pgas,rho,ne,length(rho),Threads.nthreads())
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
