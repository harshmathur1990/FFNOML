abstract type AbstractRedistributionModel end
struct NonPRD <: AbstractRedistributionModel end
struct MockPRD{T<:AbstractFloat} <: AbstractRedistributionModel
    coupling::T
end
function validate_capabilities(manifest::CapabilityManifest, redistribution::AbstractRedistributionModel, stokes::StokesSet)
    redistribution isa NonPRD || manifest.prd || throw(ArgumentError("requested PRD capability is unavailable"))
    all(c -> c in manifest.stokes, stokes.components) || throw(ArgumentError("requested Stokes capability is unavailable"))
    true
end

abstract type AbstractLineOpacityModel end
abstract type AbstractFormalSolver end
function build_muspel_line_model end

struct MuspelLineOpacityModel{L,C,V,H} <: AbstractLineOpacityModel
    line::L
    continuum::C
    voigt::V
    hydrogen_populations::H
    lower_level::Int
    upper_level::Int
    include_continuum::Bool
end
struct MuspelFormalSolver <: AbstractFormalSolver end

struct FFNOTransition{T<:AbstractFloat} <: AbstractLineOpacityModel
    species::Symbol; wavelength0_m::T; lower_level::Int; upper_level::Int
    oscillator_strength::T; atomic_mass_kg::T; damping_s::T
    function FFNOTransition(species::Symbol,wavelength0_m::T,lower::Int,upper::Int,
                            oscillator_strength::T,atomic_mass_kg::T,damping_s::T) where T<:AbstractFloat
        wavelength0_m>0 && lower>0 && upper>0 && lower!=upper || throw(ArgumentError("invalid FFNO transition"))
        oscillator_strength>0 && atomic_mass_kg>0 && damping_s>=0 || throw(ArgumentError("invalid FFNO line constants"))
        new{T}(species,wavelength0_m,lower,upper,oscillator_strength,atomic_mass_kg,damping_s)
    end
end

struct KuruczLine{T<:AbstractFloat}
    wavelength0_m::T
    species::String
    atomic_number::Int
    ion_stage::Int
    loggf::T
    lower_energy_j::T
    upper_energy_j::T
    lower_g::T
    upper_g::T
    grad_s::T
    gstark_m3_s::T
    gvdwaals_m3_s::T
    atomic_mass_kg::T
end
KuruczLine(wavelength0_m::T,species::String,loggf::T,lower_energy_j::T,upper_energy_j::T,damping_s::T) where T =
    KuruczLine(wavelength0_m,species,0,0,loggf,lower_energy_j,upper_energy_j,one(T),one(T),damping_s,zero(T),zero(T),T(56*1.66053906660e-27))
struct KuruczLTEModel{T<:AbstractFloat,E} <: AbstractLineOpacityModel
    lines::Vector{KuruczLine{T}}
    eos::E
    include_continuum::Bool
end
KuruczLTEModel(lines::Vector{KuruczLine{T}};include_continuum=true) where T=KuruczLTEModel(lines,nothing,include_continuum)

mutable struct WittmannKuruczState
    eos::WittmannEOS
    library_handle::Ptr{Cvoid}
    backend::Ptr{Cvoid}
end
function WittmannKuruczState(eos::WittmannEOS)
    library=Libdl.dlopen(eos.library)
    backend=ccall(Libdl.dlsym(library,:witt_create_backend),Ptr{Cvoid},(Cstring,),eos.partition_functions)
    backend==C_NULL && throw(ErrorException("failed to construct persistent Wittmann Kurucz backend"))
    state=WittmannKuruczState(eos,library,backend)
    finalizer(state) do value
        value.backend==C_NULL && return
        ccall(Libdl.dlsym(value.library_handle,:witt_destroy_backend),Cvoid,(Ptr{Cvoid},),value.backend)
        value.backend=C_NULL
    end
    state
end
KuruczLTEModel(lines::Vector{KuruczLine{T}},eos::WittmannEOS;include_continuum=true) where {T<:AbstractFloat}=KuruczLTEModel(lines,WittmannKuruczState(eos),include_continuum)
struct TabulatedOpacityModel{T<:AbstractFloat,A<:AbstractArray{T,2}} <: AbstractLineOpacityModel
    extinction::A; emissivity::A
end
struct KuruczLineCache{T<:AbstractFloat}
    source::String; mtime::Float64; lines::Vector{KuruczLine{T}}
end

const _EV_J=1.602176634e-19
const _HC_J_CM=6.62607015e-34*299792458.0*100.0
const _AMASS_KG = (1.008,4.003,6.941,9.012,10.811,12.011,14.007,15.999,18.998,20.179,
    22.990,24.305,26.982,28.086,30.974,32.060,35.453,39.948,39.102,40.080,
    44.956,47.900,50.941,51.996,54.938,55.847,58.933,58.710,63.546,65.370)
@inline function _air_to_vacuum_nm(λ)
    λ<200.0 && return λ
    sq=(1e7/λ)^2
    λ*(1.0000834213+2.406030e6/(1.30e10-sq)+1.5997e4/(3.89e9-sq))
end
function _kurucz_fields(line::AbstractString,line_number::Int)
    clean=strip(first(split(line,'#';limit=2))); isempty(clean) && return nothing
    fields=occursin(',',clean) ? strip.(split(clean,',')) : split(clean)
    length(fields)>=5 || throw(ArgumentError("Kurucz line $line_number needs wavelength[A], species, loggf, Elow[eV], Eup[eV]"))
    fields
end
"""Parse a portable Kurucz subset once. Optional sixth column is damping [s^-1]."""
function load_kurucz_linelist(path::AbstractString;T::Type{<:AbstractFloat}=Float64)
    lines=KuruczLine{T}[]
    open(path,"r") do io
        for (line_number,line) in enumerate(eachline(io))
            isempty(strip(line)) && continue; startswith(strip(line),'#') && continue
            if ncodeunits(line)>=97
                head=split(line[1:35]); length(head)>=4 || throw(ArgumentError("invalid K94 line $line_number"))
                λair=parse(T,head[1]); loggf=parse(T,head[2]); code=parse(T,head[3]); ei=parse(T,head[4])
                z=floor(Int,code); stage=round(Int,(code-z)*100)
                ji=parse(T,strip(line[36:41])); ej=parse(T,strip(line[54:63])); jj=parse(T,strip(line[64:69]))
                elo,eup,gl,gu = ei<=ej ? (ei,ej,2ji+1,2jj+1) : (ej,ei,2jj+1,2ji+1)
                broad=split(strip(line[80:97])); length(broad)==3 || throw(ArgumentError("invalid K94 damping at line $line_number"))
                grad,gs,gv=T(10)^parse(T,broad[1]),T(10)^parse(T,broad[2])*T(1e-6),T(10)^parse(T,broad[3])*T(1e-6)
                mass=z<=length(_AMASS_KG) ? T(_AMASS_KG[z]*1.66053906660e-27) : T(z*2*1.66053906660e-27)
                λ=T(_air_to_vacuum_nm(λair)*1e-9)
                push!(lines,KuruczLine(λ,"Z$(z).$(stage)",z,stage,loggf,T(elo*_HC_J_CM),T(eup*_HC_J_CM),gl,gu,grad,gs,gv,mass))
            else
                f=_kurucz_fields(line,line_number); f===nothing && continue
                λ=parse(T,f[1])*T(1e-10); loggf=parse(T,f[3]); elo=parse(T,f[4])*T(_EV_J); eup=parse(T,f[5])*T(_EV_J)
                damping=length(f)>=6 ? parse(T,f[6]) : zero(T)
                λ>0 && eup>=elo && damping>=0 || throw(ArgumentError("invalid Kurucz line $line_number"))
                push!(lines,KuruczLine(λ,String(f[2]),loggf,elo,eup,damping))
            end
        end
    end
    isempty(lines) && throw(ArgumentError("Kurucz line list is empty: $path"))
    sort!(lines,by=l->l.wavelength0_m)
    KuruczLineCache(String(path),mtime(path),lines)
end
function select_kurucz_lines(cache::KuruczLineCache,wavelength_m::AbstractVector;margin_m=0.0)
    isempty(wavelength_m) && return similar(cache.lines,0)
    lo,hi=extrema(wavelength_m)
    [line for line in cache.lines if lo-margin_m<=line.wavelength0_m<=hi+margin_m]
end

const _H=6.62607015e-34; const _C=299792458.0; const _KB=1.380649e-23
@inline planck_lambda(λ,T)=(2*_H*_C^2/λ^5)/expm1(_H*_C/(λ*_KB*T))
function _gaussian_profile(λ,λ0,T,vturb,mass)
    width=λ0/_C*sqrt(2*_KB*T/mass+vturb^2)
    exp(-((λ-λ0)/width)^2)/(sqrt(pi)*width)
end

const _VA_T=(0.2453407083,0.7374737285,1.2340762153,1.7385377121,2.2549740020,2.7888060584,3.3478545673,3.9447640401,4.6036824495,5.3874808900)
const _VA_W=(4.6224366960e-1,2.8667550536e-1,1.0901720602e-1,2.4810520887e-2,3.2437733422e-3,2.2833863601e-4,7.8025564785e-6,1.0860693707e-7,4.3993409922e-10,2.2293936455e-13)
const _VA_C=(0.1999999999972224,-0.1840000000029998,0.1558399999965025,-0.1216640000043988,0.0877081599940391,-0.0585141248086907,0.0362157301623914,-0.0208497654398036,0.0111960116346270,-0.56231896167109e-2,0.26487634172265e-2,-0.11732670757704e-2,0.4899519978088e-3,-0.1933630801528e-3,0.722877446788e-4,-0.256555124979e-4,0.86620736841e-5,-0.27876379719e-5,0.8566873627e-6,-0.2518433784e-6,0.709360221e-7,-0.191732257e-7,0.49801256e-8,-0.12447734e-8,0.2997777e-9,-0.696450e-10,0.156262e-10,-0.33897e-11,0.7116e-12,-0.1447e-12,0.285e-13,-0.55e-14,0.10e-14,-0.2e-15)
function _voigt_k1(a,v)
    a2=a*a; v2=v*v; u1=(v2-a2)>70 ? 0.0 : exp(a2-v2)*cos(2v*a)
    if v>5
        vi=1/v2; d1=-vi*(.5+vi*(.75+vi*(1.875+vi*(6.5625+vi*(29.53125+vi*(1162.4218+vi*1055.7421)))))); d2=(1-d1)/(2v)
    else
        b1=0.0; b2=0.0; v1=v/5; coef=4v1*v1-2; bn=0.0
        for n in length(_VA_C):-1:1; bn=coef*b1-b2+_VA_C[n]; b2=b1; b1=bn; end
        d2=v1*(bn-b2); d1=1-2v*d2
    end
    f=a*d1
    if a>1e-8
        q=1.0; an=a
        for n in 2:50
            dn=(v*d1+d2)*(-2/n); d2=d1; d1=dn
            if isodd(n); q=-q; an*=a2; g=dn*an; f+=q*g; abs(g/f)<=1e-8 && break; end
        end
    end
    u1-1.12837917*f
end
function _voigt_armstrong(a,v)
    v=abs(v)
    ((a<1&&v<4)||a<1.8/(v+1)) && return _voigt_k1(a,v)
    a2=a*a; g=0.0
    if a<2.5&&v<4
        for n in eachindex(_VA_T); t=_VA_T[n]; r=t-v; s=t+v; g+=(4t*t-2)*(r*atan(r/a)+s*atan(s/a)-.5a*(log(a2+r*r)+log(a2+s*s)))*_VA_W[n]; end
        return g/pi
    end
    for n in eachindex(_VA_T); t=_VA_T[n]; g+=(1/((v-t)^2+a2)+1/((v+t)^2+a2))*_VA_W[n]; end
    a*g/pi
end
function add_opacity_emissivity! end
function add_opacity_emissivity!(χ,η,model::TabulatedOpacityModel,wavelength,atmosphere,x,y,populations=nothing)
    size(model.extinction)==size(χ)==size(model.emissivity) || throw(DimensionMismatch("tabulated opacity shape mismatch"))
    χ.+=model.extinction; η.+=model.emissivity
end
function add_opacity_emissivity!(χ,η,model::FFNOTransition,wavelength,atmosphere,x,y,populations)
    populations===nothing && throw(ArgumentError("$(model.species) FFNO populations are required"))
    size(populations)[1:3]==size(atmosphere.temperature) || throw(DimensionMismatch("population grid differs from atmosphere"))
    size(populations,4)>=max(model.lower_level,model.upper_level) || throw(DimensionMismatch("population level missing"))
    @inbounds for k in axes(χ,1),l in axes(χ,2)
        temp=atmosphere.temperature[k,x,y]
        profile=_gaussian_profile(wavelength[l]*(1-atmosphere.vz[k,x,y]/_C),model.wavelength0_m,temp,atmosphere.vturb[k,x,y],model.atomic_mass_kg)
        nl=populations[k,x,y,model.lower_level]; nu=populations[k,x,y,model.upper_level]
        χline=max(zero(eltype(χ)),model.oscillator_strength*1e-24*(nl-nu)*profile)
        χ[k,l]+=χline; η[k,l]+=χline*planck_lambda(wavelength[l],temp)
    end
end
function add_opacity_emissivity!(χ,η,model::KuruczLTEModel,wavelength,atmosphere,x,y,populations=nothing)
    atmosphere.ne===nothing && throw(ArgumentError("LTE Kurucz synthesis requires electron density"))
    atmosphere.rho===nothing && throw(ArgumentError("LTE Kurucz synthesis requires mass density"))
    model.eos===nothing || return _add_wittmann_kurucz!(χ,η,model,wavelength,atmosphere,x,y)
    @inbounds for line in model.lines,k in axes(χ,1),l in axes(χ,2)
        temp=atmosphere.temperature[k,x,y]
        nl=atmosphere.rho[k,x,y]*exp(-line.lower_energy_j/(_KB*temp))
        profile=_gaussian_profile(wavelength[l]*(1-atmosphere.vz[k,x,y]/_C),line.wavelength0_m,temp,atmosphere.vturb[k,x,y],1.66053906660e-27*56)
        χline=max(zero(eltype(χ)),1e-2*10^line.loggf*nl*profile)
        χ[k,l]+=χline; η[k,l]+=χline*planck_lambda(wavelength[l],temp)
    end
end

function _add_wittmann_kurucz!(χ,η,model,wavelength,atmosphere,x,y)
    atmosphere.pgas===nothing && throw(ArgumentError("Wittmann Kurucz synthesis requires gas pressure"))
    nz=size(atmosphere.temperature,1); temp=Float64.(atmosphere.temperature[:,x,y]); pgas=Float64.(atmosphere.pgas[:,x,y])
    continuum=zeros(Float64,nz); lower=zeros(Float64,nz); nh=zeros(Float64,nz)
    state=model.eos isa WittmannKuruczState ? model.eos : WittmannKuruczState(model.eos)
    fn=Libdl.dlsym(state.library_handle,:witt_kurucz_state)
    continuum_done=false
    for line in model.lines
        line.atomic_number>0 || throw(ArgumentError("production Kurucz synthesis requires K94 species metadata"))
        status=ccall(fn,Cint,(Ptr{Cvoid},Ptr{Cdouble},Ptr{Cdouble},Cdouble,Cint,Cint,Cdouble,Cdouble,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Csize_t),
            state.backend,temp,pgas,line.wavelength0_m*1e10,line.atomic_number,line.ion_stage,
            line.lower_energy_j,line.lower_g,continuum,lower,nh,nz)
        status==0 || throw(ErrorException("Wittmann Kurucz state calculation failed"))
        if model.include_continuum && !continuum_done
            @inbounds for l in eachindex(wavelength)
                status=ccall(fn,Cint,(Ptr{Cvoid},Ptr{Cdouble},Ptr{Cdouble},Cdouble,Cint,Cint,Cdouble,Cdouble,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Csize_t),
                    state.backend,temp,pgas,wavelength[l]*1e10,line.atomic_number,line.ion_stage,
                    line.lower_energy_j,line.lower_g,continuum,lower,nh,nz)
                status==0 || throw(ErrorException("Wittmann continuum calculation failed"))
                for k in 1:nz
                    χ[k,l]+=continuum[k]; η[k,l]+=continuum[k]*planck_lambda(wavelength[l],temp[k])*1e-12
                end
            end
            continuum_done=true
        end
        λ0=line.wavelength0_m; Aji=6.670e13/(λ0*1e9)^2*10^line.loggf/line.upper_g
        Bji=λ0^3/(2*_H*_C)*Aji; Bij=line.upper_g/line.lower_g*Bji
        @inbounds for k in 1:nz,l in eachindex(wavelength)
            vb=sqrt(2*_KB*temp[k]/line.atomic_mass_kg+atmosphere.vturb[k,x,y]^2)
            v=(wavelength[l]/λ0-1)*_C/vb+atmosphere.vz[k,x,y]/vb
            gamma=line.grad_s+line.gstark_m3_s*atmosphere.ne[k,x,y]+line.gvdwaals_m3_s*nh[k]
            adamp=gamma*λ0/(4pi*vb); profile=_voigt_armstrong(adamp,v)/(sqrt(pi)*vb)
            nu=line.upper_g/line.lower_g*lower[k]*exp(-(_H*_C/λ0)/(_KB*temp[k]))
            factor=_H*_C/(4pi)*Bij*line.lower_g
            χline=factor*(lower[k]/line.lower_g-nu/line.upper_g)*profile
            χ[k,l]+=χline; η[k,l]+=χline*planck_lambda(wavelength[l],temp[k])*1e-12
        end
    end
    nothing
end
