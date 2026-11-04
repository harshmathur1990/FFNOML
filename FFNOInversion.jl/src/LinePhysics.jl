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
    neutral_hydrogen::Any     # Wittmann backend: neutral H in broadening from the EOS; nothing: from `hydrogen_populations`
    center_nm::Float64        # vacuum line centre of the profile: a reference wavelength, or the atom's (level energies)
    total_per_rho::Float64    # largest total of the species' levels per unit ρ, [kg⁻¹]; 0: no limit (`[physics] ffno_population_limit`)
    no_inversion::Bool        # the upper level holds at most what the line opacity allows (`[physics] ffno_no_inversion`)
end
struct MuspelFormalSolver <: AbstractFormalSolver end

function build_muspel_continuum_model end

"""The continuum of a region: Muspel's at one wavelength from T, nₑ, neutral H and protons. The hydrogen is the atmosphere's when it is given
(`h_neutral`, `h_proton`, forward only), otherwise the EOS's at (T, P) from `eos`. FFNO populations never enter."""
struct MuspelContinuumModel{C,S} <: AbstractLineOpacityModel
    continuum::C
    eos::S
end

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
    # temperature-dependent van der Waals broadening γ/n_H = vdw_coefficient*T^vdw_exponent [m^3 s^-1], added to
    # gvdwaals_m3_s: ABO (Barklem) with exponent (1-α)/2 or Unsöld with 0.3; zero when the list's γ_vdW is used
    vdw_coefficient::T
    vdw_exponent::T
    vdw_recipe::Symbol  # :abo_explicit, :abo_table, :list, :unsold or :none
end
KuruczLine(wavelength0_m::T,species::String,atomic_number::Int,ion_stage::Int,loggf::T,lower_energy_j::T,upper_energy_j::T,
           lower_g::T,upper_g::T,grad_s::T,gstark_m3_s::T,gvdwaals_m3_s::T,atomic_mass_kg::T) where T =
    KuruczLine(wavelength0_m,species,atomic_number,ion_stage,loggf,lower_energy_j,upper_energy_j,lower_g,upper_g,
               grad_s,gstark_m3_s,gvdwaals_m3_s,atomic_mass_kg,zero(T),zero(T),iszero(gvdwaals_m3_s) ? :none : :list)
KuruczLine(wavelength0_m::T,species::String,loggf::T,lower_energy_j::T,upper_energy_j::T,damping_s::T) where T =
    KuruczLine(wavelength0_m,species,0,0,loggf,lower_energy_j,upper_energy_j,one(T),one(T),damping_s,zero(T),zero(T),
               T(56*1.66053906660e-27))
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
# First and second ionization potentials [eV] (NIST) for Z = 1-28, for the Unsöld van der Waals cross-section.
const _IONPOT_EV=((13.598,0.0),(24.587,54.418),(5.392,75.640),(9.323,18.211),(8.298,25.155),(11.260,24.383),
    (14.534,29.601),(13.618,35.121),(17.423,34.971),(21.565,40.963),(5.139,47.286),(7.646,15.035),(5.986,18.829),
    (8.152,16.346),(10.487,19.770),(10.360,23.338),(12.968,23.814),(15.760,27.630),(4.341,31.630),(6.113,11.872),
    (6.561,12.800),(6.828,13.576),(6.746,14.618),(6.767,16.486),(7.434,15.640),(7.902,16.199),(7.881,17.084),
    (7.640,18.169))

"""Ionization energy [J] of `ion_stage` (0 = neutral) of element `atomic_number`, or `nothing` if not tabulated."""
function _ionization_energy(atomic_number,ion_stage)
    (1<=atomic_number<=length(_IONPOT_EV) && 0<=ion_stage<=1) || return nothing
    value=_IONPOT_EV[atomic_number][ion_stage+1]; value>0 ? value*_EV_J : nothing
end

"""Unsöld van der Waals cross-section as RH computes it (`getUnsoldcross` in RH's kurucz.c): γ_vdW = cross*T^0.3*n_H
with collisions with H and He. Zero when the ion is not tabulated or the upper level lies above the lower one's
effective radius (RH falls back to the list's γ_vdW)."""
function _unsold_cross(atomic_number,ion_stage,lower_energy_j,upper_energy_j,mass_amu)
    ionpot=_ionization_energy(atomic_number,ion_stage); ionpot===nothing && return 0.0
    rydberg=2.1798741e-18; eps4pi=4pi*8.8541878128e-12; abar_h=7.42e-41; bohr=5.29177210903e-11
    ionpot>upper_energy_j || return 0.0
    delta=(rydberg/(ionpot-upper_energy_j))^2-(rydberg/(ionpot-lower_energy_j))^2
    delta>0 || return 0.0
    z=ion_stage+1; amu=1.66053906660e-27; helium=10^(10.93-12)
    vrel(w)=(8*_KB/(pi*amu*mass_amu)*(1+mass_amu/w))^0.3
    c625=(2.5*(_EV_J^2/eps4pi)*(abar_h/eps4pi)*2pi*(z*bohr)^2/_H*delta)^0.4
    8.08*(vrel(1.008)+helium*vrel(4.003))*c625
end

"""Van der Waals broadening of a K94 line, in this order: ABO σ.α given in the γ_vdW field (a value ≥ 20: σ [a_0²] its
integer part, α its fraction, RH's convention); ABO from RH's tables for a neutral atom with s-p, p-d or d-f valence
electrons (orbitals from the level labels, RH's `RLKdet_level`); the list's log γ_vdW (non-zero); Unsöld (H and He).
As in RH the upper level lies hc/λ above the lower one, and ABO counts hydrogen only. Returns the list γ/n_H [m^3 s^-1],
the coefficient and exponent of the temperature-dependent γ/n_H, and the recipe."""
function _kurucz_vdw(z,stage,lower_j,λ_vacuum_m,log_gamma,labels,mass_amu)
    upper_j=lower_j+_H*_C/λ_vacuum_m
    if log_gamma>=20
        sigma=floor(log_gamma); alpha=log_gamma-sigma
        return 0.0,barklem_coefficient(sigma,alpha,mass_amu),(1-alpha)/2,:abo_explicit
    end
    ionpot=_ionization_energy(z,stage)
    if stage==0 && ionpot!==nothing
        neff(e)=effective_n(ionpot,e,1,mass_amu)
        abo=barklem_sigma_alpha(kurucz_label_orbital(labels[1]),kurucz_label_orbital(labels[2]),neff(lower_j),neff(upper_j))
        abo===nothing || return 0.0,barklem_coefficient(abo...,mass_amu),(1-abo[2])/2,:abo_table
    end
    log_gamma!=0 && return 10^log_gamma*1e-6,0.0,0.0,:list
    cross=_unsold_cross(z,stage,lower_j,upper_j,mass_amu)
    0.0,cross,0.3,cross>0 ? :unsold : :none
end

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
                # as RH's sscanf: fields may abut, and a predicted level's negative energy counts by its magnitude
                head=[m.match for m in eachmatch(r"[-+]?\d*\.?\d+",line[1:35])]
                length(head)>=4 || throw(ArgumentError("invalid K94 line $line_number"))
                λair=parse(T,head[1]); loggf=parse(T,head[2]); code=parse(T,head[3]); ei=abs(parse(T,head[4]))
                z=floor(Int,code); stage=round(Int,(code-z)*100)
                ji=parse(T,strip(line[36:41])); ej=abs(parse(T,strip(line[54:63]))); jj=parse(T,strip(line[64:69]))
                elo,eup,gl,gu = ei<=ej ? (ei,ej,2ji+1,2jj+1) : (ej,ei,2jj+1,2ji+1)
                # as RH: 20 characters, three numbers ending where the record's next field starts; a σ.α in the third
                # (e.g. 312.250) runs 2-3 characters past the 18 of the plain format
                broad=collect(eachmatch(r"[-+]?\d*\.?\d+",line[80:min(99,ncodeunits(line))]))
                length(broad)>=3 || throw(ArgumentError("invalid K94 damping at line $line_number"))
                lgrad,lgs,lgv=(parse(T,broad[i].match) for i in 1:3)
                mass_amu=z<=length(_AMASS_KG) ? _AMASS_KG[z] : 2z; mass=T(mass_amu*1.66053906660e-27)
                λ=T(_air_to_vacuum_nm(λair)*1e-9)
                # a zero damping constant is "not given" (as in RH): radiative -> A_ul, Stark -> 0
                aul=6.670e13/(λ*1e9)^2*10^loggf/gu
                grad=lgrad!=0 ? T(10)^lgrad : T(aul)
                gs=lgs!=0 ? T(10)^lgs*T(1e-6) : zero(T)
                # level labels (RH reads them at columns 42 and 70) in the order of the energies
                labels = ei<=ej ? (line[42:51],line[70:79]) : (line[70:79],line[42:51])
                gv,cvdw,evdw,recipe=_kurucz_vdw(z,stage,elo*_HC_J_CM,λ,lgv,labels,mass_amu)
                push!(lines,KuruczLine(λ,"Z$(z).$(stage)",z,stage,loggf,T(elo*_HC_J_CM),T(eup*_HC_J_CM),gl,gu,grad,gs,
                    T(gv),mass,T(cvdw),T(evdw),recipe))
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
"""Planck function in kW m^-2 sr^-1 nm^-1 (λ in m): the unit of every model's emissivity η (kW m^-3 sr^-1 nm^-1,
Muspel's), so lines from different models add in one region and the intensity is in kW m^-2 sr^-1 nm^-1."""
@inline source_planck(λ,T)=planck_lambda(λ,T)*1e-12
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
        profile=_gaussian_profile(wavelength[l]*(1+los_velocity(atmosphere,k,x,y)/_C),model.wavelength0_m,temp,atmosphere.vturb[k,x,y],model.atomic_mass_kg)
        nl=populations[k,x,y,model.lower_level]; nu=populations[k,x,y,model.upper_level]
        χline=max(zero(eltype(χ)),model.oscillator_strength*1e-24*(nl-nu)*profile)
        χ[k,l]+=χline; η[k,l]+=χline*source_planck(wavelength[l],temp)
    end
end
function add_opacity_emissivity!(χ,η,model::KuruczLTEModel,wavelength,atmosphere,x,y,populations=nothing)
    atmosphere.ne===nothing && throw(ArgumentError("LTE Kurucz synthesis requires electron density"))
    atmosphere.rho===nothing && throw(ArgumentError("LTE Kurucz synthesis requires mass density"))
    model.eos===nothing || return _add_wittmann_kurucz!(χ,η,model,wavelength,atmosphere,x,y)
    @inbounds for line in model.lines,k in axes(χ,1),l in axes(χ,2)
        temp=atmosphere.temperature[k,x,y]
        nl=atmosphere.rho[k,x,y]*exp(-line.lower_energy_j/(_KB*temp))
        profile=_gaussian_profile(wavelength[l]*(1+los_velocity(atmosphere,k,x,y)/_C),line.wavelength0_m,temp,atmosphere.vturb[k,x,y],1.66053906660e-27*56)
        χline=max(zero(eltype(χ)),1e-2*10^line.loggf*nl*profile)
        χ[k,l]+=χline; η[k,l]+=χline*source_planck(wavelength[l],temp)
    end
end

"""Continuum state from the persistent backend (a library built before `witt_continuum_state` falls back to a new EOS per call)."""
function continuum_state(state::WittmannKuruczState,temp::AbstractVector,pgas::AbstractVector;threads::Int=1)
    fn=Libdl.dlsym(state.library_handle,:witt_continuum_state;throw_error=false)
    fn===nothing && return continuum_state(state.eos,temp,pgas;threads)
    length(temp)==length(pgas) || throw(DimensionMismatch("temperature and pressure differ"))
    t=Float64.(temp); p=Float64.(pgas); n=length(t)
    rho=zeros(n); xna=zeros(n); xne=zeros(n); partials=zeros(17,n)
    status=ccall(fn,Cint,(Ptr{Cvoid},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Csize_t),
        state.backend,t,p,rho,xna,xne,partials,n)
    status==0 || throw(ErrorException("Wittmann continuum-state calculation failed"))
    ContinuumState(rho,xna,xne,partials)
end

"""Neutral hydrogen [m⁻³] from the EOS at (T, P), one value per element."""
function eos_neutral_hydrogen!(nh,state::WittmannKuruczState,temp,pgas)
    lower=similar(nh); fn=Libdl.dlsym(state.library_handle,:witt_kurucz_populations)
    status=ccall(fn,Cint,(Ptr{Cvoid},Ptr{Cdouble},Ptr{Cdouble},Cint,Cint,Cdouble,Cdouble,Ptr{Cdouble},Ptr{Cdouble},Csize_t),
        state.backend,temp,pgas,26,0,0.0,1.0,lower,nh,length(nh))
    status==0 || throw(ErrorException("Wittmann neutral hydrogen calculation failed"))
    nh
end

"""Protons [m⁻³] from the EOS at (T, P), one value per element."""
eos_protons(state::WittmannKuruczState,temp,pgas)=continuum_state(state,temp,pgas;threads=1).partials[2,:].*1e6

"""Add the Wittmann continuum and lines of `levels` of column (x, y); row i of `χ` and `η` is `levels[i]`."""
function _add_wittmann_kurucz!(χ,η,model,wavelength,atmosphere,x,y;levels=axes(atmosphere.temperature,1))
    atmosphere.pgas===nothing && throw(ArgumentError("Wittmann Kurucz synthesis requires gas pressure"))
    nz=length(levels); temp=Float64.(atmosphere.temperature[levels,x,y]); pgas=Float64.(atmosphere.pgas[levels,x,y])
    continuum=zeros(Float64,nz); lower=zeros(Float64,nz); nh=zeros(Float64,nz)
    state=model.eos isa WittmannKuruczState ? model.eos : WittmannKuruczState(model.eos)
    fn=Libdl.dlsym(state.library_handle,:witt_kurucz_populations)
    # Mixed synthesis already distributes independent columns over Julia
    # threads. Keep the short depth-only kernels serial within each column to
    # avoid nesting a C++ and Julia thread team inside every column task.
    given_neutral=atmosphere.h_neutral===nothing ? nothing : atmosphere.h_neutral[levels,x,y]
    continuum_state_column=model.include_continuum ? continuum_state(state,temp,pgas;threads=1) : nothing
    if model.include_continuum
        @inbounds for l in eachindex(wavelength)
            continuum.=continuum_extinction_m(continuum_state_column,temp,wavelength[l]*1e10;threaded=false)
            for k in 1:nz
                χ[k,l]+=continuum[k]; η[k,l]+=continuum[k]*source_planck(wavelength[l],temp[k])
            end
        end
    end
    for line in model.lines
        line.atomic_number>0 || throw(ArgumentError("production Kurucz synthesis requires K94 species metadata"))
        status=ccall(fn,Cint,(Ptr{Cvoid},Ptr{Cdouble},Ptr{Cdouble},Cint,Cint,Cdouble,Cdouble,Ptr{Cdouble},Ptr{Cdouble},Csize_t),
            state.backend,temp,pgas,line.atomic_number,line.ion_stage,
            line.lower_energy_j,line.lower_g,lower,nh,nz)
        status==0 || throw(ErrorException("Wittmann Kurucz state calculation failed"))
        given_neutral===nothing || (nh.=given_neutral)
        λ0=line.wavelength0_m; Aji=6.670e13/(λ0*1e9)^2*10^line.loggf/line.upper_g
        Bji=λ0^3/(2*_H*_C)*Aji; Bij=line.upper_g/line.lower_g*Bji
        @inbounds for i in 1:nz,l in eachindex(wavelength)
            k=levels[i]
            vb=sqrt(2*_KB*temp[i]/line.atomic_mass_kg+atmosphere.vturb[k,x,y]^2)
            v=(wavelength[l]/λ0-1)*_C/vb+los_velocity(atmosphere,k,x,y)/vb
            gamma=line.grad_s+line.gstark_m3_s*atmosphere.ne[k,x,y]+(line.gvdwaals_m3_s+line.vdw_coefficient*temp[i]^line.vdw_exponent)*nh[i]
            adamp=gamma*λ0/(4pi*vb); profile=_voigt_armstrong(adamp,v)/(sqrt(pi)*vb)
            nu=line.upper_g/line.lower_g*lower[i]*exp(-(_H*_C/λ0)/(_KB*temp[i]))
            factor=_H*_C/(4pi)*Bij*line.lower_g
            χline=factor*(lower[i]/line.lower_g-nu/line.upper_g)*profile
            χ[i,l]+=χline; η[i,l]+=χline*source_planck(wavelength[l],temp[i])
        end
    end
    nothing
end
