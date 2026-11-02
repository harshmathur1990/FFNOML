# Finite-difference temperature response with a smooth bump: for every log tau500 point k0 of a column,
#   δT(k) = A exp(-(log τ(k) - log τ(k0))² / 2σ²),   A = FRACTION · T(k0),
#   R(λ, k0) = (I(λ; T + δT) - I(λ; T - δT)) / 2A        [intensity per K of bump amplitude]
# pgas, rho, ne, z and chi500 (so the log tau500 scale) stay fixed. POPULATIONS is
#   fixed       H populations unchanged
#   departure   departure coefficients b = n/n_LTE unchanged: every H level (protons included) is scaled by
#               n_LTE(T + δT)/n_LTE(T), LTE from Saha-Boltzmann at fixed ne and fixed total hydrogen
# usage: julia fd_bump_response.jl FORWARD.toml PATHS.jl ATMOSPHERE.h5 OUTPUT.h5 SIGMA_DEX FRACTION POPULATIONS
#            x,y [x,y ...]   (x, y 0-based); or `grid:STEP` instead of pixels for every STEP-th pixel in x and y.
using FFNOInversion, HDF5, Printf
import Muspel
const F = FFNOInversion
config_path,paths,atmosphere_path,output=ARGS[1:4]
sigma=parse(Float64,ARGS[5]); fraction=parse(Float64,ARGS[6]); mode=ARGS[7]
mode in ("fixed","departure") || error("POPULATIONS must be fixed or departure")
include(abspath(paths)); include(joinpath(@__DIR__,"check_lib.jl"))

config=load_config(config_path)
atmosphere,populations=read_forward_atmosphere(atmosphere_path)
eos=WittmannEOS(EOS_LIBRARY,PARTITION_FUNCTIONS)
setups=build_synthesis_setup(config,atmosphere,populations,eos;atom_files=ATOM_FILES)
logtau=native_depth(atmosphere.grid); nz=length(logtau)
nx,ny=size(atmosphere.temperature)[2:3]
pixels=if startswith(ARGS[8],"grid:")
    step=parse(Int,ARGS[8][6:end]); [(x,y) for x in 0:step:nx-1 for y in 0:step:ny-1]
else
    [Tuple(parse.(Int,split(p,','))) for p in ARGS[8:end]]
end
# Every array the H line and continuum models read the populations from (one array unless the setup copied it).
hydrogen=unique(objectid,[a for s in setups for c in s.synthesizer.contributors
    for a in (c.populations,getfield(c.model,:hydrogen_populations)) if a!==nothing])

const KB=1.380649e-23; const HP=6.62607015e-34; const ME=9.1093837015e-31
atom=Muspel.read_atom(ATOM_FILES[:H]); energy=atom.χ; degeneracy=Float64.(atom.g)   # J, levels 1-5 then the proton
length(energy)==6 && energy[6]==maximum(energy) || error("expected the 6-level H atom with the continuum last")

"""Saha-Boltzmann H populations (6) at temperature t, electron density ne and total hydrogen ntot."""
function lte(t,ne,ntot)
    phi=[ne*degeneracy[i]/(2degeneracy[6])*(HP^2/(2pi*ME*KB*t))^1.5*exp((energy[6]-energy[i])/(KB*t)) for i in 1:5]
    proton=ntot/(1+sum(phi)); vcat(phi.*proton,proton)
end

"""Set the H populations of column (x, y) to saved .* n_LTE(T)/n_LTE(T0) (departure coefficients fixed)."""
function departure!(x,y,saved_populations,T0)
    for k in axes(saved_populations,1)
        n0=@view saved_populations[k,:]; ntot=sum(n0); ne=atmosphere.ne[k,x,y]
        scale=lte(atmosphere.temperature[k,x,y],ne,ntot)./lte(T0[k],ne,ntot)
        for a in hydrogen, i in 1:6; a[k,x,y,i]=n0[i]*scale[i]; end
    end
end

wavelength=reduce(vcat,[s.wavelength_vacuum_m for s in setups]); nl=length(wavelength)

function intensity!(I,x,y)
    offset=0
    for setup in setups
        w=setup.wavelength_vacuum_m; n=length(w); chi=zeros(nz,n); eta=zeros(nz,n)
        opacity!(chi,eta,setup.synthesizer,w,atmosphere,x,y)
        weight=F._log_tau_weight(atmosphere,x,y); chi.*=weight; eta.*=weight
        F.formal_solve!(@view(I[offset+1:offset+n]),setup.synthesizer.solver,chi,eta,F._log_tau_coordinate(atmosphere,x,y))
        offset+=n
    end
    I
end

R=zeros(nl,nz,length(pixels)); I0=zeros(nl,length(pixels))
Threads.@threads :dynamic for p in eachindex(pixels)
    x,y=pixels[p].+1
    T=@view atmosphere.temperature[:,x,y]; saved=copy(T); saved_populations=hydrogen[1][:,x,y,:]
    intensity!(@view(I0[:,p]),x,y)
    plus=zeros(nl); minus=zeros(nl)
    for k0 in 1:nz
        A=fraction*saved[k0]; bump=A.*exp.(-(logtau.-logtau[k0]).^2 ./(2sigma^2))
        T.=saved.+bump; mode=="departure" && departure!(x,y,saved_populations,saved); intensity!(plus,x,y)
        T.=saved.-bump; mode=="departure" && departure!(x,y,saved_populations,saved); intensity!(minus,x,y)
        R[:,k0,p].=(plus.-minus)./(2A)
    end
    T.=saved
    for a in hydrogen; a[:,x,y,:].=saved_populations; end
end

h5open(output,"w") do f
    f["R_temperature"]=R; f["intensity"]=I0; f["logtau500"]=collect(logtau)
    f["wavelength_vacuum_m"]=wavelength; f["wavelength_air_m"]=collect(config.synthesis.wavelength_m)
    f["pixels"]=reduce(hcat,[collect(p) for p in pixels])
    attributes(f["R_temperature"])["description"]="(I(T+bump)-I(T-bump))/2A per K of bump amplitude A, bump a Gaussian " *
        "in log tau500 centred on each depth; pgas, rho, ne, chi500 fixed; H populations per the populations attribute"
    attributes(f)["sigma_dex"]=sigma; attributes(f)["populations"]=mode; attributes(f)["fraction"]=fraction
end
println("wrote ",output," (",length(pixels)," pixels)")
