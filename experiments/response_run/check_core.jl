# Diagnostic for column_response.jl at a few pixels and core wavelengths:
#  1. direct finite differences: perturb T (or vz) at ONE depth k, re-solve, compare (I+ - I-)/2h with R[k]·Δlogτ(k)
#  2. τ_λ(k) at each wavelength, so the depth where τ_λ = 1 can be compared with where R peaks
#  3. split of R into its χ and η parts
# usage: julia check_core.jl FORWARD.toml PATHS.jl ATMOSPHERE.h5 RESPONSE.h5
using FFNOInversion, HDF5, Printf
import Muspel
const F = FFNOInversion
config_path,paths,atmosphere_path,response_path=ARGS
include(abspath(paths))
include(joinpath(@__DIR__,"check_lib.jl"))

config=load_config(config_path)
atmosphere,populations=read_forward_atmosphere(atmosphere_path)
eos=WittmannEOS(EOS_LIBRARY,PARTITION_FUNCTIONS)
setups=build_synthesis_setup(config,atmosphere,populations,eos;atom_files=ATOM_FILES)
wavelength_air=collect(config.synthesis.wavelength_m)
ranges=F._region_ranges(wavelength_air,setups)
logtau=native_depth(atmosphere.grid); widths=cell_widths(logtau); nz=length(logtau)
response=h5open(response_path,"r")

function solve(setup,x,y)
    nl=length(setup.wavelength_vacuum_m); chi=zeros(nz,nl); eta=zeros(nz,nl); I=zeros(nl)
    opacity!(chi,eta,setup.synthesizer,setup.wavelength_vacuum_m,atmosphere,x,y)
    w=F._log_tau_weight(atmosphere,x,y); chi.*=w; eta.*=w
    F.formal_solve!(I,setup.synthesizer.solver,chi,eta,F._log_tau_coordinate(atmosphere,x,y))
    I,chi,eta
end

for (x0,y0) in [(238,257),(380,479),(17,72),(414,478)]
    x,y=x0+1,y0+1
    for (name,centre) in [("Ha",6562.80e-10),("Ca",8542.09e-10)]
        s=findfirst(r->any(abs.(wavelength_air[r].-centre).<1e-10),ranges); setup=setups[s]; range=ranges[s]
        I0,chi,eta=solve(setup,x,y)
        # optical depth along the native coordinate (chi already per unit log10 tau500)
        coordinate=F._log_tau_coordinate(atmosphere,x,y)
        tau=zeros(nz,length(range))
        for l in axes(chi,2), k in 2:nz
            tau[k,l]=tau[k-1,l]+0.5*(chi[k,l]+chi[k-1,l])*abs(coordinate[k]-coordinate[k-1])
        end
        for dl in (-0.15,0.0,0.15)
            l=argmin(abs.(wavelength_air[range].-(centre+dl*1e-10)))
            Rt=response["R/temperature"][range[l],:,y0+1,x0+1]
            k1=findfirst(>=(1.0),tau[:,l])
            @printf("(%d,%d) %s Δλ=%+.2f  τ_λ=1 at logτ500=%s  R_T peak at %.1f  tau at logτ=-3: %.3g\n",x0,y0,name,
                (wavelength_air[range[l]]-centre)*1e10,k1===nothing ? "none" : @sprintf("%.1f",logtau[k1]),
                logtau[argmax(abs.(Rt))],tau[argmin(abs.(logtau.+3)),l])
            # direct FD at a few depths
            for target in (-6.0,-5.0,-4.0,-3.0,-2.0,-1.0)
                k=argmin(abs.(logtau.-target)); T=atmosphere.temperature; t0=T[k,x,y]; h=1e-3*t0
                T[k,x,y]=t0+h; Ip,_,_=solve(setup,x,y); T[k,x,y]=t0-h; Im,_,_=solve(setup,x,y); T[k,x,y]=t0
                fd=(Ip[l]-Im[l])/(2h); ad=Rt[k]*widths[k]
                @printf("    logτ=%5.1f  FD dI/dT_k=% .4e   R·Δ=% .4e   ratio=%.4f\n",logtau[k],fd,ad,fd/ad)
            end
        end
    end
end
