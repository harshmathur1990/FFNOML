# Rest wavelength of the H-alpha line model and the line minimum of one column with vz = 0.
using FFNOInversion, HDF5, Printf
import Muspel
const F = FFNOInversion
include(abspath("paths.jl")); include("check_lib.jl")
config=load_config("forward_multi3d.toml")
atmosphere,populations=read_forward_atmosphere("inputs/multi3d_atmosphere.h5")
eos=WittmannEOS(EOS_LIBRARY,PARTITION_FUNCTIONS)
setups=build_synthesis_setup(config,atmosphere,populations,eos;atom_files=ATOM_FILES)
setup=setups[1]; model=setup.synthesizer.contributors[1].model
println(typeof(model)); for n in fieldnames(typeof(model)); v=getfield(model,n); println(n," => ",summary(v)); end
for n in fieldnames(typeof(model))
    v=getfield(model,n); hasproperty(v,:λ0) && println("λ0 = ",v.λ0)
end
nz=95; w=setup.wavelength_vacuum_m
for (x,y) in [(414,478),(17,72),(100,200)]
    x+=1; y+=1
    function solve()
        chi=zeros(nz,length(w)); eta=zeros(nz,length(w)); I=zeros(length(w))
        opacity!(chi,eta,setup.synthesizer,w,atmosphere,x,y)
        wt=F._log_tau_weight(atmosphere,x,y); chi.*=wt; eta.*=wt
        F.formal_solve!(I,setup.synthesizer.solver,chi,eta,F._log_tau_coordinate(atmosphere,x,y)); I
    end
    I=solve(); saved=copy(atmosphere.vz[:,x,y]); atmosphere.vz[:,x,y].=0; I0=solve(); atmosphere.vz[:,x,y].=saved
    @printf("(%d,%d) vacuum min: with vz %.3f Å, vz=0 %.3f Å\n",x-1,y-1,w[argmin(I)]*1e10,w[argmin(I0)]*1e10)
end
