# V5a and the V3 smoothing measurement of docs/viewing-angle/README.md, with prescribed populations (MULTI3D H, 6 levels)
# on the Bifrost cube (heights 0..466 of the netCDF, as MULTI3D), inclined view MU, AZIMUTH:
#   (a) the library: footprint_state! (state and populations shifted by +t z per level, log for positive fields), then
#       per-column opacity and formal solution with path dz/μ — every STRIDE-th column;
#   (b) the opacity and emissivity computed on the Cartesian columns and shifted per wavelength instead of the state;
#   (c) an independent ray tracer for NRAYS random footprints: the state sampled at the true point (ξ + t z, η + t z, z)
#       of the ray, bilinear in x, y (periodic) between two levels and linear in z, at the grid levels (REFINE = 1) and
#       at REFINE = 4 points per level interval; one column per ray through the same opacity code.
# V3: ln n shifted by +t z and back by −t z at full resolution; relative error by height.
# usage: julia --project=../FFNOInversion.jl -t 64 v5a_ray_tracer.jl MU AZIMUTH STRIDE NRAYS OUT.h5
using FFNOInversion, HDF5, Printf, Random
import Muspel
const F=FFNOInversion
const BIFROST="/mn/stornext/d9/data/harshm/halpha_output/BIFROST_en024048_hion_0_504_0_504.nc"
const AUX="/mn/stornext/d9/data/harshm/halpha_output/Multi-3D-H_6_level_populations/output_aux.hdf5"
const R="/mn/stornext/d9/data/harshm/FFNOML/response_run"
include(joinpath(R,"paths.jl"))
mu=parse(Float64,ARGS[1]); azimuth=parse(Float64,ARGS[2]); stride=parse(Int,ARGS[3]); nrays=parse(Int,ARGS[4]); out=ARGS[5]
cd(R); config=load_config(joinpath(R,"forward_multi3d.toml"))
view=ViewGeometry(mu=mu,azimuth_deg=azimuth,geometry=:inclined)

bf=h5open(BIFROST,"r"); levels=1:467
z=vec(read(bf["z"]))[levels]; x=read(bf["x"]); y=read(bf["y"])
rd(name)=permutedims(Float64.(bf[name][levels,:,:,1]),(1,3,2))      # (z, x, y)
full=Dict(n=>rd(n) for n in ("temperature","velocity_x","velocity_y","velocity_z","gas_pressure","density","electron_density"))
close(bf)
# MULTI3D populations: output_aux [a, b] = column (x = a, y = 503 - b)
aux=h5open(AUX,"r"); pk="atom_H/populations"; P=read(aux[pk]); close(aux)
println("MULTI3D populations dataset $pk, Julia size $(size(P))")
# Julia (z, b, a, level) or similar: put into (z, x, y, level)
sz=size(P); il=findfirst(==(6),sz); iz=findfirst(==(467),sz)
others=[d for d in 1:4 if d!=il && d!=iz]
Pz=permutedims(P,(iz,others[2],others[1],il))                        # (z, a, b, level) with h5py (level?, a, b, z) order
Pz=Pz[:,:,end:-1:1,:]                                                # b -> y = 503 - b
P=nothing; GC.gc()
@printf("populations (z, x, y, level) %s; n1 at z=0: %.3e\n",string(size(Pz)),Pz[argmin(abs.(z)),1,1,1])
nz=length(z); nx=length(x); ny=length(y); dx=x[2]-x[1]; dy=y[2]-y[1]; t=slant(view)

# --- V3: shift and back at full resolution
plan_f=ShearPlan(z,dx,dy,t,1); plan_b=ShearPlan(z,dx,dy,t,-1)
v3=zeros(nz,6)
for l in 1:6
    a=log.(Pz[:,:,:,l]); there=shear_levels!(similar(a),a,plan_f); back=shear_levels!(similar(a),there,plan_b)
    d=abs.(exp.(back.-a).-1)
    for k in 1:nz; v3[k,l]=sort(vec(d[k,:,:]))[ceil(Int,0.5*nx*ny)]; end
end
println("V3: median |n_back/n − 1| by height (level 1, 3): ",join([@sprintf("z=%.0f km: %.1e, %.1e",z[k]/1e3,v3[k,1],v3[k,3])
    for k in (argmin(abs.(z.-h)) for h in (0.0,300e3,1000e3,2000e3,5000e3))],"; "))

# --- the stride-STRIDE cube
xs=1:stride:nx; ys=1:stride:ny; gx=collect(x[xs].-x[1]); gy=collect(y[ys].-y[1]); sx,sy=length(xs),length(ys)
sub(A)=A[:,xs,ys]
zfield=repeat(reshape(z,nz,1,1),1,sx,sy)
atmosphere(v)=Atmosphere3D(Grid3D(VerticalAxis(:z_m,z),gx,gy,v),sub(full["temperature"]),sub(full["velocity_x"]),
    sub(full["velocity_y"]),sub(full["velocity_z"]),zeros(nz,sx,sy);pgas=sub(full["gas_pressure"]),rho=sub(full["density"]),
    ne=sub(full["electron_density"]),z=copy(zfield))
pops=Pz[:,xs,ys,:]
eos=WittmannEOS(EOS_LIBRARY,PARTITION_FUNCTIONS)
wavelength=reduce(vcat,wavelengths.(config.regions)); nl=length(wavelength)
context=serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))

# (a) the library
distributed=distribute_atmosphere(Float64,atmosphere(view),context)
ws=HybridForwardWorkspace(Float64,distributed,wavelength,StokesSet(:I),Dict(:H=>6))
F.cartesian_populations(ws)[:H].=pops; F.footprint_state!(ws,distributed,context)
fa=F.synthesis_atmosphere(ws,distributed)
setups=build_synthesis_setup(config,fa,ws.populations,eos;atom_files=ATOM_FILES)
Ia=SpectralCube(zeros(nl,1,sx,sy),wavelength,StokesSet(:I)); synthesize!(Ia,setups,NonPRD(),fa,ws.populations)
println("(a) footprint synthesis done")

# (b) shift the opacities: Cartesian χ, η per column, shifted per wavelength, solved on the footprint columns
cart=atmosphere(ViewGeometry(mu=mu,azimuth_deg=azimuth,geometry=:plane_parallel))   # Cartesian columns, Doppler n·v
cpops=Dict(:H=>copy(pops)); csetups=build_synthesis_setup(config,cart,cpops,eos;atom_files=ATOM_FILES)
Ib=zeros(nl,sx,sy); vac=reduce(vcat,[s.wavelength_vacuum_m for s in csetups])
chi=zeros(nz,nl,sx,sy); eta=zeros(nz,nl,sx,sy)
Threads.@threads for c in 1:sx*sy
    i=(c-1)%sx+1; j=(c-1)÷sx+1; off=0
    for s in csetups
        n=length(s.wavelength_vacuum_m); cc=zeros(nz,n); ee=zeros(nz,n)
        for con in s.synthesizer.contributors
            add_opacity_emissivity!(cc,ee,con.model,s.wavelength_vacuum_m,cart,i,j,con.populations)
        end
        chi[:,off+1:off+n,i,j].=cc; eta[:,off+1:off+n,i,j].=ee; off+=n
    end
end
splan=ShearPlan(z,gx[2]-gx[1],gy[2]-gy[1],t,1)
Threads.@threads for l in 1:nl
    c=chi[:,l,:,:]; e=eta[:,l,:,:]
    chi[:,l,:,:].=exp.(shear_levels!(similar(c),log.(c),splan)); eta[:,l,:,:].=exp.(shear_levels!(similar(e),log.(e),splan))
end
path=z./mu
Threads.@threads for c in 1:sx*sy
    i=(c-1)%sx+1; j=(c-1)÷sx+1
    formal_solve!(@view(Ib[:,i,j]),ScalarFormalSolver(),chi[:,:,i,j],eta[:,:,i,j],path)
end
chi=eta=nothing; GC.gc()
println("(b) opacity-shift synthesis done")

# (c) independent ray tracer through the full-resolution cube, at footprints of the stride grid
rng=MersenneTwister(1); picks=[(rand(rng,1:sx),rand(rng,1:sy)) for _ in 1:nrays]
function sample(A,k,xp,yp)                           # bilinear, periodic, level k of a full-resolution (z, x, y) field
    fx=xp/dx; fy=yp/dy; i0=floor(Int,fx); j0=floor(Int,fy); wx=fx-i0; wy=fy-j0
    p(ii,jj)=A[k,mod(ii,nx)+1,mod(jj,ny)+1]
    (1-wy)*((1-wx)*p(i0,j0)+wx*p(i0+1,j0))+wy*((1-wx)*p(i0,j0+1)+wx*p(i0+1,j0+1))
end
logfields=("temperature","gas_pressure","density","electron_density")
# TRACE_SUBSAMPLED=1: trace through the same every-STRIDE-th cube as (a) and (b), isolating the method from resolution
if get(ENV,"TRACE_SUBSAMPLED","0")=="1"
    for n in keys(full); full[n]=sub(full[n]); end
    Pz=Pz[:,xs,ys,:]; nx,ny=sx,sy; dx*=stride; dy*=stride
end
logf=Dict(n=>log.(full[n]) for n in logfields); logP=log.(Pz)
function rays(refine)
    zs=Float64[]
    for k in 1:nz-1, q in 0:refine-1; push!(zs,z[k]+(z[k+1]-z[k])*q/refine); end
    push!(zs,z[end]); m=length(zs)
    T=zeros(m,nrays,1); vx_=similar(T); vy_=similar(T); vz_=similar(T); pg=similar(T); rh=similar(T); ne_=similar(T)
    pp=zeros(m,nrays,1,6)
    Threads.@threads for r in 1:nrays
        ξ=gx[picks[r][1]]; η=gy[picks[r][2]]
        for (s,zz) in enumerate(zs)
            k=min(searchsortedlast(z,zz;rev=true),nz-1); w=(z[k]-zz)/(z[k]-z[k+1])
            xp=ξ+t[1]*zz; yp=η+t[2]*zz
            lin(A)=(1-w)*sample(A,k,xp,yp)+w*sample(A,k+1,xp,yp)
            T[s,r,1]=exp(lin(logf["temperature"])); pg[s,r,1]=exp(lin(logf["gas_pressure"]))
            rh[s,r,1]=exp(lin(logf["density"])); ne_[s,r,1]=exp(lin(logf["electron_density"]))
            vx_[s,r,1]=lin(full["velocity_x"]); vy_[s,r,1]=lin(full["velocity_y"]); vz_[s,r,1]=lin(full["velocity_z"])
            for l in 1:6; pp[s,r,1,l]=exp((1-w)*sample(@view(logP[:,:,:,l]),k,xp,yp)+w*sample(@view(logP[:,:,:,l]),k+1,xp,yp)); end
        end
    end
    g=Grid3D(VerticalAxis(:z_m,zs),collect(1.0:nrays),[0.0],F.footprint_view(view))
    a=Atmosphere3D(g,T,vx_,vy_,vz_,zeros(m,nrays,1);pgas=pg,rho=rh,ne=ne_,z=repeat(reshape(zs,m,1,1),1,nrays,1))
    p=Dict(:H=>pp); st=build_synthesis_setup(config,a,p,eos;atom_files=ATOM_FILES)
    I=SpectralCube(zeros(nl,1,nrays,1),wavelength,StokesSet(:I)); synthesize!(I,st,NonPRD(),a,p)
    I.data[:,1,:,1]
end
Ic1=rays(1); println("(c) ray tracer, grid levels done"); Ic4=rays(4); println("(c) ray tracer, 4x refined done")

core=findall(abs.(wavelength.*1e9 .- 656.296).<=0.03)
pick(I)=hcat([I[:,i,j] for (i,j) in picks]...)
Ia_r=pick(Ia.data[:,1,:,:]); Ib_r=pick(Ib)
rel(A,B)=sqrt(sum(abs2,A./B.-1)/length(A))
for (name,A) in (("(a) state shift",Ia_r),("(b) opacity shift",Ib_r),("(c) ray tracer, grid levels",Ic1))
    @printf("%-28s vs (c) 4x refined: rms relative %.4f (core %.4f), max %.4f\n",name,rel(A,Ic4),rel(A[core,:],Ic4[core,:]),
        maximum(abs.(A./Ic4.-1)))
end
@printf("(a) state shift vs (b) opacity shift: rms relative %.4f (core %.4f)\n",rel(Ia_r,Ib_r),rel(Ia_r[core,:],Ib_r[core,:]))
@printf("(a) state shift vs (c) grid levels:   rms relative %.4f (core %.4f)\n",rel(Ia_r,Ic1),rel(Ia_r[core,:],Ic1[core,:]))
h5open(out,"w") do f
    f["wavelength_air_m"]=wavelength; f["a"]=Ia_r; f["b"]=Ib_r; f["c1"]=Ic1; f["c4"]=Ic4; f["v3_median"]=v3; f["z"]=z
    f["picks"]=hcat([collect(p) for p in picks]...)
end
println("wrote $out")
