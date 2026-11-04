# V6 of docs/viewing-angle/README.md: what the FFNO sees for an inclined native-log-τ atmosphere.
#
# 1. Rays toward an observer at μ, φ through every footprint of the Bifrost cube (x = ξ + t (z − z_ref)): T, vx, vy,
#    vz and χ500 (HSEMHS EOS table, from Bifrost T and P) shifted to the footprints level by level (FFNOInversion's
#    ShearPlan, the HSEMHS ViewTau convention), τ along each ray (HSEMHS tau_column!, path dz/μ), and T, v
#    interpolated onto the log τ500 axis of the forward atmosphere (-8 ... 1.4). z_ref is the mean height of
#    τ_LOS = 1 found in a first pass, so the footprints sit where HSEMHS's gauge puts z = 0.
# 2. That τ_LOS atmosphere through the HSEMHS column balance with the view (FFNOInversion's equilibrium adapter): ρ, nₑ, z.
# 3. The population mesh: features along each ray, then (inclined) shifted to their Cartesian positions — exactly the
#    arrays the FFNO receives — at the mesh levels nearest z = 0, 300 and 1200 km; also the unshifted ray features and
#    the Bifrost cube at the same heights (heights in the HSEMHS gauge: Bifrost z − z_ref).
#
# usage: julia --project=../FFNOInversion.jl -t 32 ffno_input_maps.jl MU AZIMUTH_DEG OUT.h5
using FFNOInversion, HDF5, Printf
import HSEMHS
const F=FFNOInversion
const BIFROST="/mn/stornext/d9/data/harshm/halpha_output/BIFROST_en024048_hion_0_504_0_504.nc"
const FORWARD="/mn/stornext/d9/data/harshm/FFNOML/outputs/forward_atmosphere.h5"
const CONFIG="/mn/stornext/d9/data/harshm/FFNOML/response_run/forward.toml"
const HEIGHTS=(0.0,300e3,1200e3)

mu=parse(Float64,ARGS[1]); azimuth=parse(Float64,ARGS[2]); out=ARGS[3]
view=ViewGeometry(mu=mu,azimuth_deg=azimuth,geometry=:inclined); t=slant(view)
config=load_config(CONFIG); table=HSEMHS.load_table(config.equilibrium.eos_table_file)
lt=h5read(FORWARD,"logtau_500"); nt=length(lt)

bf=h5open(BIFROST,"r")
z=vec(read(bf["z"])); x=read(bf["x"]); y=read(bf["y"]); nz=length(z); nx=length(x); ny=length(y)
dx=x[2]-x[1]; dy=y[2]-y[1]
rd(name)=permutedims(Float64.(bf[name][:,:,:,1]),(1,3,2))        # Julia view (z, y, x, t) -> (z, x, y), z top first
lT=log.(rd("temperature")); lP=log.(rd("gas_pressure")); vx=rd("velocity_x"); vy=rd("velocity_y"); vz=rd("velocity_z")
ne_b=rd("electron_density"); rho_b=rd("density")
close(bf)
@printf("μ %.3f φ %.1f°: t = (%.4f, %.4f); %d levels %.2f … %.2f Mm; %d x %d columns, dx %.1f km\n",mu,azimuth,t...,nz,
    z[1]/1e6,z[end]/1e6,nx,ny,dx/1e3)
chi=similar(lT)
Threads.@threads for i in eachindex(lT)
    c=HSEMHS.state(table,lT[i],lP[i])[3]; chi[i]=isfinite(c) ? exp(c) : 0.0
end

"""Footprint values of `A` for rays x = ξ + t (z − zref)."""
shift(A,zref)=shear_levels!(similar(A),A,ShearPlan(z .- zref,dx,dy,t,1))
"""τ along every footprint ray from the shifted χ."""
function ray_tau(schi)
    tau=similar(schi)
    Threads.@threads for i in 1:nx
        tc=zeros(nz)
        for j in 1:ny
            HSEMHS.tau_column!(tc,@view(schi[:,i,j]),z;mu=mu); tau[:,i,j].=tc
        end
    end
    tau
end
z_tau1(tau)=[HSEMHS.z_tau1(@view(tau[:,i,j]),z) for i in 1:nx, j in 1:ny]

# first pass: the mean height of τ_LOS = 1 is the ray origin (HSEMHS gauge z = 0)
zref=sum(z_tau1(ray_tau(shift(chi,0.0))))/(nx*ny)
@printf("mean z(τ_LOS = 1) of the rays through footprints at z = 0: %.1f km -> z_ref\n",zref/1e3)
tau=ray_tau(shift(chi,zref)); s_lT=shift(lT,zref); s_vx=shift(vx,zref); s_vy=shift(vy,zref); s_vz=shift(vz,zref)

function interp_at(v,lτ,L)
    L<lτ[1] && return v[1]                                   # rays whose top is below the axis minimum: hold the top
    L>lτ[end] && error("a ray does not reach log tau = $L")
    k=searchsortedlast(lτ,L); k==length(lτ) && return v[end]
    w=(L-lτ[k])/(lτ[k+1]-lτ[k]); v[k]+w*(v[k+1]-v[k])
end
T_in=zeros(nt,nx,ny); vx_in=similar(T_in); vy_in=similar(T_in); vz_in=similar(T_in)
Threads.@threads for i in 1:nx
    for j in 1:ny
        lτ=log10.(max.(@view(tau[:,i,j]),1e-300))
        for k in 1:nt
            T_in[k,i,j]=exp(interp_at(@view(s_lT[:,i,j]),lτ,lt[k])); vx_in[k,i,j]=interp_at(@view(s_vx[:,i,j]),lτ,lt[k])
            vy_in[k,i,j]=interp_at(@view(s_vy[:,i,j]),lτ,lt[k]); vz_in[k,i,j]=interp_at(@view(s_vz[:,i,j]),lτ,lt[k])
        end
    end
end
s_lT=s_vx=s_vy=s_vz=tau=chi=nothing; GC.gc()

# 2. HSEMHS column balance with the view, through the FFNOInversion adapter
grid=Grid3D(VerticalAxis(:log10tau500,lt),x .- x[1],y .- y[1],view)
atmosphere=Atmosphere3D(grid,T_in,vx_in,vy_in,vz_in,zeros(nt,nx,ny))
context=initialize_parallel(options=ParallelOptions(enabled=true,decomposition=:x_slabs,threads_per_rank=Threads.nthreads()))
distributed=distribute_atmosphere(Float64,atmosphere,context)
backend=prepare_equilibrium(config.equilibrium,distributed,context)
outcome=reconstruct_equilibrium!(backend,distributed,context)
println("HSEMHS column balance: valid=$(outcome.valid) reason=$(outcome.reason) seconds=$(round(outcome.seconds;digits=1))")
outcome.valid || error("equilibrium failed")
a=distributed.local_atmosphere

# 3. the population mesh: ray features and the Cartesian features the FFNO receives
struct Capture <: AbstractPopulationModel; seen::Vector{Any}; end
F.predict_populations!(o,m::Capture,atm::Atmosphere3D,cache=nothing)=(push!(m.seen,atm); fill!(o,1.0); o)
options=population_mesh_options(config)
model=MeshPopulationModel(Capture(Any[]),options.config,table,options.gravity)
predict_populations!(zeros(nt,nx,ny,1),model,a)
mesh=model.mesh; cart=model.inner.seen[end]; rays=model.last_features
levels=[argmin(abs.(mesh .- h)) for h in HEIGHTS]
println("mesh levels: ",[@sprintf("%.1f km",mesh[m]/1e3) for m in levels])

"""Bifrost field `A` (z, x, y) at HSEMHS-gauge height h (Bifrost z = h + zref), log-linear for positive fields."""
function bifrost_at(A,h,logarithmic)
    zz=h+zref; k=findlast(>=(zz),z); w=(z[k]-zz)/(z[k]-z[k+1])
    logarithmic ? exp.((1-w).*log.(A[k,:,:]).+w.*log.(A[k+1,:,:])) : (1-w).*A[k,:,:].+w.*A[k+1,:,:]
end
h5open(out,"w") do f
    for (h,m) in zip(HEIGHTS,levels)
        g=create_group(f,@sprintf("z%04dkm",round(Int,h/1e3)))
        g["mesh_height_m"]=mesh[m]
        g["ffno_T"]=cart.temperature[m,:,:]; g["ffno_vz"]=cart.vz[m,:,:]; g["ffno_vx"]=cart.vx[m,:,:]
        g["ffno_log10_ne"]=log10.(cart.ne[m,:,:]); g["ffno_log10_rho"]=log10.(cart.rho[m,:,:])
        g["ray_T"]=rays[1,m,:,:]; g["ray_vz"]=rays[4,m,:,:]; g["ray_log10_ne"]=rays[5,m,:,:]; g["ray_log10_rho"]=rays[6,m,:,:]
        g["bifrost_T"]=bifrost_at(exp.(lT),mesh[m],true); g["bifrost_vz"]=bifrost_at(vz,mesh[m],false)
        g["bifrost_vx"]=bifrost_at(vx,mesh[m],false)
        g["bifrost_log10_ne"]=log10.(bifrost_at(ne_b,mesh[m],true)); g["bifrost_log10_rho"]=log10.(bifrost_at(rho_b,mesh[m],true))
    end
    for (k,v) in view_attributes(view,"cartesian mesh features"); attributes(f)[k]=v; end
    attributes(f)["z_ref_bifrost_m"]=zref; attributes(f)["x_m"]=collect(x .- x[1]); attributes(f)["dx_m"]=dx
end
println("wrote $out")
finalize_parallel!(context)
