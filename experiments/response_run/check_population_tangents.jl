# CPU check of the population-tangent chain without the GPU service. A stand-in for the FFNO on the population mesh:
#   n_l(m, x, y) = P_l(m, x, y) (T/T0)^a_l (ne/ne0)^b_l (T̄/T̄0)^c,   T̄ = mean T of the 4 horizontal neighbours,
# nonlinear and coupling neighbouring columns, with its exact JVP. On an 8 x 8 crop of the forward atmosphere:
#   1. MeshPopulationModel's population_jvp! against central differences of predict_populations!
#   2. column_response.jl --population-tangents against central differences of the whole synthesis, T raised at one
#      depth in every column of the crop (populations re-predicted).
# usage: julia check_population_tangents.jl   (from response_run, FFNOML_RUN_DIR set)
using FFNOInversion, HDF5, Printf
import Muspel, HSEMHS
const F = FFNOInversion
include(abspath("paths.jl")); include(joinpath(@__DIR__,"..","FFNOInversion.jl","scripts","forward_atmosphere_io.jl"))
include(joinpath(@__DIR__,"check_lib.jl"))

const A=[1.5,4.0,6.0,7.0,7.5,-0.5]; const B=[0.2,0.6,0.8,0.9,1.0,-0.3]; const C=0.8
struct StandIn <: F.AbstractPopulationModel
    P::Array{Float64,4}; T0::Array{Float64,3}; ne0::Array{Float64,3}; Tn0::Array{Float64,3}
end
neighbours(T)=(circshift(T,(0,1,0)).+circshift(T,(0,-1,0)).+circshift(T,(0,0,1)).+circshift(T,(0,0,-1)))./4
function F.predict_populations!(out::AbstractArray{T,4},m::StandIn,atm::F.Atmosphere3D,cache=nothing) where T
    t=Float64.(atm.temperature); ne=Float64.(atm.ne); tn=neighbours(t)
    for l in 1:6
        out[:,:,:,l].=m.P[:,:,:,l].*(t./m.T0).^A[l].*(ne./m.ne0).^B[l].*(tn./m.Tn0).^C
    end
    out
end
function F.population_jvp!(out::AbstractArray{T,4},m::StandIn,atm::F.Atmosphere3D,fd::AbstractArray{<:Real,4}) where T
    p=zeros(size(out)); F.predict_populations!(p,m,atm)
    t=Float64.(atm.temperature); tdot=Float64.(fd[1,:,:,:]); lnedot=log(10).*Float64.(fd[5,:,:,:])
    tn=neighbours(t); tndot=neighbours(tdot)
    for l in 1:6
        out[:,:,:,l].=p[:,:,:,l].*(A[l].*tdot./t.+B[l].*lnedot.+C.*tndot./tn)
    end
    out
end

full,stored=read_forward_atmosphere(abspath("../outputs/forward_atmosphere.h5"))
xs=101:108; ys=201:208; crop(a)=a===nothing ? nothing : a[:,xs,ys]
g=full.grid
atm=F.Atmosphere3D(F.Grid3D(g.vertical,g.x[xs],g.y[ys]),crop(full.temperature),crop(full.vx),crop(full.vy),
    crop(full.vz),crop(full.vturb);pgas=crop(full.pgas),rho=crop(full.rho),ne=crop(full.ne),z=crop(full.z),
    logtau500=crop(full.logtau500),chi500=crop(full.chi500))
nz,nx,ny=size(atm.temperature)
config=load_config("forward.toml"); options=population_mesh_options(config)
eos_table=HSEMHS.load_table(options.eos_table_file)

# Stand-in reference populations on the mesh: the stored H populations interpolated log-linearly in height.
probe=MeshPopulationModel(F.MockPopulationModel(),options.config,eos_table,options.gravity)
f=F._mesh_features(probe,atm); mesh=probe.mesh; nm=length(mesh)
P=zeros(nm,nx,ny,6)
for x in 1:nx, y in 1:ny, l in 1:6
    z=atm.z[:,x,y]; lp=log.(stored[:H][:,xs[x],ys[y],l])
    for m in 1:nm
        P[m,x,y,l]=exp(mesh[m]>=z[1] ? lp[1] : mesh[m]<=z[end] ? lp[end] :
            (k=findlast(>=(mesh[m]),z); w=(z[k]-mesh[m])/(z[k]-z[k+1]); (1-w)*lp[k]+w*lp[k+1]))
    end
end
T0=f[1,:,:,:]; ne0=exp10.(f[5,:,:,:])
model=MeshPopulationModel(StandIn(P,T0,ne0,neighbours(T0)),options.config,eos_table,options.gravity)
H=zeros(nz,nx,ny,6); predict_populations!(H,model,atm)

# 1. mesh JVP vs central differences, 1 K at depth k0 in every column
println("1. population mesh JVP against central differences (T raised at k0 in every column)")
for k0 in (30,60,90)
    fd=zeros(6,nz,nx,ny); fd[1,k0,:,:].=1; jvp=zeros(nz,nx,ny,6); population_jvp!(jvp,model,atm,fd)
    h=0.5; Hp=zeros(nz,nx,ny,6); Hm=zeros(nz,nx,ny,6)
    atm.temperature[k0,:,:].+=h; model.last_features=nothing; predict_populations!(Hp,model,atm)
    atm.temperature[k0,:,:].-=2h; model.last_features=nothing; predict_populations!(Hm,model,atm)
    atm.temperature[k0,:,:].+=h; model.last_features=nothing
    num=(Hp.-Hm)./(2h)
    @printf("   k0=%d (log tau %.1f): max |JVP - FD| / max |FD| = %.2e,  max |ṅ/n| = %.2e per K\n",k0,
        F.native_depth(atm.grid)[k0],maximum(abs.(jvp.-num))/maximum(abs.(num)),maximum(abs.(jvp)./H))
end

# 2. end to end through column_response.jl
function write_atmosphere(path,atm,pops)
    h5open(path,"w") do file
        w(name,a)=(file[name]=reshape(permutedims(a,(1,3,2)),1,size(a,1),size(a,3),size(a,2)))
        for name in (:temperature,:vx,:vy,:vz,:vturb,:pgas,:rho,:ne,:z,:logtau500,:chi500)
            w(string(name),getfield(atm,name))
        end
        file["logtau_500"]=collect(F.native_depth(atm.grid)); file["x_m"]=collect(atm.grid.x); file["y_m"]=collect(atm.grid.y)
        for (s,p) in pops; file["populations/$s"]=permutedims(p,(4,1,3,2)); end
    end
end
mkpath("outputs/check"); pops=Dict(:H=>H,:CA=>stored[:CA][:,xs,ys,:])
write_atmosphere("outputs/check/atmosphere.h5",atm,pops)
h5open("outputs/check/tangents.h5","w") do file
    file["depth_index"]=collect(1:nz); file["logtau500"]=collect(F.native_depth(atm.grid))
    d=create_dataset(file,"tangent/H",datatype(Float32),dataspace(nz,ny,6,nz,nx))
    fd=zeros(6,nz,nx,ny); jvp=zeros(nz,nx,ny,6)
    for k0 in 1:nz
        fill!(fd,0); fd[1,k0,:,:].=1; population_jvp!(jvp,model,atm,fd); d[:,:,:,k0,:]=Float32.(permutedims(jvp,(1,3,4,2)))
    end
end
script=joinpath(@__DIR__,"..","FFNOInversion.jl","scripts","column_response.jl")
for (out,extra) in (("full.h5",["--population-tangents","outputs/check/tangents.h5"]),("fixed.h5",String[]))
    run(`$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=8 $script forward.toml paths.jl outputs/check/atmosphere.h5 outputs/check/$out --variables temperature --stride 1 $extra`)
end

setups=build_synthesis_setup(config,atm,pops,WittmannEOS(EOS_LIBRARY,PARTITION_FUNCTIONS);atom_files=ATOM_FILES)
function intensity(x,y)
    I=Float64[]
    for s in setups
        w=s.wavelength_vacuum_m; chi=zeros(nz,length(w)); eta=zeros(nz,length(w)); out=zeros(length(w))
        opacity!(chi,eta,s.synthesizer,w,atm,x,y)
        wt=F._log_tau_weight(atm,x,y); chi.*=wt; eta.*=wt
        F.formal_solve!(out,s.synthesizer.solver,chi,eta,F._log_tau_coordinate(atm,x,y)); append!(I,out)
    end
    I
end
full_R=h5read("outputs/check/full.h5","R/temperature"); fixed_R=h5read("outputs/check/fixed.h5","R/temperature")
widths=h5read("outputs/check/full.h5","dlogtau500")
println("2. column_response.jl --population-tangents against central differences of the synthesis")
x,y=4,5
for k0 in (40,60,75)
    h=0.5; saved=copy(atm.temperature)
    atm.temperature[k0,:,:].+=h; model.last_features=nothing; predict_populations!(H,model,atm); Ip=intensity(x,y)
    atm.temperature.=saved; atm.temperature[k0,:,:].-=h; model.last_features=nothing; predict_populations!(H,model,atm)
    Im=intensity(x,y); atm.temperature.=saved; model.last_features=nothing; predict_populations!(H,model,atm)
    num=(Ip.-Im)./(2h); r=full_R[:,k0,y,x].*widths[k0]; rf=fixed_R[:,k0,y,x].*widths[k0]
    @printf("   k0=%d (log tau %.1f): max |R_full Δ - FD| / max |FD| = %.2e   (populations fixed: %.2e)\n",k0,
        F.native_depth(atm.grid)[k0],maximum(abs.(r.-num))/maximum(abs.(num)),maximum(abs.(rf.-num))/maximum(abs.(num)))
end
