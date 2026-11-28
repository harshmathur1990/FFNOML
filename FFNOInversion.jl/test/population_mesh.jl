using Test
using FFNOInversion
using LinearAlgebra
import HSEMHS

import WittBifrostEOS
const MESH_EOS_TABLE=get(ENV,"HSEMHS_EOS_TABLE","/mn/stornext/d9/data/harshm/hsemhs_data/cache/eos_table_fullz.h5")

"""CPU population model with an exact VJP: ln p_l = sum_c A[l,c] phi_c + B[l] z/1e6, phi from the FFNO features."""
struct SmoothMeshTestPopulations <: AbstractPopulationModel
    A::Matrix{Float64}
    B::Vector{Float64}
end
_mesh_test_phi(a,k,x,y)=(a.temperature[k,x,y]/5000,a.vx[k,x,y]/1e4,a.vy[k,x,y]/1e4,a.vz[k,x,y]/1e4,
    log10(a.ne[k,x,y])/20,log10(a.rho[k,x,y])/10)
const _MESH_TEST_SCALE=(1/5000,1e-4,1e-4,1e-4,1/20,1/10)
function FFNOInversion.predict_populations!(out,m::SmoothMeshTestPopulations,a::Atmosphere3D,cache=nothing)
    for I in CartesianIndices(a.temperature), l in axes(out,4)
        k,x,y=Tuple(I); phi=_mesh_test_phi(a,k,x,y)
        out[k,x,y,l]=exp(sum(m.A[l,c]*phi[c] for c in 1:6)+m.B[l]*a.z[k,x,y]/1e6)
    end
    out
end
function FFNOInversion.population_vjp!(feature_bar,z_bar,m::SmoothMeshTestPopulations,a::Atmosphere3D,population_bar)
    p=zeros(size(population_bar)); FFNOInversion.predict_populations!(p,m,a)
    fill!(feature_bar,0); fill!(z_bar,0)
    for I in CartesianIndices(a.temperature), l in axes(p,4)
        k,x,y=Tuple(I); g=population_bar[k,x,y,l]*p[k,x,y,l]
        for c in 1:6; feature_bar[c,k,x,y]+=g*m.A[l,c]*_MESH_TEST_SCALE[c]; end
        z_bar[k,x,y]+=g*m.B[l]/1e6
    end
    feature_bar,z_bar
end

"""Log-tau atmosphere (top first) with photosphere-like T and rho and HSE-like heights varying by column."""
function mesh_test_atmosphere(;nx=3,ny=2,lt=collect(range(-6.0,1.4,length=38)))
    nz=length(lt); tau=exp10.(lt); f(g)=[g(k,x,y) for k in 1:nz, x in 1:nx, y in 1:ny]
    T=f((k,x,y)->(4300+1700*tau[k]^0.25)*(1+0.02x-0.03y))
    rho=f((k,x,y)->3e-4*tau[k]^0.45*(1+0.05y))
    ne=f((k,x,y)->1e19*tau[k]^0.6*(1+0.1x))
    z=f((k,x,y)->-1.2e5*log(tau[k])*(1+0.03x)+2e4*y)
    v(s)=f((k,x,y)->s*sin(0.3k+x-y))
    grid=Grid3D(VerticalAxis(:log10tau500,lt),collect(0.0:nx-1).*4.8e4,collect(0.0:ny-1).*4.8e4)
    Atmosphere3D(grid,T,v(1e3),v(-7e2),v(2e3),zeros(nz,nx,ny);rho=rho,ne=ne,z=z,
        logtau500=repeat(reshape(lt,nz,1,1),1,nx,ny))
end

@testset "Population mesh for native log-tau atmospheres" begin
    mesh=population_mesh(PopulationMeshConfig())
    @test issorted(mesh;rev=true) && mesh[1]==1.4e7 && mesh[end]==-1.0e6
    @test all(abs.(diff(mesh[end-200:end]).+2e4).<1e-6)                     # 20 km through the lower atmosphere
    @test maximum(-diff(mesh))<=1e5*1.5 && 300<length(mesh)<500
    @test_throws ArgumentError PopulationMeshConfig(bottom_m=5e6)
    @test FFNOInversion._population_mesh(Dict{String,Any}(),:log10tau500) isa PopulationMeshConfig
    @test FFNOInversion._population_mesh(Dict{String,Any}(),:z_m) isa PopulationMeshConfig
    @test FFNOInversion._population_mesh(Dict{String,Any}("mesh"=>Dict{String,Any}("enabled"=>false)),:log10tau500)===nothing
    @test FFNOInversion._population_mesh(Dict{String,Any}("mesh"=>Dict{String,Any}("spacing_m"=>1.5e4)),:log10tau500).spacing_m==1.5e4
    @test_throws ArgumentError FFNOInversion._population_mesh(Dict{String,Any}("mesh"=>Dict{String,Any}("dz"=>1)),:log10tau500)

    eos=WittBifrostEOS.load_table(MESH_EOS_TABLE)
    config=PopulationMeshConfig(bottom_m=-1e6,top_m=3e6,spacing_m=2e4,uniform_top_m=2e6,stretch=1.05,max_spacing_m=5e4)
    inner=SmoothMeshTestPopulations([0.8 0.1 -0.2 0.3 1.1 0.9; -0.5 0.2 0.1 -0.4 0.7 1.3],[0.4,-0.6])
    model=MeshPopulationModel(inner,config,eos,274.0)
    atmos=mesh_test_atmosphere(); nz,nx,ny=size(atmos.temperature)

    # inside a column the interpolation is exact for variables linear in z
    lin=deepcopy(atmos)
    for x in 1:nx, y in 1:ny, k in 1:nz
        zz=lin.z[k,x,y]/1e6
        lin.temperature[k,x,y]=exp(8.6-0.2zz); lin.vx[k,x,y]=100+50zz; lin.ne[k,x,y]=exp10(18-zz); lin.rho[k,x,y]=exp10(-4-0.8zz)
    end
    f=FFNOInversion._mesh_features(model,lin); m=model.mesh
    for x in 1:nx, y in 1:ny
        inside=findall(z->lin.z[end,x,y]<=z<=lin.z[1,x,y],m)
        @test isapprox(log.(f[1,inside,x,y]),8.6 .- 0.2 .* m[inside]./1e6;rtol=1e-12)
        @test isapprox(f[2,inside,x,y],100 .+ 50 .* m[inside]./1e6;rtol=1e-12)
        @test isapprox(f[5,inside,x,y],18 .- m[inside]./1e6;rtol=1e-12)
        @test isapprox(f[6,inside,x,y],-4 .- 0.8 .* m[inside]./1e6;rtol=1e-12)
    end

    # below the column: hydrostatic equilibrium with d ln T / d ln P = nabla, velocities held
    f=FFNOInversion._mesh_features(model,atmos); x,y=2,1
    deep=findall(<(atmos.z[end,x,y]),m); @test length(deep)>20
    lnT=log.(f[1,deep,x,y]); lnrho=f[6,deep,x,y].*log(10)
    lnP=[FFNOInversion._mesh_lnP(eos,lnT[i],lnrho[i]) for i in eachindex(deep)]
    lnPb=FFNOInversion._mesh_lnP(eos,log(atmos.temperature[end,x,y]),log(atmos.rho[end,x,y]))
    @test all(isapprox.(lnT.-log(atmos.temperature[end,x,y]),0.13 .* (lnP.-lnPb);rtol=1e-8))
    dlnP=diff(lnP)./(-diff(m[deep]))
    rate=[(exp(lnrho[i])*274/exp(lnP[i])+exp(lnrho[i+1])*274/exp(lnP[i+1]))/2 for i in 1:length(deep)-1]
    @test maximum(abs.(dlnP./rate.-1))<1e-3
    @test all(f[2,deep,x,y].==atmos.vx[end,x,y])

    # back to the column: exact for populations log-linear in z
    loglin=MeshPopulationModel(SmoothMeshTestPopulations(zeros(2,6),[0.4,-0.6]),config,eos,274.0)
    out=zeros(nz,nx,ny,2); predict_populations!(out,loglin,atmos)
    @test isapprox(out[:,:,:,1],exp.(0.4 .* atmos.z ./ 1e6);rtol=1e-12)
    @test isapprox(out[:,:,:,2],exp.(-0.6 .* atmos.z ./ 1e6);rtol=1e-12)

    # VJP against central differences in every input (temperature, velocities, ne, rho and heights)
    bar=reshape(cos.(1:nz*nx*ny*2),nz,nx,ny,2)
    objective(a)=(p=zeros(nz,nx,ny,2); predict_populations!(p,model,a); sum(bar.*p))
    feature_bar=zeros(6,nz,nx,ny); z_bar=zeros(nz,nx,ny)
    objective(atmos); population_vjp!(feature_bar,z_bar,model,atmos,bar)
    dT=0.01 .* sin.(1:nz*nx*ny); dv=reshape(cos.(2.0 .* (1:nz*nx*ny)),nz,nx,ny); dL=0.02 .* cos.(1:nz*nx*ny)
    dR=0.03 .* sin.(3.0 .* (1:nz*nx*ny)); dz=reshape(5.0 .* sin.(0.7 .* (1:nz*nx*ny)),nz,nx,ny)
    function moved(h)
        a=deepcopy(atmos)
        a.temperature.*=1 .+ h.*reshape(dT,nz,nx,ny); a.vx.+=h.*1e2 .* dv; a.vz.-=h.*3e2 .* dv
        a.ne.*=exp10.(h.*reshape(dL,nz,nx,ny)); a.rho.*=exp10.(h.*reshape(dR,nz,nx,ny)); a.z.+=h.*dz
        a
    end
    h=1e-4
    fd=(objective(moved(h))-objective(moved(-h)))/(2h)
    predicted=sum(feature_bar[1,:,:,:].*atmos.temperature.*reshape(dT,nz,nx,ny))+
        sum(feature_bar[2,:,:,:].*1e2 .* dv)-sum(feature_bar[4,:,:,:].*3e2 .* dv)+
        sum(feature_bar[5,:,:,:].*reshape(dL,nz,nx,ny))+sum(feature_bar[6,:,:,:].*reshape(dR,nz,nx,ny))+sum(z_bar.*dz)
    @test isapprox(predicted,fd;rtol=1e-5)
    @test any(!iszero,z_bar) && any(!iszero,feature_bar[6,end,:,:])          # heights and the deep extension enter

    # a native height grid bypasses the mesh
    zgrid=Grid3D(VerticalAxis(:z_m,collect(range(1e6,-1e5,length=5))),[0.0,4.8e4],[0.0])
    za=Atmosphere3D(zgrid,fill(5000.0,5,2,1),zeros(5,2,1),zeros(5,2,1),zeros(5,2,1),zeros(5,2,1);
        rho=fill(1e-5,5,2,1),ne=fill(1e17,5,2,1),z=repeat(reshape(collect(range(1e6,-1e5,length=5)),5,1,1),1,2,1))
    direct=zeros(5,2,1,2); wrapped=zeros(5,2,1,2)
    predict_populations!(direct,inner,za); predict_populations!(wrapped,model,za)
    @test wrapped==direct
end

@testset "Explicit and adaptive population meshes" begin
    parse(mesh)=FFNOInversion._population_mesh(Dict{String,Any}("mesh"=>mesh),:log10tau500)
    c=parse(Dict{String,Any}("heights_m"=>[2e6,1e6,0.0,-5e5]))
    @test population_mesh(c)==[2e6,1e6,0.0,-5e5] && !c.adaptive
    @test_throws ArgumentError parse(Dict{String,Any}("heights_m"=>[0.0,1e6]))
    @test_throws ArgumentError parse(Dict{String,Any}("refinement"=>"dynamic"))
    @test_throws ArgumentError parse(Dict{String,Any}("refinement"=>"adaptive","growth"=>1.0))
    c=parse(Dict{String,Any}("refinement"=>"adaptive","lnT_step"=>0.1,"min_spacing_m"=>5e3))
    @test c.adaptive && c.lnT_step==0.1 && c.min_spacing_m==5e3 && c.corona_temperature_k==1e6
    c=parse(Dict{String,Any}()); @test c.adaptive && c.extend && c.max_spacing_m==2e5 && c.min_spacing_m==1e4   # the default
    c=parse(Dict{String,Any}("refinement"=>"fixed")); @test !c.adaptive && c.max_spacing_m==1e5
    @test !parse(Dict{String,Any}("extend"=>false)).extend

    eos=WittBifrostEOS.load_table(MESH_EOS_TABLE)
    inner=SmoothMeshTestPopulations([0.8 0.1 -0.2 0.3 1.1 0.9; -0.5 0.2 0.1 -0.4 0.7 1.3],[0.4,-0.6])
    settings=(bottom_m=-1e6,top_m=3e6,uniform_top_m=2e6,max_spacing_m=1e5,min_spacing_m=5e3,growth=1.2)
    model=MeshPopulationModel(inner,PopulationMeshConfig(;settings...,adaptive=true),eos,274.0)
    atmos=mesh_test_atmosphere(); nz,nx,ny=size(atmos.temperature)
    out=zeros(nz,nx,ny,2); predict_populations!(out,model,atmos); m=model.mesh
    @test all(FFNOInversion.adaptive_population_mesh(model,atmos)==m for _ in 1:4)     # deterministic
    @test m[1]==3e6 && m[end]==-1e6 && issorted(m;rev=true)
    d=-diff(m)
    @test all(5e3*(1-1e-9) .<= d[1:end-1] .<= 1e5*(1+1e-9))
    @test maximum(d[2:end-1]./d[1:end-2])<=1.2*1.01 && maximum(d[1:end-2]./d[2:end-1])<=1.2*1.01
    # the criterion holds on every step that is not at the minimum spacing (to the probe's resolution)
    f=FFNOInversion._mesh_features(model,atmos); c=model.adaptive
    for i in findall(>(1.01*5e3),d[1:end-1])
        @test maximum(abs.(log.(f[1,i,:,:]./f[1,i+1,:,:])))<=1.5c.lnT_step
        @test maximum(abs.(f[6,i,:,:].-f[6,i+1,:,:]))<=1.5c.log_rho_step
    end
    # a steeper atmosphere gets more heights; a twice looser criterion fewer
    steep=deepcopy(atmos)             # T, nₑ and ρ changing 1.5 times as fast, the bottom (and the deep extension) kept
    for a in (steep.temperature,steep.ne,steep.rho); b=a[end:end,:,:]; a.=b.*(a./b).^1.5; end
    predict_populations!(out,model,steep); @test length(model.mesh)>length(m)
    loose=MeshPopulationModel(inner,PopulationMeshConfig(;settings...,adaptive=true,lnT_step=0.1,log_ne_step=0.2,
        log_rho_step=0.2,velocity_step_m_s=2e3),eos,274.0)
    predict_populations!(out,loose,atmos); @test length(loose.mesh)<length(m)
    # the VJP sees the mesh of the prediction (one mesh per atmosphere)
    predict_populations!(out,model,atmos); m=copy(model.mesh)
    population_vjp!(zeros(6,nz,nx,ny),zeros(nz,nx,ny),model,atmos,ones(nz,nx,ny,2)); @test model.mesh==m
    # extend = false: only between the columns' own lowest and highest point
    inside=MeshPopulationModel(inner,PopulationMeshConfig(;settings...,adaptive=true,extend=false),eos,274.0)
    predict_populations!(out,inside,atmos); mi=inside.mesh
    @test mi[1]==maximum(atmos.z[1,:,:]) && mi[end]==minimum(atmos.z[end,:,:]) && length(mi)<length(m)
    @test all(5e3*(1-1e-9) .<= -diff(mi)[1:end-1])
    # an explicit mesh equal to the adaptive one gives the same populations
    fixed=MeshPopulationModel(inner,PopulationMeshConfig(;settings...,heights_m=m),eos,274.0)
    ref=zeros(nz,nx,ny,2); predict_populations!(ref,fixed,atmos); predict_populations!(out,model,atmos)
    @test ref==out
    # an inversion freezes the adaptive mesh on its starting atmosphere: a mesh rebuilt for every trial changes in steps,
    # so the objective would jump between nearby controls while the gradient sees a fixed mesh
    @test !parse(Dict{String,Any}()).adapt_during_inversion
    @test parse(Dict{String,Any}("adapt_during_inversion"=>true)).adapt_during_inversion
    frozen=MeshPopulationModel(inner,PopulationMeshConfig(;settings...,adaptive=true),eos,274.0)
    freeze_population_mesh!(frozen)                       # before any prediction: the first prediction's mesh is kept
    predict_populations!(out,frozen,atmos); @test frozen.mesh==m
    predict_populations!(out,frozen,steep); @test frozen.mesh==m
    # it is the explicit mesh: same populations and the same VJP for another atmosphere
    explicit=MeshPopulationModel(inner,PopulationMeshConfig(;settings...,heights_m=m),eos,274.0)
    predict_populations!(ref,explicit,steep); @test out==ref
    bar=reshape(collect(1.0:nz*nx*ny*2),nz,nx,ny,2)./(nz*nx*ny)
    fb,zb=population_vjp!(zeros(6,nz,nx,ny),zeros(nz,nx,ny),frozen,steep,bar)
    fe,ze=population_vjp!(zeros(6,nz,nx,ny),zeros(nz,nx,ny),explicit,steep,bar)
    @test fb==fe && zb==ze && frozen.mesh==m
    # readapt: the next prediction builds the mesh for its own atmosphere and keeps it
    freeze_population_mesh!(frozen;readapt=true)
    predict_populations!(out,frozen,steep); steep_mesh=copy(frozen.mesh); @test length(steep_mesh)>length(m)
    predict_populations!(out,frozen,atmos); @test frozen.mesh==steep_mesh
    # unfrozen it follows every atmosphere again
    freeze_population_mesh!(frozen,false); predict_populations!(out,frozen,atmos); @test frozen.mesh==m
    # adapt_during_inversion = true, a fixed mesh and models without a mesh are left alone
    free=MeshPopulationModel(inner,PopulationMeshConfig(;settings...,adaptive=true,adapt_during_inversion=true),eos,274.0)
    freeze_population_mesh!(free); predict_populations!(out,free,atmos); predict_populations!(out,free,steep)
    @test !free.frozen && free.mesh==steep_mesh
    @test !freeze_population_mesh!(fixed).frozen
    @test freeze_population_mesh!(inner)===inner
    # the distributed wrappers reach the mesh model of every species
    a=MeshPopulationModel(inner,PopulationMeshConfig(;settings...,adaptive=true),eos,274.0)
    b=MeshPopulationModel(inner,PopulationMeshConfig(;settings...,adaptive=true),eos,274.0)
    freeze_population_mesh!(CompositeDistributedPopulationModel(Dict(:H=>LocalDistributedPopulationModel(a,2),
        :CA=>RootDistributedPopulationModel(b,2))))
    @test a.frozen && b.frozen
end
