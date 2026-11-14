using Test
using FFNOInversion
using LinearAlgebra
using HDF5
import HSEMHS
import WittBifrostEOS
import Muspel

# Viewing angle (docs/viewing-angle/README.md): geometry, line-of-sight Doppler shift, slanted path length, the periodic
# shear and its transpose, footprint synthesis on height grids, the inclined population mesh, configuration and the
# complete objective gradient off the vertical.

include(joinpath(@__DIR__,"test_equilibrium.jl"))
isdefined(Main,:SmoothMeshTestPopulations) || include(joinpath(@__DIR__,"population_mesh.jl"))
isdefined(Main,:Phase6TwoLevelPopulation) || include(joinpath(@__DIR__,"phase6_production_gradient.jl"))

const VIEW_EOS_TABLE=get(ENV,"HSEMHS_EOS_TABLE","/mn/stornext/d9/data/harshm/MHS/HSEMHS/cache/eos_table_fullz.h5")

"""Population model that records the (mesh) atmospheres it is given."""
struct RecordingMesh <: AbstractPopulationModel
    seen::Vector{Any}
end
FFNOInversion.predict_populations!(out,m::RecordingMesh,a::Atmosphere3D,cache=nothing)=(push!(m.seen,deepcopy(a));
    fill!(out,1.0); out)

"""Exact JVP of the smooth test model: ṗ_l = p_l Σ_c A[l,c] scale_c ḟ_c (feature units: T, v, log10 nₑ, log10 ρ)."""
function FFNOInversion.population_jvp!(out,m::SmoothMeshTestPopulations,a::Atmosphere3D,feature_dot)
    p=zeros(size(out)); FFNOInversion.predict_populations!(p,m,a)
    for I in CartesianIndices(a.temperature), l in axes(p,4)
        k,x,y=Tuple(I)
        out[k,x,y,l]=p[k,x,y,l]*sum(m.A[l,c]*_MESH_TEST_SCALE[c]*feature_dot[c,k,x,y] for c in 1:6)
    end
    out
end

with_view(a::Atmosphere3D,view)=Atmosphere3D(Grid3D(a.grid.vertical,a.grid.x,a.grid.y,view),a.temperature,a.vx,a.vy,a.vz,
    a.vturb;magnetic_field=a.magnetic_field,pgas=a.pgas,rho=a.rho,ne=a.ne,z=a.z,logtau500=a.logtau500,chi500=a.chi500)

@testset "viewing angle: geometry" begin
    v=ViewGeometry(mu=0.8,azimuth_deg=30.0,geometry=:inclined)
    n=view_direction(v); t=slant(v)
    @test isapprox(norm(collect(n)),1;rtol=1e-15) && n[3]==0.8
    @test isapprox(collect(t),[0.75*cosd(30),0.75*sind(30)];rtol=1e-14)
    @test slant(ViewGeometry(mu=0.8,azimuth_deg=30.0,geometry=:plane_parallel))==(0.0,0.0)
    @test is_vertical(ViewGeometry(mu=1.0,geometry=:inclined)) && slant(ViewGeometry(mu=1.0,geometry=:inclined))==(0.0,0.0)
    @test_throws ArgumentError ViewGeometry(mu=0.0)
    @test_throws ArgumentError ViewGeometry(mu=1.2)
    @test_throws ArgumentError ViewGeometry(mu=0.5,geometry=:slanted)
    g=Grid3D([-1.0,0.0],[0.0,1.0],[0.0])
    @test g.view==ViewGeometry() && is_vertical(g.view)
    attrs=view_attributes(v,"rays")
    @test attrs["view_mu"]==0.8 && attrs["view_geometry"]=="inclined" && attrs["view_z_ref_m"]==0.0
    @test FFNOInversion.read_view_attributes(attrs)==v
    @test FFNOInversion.read_view_attributes(Dict{String,Any}())===nothing
    @test FFNOInversion.hsemhs_view(v)==HSEMHS.ViewOptions(mu=0.8,azimuth_deg=30.0,geometry=:inclined)
    @test HSEMHS.slant(FFNOInversion.hsemhs_view(v))==t
end

@testset "viewing angle: line-of-sight Doppler shift" begin
    grid=Grid3D([-2.0,-1.0],[0.0,1e4],[0.0])
    shape=(2,2,1); w=collect(range(656.20e-9,656.36e-9,length=41))
    line=FFNOTransition(:H,656.28e-9,1,2,0.64,1.6735575e-27,1e8)
    populations=cat(fill(1e12,shape...),fill(1e11,shape...);dims=4)
    opacity(a)=(c=zeros(2,length(w)); e=zeros(2,length(w)); add_opacity_emissivity!(c,e,line,w,a,1,1,populations); c)
    base=Atmosphere3D(grid,fill(6000.0,shape),zeros(shape),zeros(shape),zeros(shape),fill(1e3,shape))
    # vertical: exactly vz
    a=deepcopy(base); a.vz.=3e3
    @test los_velocity(a,1,1,1)===a.vz[1,1,1]
    for φ in (0.0,90.0,180.0,270.0,45.0), component in (:vx,:vy,:vz)
        view=ViewGeometry(mu=0.8,azimuth_deg=φ,geometry=:inclined)
        a=with_view(deepcopy(base),view); getfield(a,component).=4e3
        n=view_direction(view); expected=4e3*n[component===:vx ? 1 : component===:vy ? 2 : 3]
        @test isapprox(los_velocity(a,1,1,1),expected;atol=1e-9)
        # the opacity equals that of a vertical atmosphere moving at n·v
        b=deepcopy(base); b.vz.=expected
        @test isapprox(opacity(a),opacity(b);rtol=1e-12)
    end
    # the adjoint distributes v̄_los over vx, vy, vz along n
    view=ViewGeometry(mu=0.6,azimuth_deg=120.0,geometry=:plane_parallel)
    a=with_view(deepcopy(base),view); bar=AtmosphereCotangent(a)
    FFNOInversion.add_los_cotangent!(bar,a,1,1,1,2.0)
    @test isapprox([bar.vx[1,1,1],bar.vy[1,1,1],bar.vz[1,1,1]],2.0 .* collect(view_direction(view));rtol=1e-15)
end

@testset "viewing angle: periodic shear and its transpose" begin
    nz,nx,ny=4,7,5; dx=2.0; dy=3.0; z=[3.0,1.5,0.0,-2.0]
    A=[sin(0.7k+1.3i)+cos(0.4j*k)+0.1i*j for k in 1:nz,i in 1:nx,j in 1:ny]
    # integer shifts are exact periodic translations, both signs, unequal dx and dy
    t=(2.0,-1.0)                                              # cells: t_x z/dx = z, t_y z/dy = -z/3
    zi=[6.0,-12.0,0.0,18.0]
    for s in (1,-1)
        out=shear_levels!(similar(A),A,ShearPlan(zi,dx,dy,t,s))
        for k in 1:nz
            sx=Int(s*t[1]*zi[k]/dx); sy=Int(s*t[2]*zi[k]/dy)
            @test out[k,:,:]==circshift(A[k,:,:],(-sx,-sy))
        end
    end
    # fractional shifts: bilinear between the four neighbours, periodic seams included
    plan=ShearPlan(z,dx,dy,(0.37,-0.81),1); out=shear_levels!(similar(A),A,plan)
    for k in 1:nz, i in 1:nx, j in 1:ny
        fx=0.37*z[k]/dx; fy=-0.81*z[k]/dy
        i0=floor(Int,fx); j0=floor(Int,fy); wx=fx-i0; wy=fy-j0
        P(ii,jj)=A[k,mod(i-1+ii,nx)+1,mod(j-1+jj,ny)+1]
        ref=(1-wy)*((1-wx)*P(i0,j0)+wx*P(i0+1,j0))+wy*((1-wx)*P(i0,j0+1)+wx*P(i0+1,j0+1))
        @test isapprox(out[k,i,j],ref;rtol=1e-14,atol=1e-14)
    end
    # the same values as HSEMHS's own shear (ViewTau.pbilin on the (y, x) plane)
    for k in 1:nz, i in 1:nx, j in 1:ny
        G=permutedims(A[k,:,:],(2,1))
        ref=HSEMHS.pbilin(G,(j-1)+(-0.81)*z[k]/dy,(i-1)+0.37*z[k]/dx)
        @test isapprox(out[k,i,j],ref;rtol=1e-14,atol=1e-14)
    end
    # transpose: <S a, b> = <a, Sᵀ b>
    B=[cos(1.1k-0.3i+0.9j) for k in 1:nz,i in 1:nx,j in 1:ny]
    At=shear_levels_transpose!(zeros(size(B)),B,plan)
    @test isapprox(dot(out,B),dot(A,At);rtol=1e-13)
    # footprint/gauge covariance: raising every height by Δ moves the footprints by t·Δ (here 2 cells in x, -1 in y)
    Δ=2.0; tg=(2.0,-3.0)                                      # t_x Δ/dx = 2, t_y Δ/dy = -2
    base=shear_levels!(similar(A),A,ShearPlan(z,dx,dy,tg,1))
    raised=shear_levels!(similar(A),A,ShearPlan(z.+Δ,dx,dy,tg,1))
    @test raised≈circshift(base,(0,-2,2))
    # the opposite shift is not the transpose, and it is not an exact inverse
    back=shear_levels!(similar(A),out,ShearPlan(z,dx,dy,(0.37,-0.81),-1))
    @test maximum(abs.(back.-A))>1e-3
    @test sum(back)≈sum(A)                                    # but it conserves the level sums
end

"""Native-z atmosphere with horizontal structure; z top first."""
function view_z_atmosphere(;nz=24,nx=8,ny=6,view=ViewGeometry())
    z=collect(range(1.2e6,-2.0e5,length=nz))
    grid=Grid3D(VerticalAxis(:z_m,z),4.0e4 .* collect(0:nx-1),5.0e4 .* collect(0:ny-1),view)
    T=[5000.0+1500.0*tanh(-z[k]/3e5)+300.0*sin(2π*(i-1)/nx)*cos(2π*(j-1)/ny)+80.0*(k%3) for k in 1:nz,i in 1:nx,j in 1:ny]
    v(s)=[s*sin(0.2k+0.9i-0.5j) for k in 1:nz,i in 1:nx,j in 1:ny]
    Atmosphere3D(grid,T,v(1.5e3),v(-1e3),v(2e3),fill(900.0,nz,nx,ny);z=repeat(reshape(z,nz,1,1),1,nx,ny))
end

@testset "viewing angle: slanted path and footprint synthesis (native z)" begin
    # exact formal solution for a source linear in τ along the ray: τ = χ (z_top − z)/μ
    nz=30; z=collect(range(2e5,0.0,length=nz)); χ0=3e-5; S0=2.0; S1=0.7
    for mu in (1.0,0.8,0.45), geometry in (:plane_parallel,:inclined)
        view=ViewGeometry(mu=mu,azimuth_deg=20.0,geometry=geometry)
        grid=Grid3D(VerticalAxis(:z_m,z),[0.0,1e4],[0.0,1e4],FFNOInversion.is_slanted(view) ? FFNOInversion.footprint_view(view) : view)
        shape=(nz,2,2); a=Atmosphere3D(grid,fill(6000.0,shape),zeros(shape),zeros(shape),zeros(shape),zeros(shape);
            z=repeat(reshape(z,nz,1,1),1,2,2))
        τ=χ0 .* (z[1] .- z) ./ mu; S=S0 .+ S1 .* τ
        model=MixedIntensitySynthesizer([OpacityContributor(TabulatedOpacityModel(fill(χ0,nz,1),reshape(χ0 .* S,nz,1)))])
        cube=SpectralCube(zeros(1,1,2,2),[500e-9],StokesSet(:I))
        synthesize!(cube,model,NonPRD(),a)
        τb=τ[end]
        exact=S0*(1-exp(-τb))+S1*(1-exp(-τb)*(1+τb))+S[end]*exp(-τb)
        @test isapprox(cube.data[1,1,1,1],exact;rtol=1e-10)
    end
    # an inclined native-z atmosphere has Cartesian columns: they are never solved as rays
    a=view_z_atmosphere(view=ViewGeometry(mu=0.8,azimuth_deg=30.0,geometry=:inclined))
    model=MixedIntensitySynthesizer([OpacityContributor(TabulatedOpacityModel(fill(1e-5,24,1),fill(1e-5,24,1)))])
    @test_throws ArgumentError synthesize!(SpectralCube(zeros(1,1,8,6),[500e-9],StokesSet(:I)),model,NonPRD(),a)

    # footprint state: the Cartesian state shifted by +t z per level (log for positive fields), heights unchanged
    view=ViewGeometry(mu=0.7,azimuth_deg=25.0,geometry=:inclined)
    atmosphere=view_z_atmosphere(view=view)
    context=serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
    distributed=distribute_atmosphere(Float64,atmosphere,context)
    TestHydrostaticEquilibrium() |> b->reconstruct_equilibrium!(b,distributed,context)
    workspace=HybridForwardWorkspace(Float64,distributed,[656.28e-9],StokesSet(:I),Dict(:H=>2))
    @test workspace.footprint!==nothing && workspace.footprint.atmosphere.grid.view.footprint
    @test FFNOInversion.cartesian_populations(workspace)!==workspace.populations
    cart=FFNOInversion.cartesian_populations(workspace)[:H]
    cart.=[1e10*(1+0.3sin(i+j+k))*l for k in 1:24,i in 1:8,j in 1:6,l in 1:2]
    FFNOInversion.footprint_state!(workspace,distributed,context)
    plan=ShearPlan(distributed.global_grid,native_depth(distributed.global_grid),1)
    fp=workspace.footprint.atmosphere; a=distributed.local_atmosphere
    @test fp.temperature≈exp.(shear_levels!(similar(a.temperature),log.(a.temperature),plan))
    @test fp.vx≈shear_levels!(similar(a.vx),a.vx,plan) && fp.z==a.z
    @test fp.pgas≈exp.(shear_levels!(similar(a.pgas),log.(a.pgas),plan))
    @test workspace.populations[:H][:,:,:,2]≈exp.(shear_levels!(similar(a.vx),log.(cart[:,:,:,2]),plan))
    # a horizontally uniform atmosphere: inclined equals plane-parallel exactly (up to rounding)
    uniform(view)=begin
        u=view_z_atmosphere(view=view); u.temperature.=u.temperature[:,1:1,1:1]; u.vx.=u.vx[:,1:1,1:1]
        u.vy.=u.vy[:,1:1,1:1]; u.vz.=u.vz[:,1:1,1:1]; u
    end
    spectra=map((:plane_parallel,:inclined)) do geometry
        v=ViewGeometry(mu=0.7,azimuth_deg=25.0,geometry=geometry)
        d=distribute_atmosphere(Float64,uniform(v),context)
        ws=HybridForwardWorkspace(Float64,d,collect(range(656.2e-9,656.36e-9,length=9)),StokesSet(:I),Dict(:H=>2))
        line=FFNOTransition(:H,656.28e-9,1,2,0.64,1.6735575e-27,1e8)
        synth=MixedIntensitySynthesizer([OpacityContributor(TabulatedOpacityModel(fill(2e-6,24,9),fill(2e8*2e-6,24,9))),
            OpacityContributor(line,ws.populations[:H])])
        m=HybridForwardModel(CompositeDistributedPopulationModel(Dict(:H=>LocalDistributedPopulationModel(
            Phase6TwoLevelPopulation(1e10),2))),NonPRD(),synth,IdentityObservation(),TestHydrostaticEquilibrium(),
            CapabilityManifest())
        forward!(ws,m,d,context).spectrum.data
    end
    @test isapprox(spectra[1],spectra[2];rtol=1e-12)
end

@testset "viewing angle: inclined population mesh" begin
    if !isfile(VIEW_EOS_TABLE)
        @info "HSEMHS EOS table not found; skipping the inclined population mesh test"
    else
    eos=WittBifrostEOS.load_table(VIEW_EOS_TABLE)
    config=PopulationMeshConfig(bottom_m=-1e6,top_m=3e6,spacing_m=2e4,uniform_top_m=2e6,stretch=1.05,max_spacing_m=5e4)
    inner=SmoothMeshTestPopulations([0.8 0.1 -0.2 0.3 1.1 0.9; -0.5 0.2 0.1 -0.4 0.7 1.3],[0.4,-0.6])
    model=MeshPopulationModel(inner,config,eos,274.0)
    vertical=mesh_test_atmosphere(nx=6,ny=5)
    view=ViewGeometry(mu=0.75,azimuth_deg=35.0,geometry=:inclined)
    atmos=with_view(vertical,view); nz,nx,ny=size(atmos.temperature)

    # the network sees the ray features at their Cartesian positions: ln T etc. shifted by −t z_m per mesh level
    rec=MeshPopulationModel(RecordingMesh(Any[]),config,eos,274.0)
    predict_populations!(zeros(nz,nx,ny,2),rec,atmos)
    f=FFNOInversion._mesh_features(rec,atmos); seen=rec.inner.seen[end]
    plan=ShearPlan(atmos.grid,rec.mesh,-1)
    @test seen.temperature≈exp.(shear_levels!(similar(seen.temperature),log.(f[1,:,:,:]),plan))
    @test seen.vy≈shear_levels!(similar(seen.vy),f[3,:,:,:],plan)
    @test log10.(seen.rho)≈shear_levels!(similar(seen.vy),f[6,:,:,:],plan)
    @test is_vertical(seen.grid.view)                         # the mesh atmosphere is Cartesian
    # a vertical or plane-parallel view feeds the ray features unshifted
    predict_populations!(zeros(nz,nx,ny,2),rec,with_view(vertical,ViewGeometry(mu=0.75,azimuth_deg=35.0)))
    @test rec.inner.seen[end].temperature==f[1,:,:,:]

    # VJP and JVP of the whole inclined chain against central differences
    bar=reshape(cos.(1:nz*nx*ny*2),nz,nx,ny,2)
    objective(a)=(p=zeros(nz,nx,ny,2); predict_populations!(p,model,a); sum(bar.*p))
    feature_bar=zeros(6,nz,nx,ny); z_bar=zeros(nz,nx,ny)
    objective(atmos); population_vjp!(feature_bar,z_bar,model,atmos,bar)
    N=nz*nx*ny
    dT=reshape(0.01 .* sin.(1:N),nz,nx,ny); dv=reshape(cos.(2.0 .* (1:N)),nz,nx,ny)
    dL=reshape(0.02 .* cos.(1:N),nz,nx,ny); dR=reshape(0.03 .* sin.(3.0 .* (1:N)),nz,nx,ny)
    function moved(h)
        a=deepcopy(atmos)
        a.temperature.*=1 .+ h.*dT; a.vx.+=h.*1e2 .* dv; a.vy.-=h.*3e2 .* dv
        a.ne.*=exp10.(h.*dL); a.rho.*=exp10.(h.*dR); a
    end
    h=1e-4
    fd=(objective(moved(h))-objective(moved(-h)))/(2h)
    predicted=sum(feature_bar[1,:,:,:].*atmos.temperature.*dT)+sum(feature_bar[2,:,:,:].*1e2 .* dv)-
        sum(feature_bar[3,:,:,:].*3e2 .* dv)+sum(feature_bar[5,:,:,:].*dL)+sum(feature_bar[6,:,:,:].*dR)
    @test isapprox(predicted,fd;rtol=1e-5)
    # JVP: the same direction as feature tangents (T, vx, vy, vz, log10 ne, log10 rho)
    fdot=zeros(6,nz,nx,ny); fdot[1,:,:,:].=atmos.temperature.*dT; fdot[2,:,:,:].=1e2 .* dv
    fdot[3,:,:,:].=-3e2 .* dv; fdot[5,:,:,:].=dL; fdot[6,:,:,:].=dR
    pdot=zeros(nz,nx,ny,2)
    FFNOInversion.population_jvp!(pdot,model,atmos,fdot)
    p(a)=(q=zeros(nz,nx,ny,2); predict_populations!(q,model,a); q)
    numeric=(p(moved(h)).-p(moved(-h)))./(2h)
    @test isapprox(pdot,numeric;rtol=1e-4)
    @test isapprox(sum(bar.*pdot),predicted;rtol=1e-4)        # JVP and VJP agree
    end
end

@testset "viewing angle: configuration and inputs" begin
    v=FFNOInversion._view(Dict{String,Any}("mu"=>0.6,"azimuth_deg"=>90.0,"geometry"=>"Inclined"))
    @test v==ViewGeometry(mu=0.6,azimuth_deg=90.0,geometry=:inclined)
    @test FFNOInversion._view(Dict{String,Any}())==ViewGeometry()
    @test_throws ArgumentError FFNOInversion._view(Dict{String,Any}("mu"=>0.6,"tilt"=>3))
    @test_throws ArgumentError FFNOInversion._view(Dict{String,Any}("mu"=>1.5))
    # the sky PSF is stretched by 1/μ along the tilt on the footprint grid
    @test FFNOInversion._footprint_psf(2e4,3e4,ViewGeometry())==(2e4,3e4)
    @test FFNOInversion._footprint_psf(2e4,3e4,ViewGeometry(mu=0.5,azimuth_deg=180.0))==(4e4,3e4)
    @test FFNOInversion._footprint_psf(2e4,3e4,ViewGeometry(mu=0.5,azimuth_deg=-90.0))==(2e4,6e4)
    @test FFNOInversion._footprint_psf(0.0,0.0,ViewGeometry(mu=0.5,azimuth_deg=45.0))==(0.0,0.0)
    @test_throws ArgumentError FFNOInversion._footprint_psf(2e4,2e4,ViewGeometry(mu=0.5,azimuth_deg=45.0))
    # vlos and Cartesian velocities are one velocity described twice
    @test_throws ArgumentError FFNOInversion._check_velocity_controls([:temperature,:vlos,:vz])
    @test FFNOInversion._check_velocity_controls([:temperature,:vlos])===nothing
    # native log-τ inputs must be built for the configured view; native-z inputs need no tag
    off=ViewGeometry(mu=0.8,azimuth_deg=10.0,geometry=:inclined)
    @test_throws ArgumentError FFNOInversion.check_input_view(nothing,off,:log10tau500,"in.h5")
    @test FFNOInversion.check_input_view(nothing,ViewGeometry(),:log10tau500,"in.h5")===nothing
    @test FFNOInversion.check_input_view(nothing,off,:z_m,"in.h5")===nothing
    @test FFNOInversion.check_input_view(off,off,:log10tau500,"in.h5")===nothing
    @test_throws ArgumentError FFNOInversion.check_input_view(ViewGeometry(mu=0.8,azimuth_deg=10.0),off,:log10tau500,"in.h5")
    @test_throws ArgumentError FFNOInversion.check_input_view(off,ViewGeometry(),:log10tau500,"in.h5")
end

"""phase6_mixed_fixture with a view, a vlos control and horizontal structure in the starting atmosphere."""
function view_mixed_fixture(view,temperature_values,vlos_values;observation=nothing,kind=:log10tau500)
    nz=kind===:z_m ? 12 : 3; nx=4; ny=3
    axis=kind===:z_m ? collect(range(8e5,-1e5,length=nz)) : [-5.0,-3.0,-1.0]
    grid=Grid3D(VerticalAxis(kind,axis),40e3 .* collect(0:nx-1),40e3 .* collect(0:ny-1),view)
    shape=(nz,nx,ny)
    T0=[5500.0+120.0*sin(1.3i)*cos(0.8j)+40.0k for k in 1:nz,i in 1:nx,j in 1:ny]
    v(s)=[s*sin(0.4k+i-0.7j) for k in 1:nz,i in 1:nx,j in 1:ny]
    zfield=kind===:z_m ? repeat(reshape(axis,nz,1,1),1,nx,ny) : nothing
    atmosphere=Atmosphere3D(grid,T0,v(400.0),v(-300.0),v(200.0),fill(800.0,shape);z=zfield)
    context=serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
    distributed=distribute_atmosphere(Float64,atmosphere,context)
    wavelength=collect(range(656.245e-9,656.315e-9,length=5))
    workspace=HybridForwardWorkspace(Float64,distributed,wavelength,StokesSet(:I),Dict(:H=>2,:CA=>2))
    continuum=TabulatedOpacityModel(fill(kind===:z_m ? 2e-6 : 2e-5,nz,5),fill(kind===:z_m ? 2e2 : 2e8,nz,5))
    transition=FFNOTransition(:H,656.28e-9,1,2,0.64,1.6735575e-27,1e8)
    ca_transition=FFNOTransition(:CA,656.300e-9,1,2,0.3,40*1.66053906660e-27,1e8)
    kurucz_line=KuruczLine(656.292e-9,"FeI",-2.5,1.0*1.602176634e-19,3.0*1.602176634e-19,1e8)
    synthesizer=MixedIntensitySynthesizer([
        OpacityContributor(continuum),OpacityContributor(transition,workspace.populations[:H]),
        OpacityContributor(ca_transition,workspace.populations[:CA]),
        OpacityContributor(KuruczLTEModel([kurucz_line]))])
    population_model=CompositeDistributedPopulationModel(Dict(
        :H=>LocalDistributedPopulationModel(Phase6TwoLevelPopulation(kind===:z_m ? 1e10 : 1e12),2),
        :CA=>LocalDistributedPopulationModel(Phase6TwoLevelPopulation(kind===:z_m ? 5e9 : 5e11),2)))
    model=HybridForwardModel(population_model,NonPRD(),synthesizer,IdentityObservation(),
        TestHydrostaticEquilibrium(),CapabilityManifest())
    nodes=kind===:z_m ? [8e5,-1e5] : [-5.0,-1.0]
    temp=NodeField(reshape(Float64.(temperature_values),2,1,1),nodes,kind)
    vl=NodeField(reshape(Float64.(vlos_values),2,1,1),nodes,kind)
    layout=ControlMapLayout(ControlMapSpec(:temperature,temp;lower=3500.0,upper=8000.0,scale=1000.0),
        ControlMapSpec(:vlos,vl;lower=-5000.0,upper=5000.0,scale=1000.0))
    regularization=RegularizationSpec(vertical=VerticalRegularizationSpec((1,1,0,0,0,0,0),1e-5,ntuple(_->1.0,7);
        length_scale_m=kind===:z_m ? 1e5 : nothing),
        horizontal=Dict(:vlos=>1e-6),scales=Dict(:temperature=>1000.0,:vlos=>1000.0),horizontal_order=1)
    if observation===nothing
        result=forward!(workspace,model,distributed,context)
        data=result.spectrum.data
        observation=ObservationCube(SpectralCube(data.*(1 .+ 0.01 .* reshape(sin.(1:length(data)),size(data))),
            wavelength,StokesSet(:I)),fill(maximum(abs,result.spectrum.data)*1e-3,size(result.spectrum.data)),
            ones(size(result.spectrum.data)))
    end
    problem=DistributedInversionProblem(model,workspace,distributed,observation,regularization,40e3,40e3,context;
        control_layout=layout,reference_id="view-test")
    (context=context,distributed=distributed,model=model,workspace=workspace,layout=layout,problem=problem,
        observation=observation)
end

@testset "viewing angle: objective gradient off the vertical ($kind, $geometry)" for kind in (:log10tau500,:z_m),
        geometry in (:plane_parallel,:inclined)
    view=ViewGeometry(mu=0.8,azimuth_deg=30.0,geometry=geometry)
    fixture=view_mixed_fixture(view,[5500.0,6000.0],[0.0,0.0];kind=kind)
    @test (fixture.workspace.footprint!==nothing)==(kind===:z_m && geometry===:inclined)
    layout=fixture.layout; parameters=initial_parameters(layout).+[200.0,-150.0,300.0,-400.0]
    production=objective_gradient!(HybridAdjointObjectiveGradient(force_balance_step=2e-5),fixture.problem,layout,
        parameters,fixture.context)
    oracle=objective_gradient!(FiniteDifferenceObjectiveGradient(step=2e-5),fixture.problem,layout,parameters,
        fixture.context)
    relative=norm(production.gradient-oracle.gradient)/max(norm(oracle.gradient),eps())
    @test relative<5e-3
    @test abs(production.gradient[3])+abs(production.gradient[4])>0          # the vlos control acts
    # v = v_prior + δ n: the perpendicular part keeps its prior on every trial (no accumulation)
    a=fixture.distributed.local_atmosphere; n=view_direction(view); ref=fixture.problem.control_reference
    apply_control_maps!(fixture.problem,layout,parameters); apply_control_maps!(fixture.problem,layout,parameters)
    δx=a.vx.-ref[:vlos_prior_vx]; δz=a.vz.-ref[:vlos_prior_vz]
    @test isapprox(δx.*n[3],δz.*n[1];atol=1e-9)
    @test isapprox(FFNOInversion.los_velocity_field(a).-(ref[:vlos_prior_vx].*n[1].+ref[:vlos_prior_vy].*n[2].+
        ref[:vlos_prior_vz].*n[3]),δz./n[3];rtol=1e-12)
    println("VIEW_GRADIENT_OK kind=$kind geometry=$geometry relative_error=$relative")
end

@testset "viewing angle: μ = 1 is the vertical code path" begin
    spectra=map((nothing,ViewGeometry(mu=1.0,azimuth_deg=40.0,geometry=:inclined),
                 ViewGeometry(mu=1.0,azimuth_deg=40.0,geometry=:plane_parallel))) do view
        fixture=view===nothing ? phase6_mixed_fixture([5200.0,6400.0],[-600.0,900.0]) : begin
            f=phase6_mixed_fixture([5200.0,6400.0],[-600.0,900.0])
            a=f.distributed.local_atmosphere
            d=distribute_atmosphere(Float64,with_view(a,view),f.context)
            ws=HybridForwardWorkspace(Float64,d,f.workspace.intrinsic.wavelength_m,StokesSet(:I),Dict(:H=>2,:CA=>2))
            @test ws.footprint===nothing
            # rebuild the synthesizer on this workspace's arrays
            c=f.model.synthesizer.contributors
            synth=MixedIntensitySynthesizer([c[1],OpacityContributor(c[2].model,ws.populations[:H]),
                OpacityContributor(c[3].model,ws.populations[:CA]),c[4]])
            m=HybridForwardModel(f.model.populations,NonPRD(),synth,f.model.observation,TestHydrostaticEquilibrium(),
                CapabilityManifest())
            (model=m,workspace=ws,distributed=d,context=f.context)
        end
        forward!(fixture.workspace,fixture.model,fixture.distributed,fixture.context).spectrum.data
    end
    @test spectra[1]==spectra[2]==spectra[3]
end

@testset "Doppler sign: positive v (toward the observer, up at μ = 1) is a blue shift in every line model" begin
    nz=2; grid=Grid3D([-2.0,-1.0],[0.0],[0.0]); shape=(nz,1,1)
    atmos(v)=Atmosphere3D(grid,reshape([5800.0,6200.0],shape),zeros(shape),zeros(shape),fill(v,shape),fill(1e3,shape);
        rho=fill(1e-4,shape),ne=reshape([1e19,2e19],shape),pgas=fill(1e4,shape),z=reshape([1e5,0.0],shape))
    centre(model,λ0,pops)=begin
        w=collect(range(λ0-0.1e-9,λ0+0.1e-9,length=2001))
        χ=zeros(nz,length(w)); η=zeros(nz,length(w))
        add_opacity_emissivity!(χ,η,model,w,atmos(5e3),1,1,pops)
        w[argmax(χ[1,:])]
    end
    pops=cat(fill(1e12,shape...),fill(1e11,shape...);dims=4)
    line=FFNOTransition(:H,656.28e-9,1,2,0.64,1.6735575e-27,1e8)
    @test isapprox(centre(line,656.28e-9,pops),656.28e-9*(1-5e3/299792458.0);atol=2e-13)
    kurucz=KuruczLTEModel([KuruczLine(630.25e-9,"FeI",-1.0,3.6*1.602176634e-19,5.6*1.602176634e-19,1e8)])
    @test isapprox(centre(kurucz,630.25e-9,nothing),630.25e-9*(1-5e3/299792458.0);atol=2e-13)
    atom=joinpath(get(ENV,"FFNO_ATOM_DIR","/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms"),"atom.h6_tiago2.yaml")
    if isfile(atom) && isdefined(Main,:Muspel)
        hp=fill(1e10,nz,1,1,6); hp[:,:,:,1].=1e16; hp[:,:,:,2].=2e14; hp[:,:,:,6].=1e15
        m=build_muspel_line_model(atmos(0.0),hp,atom,5,2,3;voigt=(a_min=1f-4,a_max=1f1,a_n=300,v_min=0f0,v_max=5f2,v_n=500),
            include_continuum=false)
        λ0=Float64(m.line.λ0)*1e-9
        @test isapprox(centre(m,λ0,hp),λ0*(1-5e3/299792458.0);atol=2e-13)
    end
end
