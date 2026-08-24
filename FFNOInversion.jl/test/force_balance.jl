include("helpers.jl")

function phase1_atmosphere(;magnetic=false)
    grid=Grid3D([-5.0,-4.0,-3.0,-2.0],[0.0,50e3,100e3],[0.0,50e3])
    shape=(4,3,2); temperature=fill(5500.0,shape); zero3=zeros(shape)
    field=nothing
    if magnetic
        bx=fill(1e-5,shape); by=zeros(shape); bz=fill(2e-5,shape)
        for k in axes(by,1); @views by[k,:,:].=(k-1)*1e-8; end
        field=MagneticField3D(bx,by,bz)
    end
    Atmosphere3D(grid,temperature,copy(zero3),copy(zero3),copy(zero3),copy(zero3);magnetic_field=field)
end

@testset "Phase 1 EOS, opacity and force balance" begin
    eos=IdealGasEOS(); opacity=ReferenceOpacity500(kappa_m2_kg=0.02)
    T=fill(6000.0,2,1,1); p=fill(10.0,2,1,1); rho=similar(p); ne=similar(p)
    thermodynamics!(rho,ne,eos,T,p)
    @test all(rho.>0) && all(ne.>0)
    kappa=similar(p); opacity500!(kappa,opacity,T,p,rho,ne)
    @test all(kappa.==0.02)

    a=phase1_atmosphere(); T0=copy(a.temperature)
    d=reconstruct_force_balance!(a,HE3DBoundaryState(1e-10,1.0,:top),eos,opacity;
        options=ForceBalanceOptions(max_iterations=200,relative_tolerance=1e-5,force_tolerance=0.6,height_tolerance_m=1.0,relaxation=0.5))
    @test d.mode==:HE3D && d.converged && d.lorentz_max_n_m3==0
    @test a.temperature==T0
    @test all(a.pgas.>0) && all(a.rho.>0) && all(a.ne.>0)
    @test all(diff(a.z[:,1,1]).<0)

    am=phase1_atmosphere(magnetic=true); B0=deepcopy(am.magnetic_field)
    dm=reconstruct_force_balance!(am,HE3DBoundaryState(1e-10,1.0,:top),eos,opacity;
        options=ForceBalanceOptions(max_iterations=200,relative_tolerance=1e-5,force_tolerance=0.6,height_tolerance_m=1.0,relaxation=0.5))
    @test dm.mode==:MHS && dm.lorentz_max_n_m3>0
    @test am.magnetic_field.Bx==B0.Bx && am.magnetic_field.By==B0.By && am.magnetic_field.Bz==B0.Bz
    @test maximum(abs.(am.pgas.-a.pgas))>0
    @test maximum(abs.(am.pgas[:,:,2].-am.pgas[:,:,1]))>0

    shape=size(am.temperature); fx=zeros(shape);fy=zeros(shape);fz=zeros(shape)
    lorentz_force!(fx,fy,fz,am.magnetic_field,am.grid,am.z)
    @test all(isfinite,fx) && all(isfinite,fy) && all(isfinite,fz)
    @test_throws ArgumentError reconstruct_force_balance!(phase1_atmosphere(),HE3DBoundaryState(1e-10,1.0,:full),eos,opacity)
end
