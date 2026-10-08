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
    @test ForceBalanceOptions().pressure_sweeps==10
    @test ForceBalanceOptions().lateral_boundary==:force_neumann
    @test ForceBalanceOptions().bottom_boundary==:force_neumann
    @test_throws ArgumentError ForceBalanceOptions(lateral_boundary=:unspecified)
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

    limited=phase1_atmosphere()
    accepted=reconstruct_force_balance!(limited,HE3DBoundaryState(1e-10,1.0,:top),eos,opacity;
        options=ForceBalanceOptions(max_iterations=1,relative_tolerance=eps(),force_tolerance=eps(),
            height_tolerance_m=eps(),relaxation=0.5,pressure_sweeps=1))
    @test accepted.mode==:HE3D && !accepted.converged && accepted.iterations==1
    @test limited.pgas!==nothing

    am=phase1_atmosphere(magnetic=true); B0=deepcopy(am.magnetic_field)
    dm=reconstruct_force_balance!(am,HE3DBoundaryState(1e-10,1.0,:top),eos,opacity;
        options=ForceBalanceOptions(max_iterations=200,relative_tolerance=5.0,force_tolerance=0.6,
            height_tolerance_m=2e7,relaxation=0.5))
    @test dm.mode==:MHS && dm.lorentz_max_n_m3>0
    @test am.magnetic_field.Bx==B0.Bx && am.magnetic_field.By==B0.By && am.magnetic_field.Bz==B0.Bz
    @test maximum(abs.(am.pgas.-a.pgas))>0
    @test maximum(abs.(am.pgas[:,:,2].-am.pgas[:,:,1]))>0
    @test_throws ErrorException reconstruct_force_balance!(phase1_atmosphere(magnetic=true),
        HE3DBoundaryState(1e-10,1.0,:top),eos,opacity;
        options=ForceBalanceOptions(max_iterations=1,relative_tolerance=eps(),force_tolerance=eps(),
            height_tolerance_m=eps(),relaxation=0.5,pressure_sweeps=1))

    shape=size(am.temperature); fx=zeros(shape);fy=zeros(shape);fz=zeros(shape)
    lorentz_force!(fx,fy,fz,am.magnetic_field,am.grid,am.z)
    @test all(isfinite,fx) && all(isfinite,fy) && all(isfinite,fz)
    @test_throws ArgumentError reconstruct_force_balance!(phase1_atmosphere(),HE3DBoundaryState(1e-10,1.0,:full),eos,opacity)

    # Manufactured solution on a strongly corrugated optical-depth mesh.
    # P depends only on physical z, so its physical horizontal derivatives
    # vanish even though P varies horizontally at fixed optical depth.
    cgrid=Grid3D([-2.0,-1.0,0.0],[0.0,2.0,4.0],[0.0,3.0,6.0])
    cshape=(3,3,3); cz=Array{Float64}(undef,cshape)
    for k in 1:3,i in 1:3,j in 1:3
        cz[k,i,j]=100.0*(k-1)+0.4*cgrid.x[i]-0.25*cgrid.y[j]
    end
    crho=fill(2e-4,cshape); cg=10.0; cp=@. 1000.0-crho*cg*cz
    cfx=zeros(cshape); cfy=zeros(cshape); cfz=zeros(cshape)
    @test FFNOInversion._force_residual(cp,crho,cz,cfx,cfy,cfz,cgrid,cg)<1e-10
    for k in 1:3,i in 1:3,j in 1:3
        dx,dy,dz=FFNOInversion._physical_derivatives(cp,cz,cgrid,k,i,j)
        @test isapprox(dx,0.0;atol=1e-12)
        @test isapprox(dy,0.0;atol=1e-12)
        @test isapprox(dz,-crho[k,i,j]*cg;atol=1e-12)
    end

    # Bx=a*z, Bz=B0 has analytic curl and verifies that Lorentz derivatives
    # use the corrugated geometry rather than a horizontally averaged z.
    ca=3e-8; cb0=2e-4
    cB=MagneticField3D(ca.*cz,zeros(cshape),fill(cb0,cshape))
    lorentz_force!(cfx,cfy,cfz,cB,cgrid,cz)
    @test maximum(abs.(cfx .- ca*cb0/FFNOInversion.MU0))<1e-12
    @test maximum(abs.(cfy))<1e-12
    @test maximum(abs.(cfz .+ ca^2 .* cz ./ FFNOInversion.MU0))<1e-12
end
