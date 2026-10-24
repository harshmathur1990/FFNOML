include("helpers.jl")

struct NonlinearPressureOpacity <: AbstractOpacity500
    base::Float64
end
function FFNOInversion.opacity500!(kappa,model::NonlinearPressureOpacity,temperature,pgas,rho,ne)
    @. kappa=model.base*(pgas+1)^0.15*(temperature/5500)^(-0.5)
    kappa
end

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
    @test ForceBalanceOptions().max_iterations==500
    @test ForceBalanceOptions().pressure_sweeps==10
    @test ForceBalanceOptions().pressure_max_sweeps==200
    @test ForceBalanceOptions().pressure_tolerance==1e-6
    @test ForceBalanceOptions().continuation_iterations==10
    @test ForceBalanceOptions().max_backtracks==8
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
    @test d.mode==:MHS && d.lorentz_max_n_m3==0
    @test d.best_force_residual<=d.initial_force_residual
    @test d.force_residual==d.best_force_residual
    @test d.absolute_force_improvement==d.initial_force_residual-d.best_force_residual
    @test a.temperature==T0
    @test all(a.pgas.>0) && all(a.rho.>0) && all(a.ne.>0)
    @test all(diff(a.z[:,1,1]).<0)
    @test all(iszero,a.z[1,:,:])

    limited=phase1_atmosphere()
    accepted=reconstruct_force_balance!(limited,HE3DBoundaryState(1e-10,1.0,:top),eos,opacity;
        options=ForceBalanceOptions(max_iterations=1,relative_tolerance=eps(),force_tolerance=eps(),
            height_tolerance_m=eps(),relaxation=0.5,pressure_sweeps=1))
    @test accepted.mode==:MHS && accepted.iterations==1
    @test accepted.best_force_residual<=accepted.initial_force_residual
    @test accepted.force_residual==accepted.best_force_residual
    @test limited.pgas!==nothing

    am=phase1_atmosphere(magnetic=true); B0=deepcopy(am.magnetic_field)
    dm=reconstruct_force_balance!(am,HE3DBoundaryState(1e-10,1.0,:top),eos,opacity;
        options=ForceBalanceOptions(max_iterations=200,relative_tolerance=5.0,force_tolerance=0.6,
            height_tolerance_m=2e7,relaxation=0.5))
    @test dm.mode==:MHS && dm.lorentz_max_n_m3>0
    @test am.magnetic_field.Bx==B0.Bx && am.magnetic_field.By==B0.By && am.magnetic_field.Bz==B0.Bz
    @test maximum(abs.(am.pgas.-a.pgas))>0
    @test maximum(abs.(am.pgas[:,:,2].-am.pgas[:,:,1]))>0
    limited_mhs=phase1_atmosphere(magnetic=true)
    accepted_mhs=reconstruct_force_balance!(limited_mhs,
        HE3DBoundaryState(1e-10,1.0,:top),eos,opacity;
        options=ForceBalanceOptions(max_iterations=1,relative_tolerance=eps(),force_tolerance=eps(),
            height_tolerance_m=eps(),relaxation=0.5,pressure_sweeps=1))
    @test accepted_mhs.mode==:MHS && !accepted_mhs.converged
    @test accepted_mhs.best_force_residual<=accepted_mhs.initial_force_residual
    @test accepted_mhs.force_residual==accepted_mhs.best_force_residual

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
    compatibility=FFNOInversion._force_integrability(cfx,cfy,cfz,crho,cz,cgrid,cg)
    @test compatibility.maximum<1e-12
    @test compatibility.rms<1e-12
    for k in 1:3,i in 1:3,j in 1:3
        dx,dy,dz=FFNOInversion._physical_derivatives(cp,cz,cgrid,k,i,j)
        @test isapprox(dx,0.0;atol=1e-12)
        @test isapprox(dy,0.0;atol=1e-12)
        @test isapprox(dz,-crho[k,i,j]*cg;atol=1e-12)
    end
    perturbed=copy(cp); perturbed[2,2,2]+=250.0
    residual_before=FFNOInversion._force_residual(perturbed,crho,cz,cfx,cfy,cfz,cgrid,cg)
    inner_sweeps=Ref(0)
    FFNOInversion._relax_pressure_3d!(perturbed,cfx,cfy,cfz,crho,cz,cgrid,
        copy(@view(cp[1,:,:])),collect(axes(cp,1)),cg,2;
        max_sweeps=80,tolerance=1e-10,sweeps_used=inner_sweeps)
    @test inner_sweeps[]>2
    @test FFNOInversion._force_residual(perturbed,crho,cz,cfx,cfy,cfz,cgrid,cg)<residual_before

    poisson=copy(cp); poisson[2,2,2]+=250.0
    poisson_result=FFNOInversion._solve_pressure_poisson_3d!(poisson,cfx,cfy,cfz,
        crho,cz,cgrid,copy(@view(cp[1,:,:])),collect(axes(cp,1)),cg;
        max_iterations=200,tolerance=1e-12)
    @test poisson_result.converged
    @test FFNOInversion._force_residual(poisson,crho,cz,cfx,cfy,cfz,cgrid,cg)<1e-8
    positive_poisson=copy(cp); positive_poisson[2,2,2]+=250.0
    positive_result=FFNOInversion._solve_pressure_poisson_positive_3d!(
        positive_poisson,cfx,cfy,cfz,crho,cz,cgrid,copy(@view(cp[1,:,:])),
        collect(axes(cp,1)),cg;max_iterations=1000,tolerance=1e-9,
        pressure_floor=1.0)
    @test positive_result.converged
    @test minimum(positive_poisson)>=1.0
    @test FFNOInversion._force_residual(positive_poisson,crho,cz,cfx,cfy,cfz,cgrid,cg)<1e-6

    # Bx=a*z, Bz=B0 has analytic curl and verifies that Lorentz derivatives
    # use the corrugated geometry rather than a horizontally averaged z.
    ca=3e-8; cb0=2e-4
    cB=MagneticField3D(ca.*cz,zeros(cshape),fill(cb0,cshape))
    lorentz_force!(cfx,cfy,cfz,cB,cgrid,cz)
    @test maximum(abs.(cfx .- ca*cb0/FFNOInversion.MU0))<1e-12
    @test maximum(abs.(cfy))<1e-12
    @test maximum(abs.(cfz .+ ca^2 .* cz ./ FFNOInversion.MU0))<1e-12

    # Production-shaped nonlinear smoke test: many depths, corrugated thermal
    # structure, pressure-dependent opacity, and no supplied pressure profile.
    pnz,pnx,pny=24,16,12
    pgrid=Grid3D(collect(range(-6.0,1.0,length=pnz)),collect(0.0:40e3:(pnx-1)*40e3),
        collect(0.0:40e3:(pny-1)*40e3))
    pshape=(pnz,pnx,pny); ptemperature=Array{Float64}(undef,pshape)
    for k in 1:pnz,i in 1:pnx,j in 1:pny
        ptemperature[k,i,j]=4800+55k+180sin(2pi*(i-1)/pnx)*cos(2pi*(j-1)/pny)
    end
    pzero=zeros(pshape)
    production=Atmosphere3D(pgrid,ptemperature,copy(pzero),copy(pzero),copy(pzero),copy(pzero))
    production_diag=reconstruct_force_balance!(production,
        HE3DBoundaryState(1e-9,0.5,:top),IdealGasEOS(),NonlinearPressureOpacity(0.02);
        options=ForceBalanceOptions(max_iterations=8,relative_tolerance=1e-4,
            force_tolerance=2.0,height_tolerance_m=1e4,relaxation=0.8,
            pressure_sweeps=5,pressure_max_sweeps=40,pressure_tolerance=1e-5))
    @test production_diag.mode==:MHS
    @test all(isfinite,production.pgas) && all(>(0),production.pgas)
    @test all(isfinite,production.rho) && all(>(0),production.rho)
    @test all(isfinite,production.z)
    @test production_diag.accepted_3d ? maximum(production.pgas)>minimum(production.pgas) : all(production.pgas.==0.5)
end

@testset "Final force diagnostics" begin
    grid=Grid3D([-2.0,-1.0,0.0],[0.0,2.0,4.0],[0.0,3.0,6.0])
    shape=(3,3,3); z=zeros(shape)
    for k in 1:3,i in 1:3,j in 1:3
        z[k,i,j]=100(k-1)+0.4grid.x[i]-0.25grid.y[j]
    end
    rho=fill(2e-4,shape); g=10.0; zero_force=zeros(shape)
    p=@. 1000-rho*g*z
    hse=FFNOInversion._directional_force_residual(p,rho,z,zero_force,zero_force,zero_force,grid,g)
    @test max(hse.x,hse.y,hse.z)<1e-10
    @test max(hse.x,hse.y,hse.z)≈FFNOInversion._force_residual(p,rho,z,
        zero_force,zero_force,zero_force,grid,g) atol=1e-12
    perturbed=copy(p)
    for i in 1:3
        @views perturbed[:,i,:].+=0.1grid.x[i]
    end
    mismatch=FFNOInversion._directional_force_residual(perturbed,rho,z,
        zero_force,zero_force,zero_force,grid,g)
    @test mismatch.x>1
    @test max(mismatch.y,mismatch.z)<1e-10

    # Analytic MHS: Bx=a*z, Bz=B0; the quadratic pressure integrates exactly
    # with the trapezoidal force on every corrugated edge.
    a=3e-8; b0=2e-4; B=MagneticField3D(a.*z,zeros(shape),fill(b0,shape))
    fx=zeros(shape); fy=zeros(shape); fz=zeros(shape)
    lorentz_force!(fx,fy,fz,B,grid,z)
    x=reshape(grid.x,1,3,1)
    p_mhs=@. 1000+a*b0/FFNOInversion.MU0*x-
        a^2*z^2/(2FFNOInversion.MU0)-rho*g*z
    mhs=FFNOInversion._directional_force_residual(p_mhs,rho,z,fx,fy,fz,grid,g)
    compatibility=FFNOInversion._force_integrability(fx,fy,fz,rho,z,grid,g)
    @test max(mhs.x,mhs.y,mhs.z)<1e-9
    @test compatibility.maximum<1e-10
    @test compatibility.faces==36
    # Fx=y has nonzero xy circulation and cannot be a scalar pressure gradient.
    for j in 1:3
        @views fx[:,:,j].=grid.y[j]
    end
    incompatible=FFNOInversion._force_integrability(fx,zero_force,zero_force,rho,z,grid,g)
    @test incompatible.maximum>0.1
    @test incompatible.rms>0

    eos=IdealGasEOS(); temperature=fill(5500.0,shape); pressure=fill(10.0,shape)
    erho=similar(pressure); ene=similar(pressure)
    thermodynamics!(erho,ene,eos,temperature,pressure)
    original=(copy(pressure),copy(erho),copy(ene))
    consistent=FFNOInversion._eos_consistency(pressure,erho,ene,temperature,eos)
    @test consistent.density==consistent.electron==0
    inconsistent=FFNOInversion._eos_consistency(pressure,1.1.*erho,0.8.*ene,temperature,eos)
    @test inconsistent.density≈0.1
    @test inconsistent.electron≈0.2
    @test (pressure,erho,ene)==original

    atmosphere=phase1_atmosphere(magnetic=true)
    diagnostics=reconstruct_force_balance!(atmosphere,HE3DBoundaryState(1e-10,1.0,:top),
        eos,ReferenceOpacity500();options=ForceBalanceOptions(max_iterations=1))
    @test max(diagnostics.force_residual_x,diagnostics.force_residual_y,
        diagnostics.force_residual_z)≈diagnostics.force_residual atol=1e-12
    expected=FFNOInversion._eos_consistency(atmosphere.pgas,atmosphere.rho,
        atmosphere.ne,atmosphere.temperature,eos)
    @test diagnostics.eos_density_relative_error==expected.density
    @test diagnostics.eos_electron_relative_error==expected.electron
    @test diagnostics.force_integrability_faces==29
end

@testset "EOS and geometry enforced at trial pressure" begin
    @test !ForceBalanceOptions().enforce_trial_consistency
    eos=IdealGasEOS(); opacity=NonlinearPressureOpacity(0.01)
    for magnetic in (false,true), distributed_mode in (false,true), limit in (1,12)
        a=phase1_atmosphere(magnetic=magnetic)
        a.temperature[:,2,:].*=1.15
        options=ForceBalanceOptions(max_iterations=limit,enforce_trial_consistency=true)
        boundary=HE3DBoundaryState(1e-10,1.,:top)
        if distributed_mode
            context=serial_context(); dist=distribute_atmosphere(Float64,a,context)
            d=reconstruct_force_balance_distributed!(dist,boundary,eos,opacity,context;
                options=options)
            a=dist.local_atmosphere
        else
            d=reconstruct_force_balance!(a,boundary,eos,opacity;options=options)
        end
        rho=similar(a.rho); ne=similar(a.ne); kappa=similar(rho); z=similar(a.z)
        thermodynamics!(rho,ne,eos,a.temperature,a.pgas)
        opacity500!(kappa,opacity,a.temperature,a.pgas,rho,ne)
        FFNOInversion._height_from_tau!(z,kappa,rho,10.0.^log_tau500_axis(a.grid),
            collect(axes(z,1)))
        @test a.rho==rho
        @test a.ne==ne
        @test a.z==z
        @test d.eos_density_relative_error==d.eos_electron_relative_error==0
        @test all(a.pgas[1,:,:].==1.)
    end
end


@testset "One MHS iteration without bootstrap" begin
    eos=IdealGasEOS();opacity=ReferenceOpacity500()
    @test !(:bootstrap_iterations in fieldnames(typeof(ForceBalanceOptions())))
    @test !any(n->startswith(String(n),"bootstrap_"),fieldnames(ForceBalanceDiagnostics{Float64}))
    for distributed_mode in (false,true)
        outputs=[]
        for explicit_zero in (false,true)
            source=phase1_atmosphere()
            shape=size(source.temperature)
            B=explicit_zero ? MagneticField3D(zeros(shape),zeros(shape),zeros(shape)) : nothing
            a=Atmosphere3D(source.grid,source.temperature,source.vx,source.vy,source.vz,source.vturb;
                magnetic_field=B)
            initial=Ref{Any}(nothing)
            options=ForceBalanceOptions(max_iterations=300)
            if distributed_mode
                context=serial_context();dist=distribute_atmosphere(Float64,a,context)
                d=reconstruct_force_balance_distributed!(dist,HE3DBoundaryState(1e-10,1.,:top),
                    eos,opacity,context;options=options,initial_callback=s->(initial[]=s))
                a=dist.local_atmosphere
            else
                d=reconstruct_force_balance!(a,HE3DBoundaryState(1e-10,1.,:top),eos,opacity;
                    options=options,initial_callback=s->(initial[]=s))
            end
            @test all(initial[].pgas.==1.) # No preliminary pressure solve.
            @test d.mode==:MHS && d.iterations>1
            @test d.lorentz_max_n_m3==0
            @test d.force_residual≈max(d.force_residual_x,d.force_residual_y,d.force_residual_z)
            @test d.force_residual_x<1e-5
            push!(outputs,(a,d))
        end
        for n in (:pgas,:rho,:ne,:z)
            @test getfield(outputs[1][1],n)==getfield(outputs[2][1],n)
        end
    end
    # Even with a supplied B, initialization must not run HSE first.
    a=phase1_atmosphere(magnetic=true);initial=Ref{Any}(nothing)
    d=reconstruct_force_balance!(a,HE3DBoundaryState(1e-10,1.,:top),eos,opacity;
        options=ForceBalanceOptions(max_iterations=1),initial_callback=s->(initial[]=s))
    @test all(initial[].pgas.==1.)
    @test d.mode==:MHS && d.iterations==1
    @test d.lorentz_max_n_m3>0
end
