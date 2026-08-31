using LinearAlgebra

struct Phase6TwoLevelPopulation{T<:AbstractFloat} <: AbstractPopulationModel
    scale::T
end

function FFNOInversion.predict_populations!(out,model::Phase6TwoLevelPopulation,
        atmosphere::Atmosphere3D,cache=nothing)
    size(out,4)==2 || throw(DimensionMismatch("fixture needs two population levels"))
    @views out[:,:,:,1].=model.scale.*atmosphere.temperature./5000
    @views out[:,:,:,2].=0.1model.scale.*atmosphere.temperature./5000
    out
end

function FFNOInversion.population_vjp!(feature_bar,z_bar,model::Phase6TwoLevelPopulation,
        atmosphere,population_bar)
    fill!(feature_bar,0); fill!(z_bar,0)
    @views feature_bar[1,:,:,:].=model.scale/5000 .* (population_bar[:,:,:,1].+
        0.1population_bar[:,:,:,2])
    feature_bar,z_bar
end

function phase6_mixed_fixture(temperature_values,vz_values;observation=nothing)
    grid=Grid3D([-5.0,-3.0,-1.0],[0.0,40e3],[0.0,40e3])
    shape=(3,2,2); zeros3=zeros(shape)
    atmosphere=Atmosphere3D(grid,fill(5500.0,shape),copy(zeros3),copy(zeros3),copy(zeros3),
        fill(800.0,shape))
    context=serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
    distributed=distribute_atmosphere(Float64,atmosphere,context)
    wavelength=collect(range(656.245e-9,656.315e-9,length=5))
    workspace=HybridForwardWorkspace(Float64,distributed,wavelength,StokesSet(:I),Dict(:H=>2,:CA=>2))
    continuum=TabulatedOpacityModel(fill(2e-5,3,5),fill(2e8,3,5))
    transition=FFNOTransition(:H,656.28e-9,1,2,0.64,1.6735575e-27,1e8)
    ca_transition=FFNOTransition(:CA,656.300e-9,1,2,0.3,40*1.66053906660e-27,1e8)
    kurucz_line=KuruczLine(656.292e-9,"FeI",-2.5,1.0*1.602176634e-19,
        3.0*1.602176634e-19,1e8)
    synthesizer=MixedIntensitySynthesizer([
        OpacityContributor(continuum),OpacityContributor(transition,workspace.populations[:H]),
        OpacityContributor(ca_transition,workspace.populations[:CA]),
        OpacityContributor(KuruczLTEModel([kurucz_line]))])
    force=ForceBalanceOptions(max_iterations=5,relative_tolerance=5.0,force_tolerance=5.0,
        height_tolerance_m=1e9,relaxation=0.5,pressure_sweeps=4)
    psf=GaussianPSFObservation(0.015e-9,20e3,20e3,40e3,40e3)
    population_model=CompositeDistributedPopulationModel(Dict(
        :H=>LocalDistributedPopulationModel(Phase6TwoLevelPopulation(1e12),2),
        :CA=>LocalDistributedPopulationModel(Phase6TwoLevelPopulation(5e11),2)))
    model=HybridForwardModel(population_model,
        NonPRD(),synthesizer,psf,IdealGasEOS(),ReferenceOpacity500(kappa_m2_kg=0.02),
        HE3DBoundaryState(1e-10,1.0,:top),force,CapabilityManifest())
    temp=NodeField(reshape(Float64.(temperature_values),2,1,1),[-5.0,-1.0])
    vz=NodeField(reshape(Float64.(vz_values),2,1,1),[-5.0,-1.0])
    layout=ControlMapLayout(
        ControlMapSpec(:temperature,temp;lower=3500.0,upper=8000.0,scale=1000.0),
        ControlMapSpec(:vz,vz;lower=-5000.0,upper=5000.0,scale=1000.0))
    regularization=RegularizationSpec(
        vertical=VerticalRegularizationSpec((1,1,0,0,0,0,1),1e-5,ntuple(_->1.0,7)),
        horizontal=Dict(:temperature=>1e-6),
        scales=Dict(:temperature=>1000.0,:vlos=>1000.0,:pgas_boundary=>0.1),
        horizontal_order=1)
    if observation===nothing
        apply_control_maps!(distributed,layout,initial_parameters(layout))
        result=forward!(workspace,model,distributed,context)
        observation=ObservationCube(SpectralCube(copy(result.spectrum.data),wavelength,StokesSet(:I)),
            fill(maximum(abs,result.spectrum.data)*1e-3,size(result.spectrum.data)),
            ones(size(result.spectrum.data)))
    end
    problem=DistributedInversionProblem(model,workspace,distributed,observation,regularization,
        40e3,40e3,context)
    (context=context,distributed=distributed,model=model,workspace=workspace,layout=layout,
        problem=problem,observation=observation)
end

@testset "Phase 6 production observation and formal-solver pullbacks" begin
    T=Float64
    grid=Grid3D(T[-3,-2,-1],T[0,1,2],T[0,1,2])
    shape=(3,3,3); zero3=zeros(T,shape)
    atmosphere=Atmosphere3D(grid,fill(T(5000),shape),copy(zero3),copy(zero3),copy(zero3),copy(zero3);
        z=repeat(reshape(T[2,1,0],3,1,1),1,3,3))
    distributed=DistributedAtmosphere(atmosphere,grid,tile_for_rank(3,3,0,1))
    context=serial_context()
    wavelength=collect(range(500e-9,500.3e-9,length=4))
    intrinsic=SpectralCube(rand(T,4,1,3,3),wavelength,StokesSet(:I))
    model=GaussianPSFObservation(0.08e-9,0.5,0.5,1.0,1.0)
    output=SpectralCube(zeros(T,4,1,3,3),wavelength,StokesSet(:I))
    apply_observation!(output,model,intrinsic)
    tangent=randn(T,size(intrinsic.data)); cotangent=randn(T,size(output.data)); input_bar=similar(tangent)
    observation_vjp!(input_bar,cotangent,model,intrinsic,distributed,context)
    epsilon=1e-6
    plus=SpectralCube(intrinsic.data.+epsilon.*tangent,wavelength,StokesSet(:I))
    minus=SpectralCube(intrinsic.data.-epsilon.*tangent,wavelength,StokesSet(:I))
    yp=SpectralCube(zeros(T,size(output.data)),wavelength,StokesSet(:I));
    ym=SpectralCube(zeros(T,size(output.data)),wavelength,StokesSet(:I))
    apply_observation!(yp,model,plus); apply_observation!(ym,model,minus)
    @test isapprox(sum(((yp.data.-ym.data)./(2epsilon)).*cotangent),sum(tangent.*input_bar);
        rtol=2e-8,atol=2e-9)

    chi=0.5 .+ rand(T,3,4); eta=0.2 .+ rand(T,3,4); z=T[3,1,0]
    output_tangent=randn(T,4); chi_tangent=randn(T,3,4); eta_tangent=randn(T,3,4); z_tangent=randn(T,3)
    chi_bar=zeros(T,3,4); eta_bar=zeros(T,3,4); z_bar=zeros(T,3)
    FFNOInversion._formal_solve_vjp!(chi_bar,eta_bar,z_bar,ScalarFormalSolver(),chi,eta,z,output_tangent)
    solve(c,e,zz)=begin out=zeros(T,4); formal_solve!(out,ScalarFormalSolver(),c,e,zz); out end
    jvp=(solve(chi.+epsilon.*chi_tangent,eta.+epsilon.*eta_tangent,z.+epsilon.*z_tangent)-
        solve(chi.-epsilon.*chi_tangent,eta.-epsilon.*eta_tangent,z.-epsilon.*z_tangent))./(2epsilon)
    @test isapprox(dot(jvp,output_tangent),sum(chi_tangent.*chi_bar)+sum(eta_tangent.*eta_bar)+dot(z_tangent,z_bar);
        rtol=2e-6,atol=2e-7)
end

@testset "Phase 6 complete mixed NLTE plus Kurucz objective pullback" begin
    truth=phase6_mixed_fixture([5200.0,6400.0],[-600.0,900.0])
    fixture=phase6_mixed_fixture([4900.0,6100.0],[-300.0,500.0];observation=truth.observation)
    parameters=initial_parameters(fixture.layout)
    production=objective_gradient!(HybridAdjointObjectiveGradient(force_balance_step=2e-5),
        fixture.problem,fixture.layout,parameters,fixture.context)
    oracle=objective_gradient!(FiniteDifferenceObjectiveGradient(step=2e-5),fixture.problem,
        fixture.layout,parameters,fixture.context)
    relative=norm(production.gradient-oracle.gradient)/max(norm(oracle.gradient),eps())
    @test relative<5e-3
    @test production.forward_evaluations==1
    direction=[0.7,-0.4,0.2,-0.5]
    report=gradient_taylor_validation(HybridAdjointObjectiveGradient(force_balance_step=2e-5),
        fixture.problem,fixture.layout,parameters,direction,fixture.context;
        steps=(2e-3,1e-3,5e-4))
    @test minimum(report.observed_orders)>1.7
    initial=production.evaluation.components.total
    result=lbfgs_invert!(fixture.problem,fixture.layout,
        HybridAdjointObjectiveGradient(force_balance_step=2e-5),fixture.context;
        options=LBFGSSolverOptions(max_iterations=4,history_length=3,gradient_tolerance=1e-8,
            initial_step=0.01,maximum_line_search_trials=20))
    @test result.state.objective<initial
    @test result.state.forward_evaluations<40
    println("PHASE6_PRODUCTION_MIXED_GRADIENT_OK relative_error=$relative forwards=$(result.state.forward_evaluations)")
end
