struct Phase5RecoverySynthesizer <: AbstractSynthesizer end

function FFNOInversion.synthesize!(cube::SpectralCube{T},::Phase5RecoverySynthesizer,
        ::NonPRD,atmosphere::Atmosphere3D,populations,cache=nothing) where T
    size(cube.data,1)==4 || throw(DimensionMismatch("Phase 5 fixture requires four wavelengths"))
    nz,nx,ny=size(atmosphere.temperature)
    for i in 1:nx,j in 1:ny
        cube.data[1,1,i,j]=atmosphere.temperature[1,i,j]/T(5000)
        cube.data[2,1,i,j]=atmosphere.temperature[nz,i,j]/T(5000)
        cube.data[3,1,i,j]=one(T)+atmosphere.vz[1,i,j]/T(5000)
        cube.data[4,1,i,j]=one(T)+atmosphere.vz[nz,i,j]/T(5000)
    end
    cube
end

function phase5_fixture()
    grid=Grid3D([-5.0,-3.0,-1.0],[0.0,50e3,100e3],[0.0,50e3])
    shape=(3,3,2); zero3=zeros(shape)
    atmosphere=Atmosphere3D(grid,fill(5500.0,shape),copy(zero3),copy(zero3),copy(zero3),copy(zero3))
    context=serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
    distributed=distribute_atmosphere(Float64,atmosphere,context)
    wave=[500.0e-9,500.1e-9,500.2e-9,500.3e-9]
    # The fixture isolates optimizer orchestration; strict force-balance
    # convergence is covered independently by the Phase 1/4 suites.
    force=ForceBalanceOptions(max_iterations=4,relative_tolerance=5.0,force_tolerance=5.0,
        height_tolerance_m=1e9,relaxation=0.5,pressure_sweeps=4)
    model=HybridForwardModel(LocalDistributedPopulationModel(MockPopulationModel(1e10),1),NonPRD(),
        Phase5RecoverySynthesizer(),IdentityObservation(),IdealGasEOS(),
        ReferenceOpacity500(kappa_m2_kg=0.02),HE3DBoundaryState(1e-10,1.0,:top),force,
        CapabilityManifest())
    workspace=HybridForwardWorkspace(Float64,distributed,wave,StokesSet(:I),1)
    zero_reg=RegularizationSpec(vertical=VerticalRegularizationSpec(ntuple(_->0,7),0.0,ntuple(_->1.0,7)),
        horizontal=Dict{Symbol,Float64}(),scales=Dict{Symbol,Float64}(),horizontal_order=1)
    (grid=grid,context=context,distributed=distributed,wave=wave,model=model,workspace=workspace,
        regularization=zero_reg)
end

function phase5_layout(temp_values,vz_values)
    temp=NodeField(reshape(Float64.(temp_values),2,2,1),[-5.0,-1.0])
    vz=NodeField(reshape(Float64.(vz_values),2,1,1),[-5.0,-1.0])
    ControlMapLayout(
        ControlMapSpec(:temperature,temp;lower=3000.0,upper=8000.0,scale=1000.0),
        ControlMapSpec(:vz,vz;lower=-3000.0,upper=3000.0,scale=1000.0))
end

function phase5_problem(fixture,root_observation)
    local_observation=distribute_observation(Float64,root_observation,size(root_observation.spectrum.data),
        fixture.wave,StokesSet(:I),fixture.context)
    DistributedInversionProblem(fixture.model,fixture.workspace,fixture.distributed,local_observation,
        fixture.regularization,50e3,50e3,fixture.context)
end

@testset "Phase 5 bounded controls and refinement" begin
    fixture=phase5_fixture()
    layout=phase5_layout([4500,4500,4500,4500],[0,0])
    parameters=initial_parameters(layout)
    @test length(parameters)==6
    @test parameter_nodefield(layout,parameters,:temperature).values==reshape([4500.0,4500,4500,4500],2,2,1)
    parameters[1]=1.0; project_parameters!(parameters,layout)
    @test parameters[1]==3000.0
    @test scaled_parameters(layout,parameters)[1]==3.0
    refined=refine_control_maps(layout,initial_parameters(layout);shapes=Dict(:temperature=>(3,2)))
    @test size(parameter_nodefield(refined.layout,refined.parameters,:temperature).values)==(2,3,2)
    apply_control_maps!(fixture.distributed,layout,initial_parameters(layout))
    @test size(fixture.distributed.local_atmosphere.temperature)==(3,3,2)
end


@testset "Phase 5 exact-model recovery, directional FD, and restart" begin
    truth_layout=phase5_layout([5000,6000,5500,6500],[-1000,1000])
    truth_parameters=initial_parameters(truth_layout)
    truth_fixture=phase5_fixture()
    apply_control_maps!(truth_fixture.distributed,truth_layout,truth_parameters)
    truth_result=forward!(truth_fixture.workspace,truth_fixture.model,truth_fixture.distributed,truth_fixture.context)
    observed=ObservationCube(SpectralCube(copy(truth_result.spectrum.data),truth_fixture.wave,StokesSet(:I)),
        fill(0.02,size(truth_result.spectrum.data)),ones(size(truth_result.spectrum.data)))

    recovered=PrototypeInversionResult[]
    for initial_temperature in (4500.0,7000.0)
        fixture=phase5_fixture()
        layout=phase5_layout(fill(initial_temperature,4),[0.0,0.0])
        problem=phase5_problem(fixture,observed)
        initial_eval=evaluate_objective!(problem,layout,initial_parameters(layout),fixture.context)
        result=prototype_invert!(problem,layout,fixture.context;
            options=PrototypeSolverOptions(max_iterations=24,initial_step=0.5,
                minimum_step=1e-4,improvement_tolerance=1e-12))
        push!(recovered,result)
        @test result.state.objective<initial_eval.components.total*1e-6
        @test maximum(abs.(result.state.parameters.-truth_parameters))<=1e-8
        @test result.state.history[1].total<=initial_eval.components.total
        @test all(record->isfinite(record.total),result.state.history)
    end
    @test recovered[1].state.parameters==recovered[2].state.parameters

    fd_fixture=phase5_fixture(); fd_layout=phase5_layout([5200,6100,5400,6400],[-800,800])
    fd_problem=phase5_problem(fd_fixture,observed); fd_parameters=initial_parameters(fd_layout)
    direction=collect(1.0:length(fd_parameters))
    report=centered_directional_validation(fd_problem,fd_layout,fd_parameters,direction,
        fd_fixture.context;steps=(1e-3,5e-4))
    @test length(report.estimates)==2
    @test report.relative_change<1e-8
    @test all(isfinite(estimate.derivative) for estimate in report.estimates)

    mktempdir() do directory
        checkpoint_path=joinpath(directory,"phase5.checkpoint")
        continuous_fixture=phase5_fixture(); continuous_layout=phase5_layout(fill(4500.0,4),[0.0,0.0])
        continuous_problem=phase5_problem(continuous_fixture,observed)
        continuous=prototype_invert!(continuous_problem,continuous_layout,continuous_fixture.context;
            options=PrototypeSolverOptions(max_iterations=8,initial_step=0.5,checkpoint_path=""))

        split_fixture=phase5_fixture(); split_layout=phase5_layout(fill(4500.0,4),[0.0,0.0])
        split_problem=phase5_problem(split_fixture,observed)
        prototype_invert!(split_problem,split_layout,split_fixture.context;
            options=PrototypeSolverOptions(max_iterations=3,initial_step=0.5,
                checkpoint_path=checkpoint_path))
        restarted=prototype_invert!(split_problem,split_layout,split_fixture.context;restart=true,
            options=PrototypeSolverOptions(max_iterations=8,initial_step=0.5,
                checkpoint_path=checkpoint_path))
        @test restarted.state.parameters==continuous.state.parameters
        @test restarted.state.objective==continuous.state.objective
        @test restarted.state.history==continuous.state.history
        @test restarted.state.evaluations==continuous.state.evaluations
        diagnostics=write_prototype_diagnostics(joinpath(directory,"diagnostics"),restarted,
            split_layout,split_fixture.context;metadata=Dict("fixture"=>"exact-model"))
        @test isfile(diagnostics.summary) && isfile(diagnostics.history)
        @test occursin("objective =",read(diagnostics.summary,String))
        @test length(readlines(diagnostics.history))==length(restarted.state.history)+1
    end
    println("PHASE5_SYNTHETIC_RECOVERY_OK initializations=2 parameters=6")
end
