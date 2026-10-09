struct Phase6RecoveryVJP <: AbstractObjectiveGradient end

function FFNOInversion.objective_gradient!(::Phase6RecoveryVJP,
        problem::DistributedInversionProblem,layout::ControlMapLayout{T},
        parameters::AbstractVector,context::ParallelContext) where T
    evaluation=evaluate_objective!(problem,layout,parameters,context)
    synthetic=problem.workspace.output.data; observation=problem.observation
    spectral_cotangent=similar(synthetic)
    @. spectral_cotangent=(T(2)/problem.active_residual_count)*
        observation.inversion_weights^2*(synthetic-observation.spectrum.data)/observation.sigma^2
    atmosphere=problem.distributed.local_atmosphere
    temperature_cotangent=zeros(T,size(atmosphere.temperature))
    vz_cotangent=zeros(T,size(atmosphere.vz)); nz=size(atmosphere.temperature,1)
    for i in axes(temperature_cotangent,2),j in axes(temperature_cotangent,3)
        temperature_cotangent[1,i,j]+=spectral_cotangent[1,1,i,j]/T(5000)
        temperature_cotangent[nz,i,j]+=spectral_cotangent[2,1,i,j]/T(5000)
        vz_cotangent[1,i,j]+=spectral_cotangent[3,1,i,j]/T(5000)
        vz_cotangent[nz,i,j]+=spectral_cotangent[4,1,i,j]/T(5000)
    end
    gradient=zeros(T,layout.parameter_count)
    accumulate_control_vjp!(gradient,layout,:temperature,temperature_cotangent,
        problem.distributed,context)
    accumulate_control_vjp!(gradient,layout,:vz,vz_cotangent,problem.distributed,context)
    ObjectiveGradientEvaluation(evaluation,gradient,1)
end

struct NegatedPhase6Gradient <: AbstractObjectiveGradient end
function FFNOInversion.objective_gradient!(::NegatedPhase6Gradient,problem,layout,parameters,context)
    result=objective_gradient!(Phase6RecoveryVJP(),problem,layout,parameters,context)
    ObjectiveGradientEvaluation(result.evaluation,-result.gradient,result.forward_evaluations)
end

function phase6_observation()
    truth_layout=phase5_layout([5000,6000,5500,6500],[-1000,1000])
    fixture=phase5_fixture(); truth=initial_parameters(truth_layout)
    apply_control_maps!(fixture.distributed,truth_layout,truth)
    result=forward!(fixture.workspace,fixture.model,fixture.distributed,fixture.context)
    ObservationCube(SpectralCube(copy(result.spectrum.data),fixture.wave,StokesSet(:I)),
        fill(0.02,size(result.spectrum.data)),ones(size(result.spectrum.data)))
end

@testset "Phase 6 matrix-free node VJP and gradient validation" begin
    fixture=phase5_fixture(); layout=phase5_layout([5200,6100,5400,6400],[-800,800])
    spec=layout.specs[1]; output_length=length(fixture.distributed.local_atmosphere.temperature)
    linearization=MatrixFreeLinearization(length(spec.initial),output_length,
        (output,tangent)->begin
            field=NodeField(reshape(collect(tangent),size(spec.initial)),spec.log_tau_nodes)
            output.=vec(expand_nodes(field,fixture.distributed.global_grid,fixture.distributed.tile))
        end,
        (output,cotangent)->begin
            local_bar=reshape(collect(cotangent),size(fixture.distributed.local_atmosphere.temperature))
            output.=vec(node_expansion_vjp(local_bar,spec,fixture.distributed.global_grid,
                fixture.distributed.tile,fixture.context))
        end)
    tangent=collect(1.0:linearization.input_length)
    cotangent=sin.(collect(1.0:linearization.output_length))
    dot_report=dot_product_validation(linearization,tangent,cotangent;rtol=1e-12,atol=1e-12)
    @test dot_report.passed
    @test dot_report.relative_error<1e-12

    observation=phase6_observation(); problem=phase5_problem(fixture,observation)
    parameters=initial_parameters(layout)
    analytic=objective_gradient!(Phase6RecoveryVJP(),problem,layout,parameters,fixture.context)
    oracle=objective_gradient!(FiniteDifferenceObjectiveGradient(step=1e-5),problem,layout,
        parameters,fixture.context)
    @test analytic.gradient≈oracle.gradient rtol=2e-8 atol=2e-8
    taylor=gradient_taylor_validation(Phase6RecoveryVJP(),problem,layout,parameters,
        collect(1.0:length(parameters)),fixture.context;steps=(1e-2,5e-3,2.5e-3))
    @test minimum(taylor.observed_orders)>1.9
end

@testset "Phase 6 bounded L-BFGS recovery, rejection safety, and restart" begin
    observation=phase6_observation(); truth=initial_parameters(
        phase5_layout([5000,6000,5500,6500],[-1000,1000]))
    fixture=phase5_fixture(); layout=phase5_layout(fill(4500.0,4),[0.0,0.0])
    problem=phase5_problem(fixture,observation)
    options=LBFGSSolverOptions(max_iterations=20,history_length=5,gradient_tolerance=1e-8,
        objective_tolerance=1e-14,maximum_line_search_trials=20)
    result=lbfgs_invert!(problem,layout,Phase6RecoveryVJP(),fixture.context;options=options)
    @test result.state.converged
    @test result.state.objective<1e-14
    @test maximum(abs.(result.state.parameters.-truth))<1e-6
    @test all(record->record.accepted,result.state.history)

    baseline_fixture=phase5_fixture(); baseline_layout=phase5_layout(fill(4500.0,4),[0.0,0.0])
    baseline_problem=phase5_problem(baseline_fixture,observation)
    baseline=prototype_invert!(baseline_problem,baseline_layout,baseline_fixture.context;
        options=PrototypeSolverOptions(max_iterations=24,initial_step=0.5,
            minimum_step=1e-4,improvement_tolerance=1e-12))
    @test result.state.forward_evaluations<baseline.state.evaluations
    @test result.state.objective<=baseline.state.objective+1e-14

    rejected_fixture=phase5_fixture(); rejected_layout=phase5_layout(fill(4500.0,4),[0.0,0.0])
    rejected_problem=phase5_problem(rejected_fixture,observation)
    rejected_initial=initial_parameters(rejected_layout)
    rejected=lbfgs_invert!(rejected_problem,rejected_layout,NegatedPhase6Gradient(),
        rejected_fixture.context;options=LBFGSSolverOptions(max_iterations=1,
            maximum_line_search_trials=4,backtracking_factor=0.25))
    @test rejected.state.termination==:line_search_failed
    @test rejected.state.parameters==rejected_initial
    @test !rejected.state.history[1].accepted
    @test rejected.state.history[1].rejected_trials==4
    expected_temperature=expand_nodes(parameter_nodefield(rejected_layout,rejected_initial,:temperature),
        rejected_fixture.distributed.global_grid,rejected_fixture.distributed.tile)
    @test rejected_fixture.distributed.local_atmosphere.temperature==expected_temperature

    mktempdir() do directory
        checkpoint_path=joinpath(directory,"phase6.checkpoint")
        continuous_fixture=phase5_fixture(); continuous_layout=phase5_layout(fill(4500.0,4),[0.0,0.0])
        continuous_problem=phase5_problem(continuous_fixture,observation)
        continuous=lbfgs_invert!(continuous_problem,continuous_layout,Phase6RecoveryVJP(),
            continuous_fixture.context;options=LBFGSSolverOptions(max_iterations=8,
                history_length=4,gradient_tolerance=1e-12,objective_tolerance=0.0))

        split_fixture=phase5_fixture(); split_layout=phase5_layout(fill(4500.0,4),[0.0,0.0])
        split_problem=phase5_problem(split_fixture,observation)
        lbfgs_invert!(split_problem,split_layout,Phase6RecoveryVJP(),split_fixture.context;
            options=LBFGSSolverOptions(max_iterations=1,history_length=4,
                gradient_tolerance=1e-12,objective_tolerance=0.0,checkpoint_path=checkpoint_path))
        restarted=lbfgs_invert!(split_problem,split_layout,Phase6RecoveryVJP(),split_fixture.context;
            restart=true,options=LBFGSSolverOptions(max_iterations=8,history_length=4,
                gradient_tolerance=1e-12,objective_tolerance=0.0,checkpoint_path=checkpoint_path))
        @test restarted.state.parameters==continuous.state.parameters
        @test restarted.state.objective==continuous.state.objective
        @test restarted.state.gradient==continuous.state.gradient
        @test restarted.state.history==continuous.state.history
        @test restarted.state.forward_evaluations==continuous.state.forward_evaluations
        diagnostics=write_lbfgs_diagnostics(joinpath(directory,"diagnostics"),restarted,
            split_layout,split_fixture.context;metadata=Dict("fixture"=>"exact-model"))
        @test isfile(diagnostics.summary) && isfile(diagnostics.history)
        @test occursin("bounded_lbfgs",read(diagnostics.summary,String))
    end
    println("PHASE6_LOCAL_GRADIENT_SOLVER_OK controls=6")
end
