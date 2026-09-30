@testset "Node corrections preserve the starting atmosphere" begin
    f=phase5_fixture(); a=f.distributed.local_atmosphere
    a.temperature[2,:,:].+=731.0 # structure not representable by endpoint nodes
    baseline=copy(a.temperature); velocity=copy(a.vz)
    layout=phase5_layout(fill(5500.0,4),[0.0,0.0])
    observation=phase6_observation()
    make_problem(id="reference-A")=DistributedInversionProblem(f.model,f.workspace,f.distributed,
        observation,f.regularization,50e3,50e3,f.context;control_layout=layout,reference_id=id)
    problem=make_problem(); initial=initial_parameters(layout)
    apply_control_maps!(problem,layout,initial)
    @test a.temperature==baseline && a.vz==velocity
    trial=copy(initial); trial[1:4].+=100
    apply_control_maps!(problem,layout,trial)
    @test a.temperature==baseline.+100
    apply_control_maps!(problem,layout,trial)
    @test a.temperature==baseline.+100 # never accumulates trials
    apply_control_maps!(problem,layout,initial)
    @test a.temperature==baseline
    e=evaluate_objective!(problem,layout,initial,f.context)
    report=gradient_taylor_validation(Phase6RecoveryVJP(),problem,layout,initial,
        [0.3,-0.4,0.2,0.1,0.1,-0.2],f.context;steps=(1e-2,5e-3,2.5e-3))
    @test minimum(report.observed_orders)>1.9
    # Node values stay inside bounds, but the unresolved bump would exceed 8000.
    trial[1:4].=7800
    rejected=evaluate_objective!(problem,layout,trial,f.context)
    @test isinf(rejected.components.total)
    evaluate_objective!(problem,layout,initial,f.context)
    @test a.temperature==baseline
    # Finite differences at a full-grid bound must use feasible one-sided steps.
    boundary=copy(initial); boundary[1:4].=8000-731
    evaluate_objective!(problem,layout,boundary,f.context)
    hp,hm=FFNOInversion._coordinate_step_limits(problem,layout,boundary,1,f.context)
    @test hp==0 && hm>0
    bounded=objective_gradient!(FiniteDifferenceObjectiveGradient(),problem,layout,boundary,f.context)
    @test all(isfinite,bounded.gradient)
    evaluate_objective!(problem,layout,initial,f.context)
    refined=refine_control_maps(problem,layout,initial,f.context;shapes=Dict(:temperature=>(3,2)))
    apply_control_maps!(refined.problem,refined.layout,refined.parameters)
    @test a.temperature==baseline
    @test_throws ArgumentError apply_control_maps!(problem,refined.layout,refined.parameters)
    rejected_solver=lbfgs_invert!(problem,layout,NegatedPhase6Gradient(),f.context;
        options=LBFGSSolverOptions(max_iterations=1,maximum_line_search_trials=2))
    @test rejected_solver.state.termination===:line_search_failed
    @test a.temperature==baseline
    mktempdir() do dir
        checkpoint=joinpath(dir,"corrections.checkpoint")
        short=lbfgs_invert!(problem,layout,Phase6RecoveryVJP(),f.context;
            options=LBFGSSolverOptions(max_iterations=1,checkpoint_path=checkpoint))
        continued=lbfgs_invert!(problem,layout,Phase6RecoveryVJP(),f.context;restart=true,
            options=LBFGSSolverOptions(max_iterations=3,checkpoint_path=checkpoint))
        uninterrupted=lbfgs_invert!(problem,layout,Phase6RecoveryVJP(),f.context;
            options=LBFGSSolverOptions(max_iterations=3))
        @test continued.state.parameters≈uninterrupted.state.parameters
        @test continued.state.objective≈uninterrupted.state.objective
        evaluate_objective!(problem,layout,initial,f.context)
        wrong=make_problem("different-atmosphere")
        @test_throws ArgumentError lbfgs_invert!(wrong,layout,Phase6RecoveryVJP(),f.context;restart=true,
            options=LBFGSSolverOptions(max_iterations=4,checkpoint_path=checkpoint))
    end
end

@testset "Production adjoint differentiates the correction mapping" begin
    truth=phase6_mixed_fixture([5200.0,6400.0],[-600.0,900.0])
    f=phase6_mixed_fixture([4900.0,6100.0],[-300.0,500.0];observation=truth.observation)
    f.distributed.local_atmosphere.temperature[2,:,:].+=200
    problem=DistributedInversionProblem(f.model,f.workspace,f.distributed,f.problem.observation,
        f.problem.regularization,40e3,40e3,f.context;control_layout=f.layout,reference_id="mixed")
    initial=initial_parameters(f.layout)
    analytic=objective_gradient!(HybridAdjointObjectiveGradient(force_balance_step=2e-5),problem,f.layout,initial,f.context)
    oracle=objective_gradient!(FiniteDifferenceObjectiveGradient(step=2e-5),problem,f.layout,initial,f.context)
    @test norm(analytic.gradient-oracle.gradient)/max(norm(oracle.gradient),eps())<5e-3
end
