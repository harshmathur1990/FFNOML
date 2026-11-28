# A line search that fails along the L-BFGS direction clears the history and retries along the projected gradient
# (solver.reset_history, on by default); the run stops with line_search_failed only when that also fails.
# The history in a checkpoint is replaced by one that proposes a huge step, as a stale history can after the
# objective's curvature has changed. Uses phase5_fixture/phase6_observation (phase 5/6 tests) and _captured_stdout
# (soft_stop_resume.jl).

@testset "L-BFGS: a failed line search clears the history and retries along the gradient" begin
    observation=phase6_observation()
    base=(history_length=4,gradient_tolerance=1e-12,objective_tolerance=0.0,
        maximum_line_search_trials=6,backtracking_factor=0.25)
    mktempdir() do directory
        checkpoint=joinpath(directory,"inversion.checkpoint")
        fixture=phase5_fixture(); layout=phase5_layout(fill(4500.0,4),[0.0,0.0])
        lbfgs_invert!(phase5_problem(fixture,observation),layout,Phase6RecoveryVJP(),fixture.context;
            options=LBFGSSolverOptions(;max_iterations=2,base...,checkpoint_path=checkpoint))
        # a history whose curvature is 1e-8 of the true one: H ~ 1e8, so its direction leaves the bounds at any trial step
        payload=restore_checkpoint(checkpoint); saved=payload.state.state
        @test length(saved.s_history)==2
        for (s,y) in zip(saved.s_history,saved.y_history)
            y.=1e-8 .* s
        end
        checkpoint!(checkpoint,payload.state;manifest=payload.manifest)
        stale=checkpoint*".stale"; cp(checkpoint,stale)

        # the default: the failed search is recorded, the history cleared, and the gradient step accepted
        reset_case=phase5_fixture()
        reset,log=_captured_stdout(()->lbfgs_invert!(phase5_problem(reset_case,observation),layout,Phase6RecoveryVJP(),
            reset_case.context;restart=true,options=LBFGSSolverOptions(;max_iterations=6,base...,checkpoint_path=checkpoint)))
        @test reset.state.termination===:maximum_iterations && reset.state.iteration==6
        third=filter(record->record.iteration==3,reset.state.history)
        @test length(third)==2
        @test !third[1].accepted && third[1].rejected_trials==6
        @test third[2].accepted
        @test third[2].total<saved.objective
        @test occursin("LBFGS iteration=3 line search failed after 6 trials along the L-BFGS direction; "*
            "history cleared, retrying along the projected gradient",log)
        @test !occursin("line_search_failed",log)
        @test reset.state.objective<third[2].total                   # the rebuilt history keeps descending
        @test all(record->record.accepted,filter(record->record.iteration>3,reset.state.history))

        # reset_history=false keeps the old stop: the run ends at the last accepted point
        cp(stale,checkpoint;force=true)
        kept_case=phase5_fixture()
        kept=lbfgs_invert!(phase5_problem(kept_case,observation),layout,Phase6RecoveryVJP(),kept_case.context;
            restart=true,options=LBFGSSolverOptions(;max_iterations=6,base...,checkpoint_path=checkpoint,reset_history=false))
        @test kept.state.termination===:line_search_failed
        @test kept.state.parameters==saved.parameters && kept.state.objective==saved.objective
        @test count(record->record.iteration==3,kept.state.history)==1

        # a failure along the gradient itself (empty history) still stops the run: no second try
        negated_case=phase5_fixture()
        negated,negated_log=_captured_stdout(()->lbfgs_invert!(phase5_problem(negated_case,observation),layout,
            NegatedPhase6Gradient(),negated_case.context;options=LBFGSSolverOptions(;max_iterations=3,base...)))
        @test negated.state.termination===:line_search_failed
        @test length(negated.state.history)==1 && negated.state.history[1].rejected_trials==6
        @test !occursin("history cleared",negated_log)
    end
    # the TOML key
    @test FFNOInversion._lbfgs_solver(Dict{String,Any}()).reset_history
    @test !FFNOInversion._lbfgs_solver(Dict{String,Any}("reset_history"=>false)).reset_history
end
