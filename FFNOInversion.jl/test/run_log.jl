# The run log on rank 0's stdout (the Slurm output): a section per iteration ("Iteration 0" is the first forward model and
# gradient) with its lines indented by a tab under it, the objective, data and regularization terms of each iteration, the
# time of each part of the gradient, and the CPU memory of each node and the memory of each GPU.
# Uses phase5_fixture/phase6_observation (phase 5/6 tests), phase6_mixed_fixture (phase6_production_gradient.jl) and
# fake_fsdp_population_backend (phase6_execution.jl).

@testset "Run log: iteration sections, objective, gradient timing, memory" begin
    observation=phase6_observation()
    fixture=phase5_fixture(); layout=phase5_layout(fill(4500.0,4),[0.0,0.0]); problem=phase5_problem(fixture,observation)
    result,log=_captured_stdout(()->lbfgs_invert!(problem,layout,Phase6RecoveryVJP(),fixture.context;
        options=LBFGSSolverOptions(max_iterations=2,history_length=4,gradient_tolerance=1e-12,objective_tolerance=0.0)))
    lines=split(chomp(log),'\n')
    @test lines[1]=="Iteration 0"
    @test "Iteration 1" in lines && "Iteration 2" in lines && !("Iteration 3" in lines)
    # every line of a section is indented under its header; the closing line is back at the margin
    last=findfirst(startswith("LBFGS done"),lines)
    @test last==length(lines)
    @test all(line->startswith(line,"Iteration ") || startswith(line,"\t"),lines[1:last-1])
    zero=lines[findfirst(startswith("\tLBFGS iteration=0 "),lines)]
    for key in ("total=","data=","regularization=","projected_gradient=","seconds=")
        @test occursin(key,zero)
    end
    @test occursin("total=$(result.state.history[1].total) ",lines[findfirst(startswith("\tLBFGS iteration=1 "),lines)])
    memory=filter(startswith("\tmemory node="),lines)
    @test length(memory)==3                                   # iteration 0, 1 and 2, one node
    @test occursin("memory node=$(gethostname()) cpu_used=",memory[1]) && occursin(" julia_ranks=1 julia_rss=",memory[1])
    @test !occursin("gpu_",memory[1])                          # no GPU service behind this population model
    # the header of iteration k comes before its lines and after iteration k-1's summary
    @test findfirst(==("Iteration 1"),lines)>findfirst(startswith("\tLBFGS iteration=0 "),lines)
    @test findfirst(startswith("\tLBFGS iteration=1 "),lines)>findfirst(==("Iteration 1"),lines)

    # the production gradient prints the slowest rank's time for each of its parts
    mixed=phase6_mixed_fixture([5100.0,6200.0],[-500.0,700.0])
    _,gradient_log=_captured_stdout(()->objective_gradient!(HybridAdjointObjectiveGradient(),mixed.problem,mixed.layout,
        initial_parameters(mixed.layout),mixed.context))
    timing=match(r"FORWARD stage=gradient seconds forward=([\d.]+) observation=([\d.]+) synthesis=([\d.]+) ffno=([\d.]+) "*
        r"regularization=([\d.]+) equilibrium=([\d.]+) controls=([\d.]+) total=([\d.]+)",gradient_log)
    @test timing!==nothing
    parts=parse.(Float64,timing.captures)
    @test all(>=(0),parts) && parts[end]>=sum(parts[1:end-1])-0.3       # rounding of each part to 0.1 s

    # GPU memory from the service: each GPU's peak allocated and reserved memory, by node
    backend,server=fake_fsdp_population_backend()
    service=FFNOInversion.population_service(backend)
    @test service isa FSDPServiceClient
    @test FFNOInversion.population_service(CompositeDistributedPopulationModel(Dict(:H=>backend)))===service
    @test FFNOInversion.population_service(fixture.model.populations)===nothing
    gpus=FFNOInversion.gpu_memory(service)
    @test length(gpus)==2 && gpus[1].host==gethostname() && gpus[2].local_rank==1
    @test gpus[1].peak_allocated==3*2^30 && gpus[1].reserved==4*2^30 && gpus[1].total==80*2^30
    _,gpu_log=_captured_stdout(()->(FFNOInversion._log_section(fixture.context,"Iteration 4");
        FFNOInversion.log_memory(fixture.context,backend); FFNOInversion._end_log_section()))
    @test occursin("Iteration 4\n\tmemory node=$(gethostname()) cpu_used=",gpu_log)
    @test occursin(" gpu_peak_allocated=[3.0, 2.0] GB gpu_reserved=[4.0, 4.0] GB of 80.0 GB each",gpu_log)
    close_fsdp_service!(service); wait(server)
end
