include("test_equilibrium.jl")
@testset "Phase 4 unified hybrid forward" begin
    grid=Grid3D([-5.0,-4,-3,-2],[0.0,50e3,100e3],[0.0,50e3,100e3])
    shape=(4,3,3); zero3=zeros(shape)
    atmosphere=Atmosphere3D(grid,fill(5500.0,shape),copy(zero3),copy(zero3),copy(zero3),copy(zero3))
    context=serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
    distributed=distribute_atmosphere(Float64,atmosphere,context)

    nodes=NodeField(reshape(collect(1.0:12.0),3,2,2),[-5.0,-3.0,-2.0])
    @test expand_nodes(nodes,grid,distributed.tile)==expand_nodes(nodes,grid)
    wave=collect(range(656.1e-9,656.5e-9,length=9))
    psf=GaussianPSFObservation(0.04e-9,50e3,50e3,50e3,50e3)
    equilibrium=TestHydrostaticEquilibrium()
    model=HybridForwardModel(LocalDistributedPopulationModel(MockPopulationModel(1e10),1),NonPRD(),
        MockIntensitySynthesizer(),psf,equilibrium,CapabilityManifest())
    workspace=HybridForwardWorkspace(Float64,distributed,wave,StokesSet(:I),1)
    result=forward!(workspace,model,distributed,context)
    @test result.valid && result.equilibrium.valid && equilibrium.calls==1
    @test size(result.spectrum.data)==(9,1,3,3)
    @test all(isfinite,result.spectrum.data)
    @test result.timings.total_seconds>=result.timings.equilibrium_seconds+
        result.timings.populations_seconds+result.timings.synthesis_seconds+result.timings.observation_seconds
    model.prepared_equilibrium[]=result.equilibrium
    prepared_result=forward!(workspace,model,distributed,context;reuse_prepared=true)
    @test prepared_result.equilibrium===result.equilibrium && equilibrium.calls==1
    @test prepared_result.timings.equilibrium_seconds==0
    @test prepared_result.timings.populations_seconds==0
    @test model.prepared_equilibrium[]===nothing
    gathered=gather_atmosphere(distributed,context)
    @test all(gathered.pgas.>0) && all(gathered.ne.>0)
    observed=ObservationCube(SpectralCube(copy(result.spectrum.data),wave,StokesSet(:I)),
        ones(size(result.spectrum.data)),ones(size(result.spectrum.data)))
    local_observed=distribute_observation(Float64,observed,size(observed.spectrum.data),wave,StokesSet(:I),context)
    @test local_observed.spectrum.data==observed.spectrum.data
    @test distributed_chi2(result.spectrum,observed,context)==0
    spec=RegularizationSpec(vertical=VerticalRegularizationSpec((1,0,0,0,0,0,0),1.0,ntuple(_->1.0,7)),
        horizontal=Dict(:temperature=>1.0),scales=Dict(:temperature=>1000.0),horizontal_order=1)
    @test distributed_regularization_penalty(distributed,spec,50e3,50e3,context).total==0
    workspace_ids=(objectid(workspace.populations),objectid(workspace.intrinsic.data),
        objectid(workspace.output.data),map(ws->objectid(ws.extinction),workspace.synthesis_cache.workspaces))
    allocation1=@allocated forward!(workspace,model,distributed,context)
    reference_output=copy(workspace.output.data)
    allocation2=@allocated forward!(workspace,model,distributed,context)
    @test allocation2<=allocation1*1.05+100_000
    @test workspace_ids==(objectid(workspace.populations),objectid(workspace.intrinsic.data),
        objectid(workspace.output.data),map(ws->objectid(ws.extinction),workspace.synthesis_cache.workspaces))
    @test reference_output==workspace.output.data
    memory=distributed_memory_report(distributed,workspace,context)
    @test memory["rank_count"]==1 && memory["maximum_owned_bytes"]>0
    provenance=parallel_provenance(context,distributed.tile;configuration_hash="fixture-config",
        model_hash="fixture-model",source_revision="fixture-revision",capabilities=["intensity","crd"])
    @test provenance["rank_layout"][1]["rank"]==0
    @test provenance["threads_per_rank"]==Threads.nthreads()
    mktemp() do path,io
        close(io)
        @test write_parallel_provenance(path,provenance)==path
        @test occursin("configuration_hash = \"fixture-config\"",read(path,String))
    end
    println("PHASE4_ALLOCATION_STABLE first=$allocation1 second=$allocation2")

    # an invalid equilibrium short-circuits populations, synthesis and observation and keeps the previous spectrum
    previous=copy(workspace.output.data); calls=equilibrium.calls
    equilibrium.invalid_above_K=5000.0
    failed=forward!(workspace,model,distributed,context)
    @test !failed.valid && !failed.equilibrium.valid && failed.equilibrium.reason==:test_invalid
    @test workspace.output.data==previous && equilibrium.calls==calls+1
    @test failed.timings.populations_seconds==0 && failed.timings.synthesis_seconds==0
    equilibrium.invalid_above_K=Inf
end
