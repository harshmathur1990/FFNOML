@testset "Phase 4 unified hybrid forward" begin
    grid=Grid3D([-5.0,-4,-3,-2],[0.0,50e3,100e3],[0.0,50e3,100e3])
    shape=(4,3,3); zero3=zeros(shape)
    atmosphere=Atmosphere3D(grid,fill(5500.0,shape),copy(zero3),copy(zero3),copy(zero3),copy(zero3))
    context=serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
    distributed=distribute_atmosphere(Float64,atmosphere,context)

    cgrid=Grid3D([-2.0,-1.0,0.0],[0.0,2.0,4.0],[0.0,3.0,6.0])
    cshape=(3,3,3); cz=Array{Float64}(undef,cshape)
    for k in 1:3,i in 1:3,j in 1:3
        cz[k,i,j]=100.0*(k-1)+0.4*cgrid.x[i]-0.25*cgrid.y[j]
    end
    ca=3e-8; cb0=2e-4; cB=MagneticField3D(ca.*cz,zeros(cshape),fill(cb0,cshape))
    catmos=Atmosphere3D(cgrid,fill(5000.0,cshape),zeros(cshape),zeros(cshape),zeros(cshape),
        zeros(cshape);magnetic_field=cB,z=cz)
    cdistributed=distribute_atmosphere(Float64,catmos,context)
    crho=fill(2e-4,cshape); cg=10.0; cp=@. 1000.0-crho*cg*cz
    cfx=zeros(cshape); cfy=zeros(cshape); cfz=zeros(cshape)
    @test FFNOInversion._distributed_force_residual(cp,crho,cz,cfx,cfy,cfz,
        cdistributed,context,cg)<1e-10
    FFNOInversion._distributed_lorentz!(cfx,cfy,cfz,cdistributed,cz,context)
    @test maximum(abs.(cfx .- ca*cb0/FFNOInversion.MU0))<1e-12
    @test maximum(abs.(cfy))<1e-12
    @test maximum(abs.(cfz .+ ca^2 .* cz ./ FFNOInversion.MU0))<1e-12

    nodes=NodeField(reshape(collect(1.0:12.0),3,2,2),[-5.0,-3.0,-2.0])
    @test expand_nodes(nodes,grid,distributed.tile)==expand_nodes(nodes,grid)
    wave=collect(range(656.1e-9,656.5e-9,length=9))
    options=ForceBalanceOptions(max_iterations=100,relative_tolerance=1e-5,force_tolerance=0.6,
        height_tolerance_m=1.0,relaxation=0.5,pressure_sweeps=20)
    psf=GaussianPSFObservation(0.04e-9,50e3,50e3,50e3,50e3)
    model=HybridForwardModel(LocalDistributedPopulationModel(MockPopulationModel(1e10),1),NonPRD(),
        MockIntensitySynthesizer(),psf,IdealGasEOS(),ReferenceOpacity500(kappa_m2_kg=0.02),
        HE3DBoundaryState(1e-10,1.0,:top),options,CapabilityManifest())
    workspace=HybridForwardWorkspace(Float64,distributed,wave,StokesSet(:I),1)
    result=forward!(workspace,model,distributed,context)
    @test result.force_balance.converged
    @test size(result.spectrum.data)==(9,1,3,3)
    @test all(isfinite,result.spectrum.data)
    @test result.timings.total_seconds>=result.timings.force_balance_seconds+
        result.timings.populations_seconds+result.timings.synthesis_seconds+result.timings.observation_seconds
    model.prepared_force_balance[]=result.force_balance
    prepared_result=forward!(workspace,model,distributed,context;reuse_prepared=true)
    @test prepared_result.force_balance===result.force_balance
    @test prepared_result.timings.force_balance_seconds==0
    @test prepared_result.timings.populations_seconds==0
    @test model.prepared_force_balance[]===nothing
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

    limited=distribute_atmosphere(Float64,atmosphere,context)
    limited_options=ForceBalanceOptions(max_iterations=1,relative_tolerance=eps(),force_tolerance=eps(),
        height_tolerance_m=eps(),relaxation=0.5,pressure_sweeps=1)
    accepted=reconstruct_force_balance_distributed!(limited,HE3DBoundaryState(1e-10,1.0,:top),
        IdealGasEOS(),ReferenceOpacity500(kappa_m2_kg=0.02),context;options=limited_options)
    @test accepted.mode==:HE3D && accepted.iterations==1
    @test accepted.best_force_residual<=accepted.initial_force_residual
    @test accepted.force_residual==accepted.best_force_residual
    @test limited.local_atmosphere.pgas!==nothing
    reference_k0,reference_k1,reference_weight=
        FFNOInversion._zero_logtau_bracket(limited.global_grid.log_tau500)
    interpolated_z0=(1-reference_weight).*limited.local_atmosphere.z[reference_k0,:,:].+
        reference_weight.*limited.local_atmosphere.z[reference_k1,:,:]
    @test maximum(abs,interpolated_z0)<1e-8

    bx=fill(1e-5,shape); by=zeros(shape); bz=fill(2e-5,shape)
    for k in axes(by,1); @views by[k,:,:].=(k-1)*1e-8; end
    magnetic=Atmosphere3D(grid,fill(5500.0,shape),copy(zero3),copy(zero3),copy(zero3),copy(zero3);
        magnetic_field=MagneticField3D(bx,by,bz))
    distributed_mhs=distribute_atmosphere(Float64,magnetic,context)
    mhs_options=ForceBalanceOptions(max_iterations=100,relative_tolerance=5.0,
        force_tolerance=0.6,height_tolerance_m=2e7,relaxation=0.5,pressure_sweeps=20)
    mhs=reconstruct_force_balance_distributed!(distributed_mhs,HE3DBoundaryState(1e-10,1.0,:top),
        IdealGasEOS(),ReferenceOpacity500(kappa_m2_kg=0.02),context;options=mhs_options)
    @test mhs.mode==:MHS && mhs.lorentz_max_n_m3>0
    @test mhs.best_force_residual<=mhs.initial_force_residual
end
