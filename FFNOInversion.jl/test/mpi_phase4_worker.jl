using FFNOInversion
using Serialization

struct CoupledPopulationFixture <: AbstractPopulationModel
    scale::Float64
end
function FFNOInversion.predict_populations!(out,model::CoupledPopulationFixture,atmosphere::Atmosphere3D)
    global_mean=sum(atmosphere.temperature)/length(atmosphere.temperature)
    @views out[:,:,:,1].=model.scale.*atmosphere.temperature./global_mean
    out
end

context=initialize_parallel(options=ParallelOptions(enabled=true,threads_per_rank=Threads.nthreads()))
try
    nx,ny=11,7; grid=Grid3D([-5.0,-4,-3,-2],collect(0.0:50e3:(nx-1)*50e3),collect(0.0:50e3:(ny-1)*50e3))
    shape=(4,nx,ny); root_atmosphere=nothing
    if isroot(context)
        zero3=zeros(shape); temperature=Array{Float64}(undef,shape)
        for k in 1:shape[1],i in 1:nx,j in 1:ny
            temperature[k,i,j]=5400+20k+3i+2j
        end
        bx=fill(1e-5,shape); by=zeros(shape); bz=fill(2e-5,shape)
        for k in axes(by,1); @views by[k,:,:].=(k-1)*1e-8; end
        root_atmosphere=Atmosphere3D(grid,temperature,copy(zero3),copy(zero3),copy(zero3),copy(zero3);
            magnetic_field=MagneticField3D(bx,by,bz))
    end
    distributed=distribute_atmosphere(Float64,root_atmosphere,context)
    root_atmosphere=nothing
    wave=collect(range(656.1e-9,656.5e-9,length=9))
    options=ForceBalanceOptions(max_iterations=120,relative_tolerance=1e-5,force_tolerance=0.6,
        height_tolerance_m=1.0,relaxation=0.5,pressure_sweeps=20)
    psf=GaussianPSFObservation(0.04e-9,50e3,50e3,50e3,50e3)
    # This fixture depends on the global temperature mean, so topology parity
    # proves that rank 0 received the complete FFNO feature volume before the
    # resulting population cube was scattered back to tile owners.
    root_model=isroot(context) ? CoupledPopulationFixture(1e10) : nothing
    model=HybridForwardModel(RootDistributedPopulationModel(root_model,1),NonPRD(),MockIntensitySynthesizer(),
        psf,IdealGasEOS(),ReferenceOpacity500(kappa_m2_kg=0.02),HE3DBoundaryState(1e-10,1.0,:top),
        options,CapabilityManifest())
    workspace=HybridForwardWorkspace(Float64,distributed,wave,StokesSet(:I),1)
    forward_start=time(); result=forward!(workspace,model,distributed,context); forward_seconds=time()-forward_start
    result.force_balance.mode==:MHS || error("MPI Phase 4 did not select MHS for supplied B")
    result.force_balance.lorentz_max_n_m3>0 || error("distributed MHS Lorentz term is zero")
    local_pixels=length(distributed.tile.xrange)*length(distributed.tile.yrange)
    allreduce_sum(local_pixels,context)==nx*ny || error("rank tiles do not uniquely own the global domain")
    context.size>1 && local_pixels>=nx*ny && error("a rank retained the complete spatial model")
    (size(workspace.populations,2),size(workspace.populations,3))==
        (length(distributed.tile.xrange),length(distributed.tile.yrange)) ||
        error("population workspace is not tile-local")
    memory=distributed_memory_report(distributed,workspace,context)
    provenance=parallel_provenance(context,distributed.tile;configuration_hash="phase4-fixture-config",
        model_hash="coupled-population-fixture",source_revision="local-worktree",
        capabilities=["intensity","crd","kurucz-lte","ffno-nlte"])
    spectrum=gather_spectrum(result.spectrum,distributed,context)
    atmosphere=gather_atmosphere(distributed,context)
    packed_populations=permutedims(workspace.populations,(1,4,2,3))
    global_packed=gather_field(DistributedField(packed_populations,
        (shape[1],1,nx,ny),distributed.tile),context;tag=380)
    populations=isroot(context) ? permutedims(global_packed,(1,3,4,2)) : nothing
    if !isroot(context)
        spectrum===nothing || error("non-root rank received a complete spectrum")
        atmosphere===nothing || error("non-root rank received a complete atmosphere")
        populations===nothing || error("non-root rank received complete populations")
    end

    root_observed=isroot(context) ? ObservationCube(SpectralCube(copy(spectrum.data),wave,StokesSet(:I)),
        ones(size(spectrum.data)),ones(size(spectrum.data))) : nothing
    observed=distribute_observation(Float64,root_observed,(length(wave),1,nx,ny),wave,StokesSet(:I),context)
    distributed_chi2(result.spectrum,observed,context)==0 || error("distributed observation/chi2 parity failed")
    regspec=RegularizationSpec(vertical=VerticalRegularizationSpec((1,0,0,0,0,0,0),1.0,ntuple(_->1.0,7)),
        horizontal=Dict(:temperature=>1e-3),scales=Dict(:temperature=>1000.0),horizontal_order=1)
    reg=distributed_regularization_penalty(distributed,regspec,50e3,50e3,context)
    isfinite(reg.total) || error("distributed regularization is not finite")
    if isroot(context)
        all(isfinite,spectrum.data) || error("gathered spectrum is invalid")
        all(atmosphere.pgas.>0) || error("gathered pressure is invalid")
        output_path=get(ENV,"PHASE4_RESULT_PATH","")
        if !isempty(output_path)
            open(output_path,"w") do io
                serialize(io,(spectrum=spectrum.data,populations=populations,temperature=atmosphere.temperature,
                    pgas=atmosphere.pgas,rho=atmosphere.rho,ne=atmosphere.ne,z=atmosphere.z,
                    Bx=atmosphere.magnetic_field.Bx,By=atmosphere.magnetic_field.By,Bz=atmosphere.magnetic_field.Bz,
                    regularization=reg,force_balance=result.force_balance,ranks=context.size,
                    threads=Threads.nthreads(),process_grid=distributed.tile.process_grid,
                    forward_seconds=forward_seconds,timings=result.timings,memory=memory,
                    provenance=provenance))
            end
        end
        println("MPI_PHASE4_OK ranks=$(context.size) threads=$(Threads.nthreads()) checksum=$(sum(spectrum.data)) reg=$(reg.total) seconds=$forward_seconds")
    end
finally
    finalize_parallel!(context)
end
