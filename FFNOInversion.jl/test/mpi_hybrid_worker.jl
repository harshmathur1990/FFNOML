using FFNOInversion

context=initialize_parallel(options=ParallelOptions(enabled=true,threads_per_rank=Threads.nthreads()))
try
    shape=(3,8,6)
    global_values=isroot(context) ? reshape(collect(Float64,1:prod(shape)),shape) : nothing
    field=distribute_field(Float64,global_values,shape,context)
    halo=exchange_halos(field,context,1)
    for j in axes(halo,3),i in axes(halo,2),k in axes(halo,1)
        gx=clamp(first(field.tile.xrange)+i-2,1,shape[2])
        gy=clamp(first(field.tile.yrange)+j-2,1,shape[3])
        expected_value=k+(gx-1)*shape[1]+(gy-1)*shape[1]*shape[2]
        halo[k,i,j]==expected_value || error("halo exchange mismatch on rank $(context.rank)")
    end

    # Exercise Julia threads only over rank-local memory. MPI remains on main.
    checks=zeros(Float64,Threads.maxthreadid())
    Threads.@threads :static for column in 1:size(field.values,2)*size(field.values,3)
        x=(column-1)%size(field.values,2)+1
        y=(column-1)÷size(field.values,2)+1
        checks[Threads.threadid()]+=sum(@view field.values[:,x,y])
    end
    local_sum=sum(checks)
    global_sum=allreduce_sum(local_sum,context)
    expected=sum(1.0:prod(shape))
    isapprox(global_sum,expected;rtol=0,atol=0) || error("distributed threaded reduction differs")

    gathered=gather_field(field,context)
    isroot(context) && gathered != global_values && error("typed scatter/gather parity failed")

    # Execute the package forward model on rank-owned tiles and reassemble the
    # spectral cube. This checks that scientific code is independent of the
    # chosen MPI decomposition.
    tile=field.tile
    logtau=[-5.0,-3.0,-1.0]
    grid=Grid3D(logtau,collect(Float64,tile.xrange),collect(Float64,tile.yrange))
    zeros_local=zeros(size(field.values))
    local_atmosphere=Atmosphere3D(grid,field.values,copy(zeros_local),copy(zeros_local),
        copy(zeros_local),copy(zeros_local))
    wave=collect(range(656.1e-9,656.5e-9,length=7))
    workspace=ForwardWorkspace(Float64,local_atmosphere,wave,StokesSet(:I))
    model=MockForwardModel(MockPopulationModel(1e10),NonPRD(),MockIntensitySynthesizer(),
        IdentityObservation(),CapabilityManifest())
    local_spectrum=forward!(workspace,model,local_atmosphere)
    spectrum_field=DistributedField(local_spectrum.data,(length(wave),1,shape[2],shape[3]),tile)
    global_spectrum=gather_field(spectrum_field,context;tag=112)
    if isroot(context)
        global_grid=Grid3D(logtau,collect(1.0:shape[2]),collect(1.0:shape[3]))
        zero_global=zeros(shape)
        global_atmosphere=Atmosphere3D(global_grid,global_values,copy(zero_global),copy(zero_global),
            copy(zero_global),copy(zero_global))
        reference_ws=ForwardWorkspace(Float64,global_atmosphere,wave,StokesSet(:I))
        reference=forward!(reference_ws,model,global_atmosphere)
        global_spectrum==reference.data || error("distributed forward parity failed")
    end

    launched=Ref(0)
    coordinator=RootGPUCoordinator(() -> (launched[]+=1; :ok))
    result=launch_gpu!(coordinator,context)
    isroot(context) && result != :ok && error("root GPU coordinator result differs")
    allreduce_sum(launched[],context)==1 || error("GPU launcher ran on more than one rank")
    isroot(context) && println("MPI_HYBRID_OK ranks=$(context.size) threads=$(Threads.nthreads())")
finally
    finalize_parallel!(context)
end
