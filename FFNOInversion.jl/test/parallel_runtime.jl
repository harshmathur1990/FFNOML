@testset "hybrid parallel runtime" begin
    serial = serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
    @test !serial.enabled
    @test isroot(serial)
    @test allreduce_sum(3.5,serial) == 3.5
    @test mpi_broadcast((2,3),serial) == (2,3)

    @test process_grid(16,200,200) == (4,4)
    tiles = [tile_for_rank(200,200,r,16) for r in 0:15]
    @test sum(length(t.xrange)*length(t.yrange) for t in tiles) == 200*200
    @test length(unique((x,y) for t in tiles for x in t.xrange for y in t.yrange)) == 200*200

    atmosphere = test_atmosphere()
    tile = tile_for_rank(2,3,1,2;grid=(2,1))
    local_atmos = local_atmosphere(atmosphere,tile)
    @test size(local_atmos.temperature) == (4,1,3)
    @test local_atmos.grid.x == atmosphere.grid.x[tile.xrange]
    field = DistributedField(local_atmos.temperature,size(atmosphere.temperature),tile)
    @test field.global_shape == (4,2,3)
    distributed = distribute_field(Float64,atmosphere.temperature,size(atmosphere.temperature),serial)
    @test gather_field(distributed,serial) == atmosphere.temperature
    padded = exchange_halos(distributed,serial,1)
    @test size(padded) == (4,4,5)
    @test padded[:,1,2:4] == atmosphere.temperature[:,1,:]

    launches = Ref(0)
    coordinator = RootGPUCoordinator(x -> (launches[] += 1; 2x))
    @test launch_gpu!(coordinator,serial,4) == 8
    @test launches[] == 1

    mktempdir() do directory
        diagnostics=initialize_gpu_control_diagnostics(serial;directory=directory,interval=0.01)
        set_diagnostic_context!(diagnostics;phase="unit_test")
        diagnostic_checkpoint!(diagnostics,"unit_test_checkpoint";value=7)
        sleep(0.04)
        stop_diagnostics!(diagnostics)
        events=read(diagnostics.event_path,String)
        @test occursin("unit_test_checkpoint",events)
        @test occursin("gpu_diagnostics_stopped",events)
        @test length(readlines(diagnostics.resource_path))>=4
    end

    cache = ThreadedSynthesisCache(Float64,4,7)
    @test length(cache.workspaces) == Threads.maxthreadid()
    @test length(unique(objectid(ws.extinction) for ws in cache.workspaces)) == Threads.maxthreadid()
end
