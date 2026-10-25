using Test
using Sockets

@testset "Phase 6 persistent FSDP service protocol" begin
    listener=listen(ip"127.0.0.1",0); port=Int(getsockname(listener)[2])
    server=@async begin
        peer=accept(listener)
        fields=split(readline(peer)); @test fields[1:2]==["PREDICT","H"]
        nz,nx,ny,levels=parse.(Int,fields[3:6])
        features=FFNOInversion._read_float32_array(peer,(6,nz,nx,ny))
        z=FFNOInversion._read_float32_array(peer,(nz,nx,ny))
        @test all(isfinite,features) && all(isfinite,z)
        println(peer,"OK PREDICT $nz $nx $ny $levels")
        FFNOInversion._write_float32_array(peer,fill(2.5f0,nz,nx,ny,levels)); flush(peer)

        fields=split(readline(peer)); @test fields[1:2]==["VJP","H"]
        nz,nx,ny,levels=parse.(Int,fields[3:6])
        FFNOInversion._read_float32_array(peer,(6,nz,nx,ny))
        FFNOInversion._read_float32_array(peer,(nz,nx,ny))
        population_bar=FFNOInversion._read_float32_array(peer,(nz,nx,ny,levels))
        @test all(population_bar.==3)
        println(peer,"OK VJP $nz $nx $ny $levels")
        FFNOInversion._write_float32_array(peer,fill(4f0,6,nz,nx,ny))
        FFNOInversion._write_float32_array(peer,fill(5f0,nz,nx,ny)); flush(peer)

        strip(readline(peer))=="SHUTDOWN" || error("missing service shutdown")
        println(peer,"BYE"); flush(peer); close(peer); close(listener)
    end

    socket=connect(ip"127.0.0.1",port)
    metadata=PopulationMetadata(FFNO_INPUT_CHANNELS,("H level 1",),"test-hash")
    undersized=FSDPServiceClient(socket,nothing,nothing,"127.0.0.1",port,"token",1,
        Dict(:H=>metadata),ReentrantLock(),false)
    @test_throws ArgumentError FSDPFFNOModel(undersized,:H,metadata,0)
    service=FSDPServiceClient(socket,nothing,nothing,"127.0.0.1",port,"token",2,
        Dict(:H=>metadata),ReentrantLock(),false)
    model=FSDPFFNOModel(service,:H,metadata,0)
    grid=Grid3D([-4.0,-2.0],[0.0,5.0e4],[0.0,5.0e4])
    shape=(2,2,2); zero3=zeros(shape)
    atmosphere=Atmosphere3D(grid,fill(5500.0,shape),copy(zero3),copy(zero3),copy(zero3),copy(zero3);
        rho=fill(1e-7,shape),ne=fill(1e17,shape),z=reshape(repeat([-2e5,0.0],4),shape))
    populations=zeros(Float64,shape...,1)
    predict_populations!(populations,model,atmosphere)
    @test all(populations.==2.5)
    feature_bar=zeros(Float64,6,shape...); z_bar=zeros(Float64,shape)
    population_vjp!(feature_bar,z_bar,model,atmosphere,fill(3.0,shape...,1))
    @test all(feature_bar.==4) && all(z_bar.==5)
    @test model.calls==2
    # Exercise the probe's composite cleanup path, including models sharing
    # one persistent service. The server expects exactly one SHUTDOWN.
    composite=CompositeDistributedPopulationModel(Dict(
        :H=>RootDistributedPopulationModel(model,1),
        :shared=>RootDistributedPopulationModel(model,1)))
    close_distributed_population_model!(composite,serial_context())
    @test service.closed && !isopen(socket)
    close_fsdp_service!(service) # closing an already closed service is harmless
    wait(server)
end

mutable struct DeadServiceTestEquilibrium <: FFNOInversion.AbstractEquilibriumBackend
    closed::Bool
end
FFNOInversion.close_equilibrium!(b::DeadServiceTestEquilibrium)=(b.closed=true; nothing)

@testset "Phase 6 FSDP service that dies mid-request" begin
    # The GPU service accepts a PREDICT, then dies (as job 2622023's workers did with
    # SIGSEGV). The prediction error must survive cleanup, and cleanup must still close the
    # socket, stop the launcher process, close the log and close the equilibrium.
    listener=listen(ip"127.0.0.1",0); port=Int(getsockname(listener)[2])
    server=@async begin
        peer=accept(listener)
        fields=split(readline(peer)); nz,nx,ny=parse.(Int,fields[3:5])
        FFNOInversion._read_float32_array(peer,(6,nz,nx,ny))
        FFNOInversion._read_float32_array(peer,(nz,nx,ny))
        close(peer); close(listener)
    end
    socket=connect(ip"127.0.0.1",port)
    log_path,log_stream=mktemp()
    process=run(`sleep 600`;wait=false)
    metadata=PopulationMetadata(FFNO_INPUT_CHANNELS,("H level 1",),"test-hash")
    service=FSDPServiceClient(socket,process,log_stream,"127.0.0.1",port,"token",2,
        Dict(:H=>metadata),ReentrantLock(),false)
    model=FSDPFFNOModel(service,:H,metadata,0)
    populations=CompositeDistributedPopulationModel(Dict(:H=>RootDistributedPopulationModel(model,1)))
    equilibrium=DeadServiceTestEquilibrium(false)
    grid=Grid3D([-4.0,-2.0],[0.0,5.0e4],[0.0,5.0e4])
    shape=(2,2,2); zero3=zeros(shape)
    atmosphere=Atmosphere3D(grid,fill(5500.0,shape),copy(zero3),copy(zero3),copy(zero3),copy(zero3);
        rho=fill(1e-7,shape),ne=fill(1e17,shape),z=reshape(repeat([-2e5,0.0],4),shape))
    out=zeros(Float64,shape...,1)

    primary=try
        try
            predict_populations!(out,model,atmosphere)
        catch
            withenv("FFNO_FSDP_SHUTDOWN_TIMEOUT"=>"1") do
                close_model_resources!(populations,equilibrium,serial_context())
            end
            rethrow()
        end
        nothing
    catch exception
        exception
    end
    wait(server)
    @test primary isa ErrorException
    @test occursin("truncated FSDP service response",primary.msg)
    @test service.closed && !isopen(socket)
    @test !process_running(process)
    @test !isopen(log_stream)
    @test equilibrium.closed
    @test isempty(close_fsdp_service!(service))           # idempotent
    rm(log_path;force=true)
end

@testset "Population model request order" begin
    models=Dict(:H=>1,:CA=>2)
    @test FFNOInversion._population_model_order(models)==[:CA,:H]
    withenv("FFNO_POPULATION_MODEL_ORDER"=>"H, CA") do
        @test FFNOInversion._population_model_order(models)==[:H,:CA]
    end
    withenv("FFNO_POPULATION_MODEL_ORDER"=>"H") do
        @test_throws ArgumentError FFNOInversion._population_model_order(models)
    end
end
