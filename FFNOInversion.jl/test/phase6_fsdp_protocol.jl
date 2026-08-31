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
    close_fsdp_service!(service)
    wait(server)
end
