using Libdl

@testset "Phase 3 mixed synthesis" begin
    mktemp() do path,io
        write(io,"# wavelength[A], species, loggf, Elow[eV], Eup[eV], damping\n")
        write(io,"6562.90, FeI, -1.2, 2.0, 3.9, 1.0e8\n")
        write(io,"8542.10 CaI -0.5 1.0 2.5\n")
        close(io)
        cache=load_kurucz_linelist(path)
        @test length(cache.lines)==2
        @test cache.lines[1].species=="FeI"
        @test length(select_kurucz_lines(cache,[656.25e-9,656.35e-9]))==1
    end

    T=Float64; nz,nx,ny,nλ=3,2,2,5
    grid=Grid3D(T[-4,-2,0],T[0,1],T[0,1])
    shape=(nz,nx,ny)
    atmos=Atmosphere3D(grid,fill(6000.0,shape),zeros(shape),zeros(shape),zeros(shape),fill(1e3,shape);
        pgas=fill(100.0,shape),rho=fill(1e-7,shape),ne=fill(1e17,shape),z=repeat(reshape(T[2e5,1e5,0],nz,1,1),1,nx,ny))
    wave=collect(range(656.20e-9,656.36e-9,length=nλ))
    cube=SpectralCube(zeros(T,nλ,1,nx,ny),wave,StokesSet(:I))
    continuum=TabulatedOpacityModel(fill(1e-5,nz,nλ),fill(2e-5,nz,nλ))
    pops=fill(1e12,nz,nx,ny,3); pops[:,:,:,3].=1e8
    halpha=FFNOTransition(:H,656.28e-9,2,3,0.64,1.6735575e-27,1e8)
    mixed=MixedIntensitySynthesizer([OpacityContributor(continuum),OpacityContributor(halpha,pops)])
    synthesize!(cube,mixed,NonPRD(),atmos)
    @test all(isfinite,cube.data)
    @test all(>(0),cube.data)
    @test_throws ArgumentError synthesize!(cube,mixed,MockPRD(0.5),atmos)

    # Contributions are additive before a single formal solution.
    only_cont=SpectralCube(zeros(T,nλ,1,nx,ny),wave,StokesSet(:I))
    synthesize!(only_cont,MixedIntensitySynthesizer([OpacityContributor(continuum)]),NonPRD(),atmos)
    @test cube.data != only_cont.data

    k94=normpath(joinpath(@__DIR__,"..","..","..","stic","input","Atoms","kurucz_6301_6302.input"))
    if isfile(k94)
        cache=load_kurucz_linelist(k94)
        @test length(cache.lines)==2
        @test all(line->line.atomic_number==26,cache.lines)
        @test all(line->line.ion_stage==0,cache.lines)
        @test cache.lines[1].wavelength0_m > 630.1e-9 # K94 air wavelengths become vacuum
        backend_dir=mktempdir()
        lib=joinpath(backend_dir,"libwitt_ffno.$(Libdl.dlext)")
        build=joinpath(@__DIR__,"..","scripts","build_wittmann_backend.jl")
        run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__,"..")) $build $lib`)
        pf=normpath(joinpath(@__DIR__,"..","..","scripts","pf_Kurucz.input"))
        if isfile(lib) && isfile(pf)
            model=KuruczLTEModel(cache.lines,WittmannEOS(lib,pf))
            kwave=collect(range(cache.lines[1].wavelength0_m-0.03e-9,cache.lines[2].wavelength0_m+0.03e-9,length=31))
            kcube=SpectralCube(zeros(T,length(kwave),1,nx,ny),kwave,StokesSet(:I))
            synthesize!(kcube,MixedIntensitySynthesizer([OpacityContributor(model)]),NonPRD(),atmos)
            @test all(isfinite,kcube.data)
            @test all(>(0),kcube.data)
            @test minimum(kcube.data[:,1,1,1]) < maximum(kcube.data[:,1,1,1])
            source=SpectralSourceConfig(:kurucz_lte,nothing,nothing,k94)
            region=SpectralRegionConfig(630.12e-9,0.01e-9,20,1.0,:none,nothing,[source])
            setup=build_synthesis_setup((regions=[region],),atmos,Dict{Symbol,Any}(),WittmannEOS(lib,pf);atom_files=Dict())
            @test length(setup)==1 # LTE-only setup neither requires nor invokes FFNO
        end
    end
end

@testset "Region synthesis (build_synthesis_setup output)" begin
    # Two regions; the cube axis is their air grids concatenated, each region is computed on its vacuum grid.
    T=Float64; nz,nx,ny=3,2,2
    grid=Grid3D(T[-4,-2,0],T[0,1],T[0,1]); shape=(nz,nx,ny)
    atmos=Atmosphere3D(grid,fill(6000.0,shape),zeros(shape),zeros(shape),fill(500.0,shape),fill(1e3,shape);
        pgas=fill(100.0,shape),rho=fill(1e-7,shape),ne=fill(1e17,shape),z=repeat(reshape(T[2e5,1e5,0],nz,1,1),1,nx,ny))
    vacuum(air)=T.(FFNOInversion._air_to_vacuum_nm.(air.*1e9).*1e-9)
    air1=collect(range(656.20e-9,656.36e-9,length=5)); air2=collect(range(854.15e-9,854.27e-9,length=4))
    populations=Dict(:H=>fill(1e12,nz,nx,ny,3),:CA=>fill(1e10,nz,nx,ny,5))
    populations[:H][:,:,:,3].=1e8; populations[:CA][:,:,:,5].=1e6
    halpha=FFNOTransition(:H,656.461e-9,2,3,0.64,1.6735575e-27,1e8)
    ca=FFNOTransition(:CA,854.444e-9,3,5,0.44,40*1.66053906660e-27,1e8)
    synth1=MixedIntensitySynthesizer([OpacityContributor(TabulatedOpacityModel(fill(1e-5,nz,5),fill(2e-5,nz,5))),
                                      OpacityContributor(halpha,populations[:H])])
    synth2=MixedIntensitySynthesizer([OpacityContributor(TabulatedOpacityModel(fill(2e-5,nz,4),fill(3e-5,nz,4))),
                                      OpacityContributor(ca,populations[:CA])])
    setups=[RegionSynthesisSetup(air1,vacuum(air1),synth1),RegionSynthesisSetup(air2,vacuum(air2),synth2)]
    wave=vcat(air1,air2); nλ=length(wave)
    cube=SpectralCube(zeros(T,nλ,1,nx,ny),wave,StokesSet(:I))
    synthesize!(cube,setups,NonPRD(),atmos,populations,ThreadedSynthesisCache(T,nz,nλ))
    @test all(isfinite,cube.data) && all(>(0),cube.data)
    parts=map(((setup,synth),)->synthesize!(SpectralCube(zeros(T,length(setup),1,nx,ny),setup,StokesSet(:I)),
        synth,NonPRD(),atmos,populations),((vacuum(air1),synth1),(vacuum(air2),synth2)))
    @test cube.data≈vcat(parts[1].data,parts[2].data) rtol=1e-12
    # the synthesis grid must be exactly the regions' grids
    short=SpectralCube(zeros(T,nλ-1,1,nx,ny),wave[1:end-1],StokesSet(:I))
    @test_throws DimensionMismatch synthesize!(short,setups,NonPRD(),atmos,populations)

    # gradient: equals the per-region gradients on their vacuum grids
    bar=reshape(sin.(1:nλ*nx*ny),nλ,1,nx,ny)
    region_bar=FFNOInversion.AtmosphereCotangent(atmos)
    pop_bar=FFNOInversion.synthesis_vjp!(region_bar,setups,NonPRD(),atmos,populations,cube,bar)
    expected_bar=FFNOInversion.AtmosphereCotangent(atmos)
    expected_pops=Dict(k=>zeros(T,size(v)) for (k,v) in populations)
    for (i,(range,synth,air)) in enumerate(((1:5,synth1,air1),(6:nλ,synth2,air2)))
        sub=SpectralCube(zeros(T,length(range),1,nx,ny),vacuum(air),StokesSet(:I))
        b=FFNOInversion.synthesis_vjp!(expected_bar,synth,NonPRD(),atmos,populations,sub,bar[range,:,:,:])
        for (k,v) in b; expected_pops[k].+=v; end
    end
    @test region_bar.z≈expected_bar.z rtol=1e-12
    @test region_bar.temperature≈expected_bar.temperature rtol=1e-12
    @test all(pop_bar[k]≈expected_pops[k] for k in keys(populations))
    @test any(!iszero,pop_bar[:CA]) && any(!iszero,pop_bar[:H])
end
