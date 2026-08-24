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
        lib=joinpath(@__DIR__,"..","deps","libwitt_ffno.$(Libdl.dlext)")
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
