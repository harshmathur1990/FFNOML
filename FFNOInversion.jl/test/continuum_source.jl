using Test
using FFNOInversion
using Libdl
import Muspel

# One continuum for every region: Wittmann's, with hydrogen from the EOS. It does not read FFNO populations and does not
# depend on the order of the region's sources.
@testset "Continuum opacity does not use FFNO hydrogen" begin
    mktempdir() do dir
        lib=joinpath(dir,"libwitt_continuum.$(Libdl.dlext)")
        run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__,"..")) $(joinpath(@__DIR__,"..","scripts","build_wittmann_backend.jl")) $lib`)
        eos=WittmannEOS(lib,joinpath(@__DIR__,"..","..","scripts","pf_Kurucz.input"))
        nz,nx,ny=5,1,1; shape=(nz,nx,ny); z=[8e5,5e5,2e5,5e4,0.0]
        field(v)=reshape(Float64.(v),shape)
        grid=Grid3D(VerticalAxis(:z_m,z),[0.0],[0.0])
        atmosphere=FFNOInversion.Atmosphere3D(grid,field([5000,4600,4700,5400,6400]),zeros(shape),zeros(shape),zeros(shape),zeros(shape);
            pgas=field([20,200,3e3,8e3,1.1e4]),rho=field([7e-7,7e-6,8e-5,2e-4,2.6e-4]),ne=field([1e17,3e17,2e18,1e19,1e20]),z=field(z))
        function hydrogen(neutral,proton)
            p=fill(1.0,nz,nx,ny,6); p[:,:,:,1].=neutral; p[:,:,:,2].=1.0; p[:,:,:,3].=1.0; p[:,:,:,6].=proton; p
        end
        fe=joinpath(dir,"fe.list")
        write(fe,"  656.9214 -0.420 26.00   53393.669  4.0 (4F)4d g5F   38175.352  3.0 (4F)4p y3D  8.36 -4.38 -7.70FMW  0 0  0 0.000  0 0.000    0    0           1287 1324\n")
        ffno=SpectralSourceConfig(:ffno,:H,:halpha,nothing); kurucz=SpectralSourceConfig(:kurucz_lte,nothing,nothing,fe)
        atom=joinpath(get(ENV,"FFNO_ATOM_DIR","/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms"),"atom.h6_tiago2.yaml")
        function spectrum(sources,H)
            region=SpectralRegionConfig(656.80e-9,0.01e-9,21,1.0,:none,nothing,sources)
            pops=Dict(:H=>H)
            setup=build_synthesis_setup((regions=[region],),atmosphere,pops,eos;atom_files=Dict(:H=>atom))
            wave=setup[1].wavelength_air_m
            cube=SpectralCube(zeros(length(wave),1,nx,ny),wave,StokesSet(:I))
            synthesize!(cube,setup,NonPRD(),atmosphere,pops,ThreadedSynthesisCache(Float64,nz,length(wave)))
            cube.data[:,1,1,1],setup,pops,cube
        end
        base,setup,pops,cube=spectrum([ffno],hydrogen(1e23,1e20))
        # exactly one source of continuum, and it is the EOS's: FFNO neutral H and protons do not change it
        continua=[c.model for c in setup[1].synthesizer.contributors if c.model.include_continuum]
        @test length(continua)==1 && only(continua) isa KuruczLTEModel
        @test isempty(only(continua).lines)
        @test all(>(0),base)
        @test spectrum([ffno],hydrogen(1e24,1e22))[1]≈base rtol=1e-5
        # the order of the sources does not matter
        @test spectrum([ffno,kurucz],hydrogen(1e23,1e20))[1]≈spectrum([kurucz,ffno],hydrogen(1e23,1e20))[1] rtol=1e-12
        # the continuum has the same value with or without an FFNO source, away from the lines
        @test spectrum([kurucz],hydrogen(1e23,1e20))[1][1]≈base[1] rtol=1e-3
        # [physics] continuum_populations = "ffno": the region's FFNO source supplies the continuum with FFNO hydrogen, whatever the
        # order of the sources; a region without an FFNO source keeps the EOS continuum
        function asked(sources,H,mode)
            region=SpectralRegionConfig(656.80e-9,0.01e-9,21,1.0,:none,nothing,sources)
            pops=Dict(:H=>H); config=(regions=[region],continuum_populations=mode)
            built=build_synthesis_setup(config,atmosphere,pops,eos;atom_files=Dict(:H=>atom))
            wave=built[1].wavelength_air_m
            c=SpectralCube(zeros(length(wave),1,nx,ny),wave,StokesSet(:I))
            synthesize!(c,built,NonPRD(),atmosphere,pops,ThreadedSynthesisCache(Float64,nz,length(wave)))
            c.data[:,1,1,1],built
        end
        withffno,built=asked([ffno],hydrogen(1e23,1e20),:ffno)
        owners=[c.model for c in built[1].synthesizer.contributors if c.model.include_continuum]
        @test length(owners)==1 && only(owners) isa MuspelLineOpacityModel
        @test !isapprox(asked([ffno],hydrogen(1e24,1e22),:ffno)[1],withffno;rtol=1e-3)     # FFNO hydrogen matters
        @test asked([ffno,kurucz],hydrogen(1e23,1e20),:ffno)[1]≈asked([kurucz,ffno],hydrogen(1e23,1e20),:ffno)[1] rtol=1e-12
        lte=asked([kurucz],hydrogen(1e23,1e20),:ffno)[2]
        @test only(c.model for c in lte[1].synthesizer.contributors if c.model.include_continuum) isa KuruczLTEModel
        @test asked([ffno],hydrogen(1e24,1e22),:eos)[1]≈base rtol=1e-5                     # eos is the default's result
        @test FFNOInversion._continuum_populations(Dict{String,Any}())===:eos
        @test FFNOInversion._continuum_populations(Dict{String,Any}("continuum_populations"=>"ffno"))===:ffno
        @test_throws ArgumentError FFNOInversion._continuum_populations(Dict{String,Any}("continuum_populations"=>"both"))
        # the adjoint sees the continuum: temperature and pressure cotangents are finite and not all zero
        bar=FFNOInversion.AtmosphereCotangent(atmosphere); weights=ones(size(cube.data))
        FFNOInversion.synthesis_vjp!(bar,setup,NonPRD(),atmosphere,pops,cube,weights)
        @test all(isfinite,bar.temperature) && any(!iszero,bar.temperature) && any(!iszero,bar.pgas)
        println("CONTINUUM_VJP_SECONDS_PER_COLUMN_5_LEVELS_21_WAVELENGTHS=",@elapsed FFNOInversion.synthesis_vjp!(bar,setup,NonPRD(),atmosphere,pops,cube,weights))
    end
end
