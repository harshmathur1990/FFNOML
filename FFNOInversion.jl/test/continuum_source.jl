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
        ffno=SpectralSourceConfig(:ffno,nothing); kurucz=SpectralSourceConfig(:kurucz_lte,fe)
        atom=joinpath(get(ENV,"FFNO_ATOM_DIR","/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms"),"atom.h6_tiago2.yaml")
        function spectrum(sources,H)
            region=SpectralRegionConfig(656.38e-9,0.01e-9,63,1.0,:none,nothing,sources)
            pops=Dict(:H=>H)
            setup=build_synthesis_setup((regions=[region],),atmosphere,pops,eos;atom_files=Dict(:H=>atom))
            wave=setup[1].wavelength_air_m
            cube=SpectralCube(zeros(length(wave),1,nx,ny),wave,StokesSet(:I))
            synthesize!(cube,setup,NonPRD(),atmosphere,pops,ThreadedSynthesisCache(Float64,nz,length(wave)))
            cube.data[:,1,1,1],setup,pops,cube
        end
        base,setup,pops,cube=spectrum([ffno],hydrogen(1e23,1e20))
        # exactly one source of continuum: Muspel's with EOS hydrogen, in every region; FFNO neutral H and protons do not change it
        continua=[c.model for c in setup[1].synthesizer.contributors if c.model isa MuspelContinuumModel]
        @test length(continua)==1 && only(continua).eos!==nothing
        @test !any(c.model isa FFNOInversion.KuruczLTEModel && c.model.include_continuum for c in setup[1].synthesizer.contributors)
        @test !any(c.model isa MuspelLineOpacityModel && c.model.include_continuum for c in setup[1].synthesizer.contributors)
        @test all(>(0),base)
        @test spectrum([ffno],hydrogen(1e24,1e22))[1]≈base rtol=1e-5
        # the order of the sources does not matter
        @test spectrum([ffno,kurucz],hydrogen(1e23,1e20))[1]≈spectrum([kurucz,ffno],hydrogen(1e23,1e20))[1] rtol=1e-12
        # the continuum has the same value with or without an FFNO source, away from the lines
        @test spectrum([kurucz],hydrogen(1e23,1e20))[1][1]≈base[1] rtol=1e-3
        # a region with no FFNO source has the same continuum; "ffno" is refused
        lte=spectrum([kurucz],hydrogen(1e23,1e20))[2]
        @test only(c.model for c in lte[1].synthesizer.contributors if c.model isa MuspelContinuumModel).eos!==nothing
        @test FFNOInversion._continuum_populations(Dict{String,Any}())===:eos
        @test FFNOInversion._continuum_populations(Dict{String,Any}("continuum_populations"=>"eos"))===:eos
        @test_throws ArgumentError FFNOInversion._continuum_populations(Dict{String,Any}("continuum_populations"=>"ffno"))
        # nothing but the line's own two levels comes from the FFNO: neutral H in broadening does not either
        core(H)=begin
            region=SpectralRegionConfig(656.20e-9,0.01e-9,21,1.0,:none,nothing,[ffno])
            pops=Dict(:H=>H); built=build_synthesis_setup((regions=[region],),atmosphere,pops,eos;atom_files=Dict(:H=>atom))
            wave=built[1].wavelength_air_m; c=SpectralCube(zeros(length(wave),1,nx,ny),wave,StokesSet(:I))
            synthesize!(c,built,NonPRD(),atmosphere,pops,ThreadedSynthesisCache(Float64,nz,length(wave))); c.data[:,1,1,1]
        end
        levels(neutral)=(p=fill(1.0,nz,nx,ny,6); p[:,:,:,1].=neutral; p[:,:,:,2].=1e15; p[:,:,:,3].=1e11; p[:,:,:,6].=1e20; p)
        @test core(levels(1e22))==core(levels(1e24))
        # [physics] line_center: the reference wavelength by default, or the atom's own; the profile peaks there
        lc(mode)=only(c.model for c in build_synthesis_setup((regions=[SpectralRegionConfig(656.20e-9,0.01e-9,21,1.0,:none,nothing,[ffno])],line_center=mode),
            atmosphere,Dict(:H=>levels(1e23)),eos;atom_files=Dict(:H=>atom))[1].synthesizer.contributors if c.model isa MuspelLineOpacityModel)
        @test lc(:reference).center_nm≈FFNOInversion._air_to_vacuum_nm(656.28)
        @test lc(:atom).center_nm==Float64(lc(:atom).line.λ0)
        @test lc(:reference).center_nm!=lc(:atom).center_nm
        for mode in (:reference,:atom)
            m=lc(mode); chi=zeros(nz,3); eta=zeros(nz,3)
            FFNOInversion.add_opacity_emissivity!(chi,eta,m,(m.center_nm.+[-0.01,0.0,0.01]).*1e-9,atmosphere,1,1,levels(1e23))
            @test all(chi[:,2].>chi[:,1])
            @test chi[:,1]≈chi[:,3] rtol=1e-2
        end
        @test FFNOInversion._line_center(Dict{String,Any}())===:reference
        @test FFNOInversion._line_center(Dict{String,Any}("line_center"=>"atom"))===:atom
        @test_throws ArgumentError FFNOInversion._line_center(Dict{String,Any}("line_center"=>"air"))
        # the line adjoint carries the EOS neutral H: against centred differences in T and P
        lp=levels(1e23); line_model=build_muspel_line_model(atmosphere,lp,atom,5,2,3;include_continuum=false,neutral_hydrogen=FFNOInversion.WittmannKuruczState(eos))
        lw=collect(range(656.40e-9,656.52e-9;length=7)); cb=Float32.(reshape(sin.(1:nz*7),nz,7)); eb=Float32.(reshape(cos.(1:nz*7),nz,7))
        function linepair(a)
            ch=zeros(Float32,nz,7); et=zeros(Float32,nz,7); add_opacity_emissivity!(ch,et,line_model,lw,a,1,1,lp)
            sum(Float64.(cb).*Float64.(ch))+sum(Float64.(eb).*Float64.(et))
        end
        lb=FFNOInversion.AtmosphereCotangent(atmosphere); FFNOInversion.add_opacity_emissivity_vjp!(lb,zeros(size(lp)),_->zeros(size(lp)),cb,eb,line_model,lw,atmosphere,1,1,lp)
        for (name,bar,h) in ((:temperature,lb.temperature,5.0),(:pgas,lb.pgas,nothing))
            k=3; f=getfield(atmosphere,name); v=f[k,1,1]; δ=h===nothing ? 2e-2*v : h
            f[k,1,1]=v+δ; up=linepair(atmosphere); f[k,1,1]=v-δ; dn=linepair(atmosphere); f[k,1,1]=v
            @test bar[k,1,1]≈(up-dn)/(2δ) rtol=0.05
        end
        # the adjoint sees the continuum: temperature and pressure cotangents are finite and not all zero
        bar=FFNOInversion.AtmosphereCotangent(atmosphere); weights=ones(size(cube.data))
        FFNOInversion.synthesis_vjp!(bar,setup,NonPRD(),atmosphere,pops,cube,weights)
        @test all(isfinite,bar.temperature) && any(!iszero,bar.temperature) && any(!iszero,bar.pgas)
        println("CONTINUUM_VJP_SECONDS_PER_COLUMN_5_LEVELS_21_WAVELENGTHS=",@elapsed FFNOInversion.synthesis_vjp!(bar,setup,NonPRD(),atmosphere,pops,cube,weights))
    end
end
