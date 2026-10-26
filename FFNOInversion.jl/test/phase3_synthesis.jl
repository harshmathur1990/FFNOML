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

@testset "Formal solver sub-steps a jump in opacity and emissivity" begin
    T=Float64
    solve(c,e,z)=begin out=zeros(T,size(c,2)); formal_solve!(out,ScalarFormalSolver(),c,e,z); out end
    # every step refined n times, χ and η log-linear: the reference for the sub-stepped coarse column
    refine(v,n)=vcat([exp.((1 .- (0:n-1)./n).*log(v[k]) .+ (0:n-1)./n.*log(v[k+1])) for k in 1:length(v)-1]...,[v[end]])
    function old_solve(c,e,z)       # the scheme without sub-steps
        nz=length(z); source=e[nz]/c[nz]; intensity=source
        for k in nz-1:-1:1
            w1,w2,w3=FFNOInversion._formal_weights(abs(z[k+1]-z[k])*(c[k]+c[k+1])/2)
            next_source=e[k]/c[k]; intensity=w1*intensity+w2*source+w3*next_source; source=next_source
        end
        intensity
    end

    # hot, transparent top (S = 1e4) over cool, opaque gas (S = 1): one optically thick step carries the jump
    z=collect(range(10,0,length=21)); hot=z.>5.2
    chi=ifelse.(hot,1e-9,1e3 .* exp.(-z)); source=ifelse.(hot,1e4,1.0); eta=chi.*source
    coarse=solve(reshape(chi,:,1),reshape(eta,:,1),z)[1]
    zfine=vcat([collect(range(z[k],z[k+1],length=201))[1:200] for k in 1:length(z)-1]...,[z[end]])
    fine=solve(reshape(refine(chi,200),:,1),reshape(refine(eta,200),:,1),zfine)[1]
    @test old_solve(chi,eta,z)>100
    @test coarse<2
    @test coarse≈fine rtol=0.05

    # smooth columns (no jump above e per step) are unchanged
    chi=1e2 .* exp.(-0.9 .* z); eta=chi.*(1 .+ 0.1 .* (10 .- z))
    @test solve(reshape(chi,:,1),reshape(eta,:,1),z)[1]==old_solve(chi,eta,z)
end

@testset "Log-tau formal integration (native log10 tau500 grid with chi500)" begin
    T=Float64; nx,ny=2,1
    column(v,nz)=repeat(reshape(T.(v),nz,1,1),1,nx,ny)
    function atmosphere(logtau,chi500,z;tau=true)
        nz=length(logtau); shape=(nz,nx,ny)
        Atmosphere3D(Grid3D(T.(logtau),T[0,1],T[0]),fill(6000.0,shape),zeros(shape),zeros(shape),zeros(shape),
            zeros(shape);z=column(z,nz),logtau500=column(logtau,nz),chi500=tau ? column(chi500,nz) : nothing)
    end
    solve(atmos,chi,eta)=begin
        cube=SpectralCube(zeros(T,1,1,nx,ny),[500e-9],StokesSet(:I))
        synthesize!(cube,MixedIntensitySynthesizer([OpacityContributor(TabulatedOpacityModel(chi,eta))]),NonPRD(),atmos)
        cube.data[1,1,1,1]
    end

    # χ = χ500 and S = a + bτ500: the emergent intensity is a + b (Eddington-Barbier is exact for linear S).
    logtau=collect(range(-6,2,length=81)); chi500=1e-5 .* exp10.(0.3 .* logtau)
    tau=exp10.(logtau); source=1 .+ 3 .* tau
    z=-cumsum(vcat(0,diff(tau)./((chi500[1:end-1].+chi500[2:end])./2)))
    atmos=atmosphere(logtau,chi500,z)
    @test FFNOInversion._tau_integration(atmos)
    @test solve(atmos,reshape(chi500,:,1),reshape(chi500.*source,:,1))≈4 rtol=1e-2

    # Transition region in one grid step: hot (S large, χ500 tiny) above cool gas, 9600 km apart in HSE height.
    # The log-tau integration uses the step's true Δτ500 (~1e-6) and returns the photospheric value. Without
    # sub-steps the z trapezoid charged the cool opacity over that height and returned the coronal source
    # function (> 100); sub-stepping the jump in χ and η now keeps the z column photospheric too.
    logtau=collect(range(-6.0,1.0,length=71)); tau=exp10.(logtau)
    hot=logtau.<=-5.85
    chi500=ifelse.(hot,1e-13,1e-4 .* tau.^0.3); source=ifelse.(hot,1e3,1 .+ tau)   # dense cool gas below
    step=diff(tau)./((chi500[1:end-1].+chi500[2:end])./2)
    step[findlast(hot)]=9.6e6                       # the HSE height of the transition step
    z=-cumsum(vcat(0,step))
    chi=reshape(chi500,:,1); eta=reshape(chi500.*source,:,1)
    with_tau=solve(atmosphere(logtau,chi500,z),chi,eta)
    with_z=solve(atmosphere(logtau,chi500,z;tau=false),chi,eta)
    @test with_tau<2.5
    @test with_z<2.5

    # VJP: χ500 and population cotangents against central differences.
    logtau=collect(range(-5,1,length=13)); nz=length(logtau); tau=exp10.(logtau)
    chi500=1e-6 .* tau.^0.8; z=-cumsum(vcat(0,diff(tau)./((chi500[1:end-1].+chi500[2:end])./2)))
    atmos=atmosphere(logtau,chi500,z); atmos.chi500[:,2,1].*=1.3
    wave=collect(range(656.20e-9,656.36e-9,length=4)); nλ=length(wave)
    pops=fill(1e12,nz,nx,ny,3); pops[:,:,:,3].=1e9
    halpha=FFNOTransition(:H,656.28e-9,2,3,0.64,1.6735575e-27,1e8)
    continuum=TabulatedOpacityModel(repeat(reshape(chi500,:,1),1,nλ),repeat(reshape(2 .* chi500 .* (1 .+ tau),:,1),1,nλ))
    synth=MixedIntensitySynthesizer([OpacityContributor(continuum),OpacityContributor(halpha,pops)])
    bar=reshape(cos.(1:nλ*nx*ny),nλ,1,nx,ny)
    objective()=begin
        cube=SpectralCube(zeros(T,nλ,1,nx,ny),wave,StokesSet(:I)); synthesize!(cube,synth,NonPRD(),atmos)
        sum(bar.*cube.data)
    end
    cube=SpectralCube(zeros(T,nλ,1,nx,ny),wave,StokesSet(:I)); synthesize!(cube,synth,NonPRD(),atmos)
    atmosphere_bar=FFNOInversion.AtmosphereCotangent(atmos)
    population_bar=FFNOInversion.synthesis_vjp!(atmosphere_bar,synth,NonPRD(),atmos,pops,cube,bar)
    @test all(iszero,atmosphere_bar.z)
    for (k,x) in ((3,1),(7,2),(11,1))
        h=1e-5*atmos.chi500[k,x,1]; c=atmos.chi500[k,x,1]
        atmos.chi500[k,x,1]=c+h; fp=objective(); atmos.chi500[k,x,1]=c-h; fm=objective(); atmos.chi500[k,x,1]=c
        @test atmosphere_bar.chi500[k,x,1]≈(fp-fm)/(2h) rtol=1e-4
        h=1e-5*pops[k,x,1,2]; c=pops[k,x,1,2]
        pops[k,x,1,2]=c+h; fp=objective(); pops[k,x,1,2]=c-h; fm=objective(); pops[k,x,1,2]=c
        @test population_bar[k,x,1,2]≈(fp-fm)/(2h) rtol=1e-4
    end
end
