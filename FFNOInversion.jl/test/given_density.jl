using Test
using FFNOInversion
using HDF5
using Libdl
import Muspel

# A given atmosphere from nₑ and ρ: no HSEMHS, the input's ρ and nₑ reach the FFNO unchanged, hydrogen from the EOS at (T, P).
@testset "A given atmosphere: nₑ and ρ, no HSEMHS" begin
    F=FFNOInversion
    mktempdir() do dir
        nz,nx,ny=8,3,2; file=joinpath(dir,"a.h5"); zv=collect(range(1e6,0.0;length=nz))
        TT=[5500.0+40.0*k+7.0*i+3.0*j for k in 1:nz,j in 1:ny,i in 1:nx]                     # (z,y,x)
        ne=[1e17*exp(0.5*k)*(1+0.01i) for k in 1:nz,j in 1:ny,i in 1:nx]
        rho=[1e-7*exp(0.8*k)*(1+0.02j) for k in 1:nz,j in 1:ny,i in 1:nx]
        pressure=[5.0*exp(0.8*k)*(1+0.02j) for k in 1:nz,j in 1:ny,i in 1:nx]
        populations=[(l==6 ? 0.05 : l==1 ? 0.95 : 1e-6)*rho[k,j,i]/F._GIVEN_MASS_PER_H_KG for l in 1:6,k in 1:nz,j in 1:ny,i in 1:nx]   # (level,z,y,x)
        h5open(file,"w") do f
            f["zm"]=zv; f["lm"]=collect(range(-3.0,0.5;length=nz)); f["lt"]=collect(range(-5.0,0.5;length=nz))
            f["x"]=4.8e4 .* collect(0:nx-1); f["y"]=4.8e4 .* collect(0:ny-1)
            f["t"]=TT; f["ne"]=ne; f["rho"]=rho; f["p"]=pressure; f["hp"]=populations; f["h"]=ones(6,nz,ny,nx); for name in ("vx","vy","vz"); f[name]=zero(TT); end
        end
        function config(coordinate,axis;extra="rho = \"rho\"\n",mode="forward",nedata="ne = \"ne\"\n",equilibrium="")
            path=joinpath(dir,"c.toml")
            write(path,"""
            mode = "$mode"
            [inputs]
            initial_atmosphere_file = "$file"
            observation_file = "$file"
            time_index = 1
            [outputs]
            synthesis_file = "$(joinpath(dir,"s.h5"))"
            atmosphere_file = "$(joinpath(dir,"o.h5"))"
            [atmosphere]
            vertical_coordinate = "$coordinate"
            [atmosphere.datasets]
            $axis
            x_m = "x"
            y_m = "y"
            temperature = "t"
            vx = "vx"
            vy = "vy"
            vz = "vz"
            $nedata$extra
            $equilibrium
            [[regions]]
            start_angstrom = 6562.0
            step_angstrom = 0.1
            count = 5
            normalization = 1.0
            psf_type = "none"
            [[regions.sources]]
            mode = "ffno"
            [physics]
            redistribution = "non_prd"
            [observation]
            stokes = ["I"]
            [population.mesh]
            enabled = false
            [solver]
            max_iterations = $(mode=="forward" ? 0 : 1)
            [parallel]
            enabled = false
            threads_per_rank = 1
            """)
            load_config(path)
        end
        lib=joinpath(dir,"libwitt.$(Libdl.dlext)")
        run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__,"..")) $(joinpath(@__DIR__,"..","scripts","build_wittmann_backend.jl")) $lib`)
        eos=WittmannEOS(lib,joinpath(@__DIR__,"..","..","scripts","pf_Kurucz.input"))
        # One Pₑ solve per cell gives exactly what the separate EOS calls give, from a cool photosphere to a hot, thin corona.
        grid_t=[t for t in (1100.0,3250.0,5778.0,9000.0,2.0e4,1.0e5,1.0e6,4.8e7),_ in 1:3,_ in 1:2]
        grid_p=[p*exp(0.1x+0.2y) for p in (1e4,1e3,1e2,1.0,0.1,0.05,0.02,0.01),x in 1:3,y in 1:2]
        for (once,separately) in zip(F._given_hydrogen_state(eos,grid_t,grid_p),F._given_hydrogen_state_separately(eos,grid_t,grid_p))
            @test once==separately
        end
        context=serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
        for (coordinate,axis,mode) in (("z","z_m = \"zm\"",:given_z),("log10m","log10m = \"lm\"",:given_logm),("log10tau500","logtau500 = \"lt\"",:given_ltau))
            cfg=config(coordinate,axis)
            @test cfg.equilibrium.backend=="given" && cfg.equilibrium.mode===mode
            root=read_inversion_inputs(cfg).atmosphere
            @test root.rho≈permutedims(rho,(1,3,2)) rtol=1e-12
            @test root.ne≈permutedims(ne,(1,3,2)) rtol=1e-12
            @test root.h_neutral===nothing && root.chi500===nothing
            expected=(root.rho./F._GIVEN_MASS_PER_H_KG.*F._GIVEN_TOTAL_ABUNDANCE.+root.ne).*F._KB_SI.*root.temperature
            @test root.pgas≈expected rtol=1e-12
            distributed=distribute_atmosphere(Float64,root,context)
            @test_throws ArgumentError prepare_equilibrium(cfg.equilibrium,distributed,context)
            backend=prepare_equilibrium(cfg.equilibrium,distributed,context;synthesis_eos=eos)
            @test backend isa GivenAtmosphereBackend
            @test backend.provenance["hydrogen_populations_given"]===false
            @test reconstruct_equilibrium!(backend,distributed,context).valid
            a=distributed.local_atmosphere
            @test all(>(0),a.chi500) && all(diff(a.z;dims=1).<0)
            @test a.rho≈permutedims(rho,(1,3,2)) rtol=1e-12        # untouched
            coordinate=="z" && @test a.z[:,1,1]≈zv rtol=1e-12
        end
        # ρ and P: no HSEMHS, both kept, nₑ from the EOS at (T, P), or given as well; column mass written
        for (coordinate,axis,mode) in (("z","z_m = \"zm\"",:given_z),("log10m","log10m = \"lm\"",:given_logm),("log10tau500","logtau500 = \"lt\"",:given_ltau))
            for withne in (false,true)
                cfg=config(coordinate,axis;nedata=withne ? "ne = \"ne\"\n" : "",extra="rho = \"rho\"\npgas = \"p\"\n")
                @test cfg.equilibrium.backend=="given" && cfg.equilibrium.mode===mode
                root=read_inversion_inputs(cfg).atmosphere
                distributed=distribute_atmosphere(Float64,root,context)
                @test_throws ArgumentError prepare_equilibrium(cfg.equilibrium,distributed,context)
                root=read_inversion_inputs(cfg).atmosphere; distributed=distribute_atmosphere(Float64,root,context)
                backend=prepare_equilibrium(cfg.equilibrium,distributed,context;synthesis_eos=eos)
                @test backend.provenance["ne_given"]===withne
                @test reconstruct_equilibrium!(backend,distributed,context).valid
                a=distributed.local_atmosphere
                @test a.rho≈permutedims(rho,(1,3,2)) rtol=1e-12
                @test a.pgas≈permutedims(pressure,(1,3,2)) rtol=1e-12
                if withne
                    @test a.ne≈permutedims(ne,(1,3,2)) rtol=1e-12
                else
                    r=zeros(size(a.temperature)); n=zeros(size(a.temperature))
                    F.thermodynamics!(r,n,eos,Array{Float64,3}(a.temperature),Array{Float64,3}(a.pgas))
                    @test a.ne≈n rtol=1e-12
                    @test !(a.ne≈permutedims(ne,(1,3,2)))
                end
                @test all(>(0),a.chi500) && all(diff(a.z;dims=1).<0) && all(isfinite,a.logtau500)
                coordinate=="z" && @test a.z[:,1,1]≈zv rtol=1e-12
                m=given_column_mass(a,274.0)
                @test all(diff(m;dims=1).>0)
                if coordinate=="log10m"
                    @test log10.(m[:,1,1])≈native_depth(a.grid) rtol=1e-10
                else
                    @test m[1,2,1]≈a.pgas[1,2,1]/274.0 rtol=1e-12
                    @test m[3,2,1]-m[2,2,1]≈(a.rho[2,2,1]+a.rho[3,2,1])/2*(a.z[2,2,1]-a.z[3,2,1]) rtol=1e-12
                end
                wave=cfg.synthesis.wavelength_m
                result=F.InversionRunResult(nothing,SpectralCube(zeros(length(wave),1,nx,ny),wave,StokesSet(:I)),a,nothing,nothing,Dict{String,Any}())
                write_inversion_outputs(cfg,result)
                @test h5open(f->read(attributes(f["column_mass"])["units"]),cfg.outputs.atmosphere_file)=="kg m^-2"
                @test h5read(cfg.outputs.atmosphere_file,"column_mass")≈F._with_time_zyx(m) rtol=1e-12
            end
        end
        refused=try config("z","z_m = \"zm\"";nedata="",extra="rho = \"rho\"\npgas = \"p\"\n",mode="inversion"); "" catch e; sprint(showerror,e) end
        @test occursin("forward synthesis only",refused)

        # every combination of ne, rho, pgas and hydrogen_populations: no HSEMHS, given fields stay as given; column sections are ignored
        @test config("z","z_m = \"zm\"";equilibrium="[equilibrium]\nmodel = \"column_hse\"\n[equilibrium.column_z]\nP_ref_pa = 1.0e5\n").equilibrium.backend=="given"
        @test_throws ArgumentError config("z","z_m = \"zm\"";equilibrium="[equilibrium]\nnot_a_key = 1\n")
        lines=Dict("ne"=>"ne = \"ne\"\n","rho"=>"rho = \"rho\"\n","pgas"=>"pgas = \"p\"\n","hp"=>"hydrogen_populations = \"hp\"\n")
        combos=[c for c in (["ne"],["rho"],["pgas"],["ne","rho"],["ne","pgas"],["rho","pgas"],["ne","rho","pgas"],["hp"],["ne","hp"],["rho","hp"],["pgas","hp"],
            ["ne","rho","hp"],["ne","pgas","hp"],["rho","pgas","hp"],["ne","rho","pgas","hp"])]
        for (coordinate,axis,mode) in (("z","z_m = \"zm\"",:given_z),("log10m","log10m = \"lm\"",:given_logm),("log10tau500","logtau500 = \"lt\"",:given_ltau)),combo in combos
            text=join(lines[k] for k in combo if k!="ne")
            @testset "$coordinate $(join(combo,"+"))" begin
                cfg=config(coordinate,axis;nedata="ne" in combo ? lines["ne"] : "",extra=text)
                @test cfg.equilibrium.backend=="given" && cfg.equilibrium.mode===mode
                distributed=distribute_atmosphere(Float64,read_inversion_inputs(cfg).atmosphere,context)
                backend=prepare_equilibrium(cfg.equilibrium,distributed,context;synthesis_eos=eos)
                @test backend isa GivenAtmosphereBackend
                @test backend.provenance["ne_given"]===("ne" in combo) && backend.provenance["hydrogen_populations_given"]===("hp" in combo)
                @test reconstruct_equilibrium!(backend,distributed,context).valid
                a=distributed.local_atmosphere
                "ne" in combo && @test a.ne≈permutedims(ne,(1,3,2)) rtol=1e-12
                "rho" in combo && @test a.rho≈permutedims(rho,(1,3,2)) rtol=1e-12
                "pgas" in combo && @test a.pgas≈permutedims(pressure,(1,3,2)) rtol=1e-12
                if "hp" in combo
                    @test a.h_neutral≈dropdims(sum(permutedims(populations,(2,4,3,1))[:,:,:,1:5];dims=4);dims=4) rtol=1e-12
                    @test a.h_proton≈permutedims(populations,(2,4,3,1))[:,:,:,6] rtol=1e-12
                else
                    @test a.h_neutral===nothing
                end
                @test all(>(0),a.ne) && all(>(0),a.rho) && all(>(0),a.pgas) && all(>(0),a.chi500)
                @test all(diff(a.z;dims=1).<0) && all(isfinite,a.logtau500)
                if "ne" in combo && "pgas" in combo && !("rho" in combo) && !("hp" in combo)     # ρ from P/kT − nₑ, which gives P back
                    @test (a.rho./F._GIVEN_MASS_PER_H_KG.*F._GIVEN_TOTAL_ABUNDANCE.+a.ne).*F._KB_SI.*a.temperature≈a.pgas rtol=1e-10
                end
                @test all(diff(given_column_mass(a,274.0);dims=1).>0)
                # a field the input leaves out comes from the EOS at (T, P), so it agrees with P there
                r=zeros(size(a.temperature)); n=zeros(size(a.temperature))
                F.thermodynamics!(r,n,eos,Array{Float64,3}(a.temperature),Array{Float64,3}(a.pgas))
                length(combo)==1 && combo[1]!="hp" && @test a.ne≈n rtol=1e-7
                length(combo)==1 && combo[1]!="hp" && @test a.rho≈r rtol=1e-7
            end
        end
        # forward only, needs ne, and one density only
        refused=try config("z","z_m = \"zm\"";mode="inversion"); "" catch e; sprint(showerror,e) end
        @test occursin("forward synthesis only",refused)

    end
end
