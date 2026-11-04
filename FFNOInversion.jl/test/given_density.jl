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
        h5open(file,"w") do f
            f["zm"]=zv; f["lm"]=collect(range(-3.0,0.5;length=nz)); f["lt"]=collect(range(-5.0,0.5;length=nz))
            f["x"]=4.8e4 .* collect(0:nx-1); f["y"]=4.8e4 .* collect(0:ny-1)
            f["t"]=TT; f["ne"]=ne; f["rho"]=rho; f["h"]=ones(6,nz,ny,nx); for name in ("vx","vy","vz"); f[name]=zero(TT); end
        end
        function config(coordinate,axis;extra="rho = \"rho\"\n",mode="forward",nedata="ne = \"ne\"\n")
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
            [[regions]]
            start_angstrom = 6562.0
            step_angstrom = 0.1
            count = 5
            normalization = 1.0
            psf_type = "none"
            [[regions.sources]]
            mode = "ffno"
            species = "H"
            line = "halpha"
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
        context=serial_context(options=ParallelOptions(threads_per_rank=1))
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
        # forward only, needs ne, and one density only
        refused=try config("z","z_m = \"zm\"";mode="inversion"); "" catch e; sprint(showerror,e) end
        @test occursin("forward synthesis only",refused)
        @test_throws ArgumentError config("z","z_m = \"zm\"";nedata="")
        @test_throws ArgumentError config("z","z_m = \"zm\"";extra="rho = \"rho\"\nhydrogen_populations = \"h\"\n")
    end
end
