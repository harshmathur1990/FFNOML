using Test
using FFNOInversion
import HDF5

# [population.given]: MULTI3D populations written mirrored (and once bottom first) beside a small snapshot; the preflight must find how,
# and the populations handed over must be the snapshot's own.
@testset "given populations" begin
    F=FFNOInversion
    nx,ny,nz,levels=12,9,5,2; z_cm=[8e7,5e7,3e7,1e7,-1e7]
    temperature(i,j)=5000+1000*((i*7919+j*104729)%97)/97
    truth(k,i,j,l)=Float32(l*1e10*(1+0.01i+0.001j+0.0001k))                 # cm^-3, on the atmosphere's columns
    tau(k,i,j)=Float32(10.0^(k-3)*exp((temperature(i,j)-5000)/1000))
    function write_snapshot(dir)
        open(joinpath(dir,"mesh"),"w") do io
            for axis in (collect(0:nx-1).*4.8e6,collect(0:ny-1).*4.8e6,z_cm); println(io,length(axis)); println(io,join(string.(axis),"  ")); end
        end
        open(joinpath(dir,"atm3d"),"w") do io
            for q in (1e15,0.0,2.0,3.0,4.0,1e-8)
                write(io,[Float32(q==0 ? temperature(i,j)*(1+0.1k) : q*(1+0.1i+0.01j+0.001k)) for i in 1:nx,j in 1:ny,k in 1:nz])
            end
        end
    end
    # the file's (a, b, k') for the atmosphere's (i, j, k)
    function write_run(dir,orientation;reversed=false,cubes=2levels+1,grid=(nx,ny,nz),value=truth)
        mkpath(dir); fx,fy=F._given_transposes(orientation) ? (ny,nx) : (nx,ny)
        place(f)=begin
            out=zeros(Float32,fx,fy,nz)
            for i in 1:nx,j in 1:ny,k in 1:nz
                a,b=F._given_source(orientation,i,j,nx,ny); out[a,b,reversed ? nz+1-k : k]=f(k,i,j)
            end
            out
        end
        open(joinpath(dir,"out_pop"),"w") do io
            for l in 1:cubes; write(io,place((k,i,j)->value(k,i,j,min(l,levels)))); end
        end
        write(joinpath(dir,"tau500"),place(tau))
        open(joinpath(dir,"multi3d.input"),"w") do io
            println(io,"             nx = $(grid[1])\n             ny =  $(grid[2])\n             nz =  $(grid[3])")
        end
        joinpath(dir,"out_pop")
    end
    mktempdir() do dir
        snap=joinpath(dir,"snap"); mkpath(snap); write_snapshot(snap)
        function config(given;mode="forward")
            p=joinpath(dir,"c.toml")
            write(p,"""mode = "$mode"\n[inputs]\ntime_index = 1\n[outputs]\nsynthesis_file = "s.h5"\natmosphere_file = "o.h5"\n[atmosphere]\natmos_model_format = "multi3d"\natmos_directory = "$snap"\n[equilibrium]\neos_table_file = "t.h5"\n$given
            [[regions]]\nstart_angstrom = 6562.0\nstep_angstrom = 0.1\ncount = 5\nnormalization = 1.0\npsf_type = "none"\n[[regions.sources]]\nmode = "ffno"
            [physics]\nredistribution = "non_prd"\n[observation]\nstokes = ["I"]\n[solver]\nmax_iterations = $(mode=="forward" ? 0 : 1)\n[parallel]\nenabled = false\nthreads_per_rank = 1\n""")
            load_config(p)
        end
        given(file;extra="")="[population.given]\n$extra[population.given.files]\nH = \"$file\"\n"
        cfg=config(given(write_run(joinpath(snap,"H"),:flip_y)))
        @test cfg.given_populations.format===:multi3d && cfg.given_populations.orientation===:auto
        @test cfg.given_populations.files[:H]==joinpath(snap,"H","out_pop")
        @test occursin("given_populations=H",dry_run_summary(cfg))
        @test config("").given_populations===nothing
        @test_throws ArgumentError config(given(joinpath(snap,"H","out_pop"));mode="inversion")
        @test_throws ArgumentError config(given("x";extra="orientation = \"rotate_45\"\n"))
        @test_throws ArgumentError config(given("x";extra="colour = 1\n"))
        @test_throws ArgumentError config("[population.given]\nformat = \"multi3d\"\n")
        # the eight orientations are bijections onto the file's columns
        for o in F.GIVEN_ORIENTATIONS
            fx,fy=F._given_transposes(o) ? (ny,nx) : (nx,ny)
            @test sort([F._given_source(o,i,j,nx,ny) for i in 1:nx for j in 1:ny])==sort([(a,b) for a in 1:fx for b in 1:fy])
        end
        @test Set(F._given_candidates((nx,ny),nx,ny))==Set((:identity,:flip_x,:flip_y,:rotate_180))
        @test Set(F._given_candidates((ny,nx),nx,ny))==Set((:transpose,:transpose_flip_x,:transpose_flip_y,:transpose_rotate_180))

        atmosphere=read_inversion_inputs(cfg).atmosphere
        result=read_given_populations(cfg.given_populations,atmosphere,Dict(:H=>levels))
        @test result.provenance["orientation"]=="flip_y" && result.provenance["levels_reversed"]==false
        @test result.provenance["correlations"]["flip_y"]>0.99
        @test any(occursin("orientation flip_y chosen",n) for n in result.notes)
        held_H=result.models[:H].populations
        @test size(held_H)==(nz,nx,ny,levels)
        @test all(held_H[k,i,j,l]≈truth(k,i,j,l)*1f6 for k in 1:nz,i in 1:nx,j in 1:ny,l in 1:levels)

        # bottom first, rotated: levels reversed, columns turned back
        cfg2=config(given(write_run(joinpath(dir,"r","H"),:rotate_180;reversed=true)))
        r2=read_given_populations(cfg2.given_populations,atmosphere,Dict(:H=>levels))
        @test r2.provenance["orientation"]=="rotate_180" && r2.provenance["levels_reversed"]==true
        @test all(r2.models[:H].populations[k,i,j,l]≈truth(k,i,j,l)*1f6 for k in 1:nz,i in 1:nx,j in 1:ny,l in 1:levels)
        # a configured orientation is used even when another correlates better
        cfg3=config(given(joinpath(snap,"H","out_pop");extra="orientation = \"identity\"\n"))
        @test (@test_logs (:warn,r"flip_y correlates best") match_mode=:any read_given_populations(cfg3.given_populations,atmosphere,
            Dict(:H=>levels))).provenance["orientation"]=="identity"

        # what the preflight refuses
        bad(path;kw...)=config(given(write_run(joinpath(dir,path,"H"),:identity;kw...))).given_populations
        @test_throws DimensionMismatch preflight_given_populations(bad("half";grid=(nx÷2,ny,nz)),atmosphere,Dict(:H=>levels))
        @test_throws ArgumentError preflight_given_populations(bad("few";cubes=1),atmosphere,Dict(:H=>levels))
        @test_throws ArgumentError preflight_given_populations(cfg.given_populations,atmosphere,Dict(:CA=>levels))
        @test_throws ArgumentError read_given_populations(bad("nan";value=(k,i,j,l)->k==2 ? NaN32 : 1f10),atmosphere,Dict(:H=>levels))
        rm(joinpath(dir,"few","H","multi3d.input"))
        truncated=joinpath(dir,"few","H","out_pop"); write(truncated,read(truncated)[1:end-4])
        @test_throws ArgumentError preflight_given_populations(config(given(truncated)).given_populations,atmosphere,Dict(:H=>levels))
        # without tau500, auto cannot choose
        notau=joinpath(dir,"notau","H"); write_run(notau,:identity); rm(joinpath(notau,"tau500"))
        @test_throws ArgumentError preflight_given_populations(config(given(joinpath(notau,"out_pop"))).given_populations,atmosphere,Dict(:H=>levels))
        @test preflight_given_populations(config(given(joinpath(notau,"out_pop");extra="orientation = \"identity\"\n")).given_populations,
            atmosphere,Dict(:H=>levels)).orientation.orientation===:identity

        # "hdf5": a forward output's populations (level, z, y, x) [m^-3] and logtau500 (time, z, y, x), here mirrored in x
        h5=joinpath(dir,"forward_atmosphere.h5")
        HDF5.h5open(h5,"w") do f
            f["populations/H"]=Float32[truth(k,nx+1-i,j,l)*1f6 for l in 1:levels,k in 1:nz,j in 1:ny,i in 1:nx]
            f["logtau500"]=Float32[log10(tau(k,nx+1-i,j)) for _ in 1:1,k in 1:nz,j in 1:ny,i in 1:nx]
        end
        r4=read_given_populations(config(given(h5;extra="format = \"hdf5\"\n")).given_populations,atmosphere,Dict(:H=>levels))
        @test r4.provenance["orientation"]=="flip_x" && r4.models[:H].populations≈held_H

        # handed over by launch_fsdp_population_models, with no service when every species is given
        context=serial_context()
        held=F.prepare_given_populations(cfg,atmosphere,Dict(:H=>levels),context)
        touch(joinpath(dir,"none.pt"))
        spec=FSDPModelSpec(:H,joinpath(dir,"none.pt"),PopulationMetadata(FFNO_INPUT_CHANNELS,("1","2"),"test"))
        model=Base.ScopedValues.with(()->launch_fsdp_population_models([spec],context),F.GIVEN_POPULATIONS=>held)
        @test F._given_in_use(model,held) && F._is_fsdp_population_backend(model)
        distributed=distribute_atmosphere(Float64,atmosphere,context)
        out=Dict(:H=>zeros(Float64,nz,nx,ny,levels))
        predict_distributed_populations!(out,model,distributed,context)
        @test out[:H]≈Float64.(held_H)
        @test close_distributed_population_model!(model,context)===nothing
    end
end
