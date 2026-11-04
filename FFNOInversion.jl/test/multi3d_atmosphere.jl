using Test
using FFNOInversion
import Muspel

# A MULTI3D snapshot: a small mesh and atm3d written here, read back through the configuration.
@testset "MULTI3D atmosphere" begin
    F=FFNOInversion
    nx,ny,nz=4,3,5; dx_cm,dy_cm=4.8e6,9.6e6; z_cm=[8e7,5e7,3e7,1e7,-1e7]         # top first, as the files are
    function write_mesh(path;x=collect(0:nx-1).*dx_cm,y=collect(0:ny-1).*dy_cm,z=z_cm,extra="")
        number(v)=string(round(v;sigdigits=6))
        open(path,"w") do io
            for axis in (x,y,z); println(io,length(axis)); println(io,join(number.(axis),"  ")); end
            print(io,extra)
        end
    end
    value(q,i,j,k)=Float32(q*(1+0.1i+0.01j+0.001k))
    # ne, T, vx, vy, vz, rho, then the hydrogen levels when there are any: the base of each
    bases(hydrogen)=hydrogen ? (1e15,5e3,2.0,3.0,4.0,1e-8,1e14,1e13,1e12,1e11,1e10,5e14) : (1e15,5e3,2.0,3.0,4.0,1e-8)
    function write_atm3d(path,hydrogen)
        open(path,"w") do io
            for q in bases(hydrogen)
                write(io,[value(q,i,j,k) for i in 1:nx,j in 1:ny,k in 1:nz])
            end
        end
    end
    mktempdir() do dir
        plain=joinpath(dir,"plain"); populated=joinpath(dir,"populated"); mkpath(plain); mkpath(populated)
        for (d,hydrogen) in ((plain,false),(populated,true))
            write_mesh(joinpath(d,"mesh")); write_atm3d(joinpath(d,"atm3d"),hydrogen)
        end
        function config(directory;extra="",atmosphere="",mode="forward",eq="[equilibrium]\neos_table_file = \"t.h5\"\n")
            p=joinpath(dir,"c.toml")
            write(p,"""mode = "$mode"\n[inputs]\ntime_index = 1\n[outputs]\nsynthesis_file = "s.h5"\natmosphere_file = "o.h5"\n[atmosphere]\natmos_model_format = "multi3d"\natmos_directory = "$directory"\n$atmosphere$eq$extra
            [[regions]]\nstart_angstrom = 6562.0\nstep_angstrom = 0.1\ncount = 5\nnormalization = 1.0\npsf_type = "none"\n[[regions.sources]]\nmode = "ffno"
            [physics]\nredistribution = "non_prd"\n[observation]\nstokes = ["I"]\n[solver]\nmax_iterations = $(mode=="forward" ? 0 : 1)\n[parallel]\nenabled = false\nthreads_per_rank = 1\n""")
            load_config(p)
        end
        cfg=config(plain)
        @test cfg.atmosphere.model_format===:multi3d && cfg.atmosphere.vertical_coordinate===:z_m
        @test cfg.equilibrium.backend=="given"                                    # nₑ and rho are given: no HSEMHS
        @test cfg.atmosphere.ne_dataset!==nothing && cfg.atmosphere.rho_dataset!==nothing
        @test cfg.atmosphere.hydrogen_populations_dataset===nothing
        @test config(populated).atmosphere.hydrogen_populations_dataset!==nothing
        @test cfg.population_mesh!==nothing                                       # on by default
        @test cfg.synthesis.dx_m≈48e3 && cfg.synthesis.dy_m≈96e3                  # cm in the mesh, not assumed square
        @test cfg.atmosphere.temperature_floor_k==0.0                             # the file's temperatures
        a=read_inversion_inputs(cfg).atmosphere
        @test size(a.temperature)==(nz,nx,ny)
        @test native_depth(a.grid)≈z_cm.*1e-2                                     # already top first: nothing is reversed
        @test a.grid.x≈(0:nx-1).*48e3 && a.grid.y≈(0:ny-1).*96e3
        # (k, x, y) is the file's (x, y, k), with nothing flipped, in SI
        for (i,j,k) in ((1,1,1),(4,2,3),(2,3,5))
            @test a.ne[k,i,j]≈value(1e15,i,j,k)*1e6 rtol=1e-6
            @test a.temperature[k,i,j]≈value(5e3,i,j,k) rtol=1e-6
            @test a.vx[k,i,j]≈value(2.0,i,j,k)*1e3 rtol=1e-6
            @test a.vy[k,i,j]≈value(3.0,i,j,k)*1e3 rtol=1e-6
            @test a.vz[k,i,j]≈value(4.0,i,j,k)*1e3 rtol=1e-6
            @test a.rho[k,i,j]≈value(1e-8,i,j,k)*1e3 rtol=1e-6
        end
        @test a.pgas!==nothing && all(>(0),a.pgas)                                # from nₑ and rho
        @test a.h_neutral===nothing && a.chi500===nothing                         # no populations: the EOS gives these later
        # with populations: the levels are the file's, the last the protons, and rho stays as given
        b=read_inversion_inputs(config(populated)).atmosphere
        @test b.temperature==a.temperature && b.ne==a.ne && b.rho==a.rho
        levels=bases(true)[7:end]
        for (i,j,k) in ((1,1,1),(4,2,3),(2,3,5))
            @test b.h_neutral[k,i,j]≈sum(value(q,i,j,k) for q in levels[1:5])*1e6 rtol=1e-5
            @test b.h_proton[k,i,j]≈value(levels[6],i,j,k)*1e6 rtol=1e-5
        end
        @test b.chi500!==nothing
        # options: a floor changes only the temperature; vz_positive; storage_order means nothing here
        cut=sort(vec(a.temperature))[10]
        raised=read_inversion_inputs(config(plain;atmosphere="temperature_floor_k = $(Float64(cut))\n")).atmosphere
        @test raised.temperature==max.(a.temperature,cut) && count(<(cut),a.temperature)>0 && raised.rho==a.rho && raised.ne==a.ne
        down=read_inversion_inputs(config(plain;atmosphere="vz_positive = \"down\"\n")).atmosphere
        @test down.vz==-a.vz && down.vx==a.vx && down.vy==a.vy
        @test read_inversion_inputs(config(plain;atmosphere="storage_order = \"python\"\n")).atmosphere.temperature==a.temperature
        # y falls in most snapshots' meshes: only the spacing's size is used, and nothing is flipped
        falling=mktempdir(dir); write_mesh(joinpath(falling,"mesh");y=collect(0:ny-1).*-dy_cm); write_atm3d(joinpath(falling,"atm3d"),false)
        c=config(falling); @test c.synthesis.dy_m≈96e3
        f=read_inversion_inputs(c).atmosphere
        @test f.grid.y≈a.grid.y && f.temperature==a.temperature && f.vy==a.vy
        # [grid] must agree, to the digits the mesh has
        @test config(plain;extra="[grid]\ndx_m = 48000.0\n").synthesis.dx_m≈48e3
        @test_throws ArgumentError config(plain;extra="[grid]\ndx_m = 96000.0\n")
        # refusals
        refused=try config(plain;mode="inversion"); "" catch e; sprint(showerror,e) end
        @test occursin("forward synthesis only",refused) && occursin("multi3d",refused)
        @test_throws ArgumentError config(plain;eq="")                            # the mesh needs the table
        @test_throws ArgumentError config(joinpath(dir,"none"))
        @test_throws ArgumentError config(plain;atmosphere="vertical_coordinate = \"log10tau500\"\n")
        function broken(edit)
            d=mktempdir(dir); write_mesh(joinpath(d,"mesh")); write_atm3d(joinpath(d,"atm3d"),false); edit(d); d
        end
        missing_atm3d=broken(d->rm(joinpath(d,"atm3d")))
        refused=try config(missing_atm3d); "" catch e; sprint(showerror,e) end
        @test occursin("atm3d",refused)
        short=broken(d->write(joinpath(d,"atm3d"),zeros(Float32,nx*ny*nz*6-1)))
        refused=try config(short); "" catch e; sprint(showerror,e) end
        @test occursin("bytes",refused)                                           # neither 6 nor 12 fields
        @test_throws ArgumentError config(broken(d->write_mesh(joinpath(d,"mesh");z=reverse(z_cm))))     # bottom first
        @test_throws ArgumentError config(broken(d->write_mesh(joinpath(d,"mesh");extra="7\n")))         # numbers left over
        @test_throws ArgumentError config(broken(d->write_mesh(joinpath(d,"mesh");x=[0.0,1e6,2e6,9e6]))) # not uniform
        zero_ne=broken() do d
            cells=zeros(Float32,nx*ny*nz*6); cells.=1f0; write(joinpath(d,"atm3d"),cells)
            open(joinpath(d,"atm3d"),"r+") do io; write(io,0f0); end
        end
        @test_throws ArgumentError read_inversion_inputs(config(zero_ne))
    end
end
