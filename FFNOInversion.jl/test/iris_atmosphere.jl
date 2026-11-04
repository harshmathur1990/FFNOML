using Test
using FFNOInversion

# An IRIS/MURaM FITS atmosphere: a small cube written here, read back through the configuration.
@testset "IRIS FITS atmosphere" begin
    F=FFNOInversion
    card(key,value)=rpad(rpad(key,8)*"= "*value,80)
    function block(cards)
        text=join(cards)*rpad("END",80)
        text*" "^(mod(-length(text),2880))
    end
    function write_fits(path,cube,bunit;zmm)
        nx,ny,nz=size(cube)
        open(path,"w") do io
            write(io,block([card("SIMPLE","T"),card("BITPIX","-32"),card("NAXIS","3"),card("NAXIS1",string(nx)),card("NAXIS2",string(ny)),card("NAXIS3",string(nz)),
                card("BUNIT","'$bunit'"),card("CDELT1","0.192000 / [Mm]"),card("CDELT2","0.192000"),card("CUNIT1","'Mm'"),card("CUNIT2","'Mm'"),card("CUNIT3","'Mm'")]))
            payload=hton.(Float32.(cube)); write(io,payload); write(io,zeros(UInt8,mod(-sizeof(payload),2880)))
            write(io,block([card("XTENSION","'IMAGE'"),card("BITPIX","-32"),card("NAXIS","1"),card("NAXIS1",string(nz)),card("EXTNAME","' z-coordinate'"),card("BUNIT","'Mm'")]))
            heights=hton.(Float32.(zmm)); write(io,heights); write(io,zeros(UInt8,mod(-sizeof(heights),2880)))
        end
    end
    mktempdir() do dir
        atmos=joinpath(dir,"atmos"); mkpath(atmos)
        nx,ny,nz=4,3,5; zmm=collect(range(-1.0,3.0;length=nz))                   # bottom first, as the files are
        value(q,i,j,k)=q+0.1i+0.01j+0.001k
        make(q)=[value(q,i,j,k) for i in 1:nx,j in 1:ny,k in 1:nz]
        for (q,base,unit) in (("lgtg",3.7,"K"),("lgr",-6.0,"kg m^(-3)"),("lgp",3.0,"N m^(-2)"),("ux",100.0,"m s^(-1)"),("uy",200.0,"m s^(-1)"),("uz",300.0,"m s^(-1)"))
            write_fits(joinpath(atmos,"MURaM_run_$(q)_12.fits"),make(base),unit;zmm=zmm)
        end
        function config(extra="";atmosphere="atmos_model_format = \"iris\"\natmos_directory = \"$atmos\"\n",mode="forward",eq="[equilibrium]\neos_table_file = \"t.h5\"\n")
            p=joinpath(dir,"c.toml")
            write(p,"""mode = "$mode"\n[inputs]\ntime_index = 1\n[outputs]\nsynthesis_file = "s.h5"\natmosphere_file = "o.h5"\n[atmosphere]\n$atmosphere$eq$extra
            [[regions]]\nstart_angstrom = 6562.0\nstep_angstrom = 0.1\ncount = 5\nnormalization = 1.0\npsf_type = "none"\n[[regions.sources]]\nmode = "ffno"\nspecies = "H"\nline = "halpha"
            [physics]\nredistribution = "non_prd"\n[observation]\nstokes = ["I"]\n[solver]\nmax_iterations = $(mode=="forward" ? 0 : 1)\n[parallel]\nenabled = false\nthreads_per_rank = 1\n""")
            load_config(p)
        end
        cfg=config()
        @test cfg.atmosphere.model_format===:iris && cfg.atmosphere.vertical_coordinate===:z_m
        @test cfg.equilibrium.backend=="given"                                    # rho and P are given: no HSEMHS
        @test cfg.population_mesh!==nothing                                       # on by default
        @test cfg.synthesis.dx_m≈192e3 && cfg.synthesis.dy_m≈192e3
        a=read_inversion_inputs(cfg).atmosphere
        @test size(a.temperature)==(nz,nx,ny)
        @test native_depth(a.grid)≈reverse(zmm).*1e6 rtol=1e-6                    # top first
        @test all(diff(native_depth(a.grid)).<0)
        # the top of column (x, y) of the output is the file's last level at (x, reversed y)
        for (a_i,b_i,k_i) in ((1,1,1),(4,2,3),(2,3,5))
            file_y=ny+1-b_i; file_k=nz+1-k_i
            @test a.temperature[k_i,a_i,b_i]≈exp10(value(3.7,a_i,file_y,file_k)) rtol=1e-5
            @test a.rho[k_i,a_i,b_i]≈exp10(value(-6.0,a_i,file_y,file_k)) rtol=1e-5
            @test a.pgas[k_i,a_i,b_i]≈exp10(value(3.0,a_i,file_y,file_k)) rtol=1e-5
            @test a.vx[k_i,a_i,b_i]≈value(100.0,a_i,file_y,file_k) rtol=1e-5
            @test a.vy[k_i,a_i,b_i]≈value(200.0,a_i,file_y,file_k) rtol=1e-5     # uy is kept as it is
            @test a.vz[k_i,a_i,b_i]≈value(300.0,a_i,file_y,file_k) rtol=1e-5
        end
        @test a.ne===nothing && a.chi500===nothing                                # the EOS gives these later
        # the temperature floor: 3250 K unless configured, only the temperature, and storage_order means nothing here
        @test cfg.atmosphere.temperature_floor_k==3250.0
        temperature(extra)=read_inversion_inputs(config(;atmosphere="atmos_model_format = \"iris\"\natmos_directory = \"$atmos\"\n$extra")).atmosphere
        @test temperature("").temperature==a.temperature                          # this cube is warmer than the floor
        cut=sort(vec(a.temperature))[10]; raised=temperature("temperature_floor_k = $(Float64(cut))\n")
        @test raised.temperature==max.(a.temperature,cut) && count(<(cut),a.temperature)>0
        @test raised.rho==a.rho && raised.pgas==a.pgas && raised.vx==a.vx
        @test temperature("temperature_floor_k = 0.0\n").temperature==a.temperature
        @test temperature("storage_order = \"python\"\n").temperature==a.temperature
        @test_throws ArgumentError config(;atmosphere="atmos_model_format = \"iris\"\natmos_directory = \"$atmos\"\ntemperature_floor_k = -1.0\n")
        refused=try config(;atmosphere="temperature_floor_k = 3250.0\n"); "" catch e; sprint(showerror,e) end
        @test occursin("temperature_floor_k",refused)                             # not for an HDF5 atmosphere
        # the mesh can be turned off; [grid] must agree; refusals
        @test config("[population.mesh]\nenabled = false\n").population_mesh===nothing
        @test config("[grid]\ndx_m = 192000.0\n").synthesis.dx_m≈192e3
        @test_throws ArgumentError config("[grid]\ndx_m = 48000.0\n")
        @test_throws ArgumentError config(eq="")                                  # the mesh needs the table
        refused=try config(mode="inversion"); "" catch e; sprint(showerror,e) end
        @test occursin("forward synthesis only",refused)
        @test_throws ArgumentError config(atmosphere="atmos_model_format = \"unknown\"\n")
        @test_throws ArgumentError config(atmosphere="atmos_model_format = \"iris\"\natmos_directory = \"$dir/none\"\n")
        write_fits(joinpath(atmos,"MURaM_run_lgtg_13.fits"),make(3.7),"K";zmm=zmm)
        @test_throws ArgumentError config()                                       # two snapshots
        rm(joinpath(atmos,"MURaM_run_lgtg_13.fits"))
        write_fits(joinpath(atmos,"MURaM_run_lgr_12.fits"),make(-6.0),"parsec";zmm=zmm)
        @test_throws ArgumentError read_inversion_inputs(config())                # an unknown unit is an error
    end
end
