using Test
using FFNOInversion
using HDF5
include(joinpath(@__DIR__,"..","scripts","forward_atmosphere_io.jl"))

# [atmosphere] vz_positive = "down": only the sign of vz in the files differs. z is positive upward and tau increases
# downward whatever the setting, and nothing but vz (and its control bounds) changes.
@testset "vz_positive: the input's vz sign, nothing else" begin
    isdefined(Main,:write_native_z_file) || include(joinpath(@__DIR__,"unit_coordinates.jl"))
    mktempdir() do dir
        up=joinpath(dir,"up.h5"); write_native_z_file(up)
        down=joinpath(dir,"down.h5")
        h5open(up,"r") do f
            h5open(down,"w") do g
                for name in keys(f); g[name]=name=="vz" ? -read(f[name]) : read(f[name]); end
            end
        end
        view="[view]\nmu = 0.8\nazimuth_deg = 30.0\ngeometry = \"inclined\""
        cu=load_config(native_z_config(dir,up;extra=view))
        cd_=load_config(native_z_config(dir,down;extra_atmos="vz_positive = \"down\"",extra=view))
        @test cu.vz_positive===:up && cd_.vz_positive===:down
        @test cd_.view==cu.view && cd_.view.azimuth_deg==30.0            # the azimuth is not mirrored
        a=read_inversion_inputs(cu).atmosphere; b=read_inversion_inputs(cd_).atmosphere
        @test native_depth(a.grid)==native_depth(b.grid)                 # z is not negated
        @test a.grid.x==b.grid.x && a.grid.y==b.grid.y                   # y is not reversed
        for name in (:temperature,:vx,:vy,:vz,:z); @test getfield(a,name)==getfield(b,name); end

        # outputs: vz back in the file's sign, everything else as in the code
        nλ,nx,ny=3,size(b.vz,2),size(b.vz,3)
        spectrum=reshape(collect(1.0:nλ*nx*ny),nλ,1,nx,ny)
        result=InversionRunResult(nothing,SpectralCube(spectrum,[1e-7,2e-7,3e-7],StokesSet(:I)),b,nothing,nothing,Dict{String,Any}())
        write_inversion_outputs(cd_,result)
        h5open(cd_.outputs.atmosphere_file,"r") do f
            @test read(f["z_m"])==native_depth(b.grid)
            @test read(f["y_m"])==b.grid.y && read(f["x_m"])==b.grid.x
            @test read(f["vz"])[1,:,:,:]==permutedims(-b.vz,(1,3,2))
            @test read(f["vy"])[1,:,:,:]==permutedims(b.vy,(1,3,2))
            @test read(f["temperature"])[1,:,:,:]==permutedims(b.temperature,(1,3,2))
            @test read(attributes(f)["vz_positive"])=="down"
            @test !haskey(attributes(f),"frame")
            @test read(attributes(f)["view_azimuth_deg"])==30.0
        end
        h5open(cd_.outputs.synthesis_file,"r") do f
            scale=output_intensity_scale(cd_.regions,cd_.outputs.intensity_units)
            @test read(f["intensity"])[1,1,:,:,:]≈permutedims((spectrum.*scale)[:,1,:,:],(1,3,2)) rtol=1e-14   # not reversed along y
        end

        # controls: vz bounds in the file's sign are mirrored; other bounds and every node stay
        raw=FFNOInversion.TOML.parse("""
[inversion]
[[inversion.controls]]
variable = "vz"
z_nodes_m = [2.0e5, 1.0e6]
lower = -1000.0
upper = 3000.0
scale = 1000.0
[[inversion.controls]]
variable = "vy"
z_nodes_m = [2.0e5, 1.0e6]
lower = -1000.0
upper = 3000.0
scale = 1000.0
""")
        controls=FFNOInversion._controls_in_code_sign(FFNOInversion._controls(raw["inversion"],:z_m),:down)
        @test controls[1].lower==-3000.0 && controls[1].upper==1000.0 && controls[1].nodes==[2.0e5,1.0e6]
        @test controls[2].lower==-1000.0 && controls[2].upper==3000.0 && controls[2].nodes==[2.0e5,1.0e6]

        # the forward-atmosphere reader returns the code's vz from a file written with vz_positive = "down"
        shape=(3,2,2); grid=Grid3D([-5.0,-3.0,-1.0],[0.0,4e4],[0.0,4e4])
        vz=reshape(collect(1.0:12.0),shape)
        atmosphere=Atmosphere3D(grid,fill(5000.0,shape),zeros(shape),zeros(shape),vz,zeros(shape))
        path=joinpath(dir,"forward_down.h5")
        write_forward_atmosphere(path,atmosphere,Dict(:H=>zeros(3,2,2,6));attributes_=Dict("vz_positive"=>"down"))
        @test read_forward_atmosphere(path)[1].vz==-vz
        write_forward_atmosphere(path,atmosphere,Dict(:H=>zeros(3,2,2,6)))
        @test read_forward_atmosphere(path)[1].vz==vz

        # the option's values, and the removed option is refused rather than ignored
        @test_throws ArgumentError FFNOInversion._vz_positive(Dict("vz_positive"=>"sideways"))
        @test_throws ArgumentError FFNOInversion._vz_positive(Dict("frame"=>"z_down"))
        @test FFNOInversion._vz_positive(Dict{String,Any}())===:up
    end
end
