using Test
using FFNOInversion
using HDF5
isdefined(Main,:write_native_z_file) || include(joinpath(@__DIR__,"unit_coordinates.jl"))

mutable struct RecordingHeights <: AbstractPopulationModel
    nz::Int
end
FFNOInversion.predict_populations!(out,m::RecordingHeights,atmosphere,cache=nothing)=
    (m.nz=size(atmosphere.temperature,1); out.=1e10 .* reshape(atmosphere.temperature./5000,size(atmosphere.temperature)...,1); out)

# [population.mesh] on a native z atmosphere: off by default (FFNO on the native grid), on when asked for. The FFNO then
# sees the mesh, and the populations come back on the input's heights; z is the input's, and columns are never sheared.
@testset "Population mesh on a native z atmosphere" begin
    mktempdir() do dir
        path=joinpath(dir,"z.h5"); write_native_z_file(path)
        @test load_config(native_z_config(dir,path)).population_mesh===nothing
        @test population_mesh_options(load_config(native_z_config(dir,path)))===nothing
        on=load_config(native_z_config(dir,path;extra="[population.mesh]\nenabled = true\nspacing_m = 5.0e4"))
        @test on.population_mesh.spacing_m==5.0e4
        options=population_mesh_options(on)
        @test options.native_z===true
        @test load_config(native_z_config(dir,path;extra="[population.mesh]\nenabled = false")).population_mesh===nothing

        # the model on heights: T(z) for two identical columns on 7 uneven levels
        nz,nx,ny=7,2,2; z=collect(range(2.0e6,-0.1e6;length=nz)); z[2]-=2e5; T=[5000.0+1000.0*abs(zk)/1e6 for zk in z]
        field(v)=repeat(reshape(v,nz,1,1),1,nx,ny); zero3=zeros(nz,nx,ny)
        for view in (ViewGeometry(),ViewGeometry(mu=0.8,azimuth_deg=30.0,geometry=:inclined))
            grid=Grid3D(VerticalAxis(:z_m,z),[0.0,5e4],[0.0,5e4],view)
            a=Atmosphere3D(grid,field(T),copy(zero3),copy(zero3),copy(zero3),copy(zero3);
                pgas=field(1e3 .* exp.(-(z.-z[end])./2e5)),rho=field(1e-5 .* exp.(-(z.-z[end])./2e5)),
                ne=field(1e18 .* exp.(-(z.-z[end])./2e5)),z=field(z))
            inner=RecordingHeights(0)
            eos=(mass_per_H=2.2e-27,)                # not used: the mesh below stays inside the column
            config=PopulationMeshConfig(bottom_m=-0.1e6,top_m=2.0e6,spacing_m=2.0e4,uniform_top_m=2.0e6)
            model=MeshPopulationModel(inner,config,eos,274.0;native_z=true)
            out=zeros(nz,nx,ny,1)
            predict_populations!(out,model,a)
            @test inner.nz==length(population_mesh(config))>nz          # the FFNO saw the mesh, not the 7 levels
            @test isapprox(out[:,1,1,1],1e10 .* T./5000;rtol=2e-3)      # T(z) comes back on the input's levels
            @test model.last_cartesian===nothing                        # z columns are Cartesian: no shift, also inclined
        end
    end
end
