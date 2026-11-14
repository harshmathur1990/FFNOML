using Test
using FFNOInversion
using HDF5
import HSEMHS
import WittBifrostEOS
isdefined(Main,:write_native_z_file) || include(joinpath(@__DIR__,"unit_coordinates.jl"))

mutable struct RecordingHeights <: AbstractPopulationModel
    nz::Int
end
FFNOInversion.predict_populations!(out,m::RecordingHeights,atmosphere,cache=nothing)=
    (m.nz=size(atmosphere.temperature,1); out.=1e10 .* reshape(atmosphere.temperature./5000,size(atmosphere.temperature)...,1); out)

# [population.mesh] on a native z atmosphere: on by default, off when asked for (FFNO on the native grid). The FFNO then
# sees the mesh, and the populations come back on the input's heights; z is the input's, and columns are never sheared.
@testset "Population mesh on a native z atmosphere" begin
    mktempdir() do dir
        path=joinpath(dir,"z.h5"); write_native_z_file(path)
        @test load_config(native_z_config(dir,path)).population_mesh isa PopulationMeshConfig
        @test population_mesh_options(load_config(native_z_config(dir,path))).native_z===true
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

mutable struct RecordingColumn <: AbstractPopulationModel
    seen::Any
end
FFNOInversion.predict_populations!(out,m::RecordingColumn,a,cache=nothing)=(m.seen=deepcopy(a); fill!(out,1e10); out)

# Where the mesh must extrapolate above a column: a transition region to a coronal temperature, hydrostatic from the column's top.
@testset "Corona above a column's top on the population mesh" begin
    table=WittBifrostEOS.load_table(get(ENV,"HSEMHS_TEST_TABLE","/mn/stornext/d9/data/harshm/hsemhs_env112/work/e2e/inputs/eos_table_fullz.h5"))
    # configuration: on by default wherever the mesh is used; 0 keeps the top values
    @test FFNOInversion._population_mesh(Dict{String,Any}(),:log10tau500).corona_temperature_k==1e6
    @test FFNOInversion._population_mesh(Dict{String,Any}("mesh"=>Dict{String,Any}("enabled"=>true)),:z_m).corona_temperature_k==1e6
    @test FFNOInversion._population_mesh(Dict{String,Any}("mesh"=>Dict{String,Any}("corona_temperature_k"=>0.0)),:log10tau500).corona_temperature_k==0
    @test PopulationMeshConfig().corona_temperature_k==0
    @test_throws ArgumentError PopulationMeshConfig(corona_transition_m=-1.0)

    nz,nx,ny=9,2,2; z=collect(range(2.0e6,-0.1e6;length=nz)); field(v)=repeat(reshape(v,nz,1,1),1,nx,ny)
    T=[1.0e5,2.0e4,8000.0,7000.0,6000.0,5000.0,5000.0,6000.0,9000.0]; lnP=log.(0.02 .* exp.(range(0,log(1e6);length=nz)))
    states=[WittBifrostEOS.state(table,log(T[k]),lnP[k]) for k in 1:nz]; rho=[exp(s[1]) for s in states]; ne=[exp(s[2]) for s in states]
    grid=Grid3D(VerticalAxis(:z_m,z),[0.0,5e4],[0.0,5e4])
    a=Atmosphere3D(grid,field(T),field(fill(300.0,nz)),field(fill(-200.0,nz)),field(fill(100.0,nz)),zeros(nz,nx,ny);pgas=field(exp.(lnP)),rho=field(rho),ne=field(ne),z=field(z))
    config(Tc)=PopulationMeshConfig(bottom_m=-0.3e6,top_m=6.0e6,spacing_m=5.0e4,uniform_top_m=6.0e6,corona_temperature_k=Tc,corona_transition_m=2.4e5)
    function features(Tc)
        inner=RecordingColumn(nothing); m=MeshPopulationModel(inner,config(Tc),table,274.0;native_z=true)
        predict_populations!(zeros(nz,nx,ny,1),m,a); inner.seen,m
    end
    held,_=features(0.0); s,model=features(1e6); mesh=population_mesh(config(1e6)); above=findall(>(z[1]),mesh)
    @test all(t->isapprox(t,T[1];rtol=1e-12),held.temperature[above,1,1])  # off: the top values, as before
    Tm=s.temperature[:,1,1]; rm=s.rho[:,1,1]; nm=s.ne[:,1,1]
    inside=findall(<=(z[1]),mesh)
    @test Tm[inside]==held.temperature[inside,1,1] && rm[inside]==held.rho[inside,1,1]    # at and below the top: unchanged
    @test all(diff(Tm[above]).<=0)                                      # top first: T falls with depth down to the column
    @test Tm[1]≈1e6 rtol=1e-9
    @test Tm[findlast(m->mesh[m]>=z[1]+2.4e5,above)]≈1e6 rtol=1e-9
    @test maximum(s.vx[above,1,1])==300.0 && maximum(s.vz[above,1,1])==100.0
    # hydrostatic: d ln P/dz = -rho g / P along the corona; ne and rho from the EOS
    lnp=[FFNOInversion._mesh_lnP(table,log(Tm[m]),log(rm[m])) for m in above]
    for i in 2:length(above)-1
        mesh[above[i+1]]>z[1]+2.4e5 || continue                         # where T is constant: the slope has no kink
        slope=(lnp[i-1]-lnp[i+1])/(mesh[above[i-1]]-mesh[above[i+1]])
        @test slope≈-rm[above[i]]*274.0/exp(lnp[i]) rtol=0.02
        @test log(nm[above[i]])≈WittBifrostEOS.state(table,log(Tm[above[i]]),lnp[i])[2] atol=1e-6
    end
    @test maximum(s.rho[above,1,1])<rho[1]*1.01 && minimum(s.rho[above,1,1])<rho[1]/5  # thin: not a dense layer held to the top
    # the tangent and transpose agree through the corona
    f=FFNOInversion._mesh_features(model,a); fd=[sin(0.7c+0.3k)*(c==1 ? 50.0 : c<=4 ? 10.0 : 0.02) for c in 1:6,k in 1:nz]
    out=zeros(6,length(mesh)); FFNOInversion._mesh_column_jvp!(out,model,a,1,1,@view(f[:,:,1,1]),fd)
    fb=[cos(0.5c+0.02m)*(c==1 ? 1e-3 : 1.0) for c in 1:6,m in 1:length(mesh)]
    fbar=zeros(6,nz); zbar=zeros(nz); FFNOInversion._mesh_column_vjp!(fbar,zbar,model,a,1,1,@view(f[:,:,1,1]),fb)
    @test sum(out.*fb)≈sum(fbar.*fd) rtol=1e-4
    # a column already hotter than the corona keeps its temperature, hydrostatically
    hot=FFNOInversion._mesh_corona!(zeros(3,2),table,274.0,2e6,log(8e5),log(1e-12),5e5,2.4e5,[3e6,2.5e6])
    @test hot[1,:]≈[log(8e5),log(8e5)]
    @test hot[3,1]<hot[3,2]
end
