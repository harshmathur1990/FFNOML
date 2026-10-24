using HDF5
using LinearAlgebra

# Native vertical coordinates (z in metres or log10 tau500 in dex): axis rules, node interpolation on the native
# coordinate and its transpose, regularization length scales, configuration and input reading, x slabs.

"""A small native-z atmosphere file in storage order `order` (:python reverses all dimensions) with the vertical axis
stored `top_first` or bottom first, and an explicit x origin."""
function write_native_z_file(path;order=:julia,top_first=true,nz=7,nx=5,ny=4,x0=1.2e6,dx=48_000.0,dy=48_000.0)
    z=collect(range(2.0e6,-0.4e6;length=nz)); z[3]+=1.0e4                       # non-uniform, top first
    T=[6000.0+300.0*k^1.5+20.0*j+7.0*i for k in 1:nz,j in 1:ny,i in 1:nx]       # (z,y,x), non-symmetric in z
    fields=Dict("temperature"=>T,"vx"=>0.1 .* T,"vy"=>0.2 .* T,"vz"=>0.3 .* T)
    if !top_first
        z=reverse(z); for k in keys(fields); fields[k]=reverse(fields[k];dims=1); end
    end
    store(a)=order===:python ? permutedims(a,reverse(1:ndims(a))) : a
    h5open(path,"w") do f
        f["z_m"]=z; f["x_m"]=x0 .+ dx .* collect(0:nx-1); f["y_m"]=dy .* collect(0:ny-1)
        for (k,v) in fields; f[k]=store(v); end
    end
    (z=top_first ? z : reverse(z),T=top_first ? fields["temperature"] : reverse(fields["temperature"];dims=1))
end

function native_z_config(dir,file;order="julia",extra_atmos="",extra="",grid="",controls="")
    path=joinpath(dir,"cfg_$(hash((file,order,extra_atmos,extra,grid,controls))).toml")
    write(path,"""
mode = "forward"
[inputs]
initial_atmosphere_file = "$file"
[outputs]
synthesis_file = "$(joinpath(dir,"syn.h5"))"
atmosphere_file = "$(joinpath(dir,"atm.h5"))"
[atmosphere]
vertical_coordinate = "z"
storage_order = "$order"
pressure_top_pa = 0.1
$extra_atmos
[atmosphere.datasets]
z_m = "z_m"
x_m = "x_m"
y_m = "y_m"
temperature = "temperature"
vx = "vx"
vy = "vy"
vz = "vz"
$grid
[[regions]]
start_angstrom = 6563.0
step_angstrom = 0.1
count = 3
normalization = 1.0
psf_type = "none"
[physics]
redistribution = "non_prd"
$extra
$controls
""")
    path
end

@testset "native vertical coordinates" begin
    @testset "vertical axis rules" begin
        @test VerticalAxis(:z_m,[2.0,1.0,0.0]).values==[2.0,1.0,0.0]
        @test VerticalAxis(:log10tau500,[-4.0,-2.0,0.0]).kind===:log10tau500
        @test_throws ArgumentError VerticalAxis(:z_m,[0.0,1.0,2.0])            # bottom first
        @test_throws ArgumentError VerticalAxis(:log10tau500,[0.0,-2.0])        # bottom first
        @test_throws ArgumentError VerticalAxis(:z_m,[1.0])
        @test_throws ArgumentError VerticalAxis(:z_m,[2.0,NaN,0.0])
        @test_throws ArgumentError VerticalAxis(:height,[2.0,1.0])
        zg=Grid3D(VerticalAxis(:z_m,[2.0e6,1.0e6,0.0]),[0.0,1.0],[0.0])
        @test vertical_coordinate(zg)===:z_m && native_depth(zg)==[2.0e6,1.0e6,0.0] && nlevels(zg)==3
        @test z_axis(zg)==[2.0e6,1.0e6,0.0]
        err=try log_tau500_axis(zg); "" catch e; sprint(showerror,e) end
        @test occursin(":z_m",err) && occursin("[m]",err)
        # the legacy constructor builds a log-tau axis and never reverses fields it cannot see
        @test log_tau500_axis(Grid3D([-4.0,-2.0,0.0],[0.0,1.0],[0.0]))==[-4.0,-2.0,0.0]
        err=try Grid3D([0.0,-2.0,-4.0],[0.0,1.0],[0.0]); "" catch e; sprint(showerror,e) end
        @test occursin("reverse the axis and every field together",err)
        @test_throws ArgumentError Grid3D([-4.0,0.0],[1.0,0.0],[0.0])          # decreasing x
        @test_throws ArgumentError Grid3D([-4.0,0.0],[0.0,1.0],[2.0,1.0])      # decreasing y
    end

    @testset "coordinate consistency" begin
        g=Grid3D(VerticalAxis(:z_m,[2.0,1.0,0.0]),[0.0,1.0],[0.0,1.0])
        T=fill(6000.0,3,2,2); zfield=repeat(reshape([2.0,1.0,0.0],:,1,1),1,2,2)
        a=Atmosphere3D(g,T,0T,0T,0T,0T;z=zfield,logtau500=rand(3,2,2))
        @test check_coordinate_consistency(a)
        zbad=copy(zfield); zbad[2,1,2]=1.5
        @test_throws ArgumentError check_coordinate_consistency(Atmosphere3D(g,T,0T,0T,0T,0T;z=zbad))
    end

    @testset "node interpolation on the native coordinate" begin
        z=[1.0e6,7.0e5,3.0e5,1.0e5,0.0,-1.0e5,-2.0e5]                               # non-uniform, top first
        g=Grid3D(VerticalAxis(:z_m,z),[0.0,1.0,2.0],[0.0,1.0])
        nodes=[-2.0e5,0.0,3.0e5,7.0e5]                                               # bottom first is allowed
        f(zz)=2.0+3.0e-6*zz
        values=repeat(reshape(f.(nodes),:,1,1),1,2,2)
        expanded=expand_nodes(NodeField(values,nodes,:z_m),g)
        inside=findall(zz->nodes[1]<=zz<=nodes[end],z)
        @test maximum(abs.(expanded[inside,:,:].-reshape(f.(z[inside]),:,1,1)))<=1e-12
        @test all(expanded[1,:,:].==f(7.0e5)) && all(expanded[end,:,:].==f(-2.0e5))  # clamped outside the nodes
        # the same numbers as log-tau nodes are rejected on a z grid, and vice versa
        @test_throws ArgumentError expand_nodes(NodeField(values,nodes,:log10tau500),g)
        tg=Grid3D([-4.0,-2.0,0.0],[0.0,1.0,2.0],[0.0,1.0])
        @test_throws ArgumentError expand_nodes(NodeField(values,nodes,:z_m),tg)
        tile=FFNOInversion.tile_for_rank(3,2,0,1)
        @test_throws ArgumentError expand_nodes(NodeField(values,nodes,:log10tau500),g,tile)
        err=try expand_nodes(NodeField(values,nodes,:log10tau500),g); "" catch e; sprint(showerror,e) end
        @test occursin("[dex]",err) && occursin("[m]",err)
    end

    @testset "node expansion transpose (dot product)" for kind in (:z_m,:log10tau500), (ncx,ncy) in ((1,1),(3,1),(3,2))
        depth=kind===:z_m ? [9.0e5,6.0e5,2.5e5,1.0e5,0.0,-1.5e5] : [-5.0,-3.5,-2.0,-1.2,-0.3,0.4]
        nodes=kind===:z_m ? [-1.0e5,2.0e5,7.0e5] : [-4.0,-1.0,0.2]
        g=Grid3D(VerticalAxis(kind,depth),collect(0.0:4.0),collect(0.0:2.0))      # non-square 5×3
        spec=ControlMapSpec(:temperature,NodeField(fill(5000.0,3,ncx,ncy),nodes,kind);lower=1000,upper=1e5,scale=1000)
        v=randn(3,ncx,ncy); w=randn(6,5,3)
        Ev=expand_nodes(NodeField(v,nodes,kind),g)
        context=serial_context()
        for tile in (FFNOInversion.tile_for_rank(5,3,0,1),FFNOInversion.tile_for_rank(5,3,1,2;grid=(2,1)))
            wt=w[:,tile.xrange,tile.yrange]
            Etv=expand_nodes(NodeField(v,nodes,kind),g,tile)
            @test Etv==Ev[:,tile.xrange,tile.yrange]
            ETw=node_expansion_vjp(wt,spec,g,tile,context)
            @test abs(dot(Etv,wt)-dot(v,ETw))<=1e-12*max(abs(dot(Etv,wt)),1)
        end
    end

    @testset "regularization length scales" begin
        z=[1.0e6,6.0e5,3.0e5,0.0,-2.0e5]
        mk(zs,scale)=begin
            g=Grid3D(VerticalAxis(:z_m,zs),[0.0,1.0,2.0],[0.0,1.0])
            T=[5000.0+60.0*k^2+40.0*i for k in 1:5,i in 1:3,j in 1:2]
            a=Atmosphere3D(g,T,0T,0T,0T,0T)
            vspec=VerticalRegularizationSpec((4,0,0,0,0,0,0),1.0,ntuple(_->1.0,7);length_scale_m=scale)
            regularization_penalty(a,RegularizationSpec(vertical=vspec,horizontal=Dict{Symbol,Float64}(),
                scales=Dict(:temperature=>1000.0),horizontal_order=1),1.0,1.0).total
        end
        # changing the unit of z and of the length scale together leaves the penalty unchanged
        @test isapprox(mk(z,1.0e5),mk(1000 .* z,1.0e8);rtol=1e-12)
        @test_throws ArgumentError mk(z,nothing)
        # log-tau grids keep the historical penalty (length_scale_dex = 1)
        g=Grid3D([-4.0,-2.5,-1.0,0.0],[0.0,1.0],[0.0])
        T=[5000.0+300.0*k^2+10i for k in 1:4,i in 1:2,j in 1:1]
        a=Atmosphere3D(g,T,0T,0T,0T,0T)
        v1=VerticalRegularizationSpec((1,0,0,0,0,0,0),1.0,ntuple(_->1.0,7))
        manual=sum(abs2,diff(T./1000;dims=1)./reshape(diff([-4.0,-2.5,-1.0,0.0]),:,1,1))/(3*2)
        @test regularization_penalty(a,RegularizationSpec(vertical=v1,horizontal=Dict{Symbol,Float64}(),
            scales=Dict(:temperature=>1000.0),horizontal_order=1),1.0,1.0).total==manual
    end

    @testset "configuration rules" begin
        mktempdir() do dir
            file=joinpath(dir,"atm.h5"); write_native_z_file(file)
            cfg=load_config(native_z_config(dir,file))
            @test cfg.atmosphere.vertical_coordinate===:z_m && cfg.atmosphere.vertical_dataset=="z_m"
            @test cfg.synthesis.dx_m==48_000.0                                  # from the explicit x_m dataset
            @test occursin("vertical_coordinate=z_m",dry_run_summary(cfg)) && occursin("nodes_unit=m",dry_run_summary(cfg))
            bad(msg;kw...) = (err=try load_config(native_z_config(dir,file;kw...)); "" catch e; sprint(showerror,e) end;
                              @test occursin(msg,err))
            p=native_z_config(dir,file); txt=read(p,String)
            write(p,replace(txt,"z_m = \"z_m\""=>"z_m = \"z_m\"\nlogtau500 = \"z_m\""))
            @test_throws ArgumentError load_config(p)                            # z with a logtau dataset
            write(p,replace(txt,"x_m = \"x_m\"\n"=>""))
            @test_throws ArgumentError load_config(p)                            # z without x_m
            bad("differs from the spacing";grid="[grid]\ndx_m = 50000.0\ndy_m = 48000.0")
            inv="""[solver]\nmethod = "lbfgs"\n"""
            ctl(key)="""[[inversion.controls]]\nvariable = "temperature"\n$key = [-2.0e5, 0.0, 7.0e5]\nlower = 1000.0\nupper = 20000.0\nscale = 1000.0"""
            p2=native_z_config(dir,file;extra=inv,controls=ctl("z_nodes_m"))
            t2=replace(read(p2,String),"mode = \"forward\""=>"mode = \"inversion\"")
            t2=replace(t2,"[outputs]"=>"[observation.datasets]\nintensity = \"i\"\nsigma = \"s\"\nwavelength_weights = \"ww\"\nspatial_weights = \"sw\"\n[outputs]")
            t2=replace(t2,"[inputs]"=>"[inputs]\nobservation_file = \"$(joinpath(dir,"obs.h5"))\"")
            write(p2,t2); c2=load_config(p2)
            @test c2.controls[1].kind===:z_m && c2.controls[1].nodes==[-2.0e5,0.0,7.0e5]
            write(p2,replace(t2,"z_nodes_m"=>"log_tau_nodes")); @test_throws ArgumentError load_config(p2)
            write(p2,replace(t2,"[solver]"=>"[regularization.vertical]\ntypes = [1,0,0,0,0,0,0]\nregularize = 1.0\n[solver]"))
            err=try load_config(p2); "" catch e; sprint(showerror,e) end
            @test occursin("length_scale_m",err)
            write(p2,replace(t2,"[solver]"=>"[regularization.vertical]\ntypes = [1,0,0,0,0,0,0]\nregularize = 1.0\nlength_scale_m = 1.0e5\n[solver]"))
            @test load_config(p2).regularization.vertical.length_scale_m==1.0e5
            write(p2,replace(t2,"[solver]"=>"[parallel]\ndecomposition = \"x_slabs\"\n[solver]"))
            @test load_config(p2).parallel.decomposition===:x_slabs
        end
        # legacy example configurations still parse unchanged as log-tau inputs
        for example in ("forward.toml","inversion.toml")
            path=joinpath(@__DIR__,"..","..","examples","inversion_run",example)
            isfile(path) || continue
            c=load_config(path)
            @test c.atmosphere.vertical_coordinate===:log10tau500 && c.atmosphere.vertical_dataset=="logtau_500"
            all(cc->cc.kind===:log10tau500,c.controls) && @test true
        end
    end

    @testset "native-z input: storage order, vertical order, explicit x/y" begin
        mktempdir() do dir
            reads=map(((:julia,true),(:python,true),(:julia,false),(:python,false))) do (order,top)
                file=joinpath(dir,"atm_$(order)_$top.h5"); ref=write_native_z_file(file;order=order,top_first=top)
                cfg=load_config(native_z_config(dir,file;order=string(order)))
                (read_inversion_inputs(cfg).atmosphere,ref)
            end
            a0,ref=reads[1]
            @test native_depth(a0.grid)==ref.z && vertical_coordinate(a0.grid)===:z_m
            @test a0.grid.x[1]==1.2e6 && a0.grid.y[1]==0.0                     # physical origin preserved
            @test a0.temperature[:,3,2]==ref.T[:,2,3]                           # (z,x,y) from (z,y,x)
            @test a0.z!==nothing && check_coordinate_consistency(a0)
            for (a,_) in reads[2:end]
                @test native_depth(a.grid)==native_depth(a0.grid) && a.temperature==a0.temperature && a.z==a0.z
            end
            # non-uniform x is rejected with the reason
            file=joinpath(dir,"nonuniform.h5"); write_native_z_file(file)
            h5open(file,"r+") do f; x=read(f["x_m"]); x[3]+=10.0; delete_object(f,"x_m"); f["x_m"]=x; end
            err=try load_config(native_z_config(dir,file)); "" catch e; sprint(showerror,e) end
            @test occursin("uniformly spaced",err)
        end
    end

    @testset "output round trip and fingerprints" begin
        mktempdir() do dir
            file=joinpath(dir,"atm.h5"); write_native_z_file(file)
            cfg=load_config(native_z_config(dir,file))
            a=read_inversion_inputs(cfg).atmosphere
            a.logtau500=fill(-1.0,size(a.temperature))                          # a derived field next to the native axis
            result=InversionRunResult(nothing,SpectralCube(zeros(3,1,5,4),[1e-7,2e-7,3e-7],StokesSet(:I)),a,nothing,nothing,
                Dict{String,Any}())
            write_inversion_outputs(cfg,result)
            h5open(cfg.outputs.atmosphere_file,"r") do f
                @test read(f["z_m"])==native_depth(a.grid) && read(f["x_m"])==a.grid.x && read(f["y_m"])==a.grid.y
                @test haskey(f,"logtau500") && haskey(f,"z")
                @test read(attributes(f)["vertical_coordinate"])=="z_m" && read(attributes(f)["vertical_unit"])=="m"
                @test read(attributes(f)["vertical_order"])=="top_first"
            end
            # the reference ID and the control-layout signature depend on the coordinate kind, not only its numbers
            g1=Grid3D(VerticalAxis(:log10tau500,[-2.0,-1.0,0.0]),[0.0,1.0],[0.0])
            g2=Grid3D(VerticalAxis(:z_m,[0.0,-1.0,-2.0]),[0.0,1.0],[0.0])
            T=fill(5000.0,3,2,1)
            id(g)=FFNOInversion._atmosphere_reference_id(Atmosphere3D(g,T,0T,0T,0T,0T))
            @test id(g1)!=id(Grid3D(VerticalAxis(:log10tau500,[-2.0,-1.0,0.5]),[0.0,1.0],[0.0]))
            @test id(Grid3D(VerticalAxis(:z_m,[2.0,1.0,0.0]),[0.0,1.0],[0.0]))!=id(Grid3D(VerticalAxis(:log10tau500,[0.0,1.0,2.0]),[0.0,1.0],[0.0]))
            s(kind)=FFNOInversion._layout_signature(ControlMapLayout([ControlMapSpec(:temperature,
                NodeField(fill(5000.0,2,1,1),[0.0,1.0],kind);lower=1000,upper=1e4,scale=1000)]))
            @test s(:z_m)!=s(:log10tau500)
        end
    end

    @testset "x slabs" begin
        for nx in (8,11), nranks in (1,2,3,4)
            nx ÷ nranks >= 2 || continue
            grid=FFNOInversion.process_grid(nranks,nx,7,:x_slabs)
            @test grid==(nranks,1)
            tiles=[FFNOInversion.tile_for_rank(nx,7,r,nranks;grid=grid) for r in 0:nranks-1]
            @test sort(vcat(collect.(getfield.(tiles,:xrange))...))==collect(1:nx)
            @test all(t->t.yrange==1:7,tiles)
        end
        @test_throws ArgumentError FFNOInversion.process_grid(3,5,7,:x_slabs;min_columns=2)
        @test FFNOInversion.process_grid(4,8,8,:cartesian_2d)==FFNOInversion.process_grid(4,8,8)
        ctx=serial_context(options=ParallelOptions(decomposition=:x_slabs))
        @test FFNOInversion.local_tile(ctx,9,4).yrange==1:4
    end
end
