using HDF5
include("test_equilibrium.jl")

# The HSEMHS equilibrium backend (column modes, HSEMHS integration Phase 3): configuration, the adapter against
# direct HSEMHS calls, coordinate semantics, physics, failures, provenance and lifecycle. Uses the real HSEMHS EOS
# table (HSEMHS_EOS_TABLE, default the MHS checkout's cache); multi-rank checks are in mpi_equilibrium_worker.jl.

const HSEMHS_TABLE=get(ENV,"HSEMHS_EOS_TABLE","/mn/stornext/d9/data/harshm/MHS/HSEMHS/cache/eos_table_fullz.h5")
const H=FFNOInversion.HSEMHS

"""A non-square atmosphere with T varying independently along z/τ, x and y, optional B."""
function equilibrium_atmosphere(kind;nz=40,nx=12,ny=7,magnetic=false)
    axis=kind===:z_m ? collect(range(1.5e6,-4.0e5;length=nz)) : collect(range(-7.0,1.4;length=nz))
    grid=Grid3D(VerticalAxis(kind,axis),4.8e4 .* collect(0:nx-1),4.8e4 .* collect(0:ny-1))
    T=[(kind===:z_m ? 6000.0+1500.0*tanh(axis[k]/4.0e5) :
        4500.0+1500.0*exp10(min(axis[k],0.5)/2)) + 25.0*i + 9.0*j for k in 1:nz,i in 1:nx,j in 1:ny]
    zero3=zeros(nz,nx,ny)
    B=magnetic ? MagneticField3D([1e-2*cos(0.3i) for k in 1:nz,i in 1:nx,j in 1:ny],
        [5e-3*sin(0.4j) for k in 1:nz,i in 1:nx,j in 1:ny],fill(2e-2,nz,nx,ny)) : nothing
    Atmosphere3D(grid,T,copy(zero3),copy(zero3),copy(zero3),copy(zero3);magnetic_field=B)
end

column_options(kind)=kind===:z_m ? H.ColumnZOptions(anchor=:top,P_ref=0.3) : H.ColumnTauOptions(align="ptot")
equilibrium_config(kind)=EquilibriumConfig("HSEMHS",:column_hse,kind===:z_m ? :column_hse_z : :column_hse_ltau,274.0,
    HSEMHS_TABLE,:reject_trial,column_options(kind),nothing)

if !isfile(HSEMHS_TABLE)
    @info "HSEMHS EOS table not found; skipping the HSEMHS equilibrium tests" HSEMHS_TABLE
else
@testset "HSEMHS equilibrium backend (single rank)" begin
    context=initialize_parallel(options=ParallelOptions(threads_per_rank=Threads.nthreads()))   # managed single-rank MPI
    @test !context.enabled && FFNOInversion.MPI.Initialized() && FFNOInversion.MPI.Query_thread()>=FFNOInversion.MPI.THREAD_FUNNELED
    table=H.load_table(HSEMHS_TABLE)

    @testset "configuration" begin
        raw(model;extra...)=Dict{String,Any}("backend"=>"HSEMHS","model"=>model,"eos_table_file"=>"t.h5",
            (string(k)=>v for (k,v) in extra)...)
        cz=Dict("P_ref_pa"=>1e5)
        @test FFNOInversion._equilibrium_config(raw("column_hse";column_z=cz),:z_m,"/r").mode===:column_hse_z
        @test FFNOInversion._equilibrium_config(raw("column_hse"),:log10tau500,"/r").mode===:column_hse_ltau
        @test FFNOInversion._equilibrium_config(raw("column_hse"),:log10tau500,"/r").eos_table_file=="/r/t.h5"
        bad(msg,args...) = (err=try FFNOInversion._equilibrium_config(args...); "" catch e; sprint(showerror,e) end;
                            @test occursin(msg,err))
        bad("Phase 4",raw("mhs"),:log10tau500,"/r")
        bad("only equilibrium backend",merge(raw("column_hse"),Dict("backend"=>"legacy")),:log10tau500,"/r")
        bad("P_ref_pa is required",raw("column_hse"),:z_m,"/r")
        bad("does not apply to a native log-tau run",raw("column_hse";column_z=cz),:log10tau500,"/r")
        bad("does not apply to a native-z run",raw("column_hse";column_z=cz,column_ltau=Dict("align"=>"none")),:z_m,"/r")
        bad("unknown key",merge(raw("column_hse"),Dict("pressure_top_pa"=>0.1)),:log10tau500,"/r")
        bad("unknown key",raw("column_hse";column_ltau=Dict("cap"=>1.0)),:log10tau500,"/r")
        bad("align",raw("column_hse";column_ltau=Dict("align"=>"median")),:log10tau500,"/r")
        bad("anchor",raw("column_hse";column_z=Dict("P_ref_pa"=>1e5,"anchor"=>"middle")),:z_m,"/r")
        # pressure_top in [atmosphere] and cartesian_2d with MPI are rejected at load_config
        mktempdir() do dir
            path=joinpath(dir,"c.toml")
            base="""mode = "forward"\n[inputs]\ninitial_atmosphere_file = "a.h5"\n[outputs]\nsynthesis_file = "s.h5"\natmosphere_file = "o.h5"
[equilibrium]\nbackend = "HSEMHS"\nmodel = "column_hse"\neos_table_file = "t.h5"\n[atmosphere]\nPTOP\n[atmosphere.datasets]
logtau500 = "l"\ntemperature = "t"\nvx = "x"\nvy = "y"\nvz = "z"\n[grid]\ndx_m = 1.0\ndy_m = 1.0
[[regions]]\nstart_angstrom = 6563.0\nstep_angstrom = 0.1\ncount = 2\nnormalization = 1.0\npsf_type = "none"
[physics]\nredistribution = "non_prd"\n[parallel]\nenabled = true\ndecomposition = "DECOMP"\n"""
            write(path,replace(base,"PTOP"=>"pressure_top_pa = 0.1","DECOMP"=>"x_slabs"))
            err=try load_config(path); "" catch e; sprint(showerror,e) end
            @test occursin("set by HSEMHS",err)
            write(path,replace(base,"PTOP"=>"","DECOMP"=>"cartesian_2d"))
            err=try load_config(path); "" catch e; sprint(showerror,e) end
            @test occursin("x_slabs",err)
            write(path,replace(base,"PTOP"=>"","DECOMP"=>"x_slabs"))
            @test load_config(path).equilibrium.mode===:column_hse_ltau
            write(path,replace(replace(base,"PTOP"=>"","DECOMP"=>"x_slabs"),r"\[equilibrium\][^\[]*"=>""))
            err=try load_config(path); "" catch e; sprint(showerror,e) end
            @test occursin("[equilibrium]",err)
        end
    end

    @testset "adapter equals direct HSEMHS ($kind)" for kind in (:z_m,:log10tau500)
        atmosphere=equilibrium_atmosphere(kind;magnetic=kind===:log10tau500)
        distributed=distribute_atmosphere(Float64,atmosphere,context)
        backend=prepare_equilibrium(equilibrium_config(kind),distributed,context)
        outcome=reconstruct_equilibrium!(backend,distributed,context)
        @test outcome.valid && outcome.reason===:converged && outcome.effective_mode===(kind===:z_m ? :column_hse_z : :column_hse_ltau)
        a=distributed.local_atmosphere
        # hand-packed direct call on (vertical, y, x)
        ws=H.prepare_embedded(FFNOInversion.MPI.COMM_SELF,kind===:z_m ? :column_hse_z : :column_hse_ltau;
            axis=native_depth(a.grid),x=a.grid.x,y=a.grid.y,eos=table,g=274.0,column=column_options(kind))
        lnT=permutedims(log.(atmosphere.temperature),(1,3,2))
        B=atmosphere.magnetic_field
        res=B===nothing ? H.solve_embedded!(ws,lnT) : H.solve_embedded!(ws,lnT,permutedims(B.Bx,(1,3,2)),
            permutedims(B.By,(1,3,2)),permutedims(B.Bz,(1,3,2)))
        for (mine,theirs) in ((a.pgas,res.Pgas),(a.rho,res.rho),(a.ne,res.ne),(a.z,res.z),(a.logtau500,res.logtau500))
            @test maximum(abs.(mine .- permutedims(theirs,(1,3,2))) ./ abs.(mine))<=1e-13
        end
        # coordinate semantics: the native field is the axis, the derived one varies by column
        @test check_coordinate_consistency(a)
        derived=kind===:z_m ? a.logtau500 : a.z
        @test maximum(abs.(derived[20,:,:] .- derived[20,1,1]))>0
        @test all(diff(a.z;dims=1).<0) && all(diff(a.logtau500;dims=1).>0)
        # T, B, velocities untouched
        @test a.temperature==atmosphere.temperature
        B===nothing || @test a.magnetic_field.Bx==atmosphere.magnetic_field.Bx
        H.free_embedded!(ws); close_equilibrium!(backend)
        @test backend.closed
    end

    @testset "hydrostatic physics (zero B, height grid)" begin
        atmosphere=equilibrium_atmosphere(:z_m;nz=400,nx=4,ny=3)
        distributed=distribute_atmosphere(Float64,atmosphere,context)
        backend=prepare_equilibrium(equilibrium_config(:z_m),distributed,context)
        @test reconstruct_equilibrium!(backend,distributed,context).valid
        a=distributed.local_atmosphere; z=native_depth(a.grid)
        # centred dP/dz + ρg relative to ρg is small (RK4 column integration with the same table)
        r=[((a.pgas[k-1,i,j]-a.pgas[k+1,i,j])/(z[k-1]-z[k+1])+a.rho[k,i,j]*274.0)/(a.rho[k,i,j]*274.0)
           for k in 2:399,i in 1:4,j in 1:3]
        @test maximum(abs,r)<2e-3
        @test all(a.pgas[1,:,:].≈0.3)                                       # top anchor (HSEMHS P_ref)
        close_equilibrium!(backend)
    end

    @testset "vertical order of the input file" begin
        mktempdir() do dir
            nz,nx,ny=30,5,4
            z=collect(range(1.2e6,-3.0e5;length=nz))
            T=[5000.0+2000.0*exp(-max(z[k],0)/4e5)+10.0*i+3.0*j for k in 1:nz,j in 1:ny,i in 1:nx]   # (z,y,x)
            pgas=Dict{Bool,Array{Float64,3}}()
            for top in (true,false)
                f=joinpath(dir,"a_$top.h5")
                h5open(f,"w") do h
                    h["z_m"]=top ? z : reverse(z); h["x_m"]=4.8e4 .* collect(0:nx-1); h["y_m"]=4.8e4 .* collect(0:ny-1)
                    for name in ("temperature","vx","vy","vz")
                        v=name=="temperature" ? T : zero(T); h[name]=top ? v : reverse(v;dims=1)
                    end
                end
                cfgpath=joinpath(dir,"c_$top.toml")
                write(cfgpath,"""mode = "forward"\n[inputs]\ninitial_atmosphere_file = "$f"\n[outputs]\nsynthesis_file = "s.h5"\natmosphere_file = "o.h5"
[equilibrium]\nbackend = "HSEMHS"\nmodel = "column_hse"\neos_table_file = "$HSEMHS_TABLE"\n[equilibrium.column_z]\nanchor = "top"\nP_ref_pa = 0.3
[atmosphere]\nvertical_coordinate = "z"\n[atmosphere.datasets]\nz_m = "z_m"\nx_m = "x_m"\ny_m = "y_m"\ntemperature = "temperature"\nvx = "vx"\nvy = "vy"\nvz = "vz"
[[regions]]\nstart_angstrom = 6563.0\nstep_angstrom = 0.1\ncount = 2\nnormalization = 1.0\npsf_type = "none"\n[physics]\nredistribution = "non_prd"\n""")
                cfg=load_config(cfgpath)
                d=distribute_atmosphere(Float64,read_inversion_inputs(cfg).atmosphere,context)
                b=prepare_equilibrium(cfg,d,context); @test reconstruct_equilibrium!(b,d,context).valid
                pgas[top]=copy(d.local_atmosphere.pgas); close_equilibrium!(b)
            end
            @test pgas[true]==pgas[false]
        end
    end

    @testset "failures and provenance" begin
        atmosphere=equilibrium_atmosphere(:log10tau500)
        distributed=distribute_atmosphere(Float64,atmosphere,context)
        backend=prepare_equilibrium(equilibrium_config(:log10tau500),distributed,context)
        reference=reconstruct_equilibrium!(backend,distributed,context); @test reference.valid
        pgas=copy(distributed.local_atmosphere.pgas)
        distributed.local_atmosphere.temperature[:,3,2].=5.0e7                     # far outside the table
        bad=reconstruct_equilibrium!(backend,distributed,context)
        @test !bad.valid && bad.reason===:eos_out_of_envelope
        @test distributed.local_atmosphere.pgas==pgas                              # fields untouched on failure
        distributed.local_atmosphere.temperature.=atmosphere.temperature
        @test reconstruct_equilibrium!(backend,distributed,context).valid
        @test backend.provenance["pf_sha256"]==bytes2hex(open(FFNOInversion.SHA.sha256,
            joinpath(@__DIR__,"..","..","scripts","pf_Kurucz.input")))
        @test backend.provenance["mode"]=="column-hse-ltau" && haskey(backend.provenance,"eos_table_sha256")
        close_equilibrium!(backend)
        # synthesis EOS compatibility: same partition functions accepted, different rejected
        pf=joinpath(@__DIR__,"..","..","scripts","pf_Kurucz.input")
        ok=prepare_equilibrium(equilibrium_config(:log10tau500),distributed,context;synthesis_eos=WittmannEOS(pf,pf))
        @test ok.provenance["synthesis_pf_sha256"]==ok.provenance["pf_sha256"]; close_equilibrium!(ok)
        mktempdir() do dir
            other=joinpath(dir,"pf.input"); write(other,read(pf,String)*"\n# changed\n")
            err=try prepare_equilibrium(equilibrium_config(:log10tau500),distributed,context;
                synthesis_eos=WittmannEOS(other,other)); "" catch e; sprint(showerror,e) end
            @test occursin("partition functions",err)
        end
        # a missing table and a grid of the wrong kind are rejected before any solver state exists
        missing=EquilibriumConfig("HSEMHS",:column_hse,:column_hse_ltau,274.0,"/nonexistent/t.h5",:reject_trial,
            column_options(:log10tau500),nothing)
        @test_throws ArgumentError prepare_equilibrium(missing,distributed,context)
        @test_throws ArgumentError prepare_equilibrium(equilibrium_config(:z_m),distributed,context)
    end

    @testset "forward model with the HSEMHS backend" begin
        atmosphere=equilibrium_atmosphere(:log10tau500;nx=6,ny=5)
        distributed=distribute_atmosphere(Float64,atmosphere,context)
        backend=prepare_equilibrium(equilibrium_config(:log10tau500),distributed,context)
        model=HybridForwardModel(LocalDistributedPopulationModel(MockPopulationModel(1e10),1),NonPRD(),
            MockIntensitySynthesizer(),IdentityObservation(),backend,CapabilityManifest())
        workspace=HybridForwardWorkspace(Float64,distributed,[656.2e-9,656.3e-9],StokesSet(:I),1)
        result=forward!(workspace,model,distributed,context)
        @test result.valid && all(isfinite,result.spectrum.data) && result.timings.equilibrium_seconds>0
        previous=copy(workspace.output.data)
        distributed.local_atmosphere.temperature[:,1,1].=5.0e7
        failed=forward!(workspace,model,distributed,context)
        @test !failed.valid && workspace.output.data==previous && failed.timings.populations_seconds==0
        close_equilibrium!(backend)
    end

    @testset "lifecycle: 20 prepare/reconstruct/close cycles" begin
        atmosphere=equilibrium_atmosphere(:log10tau500;nx=4,ny=3,nz=20)
        distributed=distribute_atmosphere(Float64,atmosphere,context)
        ok=true
        for _ in 1:20
            b=prepare_equilibrium(equilibrium_config(:log10tau500),distributed,context)
            ok&=reconstruct_equilibrium!(b,distributed,context).valid
            close_equilibrium!(b); close_equilibrium!(b)                          # idempotent
        end
        @test ok
    end
end
end
