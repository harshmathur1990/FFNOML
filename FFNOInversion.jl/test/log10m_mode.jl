using HDF5
using LinearAlgebra
import Muspel
@isdefined(Phase6TwoLevelPopulation) || include("phase6_production_gradient.jl")

# The native log10 column-mass grid: axis rules, configuration and input reading, the formal solution on the mass axis
# and its VJP, and the column-hse-logm HSEMHS mode. Multi-rank checks are in mpi_equilibrium_worker.jl.

const LOGM_TABLE=get(ENV,"HSEMHS_EOS_TABLE","/mn/stornext/d9/data/harshm/MHS/HSEMHS/cache/eos_table_fullz.h5")

function logm_config_text(file;datasets="log10m = \"lm\"",equilibrium="",coordinate="log10m",mode="forward",extra="")
    """mode = "$mode"\n[inputs]\ninitial_atmosphere_file = "$file"\n[outputs]\nsynthesis_file = "s.h5"\natmosphere_file = "o.h5"
[equilibrium]\nbackend = "HSEMHS"\nmodel = "column_hse"\neos_table_file = "t.h5"\n$equilibrium
[atmosphere]\nvertical_coordinate = "$coordinate"\n[atmosphere.datasets]\n$datasets\ntemperature = "t"\nvx = "vx"\nvy = "vy"\nvz = "vz"
[grid]\ndx_m = 48000.0\ndy_m = 48000.0\n[[regions]]\nstart_angstrom = 6563.0\nstep_angstrom = 0.1\ncount = 2\nnormalization = 1.0\npsf_type = "none"
[physics]\nredistribution = "non_prd"\n$extra\n"""
end

@testset "Native log10 column mass" begin
    @testset "axis" begin
        axis=VerticalAxis(:log10m,[-3.0,0.0,2.0])
        @test :log10m in VERTICAL_KINDS && occursin("kg",vertical_unit(:log10m))
        @test_throws ArgumentError VerticalAxis(:log10m,[2.0,0.0,-3.0])
        grid=Grid3D(axis,[0.0,1.0],[0.0])
        @test log10m_axis(grid)==[-3.0,0.0,2.0] && vertical_coordinate(grid)===:log10m
        @test_throws ArgumentError log_tau500_axis(grid)
        @test_throws ArgumentError z_axis(grid)
        @test FFNOInversion.view_columns(grid)=="rays"
        shape=(3,2,1); zero3()=zeros(shape)
        # logtau500 and z are both derived here, so they may vary by column
        a=Atmosphere3D(grid,fill(6000.0,shape),zero3(),zero3(),zero3(),zero3();logtau500=[Float64(k+i) for k in 1:3,i in 1:2,j in 1:1])
        @test check_coordinate_consistency(a)
        @test FFNOInversion._mesh_active(nothing,a)
        field=NodeField(ones(2,1,1),[-3.0,2.0],:log10m)
        @test_throws ArgumentError expand_nodes(field,Grid3D(VerticalAxis(:log10tau500,[-3.0,0.0,2.0]),[0.0,1.0],[0.0]))
    end

    @testset "configuration and input" begin
        mktempdir() do dir
            nz,nx,ny=5,3,2; lm=collect(range(-3.0,2.0;length=nz))
            T=[6000.0+100.0*k+7.0*i+3.0*j for k in 1:nz,j in 1:ny,i in 1:nx]         # (z,y,x)
            write_file(path;axis=lm,fields=T,units=nothing)=h5open(path,"w") do f
                f["lm"]=axis; units===nothing || (HDF5.attributes(f["lm"])["units"]=units)
                f["t"]=fields; for name in ("vx","vy","vz"); f[name]=0.1 .* fields; end
            end
            config(text)=(p=joinpath(dir,"c$(hash(text)).toml"); write(p,text); load_config(p))
            file=joinpath(dir,"a.h5"); write_file(file)
            cfg=config(logm_config_text(file))
            @test cfg.atmosphere.vertical_coordinate===:log10m && cfg.equilibrium.mode===:column_hse_logm
            @test cfg.population_mesh!==nothing                                       # on by default, as for log tau
            a=read_inversion_inputs(cfg).atmosphere
            @test log10m_axis(a.grid)==lm && a.temperature[:,1,1]==T[:,1,1] && a.z===nothing
            # bottom-first input is normalized, cgs is converted, an unknown unit is an error
            flipped=joinpath(dir,"b.h5"); write_file(flipped;axis=reverse(lm),fields=reverse(T;dims=1))
            b=read_inversion_inputs(config(logm_config_text(flipped))).atmosphere
            @test log10m_axis(b.grid)==lm && b.temperature==a.temperature
            cgs=joinpath(dir,"cgs.h5"); write_file(cgs;units="g cm-2")
            @test log10m_axis(read_inversion_inputs(config(logm_config_text(cgs))).atmosphere.grid)≈lm.+1
            odd=joinpath(dir,"odd.h5"); write_file(odd;units="Pa")
            @test_throws ArgumentError read_inversion_inputs(config(logm_config_text(odd)))
            # the [equilibrium] subsections and the datasets of other coordinates are rejected
            bad(msg,text)=(err=try config(text); "" catch e; sprint(showerror,e) end; @test occursin(msg,err))
            bad("does not apply to a native log-m run",logm_config_text(file;equilibrium="[equilibrium.column_z]\nP_ref_pa = 1.0"))
            bad("does not apply to a native log-m run",logm_config_text(file;equilibrium="[equilibrium.column_ltau]\nalign = \"none\""))
            bad("does not apply to a native log-tau run",logm_config_text(file;datasets="logtau500 = \"lm\"",coordinate="log10tau500",equilibrium="[equilibrium.column_logm]\nalign = \"none\""))
            err=try FFNOInversion._equilibrium_config(Dict{String,Any}("model"=>"column_hse","eos_table_file"=>"t.h5","column_z"=>Dict("P_ref_pa"=>1.0),"column_logm"=>Dict("align"=>"none")),:z_m,"/r"); "" catch e; sprint(showerror,e) end
            @test occursin("does not apply to a native-z run",err)
            bad("requires datasets.log10m",logm_config_text(file;datasets=""))
            bad("logtau500 is not allowed",logm_config_text(file;datasets="log10m = \"lm\"\nlogtau500 = \"lm\""))
            bad("log10m is not allowed",logm_config_text(file;datasets="log10m = \"lm\"\nlogtau500 = \"lm\"",coordinate="log10tau500"))
            bad("z_m is not allowed",logm_config_text(file;datasets="log10m = \"lm\"\nz_m = \"lm\""))
            bad("must be",logm_config_text(file;coordinate="mass"))
            @test occursin("align",try config(logm_config_text(file;equilibrium="[equilibrium.column_logm]\nalign = \"median\"")); "" catch e; sprint(showerror,e) end)
            @test config(logm_config_text(file;equilibrium="[equilibrium.column_logm]\nalign = \"none\"")).equilibrium.column.align=="none"
            # control nodes and the regularization length use the dex of the axis
            raw(key)=Dict{String,Any}("controls"=>Any[Dict{String,Any}("variable"=>"temperature",key=>[-3.0,2.0],"lower"=>3000.0,"upper"=>1.0e5,"scale"=>100.0)])
            controls=FFNOInversion._controls(raw("log_m_nodes"),:log10m)
            @test controls[1].kind===:log10m && controls[1].nodes==[-3.0,2.0]
            for other in ("log_tau_nodes","z_nodes_m")
                err=try FFNOInversion._controls(raw(other),:log10m); "" catch e; sprint(showerror,e) end
                @test occursin("not valid",err)
            end
            spec=VerticalRegularizationSpec((1,0,0,0,0,0,0),1.0,ntuple(_->1.0,7);length_scale_dex=2.0)
            @test FFNOInversion._vertical_spacing_axis(a.grid,spec)≈lm./2
        end
    end

    T=Float64; nx,ny=2,1
    column(v,nz)=repeat(reshape(T.(v),nz,1,1),1,nx,ny)
    function mass_atmosphere(logm,rho,z)
        nz=length(logm); shape=(nz,nx,ny)
        Atmosphere3D(Grid3D(VerticalAxis(:log10m,T.(logm)),T[0,1],T[0]),fill(6000.0,shape),zeros(shape),zeros(shape),zeros(shape),
            zeros(shape);z=column(z,nz),rho=column(rho,nz))
    end
    solve(atmos,chi,eta)=begin
        cube=SpectralCube(zeros(T,1,1,nx,ny),[500e-9],StokesSet(:I))
        synthesize!(cube,MixedIntensitySynthesizer([OpacityContributor(TabulatedOpacityModel(chi,eta))]),NonPRD(),atmos)
        cube.data[1,1,1,1]
    end

    @testset "formal solution on the mass axis" begin
        # χ = κρ and S = a + bτ with τ = κm: the emergent intensity is a + b, and the step optical depths are those of
        # the log-tau integration of the same column
        logm=collect(range(-4,3,length=71)); kappa=0.1; m=exp10.(logm); rho=1e-7 .* m.^0.7
        tau=kappa.*m; source=1 .+ 3 .* tau
        z=-cumsum(vcat(0,diff(m)./((rho[1:end-1].+rho[2:end])./2)))
        atmos=mass_atmosphere(logm,rho,z)
        @test FFNOInversion._mass_integration(atmos) && !FFNOInversion._tau_integration(atmos)
        chi=reshape(kappa.*rho,:,1); eta=reshape(kappa.*rho.*source,:,1)
        on_mass=solve(atmos,chi,eta)
        @test on_mass≈4 rtol=1e-2
        shape=(length(m),nx,ny)
        chi500=kappa.*rho
        tau_atmosphere=Atmosphere3D(Grid3D(T.(log10.(tau)),T[0,1],T[0]),fill(6000.0,shape),zeros(shape),zeros(shape),zeros(shape),
            zeros(shape);z=column(z,length(m)),logtau500=column(log10.(tau),length(m)),chi500=column(chi500,length(m)))
        @test on_mass≈solve(tau_atmosphere,chi,eta) rtol=1e-12
        # no rho: the height path, or an error without heights
        no_rho=Atmosphere3D(atmos.grid,atmos.temperature,atmos.vx,atmos.vy,atmos.vz,atmos.vturb;z=atmos.z)
        @test !FFNOInversion._axis_integration(no_rho)
        # the axis is along the line of sight: the view changes nothing in the formal solution
        slanted=Atmosphere3D(Grid3D(VerticalAxis(:log10m,T.(logm)),T[0,1],T[0],ViewGeometry(mu=0.5)),atmos.temperature,atmos.vx,atmos.vy,atmos.vz,
            atmos.vturb;z=atmos.z,rho=atmos.rho)
        @test solve(slanted,chi,eta)==on_mass

        # Transition region in one grid step: hot gas of tiny χ/ρ above cool gas; the step's Δm stays optically thin
        logm=collect(range(-4.0,3.0,length=71)); m=exp10.(logm); hot=logm.<=-3.85
        rho=ifelse.(hot,1e-13,1e-7 .* m.^0.7); opacity=ifelse.(hot,1e-6,0.1); source=ifelse.(hot,1e3,1 .+ 0.1 .* m)
        step=diff(m)./((rho[1:end-1].+rho[2:end])./2); step[findlast(hot)]=9.6e6; z=-cumsum(vcat(0,step))
        @test solve(mass_atmosphere(logm,rho,z),reshape(opacity.*rho,:,1),reshape(opacity.*rho.*source,:,1))<2.5
    end

    @testset "VJP on the mass axis" begin
        logm=collect(range(-3,2,length=13)); nz=length(logm); m=exp10.(logm)
        rho=1e-7 .* m.^0.8; z=-cumsum(vcat(0,diff(m)./((rho[1:end-1].+rho[2:end])./2)))
        atmos=mass_atmosphere(logm,rho,z); atmos.rho[:,2,1].*=1.3
        wave=collect(range(656.20e-9,656.36e-9,length=4)); nλ=length(wave)
        pops=fill(1e12,nz,nx,ny,3); pops[:,:,:,3].=1e9
        halpha=FFNOTransition(:H,656.28e-9,2,3,0.64,1.6735575e-27,1e8)
        tabulated=TabulatedOpacityModel(repeat(reshape(0.1 .* rho,:,1),1,nλ),repeat(reshape(0.2 .* rho .* (1 .+ m),:,1),1,nλ))
        synth=MixedIntensitySynthesizer([OpacityContributor(tabulated),OpacityContributor(halpha,pops)])
        bar=reshape(cos.(1:nλ*nx*ny),nλ,1,nx,ny)
        objective()=begin
            cube=SpectralCube(zeros(T,nλ,1,nx,ny),wave,StokesSet(:I)); synthesize!(cube,synth,NonPRD(),atmos)
            sum(bar.*cube.data)
        end
        cube=SpectralCube(zeros(T,nλ,1,nx,ny),wave,StokesSet(:I)); synthesize!(cube,synth,NonPRD(),atmos)
        atmosphere_bar=FFNOInversion.AtmosphereCotangent(atmos)
        population_bar=FFNOInversion.synthesis_vjp!(atmosphere_bar,synth,NonPRD(),atmos,pops,cube,bar)
        @test all(iszero,atmosphere_bar.z) && all(iszero,atmosphere_bar.chi500)
        for (k,x) in ((3,1),(7,2),(11,1))
            h=1e-5*atmos.rho[k,x,1]; c=atmos.rho[k,x,1]
            atmos.rho[k,x,1]=c+h; fp=objective(); atmos.rho[k,x,1]=c-h; fm=objective(); atmos.rho[k,x,1]=c
            @test atmosphere_bar.rho[k,x,1]≈(fp-fm)/(2h) rtol=1e-4
            h=1e-5*pops[k,x,1,2]; c=pops[k,x,1,2]
            pops[k,x,1,2]=c+h; fp=objective(); pops[k,x,1,2]=c-h; fm=objective(); pops[k,x,1,2]=c
            @test population_bar[k,x,1,2]≈(fp-fm)/(2h) rtol=1e-4
        end
    end

    if !isfile(LOGM_TABLE)
        @info "HSEMHS EOS table not found; skipping the column-hse-logm tests" LOGM_TABLE
    else
    @testset "column-hse-logm through the HSEMHS backend" begin
        H=FFNOInversion.HSEMHS; g=274.0
        context=initialize_parallel(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
        function logm_atmosphere(;nz=60,nx=8,ny=5,view=ViewGeometry(),magnetic=true,ne=nothing)
            axis=collect(range(-3.5,3.0;length=nz))
            grid=Grid3D(VerticalAxis(:log10m,axis),4.8e4 .* collect(0:nx-1),4.8e4 .* collect(0:ny-1),view)
            temperature=[4600.0+1400.0*exp10(min(axis[k],2.0)/8)+25.0*i+9.0*j for k in 1:nz,i in 1:nx,j in 1:ny]
            B=magnetic ? MagneticField3D([1e-2*cos(0.3i) for k in 1:nz,i in 1:nx,j in 1:ny],[5e-3*sin(0.4j) for k in 1:nz,i in 1:nx,j in 1:ny],fill(2e-2,nz,nx,ny)) : nothing
            zero3=zeros(nz,nx,ny)
            Atmosphere3D(grid,temperature,copy(zero3),copy(zero3),copy(zero3),copy(zero3);magnetic_field=B,ne=ne)
        end
        config(align="ptot")=EquilibriumConfig("HSEMHS",:column_hse,:column_hse_logm,g,LOGM_TABLE,:reject_trial,H.ColumnMassOptions(align=align),nothing)

        @testset "adapter equals direct HSEMHS, P = g m" begin
            atmosphere=logm_atmosphere()
            distributed=distribute_atmosphere(Float64,atmosphere,context)
            backend=prepare_equilibrium(config(),distributed,context)
            @test backend.provenance["mode"]=="column-hse-logm"
            outcome=reconstruct_equilibrium!(backend,distributed,context)
            @test outcome.valid && outcome.effective_mode===:column_hse_logm
            a=distributed.local_atmosphere; m=exp10.(native_depth(a.grid))
            ws=H.prepare_embedded(FFNOInversion.MPI.COMM_SELF,:column_hse_logm;axis=native_depth(a.grid),x=a.grid.x,y=a.grid.y,
                eos=LOGM_TABLE,g=g,column=H.ColumnMassOptions(align="ptot"))
            B=atmosphere.magnetic_field; zyx(f)=permutedims(f,(1,3,2))
            res=H.solve_embedded!(ws,zyx(log.(atmosphere.temperature)),zyx(B.Bx),zyx(B.By),zyx(B.Bz))
            for (mine,theirs) in ((a.pgas,res.Pgas),(a.rho,res.rho),(a.ne,res.ne),(a.z,res.z),(a.logtau500,res.logtau500),(a.chi500,res.chi500))
                @test maximum(abs.(mine .- zyx(theirs)) ./ abs.(mine))<=1e-13
            end
            H.free_embedded!(ws)
            @test maximum(abs.(a.pgas ./ reshape(g .* m,:,1,1) .- 1))<1e-10
            @test all(diff(a.z;dims=1).<0) && all(diff(a.logtau500;dims=1).>0)
            @test maximum(abs.(a.logtau500[30,:,:] .- a.logtau500[30,1,1]))>0                  # derived, varies by column
            @test check_coordinate_consistency(a)
            @test a.temperature==atmosphere.temperature && a.magnetic_field.Bx==B.Bx
            # heights agree with dz = -dm/ρ, up to T being linear in the axis within a step
            dz=-diff(a.z[:,3,2]); dm=diff(m); ρ=a.rho[:,3,2]
            @test maximum(abs.(dz./(dm./((ρ[1:end-1].+ρ[2:end])./2)).-1))<0.02
            close_equilibrium!(backend)
        end

        @testset "line of sight: P = μ g m" begin
            atmosphere=logm_atmosphere(;view=ViewGeometry(mu=0.5),magnetic=false)
            distributed=distribute_atmosphere(Float64,atmosphere,context)
            backend=prepare_equilibrium(config("none"),distributed,context)
            @test reconstruct_equilibrium!(backend,distributed,context).valid
            a=distributed.local_atmosphere
            @test maximum(abs.(a.pgas ./ reshape(0.5 .* g .* exp10.(native_depth(a.grid)),:,1,1) .- 1))<1e-10
            close_equilibrium!(backend)
        end

        @testset "objective gradient against finite differences ($(view.mu))" for view in (ViewGeometry(),
                ViewGeometry(mu=0.8,azimuth_deg=30.0,geometry=:inclined))
            nz,nx,ny=24,4,3; axis=collect(range(-3.0,2.5;length=nz)); shape=(nz,nx,ny)
            grid=Grid3D(VerticalAxis(:log10m,axis),4.0e4 .* collect(0:nx-1),4.0e4 .* collect(0:ny-1),view)
            T0=[4800.0+900.0*exp10(min(axis[k],2.0)/6)+20.0*i-15.0*j for k in 1:nz,i in 1:nx,j in 1:ny]
            atmosphere=Atmosphere3D(grid,T0,fill(400.0,shape),fill(-300.0,shape),fill(200.0,shape),fill(800.0,shape))
            distributed=distribute_atmosphere(Float64,atmosphere,context)
            wavelength=collect(range(656.245e-9,656.315e-9,length=5))
            workspace=HybridForwardWorkspace(Float64,distributed,wavelength,StokesSet(:I),Dict(:H=>2))
            synthesizer=MixedIntensitySynthesizer([OpacityContributor(TabulatedOpacityModel(fill(2e-5,nz,5),fill(2e8,nz,5))),
                OpacityContributor(FFNOTransition(:H,656.28e-9,1,2,0.64,1.6735575e-27,1e8),workspace.populations[:H])])
            populations=CompositeDistributedPopulationModel(Dict(:H=>LocalDistributedPopulationModel(Phase6TwoLevelPopulation(1e12),2)))
            backend=prepare_equilibrium(config("none"),distributed,context)
            model=HybridForwardModel(populations,NonPRD(),synthesizer,IdentityObservation(),backend,CapabilityManifest())
            layout=ControlMapLayout(ControlMapSpec(:temperature,NodeField(reshape([5200.0,6000.0],2,1,1),[-2.0,1.5],:log10m);
                lower=3500.0,upper=8000.0,scale=1000.0))
            regularization=RegularizationSpec(vertical=VerticalRegularizationSpec((1,0,0,0,0,0,0),1e-5,ntuple(_->1.0,7)),
                horizontal=Dict(:temperature=>1e-6),scales=Dict(:temperature=>1000.0),horizontal_order=1)
            data=forward!(workspace,model,distributed,context).spectrum.data
            observation=ObservationCube(SpectralCube(data.*(1 .+ 0.01 .* reshape(sin.(1:length(data)),size(data))),wavelength,StokesSet(:I)),
                fill(maximum(abs,data)*1e-3,size(data)),ones(size(data)))
            problem=DistributedInversionProblem(model,workspace,distributed,observation,regularization,4.0e4,4.0e4,context;
                control_layout=layout,reference_id="logm-test")
            parameters=initial_parameters(layout).+[150.0,-100.0]
            production=objective_gradient!(HybridAdjointObjectiveGradient(force_balance_step=2e-5),problem,layout,parameters,context)
            oracle=objective_gradient!(FiniteDifferenceObjectiveGradient(step=2e-5),problem,layout,parameters,context)
            @test norm(production.gradient-oracle.gradient)/max(norm(oracle.gradient),eps())<5e-3
            @test norm(oracle.gradient)>0
            close_equilibrium!(backend)
        end

        @testset "wrong grid" begin
            distributed=distribute_atmosphere(Float64,logm_atmosphere(),context)
            tau=EquilibriumConfig("HSEMHS",:column_hse,:column_hse_ltau,g,LOGM_TABLE,:reject_trial,H.ColumnTauOptions(),nothing)
            @test_throws ArgumentError prepare_equilibrium(tau,distributed,context)
        end

        @testset "a given nₑ alone on the mass axis: the EOS at (T, nₑ)" begin
            table=H.load_table(LOGM_TABLE); plain=logm_atmosphere(;magnetic=false)
            ne=1e18 .* exp10.(0.3 .* reshape(native_depth(plain.grid),:,1,1)) .* ones(60,8,5)
            given=distribute_atmosphere(Float64,Atmosphere3D(plain.grid,plain.temperature,plain.vx,plain.vy,plain.vz,plain.vturb;ne=copy(ne)),context)
            backend=prepare_equilibrium(config(),given,context)
            @test reconstruct_equilibrium!(backend,given,context).valid
            a=given.local_atmosphere
            states=[H.state(table,log(a.temperature[k,1,1]),H.lnP_from_lnne(table,log(a.temperature[k,1,1]),log(ne[k,1,1]))) for k in 1:60]
            @test a.rho[:,1,1]≈[exp(s[1]) for s in states] rtol=1e-12
            @test a.chi500[:,1,1]≈[exp(s[3]) for s in states] rtol=1e-12
            @test a.ne==ne && all(diff(a.z;dims=1).<0)
            a.ne[10,2,2]=1e60          # an nₑ no EOS state reaches: invalid, not a throw
            @test reconstruct_equilibrium!(backend,given,context).reason===:eos_out_of_envelope
            close_equilibrium!(backend)
        end
    end
    end
end

@testset "A given atmosphere: nₑ and hydrogen populations, no HSEMHS" begin
    F=FFNOInversion
    # RH's convertScales written out, one column, three axes
    n=40; lm=collect(range(-4.0,1.0;length=n)); m=exp10.(lm); rho=1e-5 .* m.^0.9; chi=3e-6 .* m.^0.6
    z=zeros(n); lt=zeros(n)
    given_geometry!(z,lt,:log10m,lm,rho,chi,1.0)
    zr=zeros(n); tr=zeros(n); tr[1]=chi[1]/rho[1]*m[1]
    for k in 2:n
        zr[k]=zr[k-1]-2*(m[k]-m[k-1])/(rho[k-1]+rho[k]); tr[k]=tr[k-1]+0.5*(chi[k-1]+chi[k])*(zr[k-1]-zr[k])
    end
    j=findfirst(>(1),tr); h0=zr[j-1]+(zr[j]-zr[j-1])*(1-tr[j-1])/(tr[j]-tr[j-1])
    @test z≈zr.-h0 rtol=1e-12
    @test lt≈log10.(tr) rtol=1e-12
    # μ: the same column along the ray: τ_LOS unchanged for the same m_LOS and ρ, heights × μ
    z2=zeros(n); lt2=zeros(n); given_geometry!(z2,lt2,:log10m,lm,rho,chi,0.5)
    @test z2≈0.5 .* z rtol=1e-12
    @test lt2≈lt rtol=1e-12
    # the τ axis returns that column's heights given its own τ
    logtau=log10.(tr); zt=zeros(n); ltt=zeros(n); given_geometry!(zt,ltt,:log10tau500,logtau,rho,chi,1.0)
    @test ltt==logtau && all(diff(zt).<0)
    @test abs(zt[j]+(zt[j-1]-zt[j])*(tr[j]-1)/(tr[j]-tr[j-1]))<1e-6                 # z(τ = 1) = 0 here too
    # the height axis keeps its heights
    zaxis=collect(range(2e6,-1e5;length=n)); zz=zeros(n); ltz=zeros(n); given_geometry!(zz,ltz,:z_m,zaxis,rho,fill(1e-6,n),1.0)
    @test zz==zaxis && all(diff(ltz).>0)
    @test exp10(ltz[3])≈1e-6*(zaxis[1]-zaxis[2])/2+1e-6*(zaxis[1]-zaxis[3]) rtol=1e-12

    # the abundances are Wittmann's
    source=read(joinpath(@__DIR__,"..","..","scripts","witt_eos_cpp.cpp"),String)
    numbers(name)=parse.(Float64,split(replace(match(Regex(name*" = \\{\\{(.*?)\\}\\}","s"),source)[1],"\n"=>""),","))
    ab=exp10.(numbers("ABUND_RAW")); weights=numbers("AMASS")
    @test F._GIVEN_TOTAL_ABUNDANCE≈sum(ab)/ab[1] rtol=1e-7
    @test F._GIVEN_MASS_PER_H_KG≈sum(ab.*weights)/ab[1]*1.66053906660e-27 rtol=1e-3
    isfile(LOGM_TABLE) && @test F._GIVEN_MASS_PER_H_KG≈F.HSEMHS.mass_per_H(F.HSEMHS.load_table(LOGM_TABLE)) rtol=1e-12

    # configuration: nothing of HSEMHS is needed, on any axis, and its options are refused
    mktempdir() do dir
        nz,nx,ny=8,3,2; depth=collect(range(-3.0,0.5;length=nz)); file=joinpath(dir,"a.h5")
        TT=[5500.0+40.0*k+7.0*i+3.0*j for k in 1:nz,j in 1:ny,i in 1:nx]
        pops=[(l==6 ? 1e17 : 1e21/10.0^l)*exp10(0.8*depth[k])*(1+0.01i) for l in 1:6,k in 1:nz,j in 1:ny,i in 1:nx]    # (level,z,y,x)
        ne=[1e17*exp10(0.3*depth[k]) for k in 1:nz,j in 1:ny,i in 1:nx]
        zv=collect(range(1e6,0.0;length=nz))
        h5open(file,"w") do f
            f["lm"]=depth; f["lt"]=depth; f["zm"]=zv; f["x"]=4.8e4 .* collect(0:nx-1); f["y"]=4.8e4 .* collect(0:ny-1)
            f["t"]=TT; f["ne"]=ne; f["h"]=pops; for name in ("vx","vy","vz"); f[name]=zero(TT); end
        end
        datasets(key,name)="$key = \"$name\"\nne = \"ne\"\nhydrogen_populations = \"h\"\nx_m = \"x\"\ny_m = \"y\""
        function text(coordinate,key,name;equilibrium="[population.mesh]\nenabled = false")
            t=logm_config_text(file;coordinate=coordinate,datasets=datasets(key,name),extra=equilibrium)
            replace(t,r"\[equilibrium\][^\[]*"=>"")
        end
        config(t)=(p=joinpath(dir,"c$(hash(t)).toml"); write(p,t); load_config(p))
        for (coordinate,key,name,mode) in (("log10m","log10m","lm",:given_logm),("log10tau500","logtau500","lt",:given_ltau),("z","z_m","zm",:given_z))
            cfg=config(text(coordinate,key,name))
            @test cfg.equilibrium.backend=="given" && cfg.equilibrium.mode===mode && cfg.equilibrium.eos_table_file==""
            a=read_inversion_inputs(cfg).atmosphere
            nh=[sum(pops[:,k,y,x]) for k in 1:nz,x in 1:nx,y in 1:ny]
            @test a.rho≈nh.*F._GIVEN_MASS_PER_H_KG rtol=1e-12
            @test a.pgas≈(nh.*F._GIVEN_TOTAL_ABUNDANCE.+permutedims(ne,(1,3,2))).*1.380649e-23.*a.temperature rtol=1e-12
            @test all(>(0),a.chi500)
            @test a.h_neutral≈[sum(pops[1:5,k,y,x]) for k in 1:nz,x in 1:nx,y in 1:ny] rtol=1e-12
            @test a.h_proton≈[pops[6,k,y,x] for k in 1:nz,x in 1:nx,y in 1:ny] rtol=1e-12
            context=serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
            distributed=distribute_atmosphere(Float64,a,context)
            backend=prepare_equilibrium(cfg.equilibrium,distributed,context)
            @test backend isa GivenAtmosphereBackend
            outcome=reconstruct_equilibrium!(backend,distributed,context)
            @test outcome.valid && outcome.effective_mode===mode
            b=distributed.local_atmosphere
            @test all(diff(b.z;dims=1).<0) && all(diff(b.logtau500;dims=1).>0) && check_coordinate_consistency(b)
            @test b.ne==a.ne && b.temperature==a.temperature && b.h_neutral==a.h_neutral
            # the continuum comes from the given hydrogen: positive, Planck-weighted, and more neutral H gives more of it
            wave=[656.0e-9,656.5e-9]; chi_c=zeros(nz,2); eta_c=zeros(nz,2)
            F.add_opacity_emissivity!(chi_c,eta_c,build_muspel_continuum_model(656.25),wave,b,1,1,nothing)
            @test all(>(0),chi_c) && chi_c[:,1]==chi_c[:,2]
            @test eta_c[3,1]≈chi_c[3,1]*F.source_planck(wave[1],b.temperature[3,1,1]) rtol=1e-6
            more=deepcopy(b); more.h_neutral.*=2; chi_m=zeros(nz,2); eta_m=zeros(nz,2)
            F.add_opacity_emissivity!(chi_m,eta_m,build_muspel_continuum_model(656.25),wave,more,1,1,nothing)
            @test all(chi_m.>chi_c)
        end
        # the mesh needs the table; HSEMHS options and an inversion are refused; a level count below two is an error
        err=try config(text("log10m","log10m","lm";equilibrium="")); "" catch e; sprint(showerror,e) end
        @test occursin("eos_table_file",err)
        with_column=logm_config_text(file;coordinate="log10m",datasets=datasets("log10m","lm"),equilibrium="[equilibrium.column_logm]\nalign = \"none\"",extra="[population.mesh]\nenabled = false")
        @test config(with_column).equilibrium.backend=="given"   # the column sections configure HSEMHS: ignored here
        inversion=logm_config_text(file;coordinate="log10m",datasets=datasets("log10m","lm"),mode="inversion",extra="[population.mesh]\nenabled = false")
        err=try config(inversion); "" catch e; sprint(showerror,e) end
        @test occursin("forward synthesis only",err)
    end
end
