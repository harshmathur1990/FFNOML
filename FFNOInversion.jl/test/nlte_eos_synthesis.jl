using Test
using FFNOInversion
using Libdl
using LinearAlgebra
import HSEMHS
import WittBifrostEOS
import Muspel
isdefined(Main,:TestHydrostaticEquilibrium) || include(joinpath(@__DIR__,"test_equilibrium.jl"))
isdefined(Main,:Phase6TwoLevelPopulation) || include(joinpath(@__DIR__,"phase6_production_gradient.jl"))

# Synthesis with the NLTE EOS ([equilibrium] eos_model = "nlte"): the continuum's hydrogen and neutral H in broadening from the HSEMHS
# NLTE table, the Kurucz metals in LTE at the atmosphere's nₑ (witt_kurucz_populations_given_ne). Their adjoints against finite
# differences, and the objective gradient through the production adjoint. HSEMHS_NLTE_EOS_TABLE as in hsemhs_equilibrium.jl.
const NLTE_SYNTHESIS_TABLE=get(ENV,"HSEMHS_NLTE_EOS_TABLE","/mn/stornext/d9/data/harshm/MHS_nlte/HSEMHS/cache/eos_table_nlte.h5")
const LTE_SYNTHESIS_TABLE=get(ENV,"HSEMHS_EOS_TABLE","/mn/stornext/d9/data/harshm/MHS/HSEMHS/cache/eos_table_fullz.h5")

if !isfile(NLTE_SYNTHESIS_TABLE)
    @info "HSEMHS NLTE EOS table not found; skipping the NLTE synthesis tests" NLTE_SYNTHESIS_TABLE
else
@testset "Synthesis with the NLTE EOS" begin
    F=FFNOInversion
    table=WittBifrostEOS.load_table(NLTE_SYNTHESIS_TABLE;model="nlte"); state=NLTEHydrogenState(table)
    mktempdir() do dir
        lib=joinpath(dir,"libwitt_nlte.$(Libdl.dlext)")
        run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__,"..")) $(joinpath(@__DIR__,"..","scripts","build_wittmann_backend.jl")) $lib`)
        eos=WittmannEOS(lib,joinpath(@__DIR__,"..","..","scripts","pf_Kurucz.input")); witt=WittmannKuruczState(eos)
        function column(nz;ne=nothing)
            z=collect(range(8e5,0.0;length=nz)); shape=(nz,1,1); f(v)=reshape(collect(Float64,v),shape)
            Temp=vcat(1.5e4,9.0e3,range(4400.0,6400.0;length=nz-2)); P=10 .^ range(1.3,4.0;length=nz)
            F.Atmosphere3D(Grid3D(VerticalAxis(:z_m,z),[0.0],[0.0]),f(Temp),f(100 .* (1:nz)),f(-50 .* (1:nz)),f(800 .* sin.(1:nz)),
                f(1000 .+ 30 .* (1:nz));pgas=f(P),rho=f(3e-8 .* P),ne=ne===nothing ? f(1e17 .* P.^0.6) : f(ne),z=f(z))
        end
        nz=10; a=column(nz); T=a.temperature[:,1,1]; P=a.pgas[:,1,1]

        @testset "NLTE hydrogen state" begin
            hi=F.eos_neutral_hydrogen!(zeros(nz),state,T,P); hp=F.eos_protons(state,T,P)
            for k in 1:nz
                x,y,_=WittBifrostEOS.hydrogen(table,log(T[k]),log(P[k]))
                @test hi[k]==exp(x) && hp[k]==exp(y)
            end
            @test hp[1]>hi[1] && hi[end]>hp[end]                  # ionized at 15 kK, neutral in the photosphere
            # below the table: the model it was built from
            v,_,ok=WittBifrostEOS.model_fields(WittBifrostEOS.table_model(table),log(1e3),log(1e3)); @test ok
            @test F.eos_neutral_hydrogen!(zeros(1),state,[1e3],[1e3])[1]==exp(v[4])
            isfile(LTE_SYNTHESIS_TABLE) && @test_throws ArgumentError NLTEHydrogenState(WittBifrostEOS.load_table(LTE_SYNTHESIS_TABLE))
        end

        fe=joinpath(dir,"fe.list")
        write(fe,"  656.9214 -0.420 26.00   53393.669  4.0 (4F)4d g5F   38175.352  3.0 (4F)4p y3D  8.36 -4.38 -7.70FMW  0 0  0 0.000  0 0.000    0    0           1287 1324\n")
        lines=load_kurucz_linelist(fe).lines; line=only(lines)
        wave=collect(range(656.9e-9,657.2e-9;length=5))

        @testset "metals at a given nₑ" begin
            fn=Libdl.dlsym(witt.library_handle,:witt_kurucz_populations); fn_ne=Libdl.dlsym(witt.library_handle,:witt_kurucz_populations_given_ne)
            lte(t,p)=(l=zeros(length(t)); h=zeros(length(t)); @test 0==ccall(fn,Cint,(Ptr{Cvoid},Ptr{Cdouble},Ptr{Cdouble},Cint,Cint,Cdouble,Cdouble,Ptr{Cdouble},Ptr{Cdouble},Csize_t),
                witt.backend,t,p,line.atomic_number,line.ion_stage,line.lower_energy_j,line.lower_g,l,h,length(t)); l)
            given(t,p,n)=(l=zeros(length(t)); @test 0==ccall(fn_ne,Cint,(Ptr{Cvoid},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Cint,Cint,Cdouble,Cdouble,Ptr{Cdouble},Csize_t),
                witt.backend,t,p,n,line.atomic_number,line.ion_stage,line.lower_energy_j,line.lower_g,l,length(t)); l)
            rho=zeros(nz); ne_lte=zeros(nz); F.thermodynamics!(reshape(rho,nz,1,1),reshape(ne_lte,nz,1,1),eos,reshape(T,nz,1,1),reshape(P,nz,1,1))
            # at Wittmann's own LTE nₑ the given-nₑ populations are the LTE ones
            @test given(T,P,ne_lte)≈lte(T,P) rtol=1e-10
            # Fe I, mostly ionized above 5 kK: more electrons, more neutral iron
            @test all(given(T,P,2 .* ne_lte)[3:end].>lte(T,P)[3:end])
            bad=zeros(1)
            @test 1==ccall(fn_ne,Cint,(Ptr{Cvoid},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Cint,Cint,Cdouble,Cdouble,Ptr{Cdouble},Csize_t),
                witt.backend,[5000.0],[1e3],[0.0],26,0,0.0,1.0,bad,1)

            # the NLTE Kurucz model at the LTE nₑ, with a given neutral H in both, is the LTE model
            g=column(nz;ne=ne_lte); h=F.Atmosphere3D(g.grid,g.temperature,g.vx,g.vy,g.vz,g.vturb;pgas=g.pgas,rho=g.rho,ne=g.ne,z=g.z,
                h_neutral=fill(1e21,nz,1,1),h_proton=fill(1e18,nz,1,1))
            mlte=KuruczLTEModel(lines,eos;include_continuum=false); mnlte=KuruczLTEModel(lines,eos,state)
            @test mnlte.eos isa NLTEKuruczState && !mnlte.include_continuum
            @test_throws ArgumentError KuruczLTEModel(lines,eos,state;include_continuum=true)
            c1=zeros(nz,5),zeros(nz,5); c2=zeros(nz,5),zeros(nz,5)
            F.add_opacity_emissivity!(c1...,mlte,wave,h,1,1); F.add_opacity_emissivity!(c2...,mnlte,wave,h,1,1)
            @test c2[1]≈c1[1] rtol=1e-10
            @test c2[2]≈c1[2] rtol=1e-10
            # with the atmosphere's own nₑ and the NLTE neutral H it differs
            c3=zeros(nz,5),zeros(nz,5); F.add_opacity_emissivity!(c3...,mnlte,wave,a,1,1)
            @test all(isfinite,c3[1]) && c3[1]!=c1[1]
        end

        @testset "adjoints against finite differences" begin
            cb=reshape(sin.(1:nz*5),nz,5); eb=reshape(cos.(1:nz*5),nz,5)
            kurucz=KuruczLTEModel(lines,eos,state); continuum=build_muspel_continuum_model(656.25;eos=state)
            for (model,fields) in ((kurucz,(:temperature,:pgas,:ne,:vturb,:vz)),(continuum,(:temperature,:pgas,:ne)))
                pair(b)=(ch=zeros(nz,5); et=zeros(nz,5); F.add_opacity_emissivity!(ch,et,model,wave,b,1,1,nothing); sum(cb.*ch)+sum(eb.*et))
                bar=F.AtmosphereCotangent(a); F.add_opacity_emissivity_vjp!(bar,nothing,_->nothing,cb,eb,model,wave,a,1,1,nothing)
                for name in fields
                    f=getfield(a,name); cot=getfield(bar,name)
                    for k in (1,2,4,7,10)
                        v=f[k,1,1]; h=1e-4*max(abs(v),1.0); f[k,1,1]=v+h; up=pair(a); f[k,1,1]=v-h; dn=pair(a); f[k,1,1]=v
                        @test cot[k,1,1]≈(up-dn)/(2h) rtol=0.02 atol=1e-3*max(maximum(abs,cot),1e-300)
                    end
                end
            end
        end

        @testset "synthesis setup" begin
            region=SpectralRegionConfig(656.90e-9,0.05e-9,7,1.0,:none,nothing,[SpectralSourceConfig(:kurucz_lte,fe)])
            setup=only(build_synthesis_setup((regions=[region],),a,Dict{Symbol,Any}(),eos;atom_files=Dict(),nlte_hydrogen=state))
            models=[c.model for c in setup.synthesizer.contributors]
            @test models[1] isa MuspelContinuumModel && models[1].eos===state
            @test models[2] isa KuruczLTEModel && models[2].eos isa NLTEKuruczState
            lte=only(build_synthesis_setup((regions=[region],),a,Dict{Symbol,Any}(),eos;atom_files=Dict()))
            @test [c.model for c in lte.synthesizer.contributors][1].eos isa WittmannKuruczState
            # an NLTE equilibrium without the NLTE hydrogen is refused rather than synthesized in LTE
            eq=EquilibriumConfig("HSEMHS",:column_hse,:column_hse_ltau,274.0,NLTE_SYNTHESIS_TABLE,:reject_trial,HSEMHS.ColumnTauOptions(),nothing,"nlte")
            err=try build_synthesis_setup((regions=[region],equilibrium=eq),a,Dict{Symbol,Any}(),eos;atom_files=Dict()); "" catch e; sprint(showerror,e) end
            @test occursin("nlte_hydrogen",err)
        end

        # the whole objective gradient with the NLTE continuum and Kurucz line, through the production adjoint, against finite differences
        @testset "objective gradient ($kind)" for kind in (:log10tau500,:z_m)
            nl=12; zs=collect(range(8e5,-1e5;length=nl)); axis=kind===:z_m ? zs : collect(range(-5.0,0.0;length=nl)); shape=(nl,4,3)
            grid=Grid3D(VerticalAxis(kind,axis),40e3 .* collect(0:3),40e3 .* collect(0:2))
            T0=[5500.0+120.0*sin(1.3i)*cos(0.8j)+40.0k for k in 1:nl,i in 1:4,j in 1:3]; fv=fill(800.0,shape)
            atmosphere=F.Atmosphere3D(grid,T0,zeros(shape),zeros(shape),zeros(shape),fv;z=kind===:z_m ? repeat(reshape(axis,nl,1,1),1,4,3) : nothing)
            context=serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
            distributed=distribute_atmosphere(Float64,atmosphere,context)
            w=collect(range(656.88e-9,656.96e-9;length=5))
            workspace=HybridForwardWorkspace(Float64,distributed,w,StokesSet(:I),Dict(:H=>2))
            synth=MixedIntensitySynthesizer([OpacityContributor(build_muspel_continuum_model(656.92;eos=state)),
                OpacityContributor(KuruczLTEModel(lines,eos,state)),
                OpacityContributor(FFNOTransition(:H,656.92e-9,1,2,0.64,1.6735575e-27,1e8),workspace.populations[:H])])
            pops=CompositeDistributedPopulationModel(Dict(:H=>LocalDistributedPopulationModel(Phase6TwoLevelPopulation(1e12),2)))
            forward_model=HybridForwardModel(pops,NonPRD(),synth,IdentityObservation(),TestHydrostaticEquilibrium(electron_fraction=1e-3),CapabilityManifest())
            nodes=kind===:z_m ? [8e5,-1e5] : [-5.0,-1.0]
            layout=ControlMapLayout(ControlMapSpec(:temperature,NodeField(reshape([5200.0,6000.0],2,1,1),nodes,kind);lower=3500.0,upper=8000.0,scale=1000.0))
            reg=RegularizationSpec(vertical=VerticalRegularizationSpec((1,0,0,0,0,0,0),1e-5,ntuple(_->1.0,7);length_scale_m=kind===:z_m ? 1e5 : nothing),
                horizontal=Dict(:temperature=>1e-6),scales=Dict(:temperature=>1000.0),horizontal_order=1)
            data=forward!(workspace,forward_model,distributed,context).spectrum.data
            obs=ObservationCube(SpectralCube(data.*(1 .+ 0.01 .* reshape(sin.(1:length(data)),size(data))),w,StokesSet(:I)),fill(maximum(abs,data)*1e-3,size(data)),ones(size(data)))
            problem=DistributedInversionProblem(forward_model,workspace,distributed,obs,reg,40e3,40e3,context;control_layout=layout,reference_id="nlte-synthesis")
            parameters=initial_parameters(layout).+[150.0,-100.0]
            production=objective_gradient!(HybridAdjointObjectiveGradient(force_balance_step=2e-5),problem,layout,parameters,context)
            oracle=objective_gradient!(FiniteDifferenceObjectiveGradient(step=2e-5),problem,layout,parameters,context)
            @test norm(oracle.gradient)>0
            @test norm(production.gradient-oracle.gradient)/norm(oracle.gradient)<5e-3
        end
    end
end
end
