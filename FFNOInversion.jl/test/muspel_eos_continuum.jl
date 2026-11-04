using Test
using FFNOInversion
using Libdl
using LinearAlgebra
import Muspel
isdefined(Main,:TestHydrostaticEquilibrium) || include(joinpath(@__DIR__,"test_equilibrium.jl"))
isdefined(Main,:Phase6TwoLevelPopulation) || include(joinpath(@__DIR__,"phase6_production_gradient.jl"))

# The continuum in forward and inversion: Muspel's with hydrogen from the EOS at (T, P). No FFNO population enters it.
@testset "Muspel continuum with EOS hydrogen" begin
    F=FFNOInversion
    mktempdir() do dir
        lib=joinpath(dir,"libwitt_continuum.$(Libdl.dlext)")
        run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__,"..")) $(joinpath(@__DIR__,"..","scripts","build_wittmann_backend.jl")) $lib`)
        eos=WittmannEOS(lib,joinpath(@__DIR__,"..","..","scripts","pf_Kurucz.input")); state=WittmannKuruczState(eos)
        function column(nz)
            z=collect(range(8e5,0.0;length=nz)); shape=(nz,1,1); f(v)=reshape(collect(Float64,v),shape)
            Temp=vcat(1.5e4,4.0e4,range(4400.0,6400.0;length=nz-2)); P=10 .^ range(1.3,4.0;length=nz)
            F.Atmosphere3D(Grid3D(VerticalAxis(:z_m,z),[0.0],[0.0]),f(Temp),f(zeros(nz)),f(zeros(nz)),f(zeros(nz)),f(fill(1e3,nz));
                pgas=f(P),rho=f(3e-8 .* P),ne=f(1e17 .* P.^0.6),z=f(z))
        end
        a=column(12); T=a.temperature[:,1,1]; P=a.pgas[:,1,1]; wave=collect(range(656.0e-9,656.5e-9;length=4)); nz=12
        model=build_muspel_continuum_model(656.25;eos=state)
        @test model isa MuspelContinuumModel

        # the EOS hydrogen: protons from partials, neutrals from the same state; ionized at the hot levels, neutral at the cool ones
        hi=F.eos_neutral_hydrogen!(zeros(nz),state,T,P); hp=F.eos_protons(state,T,P)
        @test all(>(0),hi) && all(>(0),hp) && hp[2]>hi[2] && hi[end÷2]>hp[end÷2]

        # the interpolation on the fixed range is close to one built for this atmosphere's own range
        c=zeros(nz,1); e=zeros(nz,1); F.add_opacity_emissivity!(c,e,model,[wave[1]],a,1,1,nothing)
        p(A)=Float32.(repeat(reshape(A,:,1,1),1,1,1))
        m=Muspel.Atmosphere3D(1,1,nz,Float32[0],Float32[0],p(a.z[:,1,1]),p(T),p(zeros(nz)),p(zeros(nz)),p(zeros(nz)),p(a.ne[:,1,1]),p(hi),p(hp))
        files=[joinpath(Muspel.AtomicData.get_atom_dir(),n) for n in ("Al.yaml","C.yaml","Ca.yaml","Fe.yaml","H_6.yaml","He.yaml","KI.yaml","Mg.yaml","N.yaml","Na.yaml","NiI.yaml","O.yaml","S.yaml","Si.yaml")]
        own=Muspel.get_σ_itp(m,656.25f0,files)
        direct=[Muspel.α_cont(own,T[k],a.ne[k,1,1],hi[k],hp[k]) for k in 1:nz]
        @test maximum(abs.(c[:,1]./direct.-1))<2e-3
        # S = Planck
        @test e[:,1]≈c[:,1].*F.source_planck.(wave[1],T) rtol=1e-12

        # adjoint against centred differences in T, nₑ and P at every level
        cb=reshape(sin.(1:nz*4),nz,4); eb=reshape(cos.(1:nz*4),nz,4)
        pair(b)=(ch=zeros(nz,4); et=zeros(nz,4); F.add_opacity_emissivity!(ch,et,model,wave,b,1,1,nothing); sum(cb.*ch)+sum(eb.*et))
        bar=F.AtmosphereCotangent(a); F.add_opacity_emissivity_vjp!(bar,nothing,_->nothing,cb,eb,model,wave,a,1,1,nothing)
        for (name,cot,rel) in ((:temperature,bar.temperature,2e-3),(:ne,bar.ne,1e-2),(:pgas,bar.pgas,1e-2))
            f=getfield(a,name)
            for k in (1,3,6,9,12)
                v=f[k,1,1]; h=rel*v; f[k,1,1]=v+h; up=pair(a); f[k,1,1]=v-h; dn=pair(a); f[k,1,1]=v
                fd=(up-dn)/(2h); @test cot[k,1,1]≈fd rtol=0.02 atol=1e-3*maximum(abs,cot)
            end
        end
        @test_throws ArgumentError F.add_opacity_emissivity_vjp!(bar,nothing,_->nothing,cb,eb,MuspelContinuumModel(model.continuum,state),wave,
            F.Atmosphere3D(a.grid,a.temperature,a.vx,a.vy,a.vz,a.vturb;pgas=a.pgas,rho=a.rho,ne=a.ne,z=a.z,h_neutral=fill(1e20,nz,1,1),h_proton=fill(1e18,nz,1,1)),1,1,nothing)

        # a given atmosphere's hydrogen replaces the EOS's
        g=F.Atmosphere3D(a.grid,a.temperature,a.vx,a.vy,a.vz,a.vturb;pgas=a.pgas,rho=a.rho,ne=a.ne,z=a.z,h_neutral=reshape(hi,:,1,1).*2,h_proton=reshape(hp,:,1,1))
        c2=zeros(nz,1); e2=zeros(nz,1); F.add_opacity_emissivity!(c2,e2,build_muspel_continuum_model(656.25),[wave[1]],g,1,1,nothing)
        @test all(c2[3:end,1].>c[3:end,1])
        @test_throws ArgumentError F.add_opacity_emissivity!(zeros(nz,1),zeros(nz,1),build_muspel_continuum_model(656.25),[wave[1]],a,1,1,nothing)

        # the whole objective gradient with this continuum, through the production adjoint, against finite differences
        @testset "objective gradient ($kind)" for kind in (:log10tau500,:z_m)
            nl=12; zs=collect(range(8e5,-1e5;length=nl)); axis=kind===:z_m ? zs : collect(range(-5.0,0.0;length=nl)); shape=(nl,4,3)
            grid=Grid3D(VerticalAxis(kind,axis),40e3 .* collect(0:3),40e3 .* collect(0:2))
            T0=[5500.0+120.0*sin(1.3i)*cos(0.8j)+40.0k for k in 1:nl,i in 1:4,j in 1:3]; fv=fill(800.0,shape)
            atmosphere=F.Atmosphere3D(grid,T0,zeros(shape),zeros(shape),zeros(shape),fv;z=kind===:z_m ? repeat(reshape(axis,nl,1,1),1,4,3) : nothing)
            context=serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
            distributed=distribute_atmosphere(Float64,atmosphere,context)
            w=collect(range(656.245e-9,656.315e-9;length=5))
            workspace=HybridForwardWorkspace(Float64,distributed,w,StokesSet(:I),Dict(:H=>2))
            synth=MixedIntensitySynthesizer([OpacityContributor(build_muspel_continuum_model(656.28;eos=state)),
                OpacityContributor(FFNOTransition(:H,656.28e-9,1,2,0.64,1.6735575e-27,1e8),workspace.populations[:H])])
            pops=CompositeDistributedPopulationModel(Dict(:H=>LocalDistributedPopulationModel(Phase6TwoLevelPopulation(1e12),2)))
            forward_model=HybridForwardModel(pops,NonPRD(),synth,IdentityObservation(),TestHydrostaticEquilibrium(),CapabilityManifest())
            nodes=kind===:z_m ? [8e5,-1e5] : [-5.0,-1.0]
            layout=ControlMapLayout(ControlMapSpec(:temperature,NodeField(reshape([5200.0,6000.0],2,1,1),nodes,kind);lower=3500.0,upper=8000.0,scale=1000.0))
            reg=RegularizationSpec(vertical=VerticalRegularizationSpec((1,0,0,0,0,0,0),1e-5,ntuple(_->1.0,7);length_scale_m=kind===:z_m ? 1e5 : nothing),
                horizontal=Dict(:temperature=>1e-6),scales=Dict(:temperature=>1000.0),horizontal_order=1)
            data=forward!(workspace,forward_model,distributed,context).spectrum.data
            obs=ObservationCube(SpectralCube(data.*(1 .+ 0.01 .* reshape(sin.(1:length(data)),size(data))),w,StokesSet(:I)),fill(maximum(abs,data)*1e-3,size(data)),ones(size(data)))
            problem=DistributedInversionProblem(forward_model,workspace,distributed,obs,reg,40e3,40e3,context;control_layout=layout,reference_id="muspel-continuum")
            parameters=initial_parameters(layout).+[150.0,-100.0]
            production=objective_gradient!(HybridAdjointObjectiveGradient(force_balance_step=2e-5),problem,layout,parameters,context)
            oracle=objective_gradient!(FiniteDifferenceObjectiveGradient(step=2e-5),problem,layout,parameters,context)
            @test norm(oracle.gradient)>0
            @test norm(production.gradient-oracle.gradient)/norm(oracle.gradient)<5e-3
        end

        # cost per column at 95 levels and 88 wavelengths, against the Wittmann continuum
        n95=column(95); w88=collect(range(656.0e-9,656.5e-9;length=88)); cb95=ones(95,88); eb95=ones(95,88)
        m=model; wc=KuruczLTEModel(KuruczLine{Float64}[],eos;include_continuum=true); b95=F.AtmosphereCotangent(n95)
        forward(md)=@elapsed F.add_opacity_emissivity!(zeros(95,88),zeros(95,88),md,w88,n95,1,1,nothing)
        adjoint(md)=@elapsed F.add_opacity_emissivity_vjp!(b95,nothing,_->nothing,cb95,eb95,md,w88,n95,1,1,nothing)
        forward(m); adjoint(m); forward(wc); adjoint(wc)
        println("CONTINUUM_COST per column: Muspel forward ",minimum(forward(m) for _ in 1:5)," adjoint ",minimum(adjoint(m) for _ in 1:5),
            "; Wittmann forward ",minimum(forward(wc) for _ in 1:5)," adjoint ",minimum(adjoint(wc) for _ in 1:5))
    end
end
