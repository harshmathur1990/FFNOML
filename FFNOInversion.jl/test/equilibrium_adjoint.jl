# The equilibrium adjoint in the objective gradient (docs/equilibrium-adjoint): the regularization's analytic gradient,
# the HSEMHS equilibrium VJP (HSEMHS.adjoint_embedded!) and the objective gradient built from them, each against central
# differences. Serial; the regularization's halo transpose on several ranks is in mpi_equilibrium_worker.jl.
using LinearAlgebra, Random, Statistics
@isdefined(HSEMHS_TABLE) || include("hsemhs_equilibrium.jl")

"""
Two-level population fixture with an exact VJP that depends on the equilibrium through ρ (as the FFNO does through its
log10 ρ input): n₁ = s (T/5000) (ρ/10⁻⁴ kg m⁻³)^0.2, n₂ = n₁/10.
"""
struct AdjointTwoLevelPopulation{T<:AbstractFloat} <: AbstractPopulationModel
    scale::T
end
_rho_factor(atmosphere)=(atmosphere.rho./1e-4).^0.2
function FFNOInversion.predict_populations!(out,model::AdjointTwoLevelPopulation,atmosphere::Atmosphere3D,cache=nothing)
    f=_rho_factor(atmosphere)
    @views out[:,:,:,1].=model.scale.*atmosphere.temperature./5000 .* f
    @views out[:,:,:,2].=0.1model.scale.*atmosphere.temperature./5000 .* f
    out
end
function FFNOInversion.population_vjp!(feature_bar,z_bar,model::AdjointTwoLevelPopulation,atmosphere,population_bar)
    fill!(feature_bar,0); fill!(z_bar,0)
    f=_rho_factor(atmosphere); pb=population_bar[:,:,:,1].+0.1population_bar[:,:,:,2]
    @views feature_bar[1,:,:,:].=model.scale/5000 .* f .* pb
    @views feature_bar[6,:,:,:].=0.2log(10) .* model.scale .* atmosphere.temperature./5000 .* f .* pb   # ∂/∂log10 ρ
    feature_bar,z_bar
end

"""Best relative error of a central difference of `f` along `d` against `pairing` over the steps `hs`."""
function adjoint_fd_error(f,pairing;hs=(1e-3,1e-4,1e-5,1e-6))
    best=Inf
    for h in hs
        d=(f(h)-f(-h))/2h
        best=min(best,abs(d-pairing)/max(abs(d),abs(pairing),1e-300))
    end
    best
end

const ADJ_FIELDS=(:temperature,:vx,:vy,:vz,:vturb,:pgas)

if isfile(HSEMHS_TABLE)
@testset "equilibrium adjoint in the objective gradient" begin
    context=initialize_parallel(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
    rng=MersenneTwister(11)

    @testset "regularization: analytic gradient ($(view.mu), order $order, kind $kind)" for
            view in (ViewGeometry(),ViewGeometry(mu=0.8,azimuth_deg=30.0,geometry=:inclined)),order in (1,2),
            kind in (:log10tau500,:z_m)
        nz,nx,ny=12,5,4
        axis=kind===:z_m ? collect(range(1.5e6,-4.0e5;length=nz)) : collect(range(-6.0,1.0;length=nz))
        grid=Grid3D(VerticalAxis(kind,axis),4.0e4 .* collect(0:nx-1),4.0e4 .* collect(0:ny-1),view)
        r()=randn(rng,nz,nx,ny)
        B=MagneticField3D(0.01 .* r(),0.01 .* r(),0.02 .+ 0.01 .* r())
        a=Atmosphere3D(grid,5000 .+ 300 .* r(),300 .* r(),300 .* r(),300 .* r(),800 .+ 100 .* r();magnetic_field=B,
            pgas=exp.(5 .+ 0.1 .* r()))
        distributed=distribute_atmosphere(Float64,a,context)
        spec=RegularizationSpec(vertical=VerticalRegularizationSpec((4,1,2,3,1,2,1),0.7,(1.0,0.5,0.3,0.2,0.4,0.6,0.8);
                length_scale_m=kind===:z_m ? 1e5 : nothing),
            horizontal=Dict(:temperature=>1e-3,:vz=>2e-4,:vlos=>1e-4,:Bx=>0.5,:pgas=>1e-3),
            scales=Dict(:temperature=>1000.0,:vlos=>1000.0,:vturb=>1000.0,:B=>0.1,:inc=>1.0,:azi=>1.0,:pgas_boundary=>100.0,
                :vz=>1000.0,:Bx=>0.1,:pgas=>100.0),horizontal_order=order)
        dx=4.0e4; dy=4.0e4
        bar=AtmosphereCotangent(distributed.local_atmosphere)
        distributed_regularization_vjp!(bar,distributed,spec,dx,dy,context)
        loc=distributed.local_atmosphere
        dirs=Dict(f=>randn(rng,size(getfield(loc,f))).*std(getfield(loc,f)) for f in ADJ_FIELDS)
        dB=(0.01 .* r(),0.01 .* r(),0.01 .* r())
        base=Dict(f=>copy(getfield(loc,f)) for f in ADJ_FIELDS); B0=(copy(B.Bx),copy(B.By),copy(B.Bz))
        function pen(h)
            for f in ADJ_FIELDS; getfield(loc,f).=base[f].+h.*dirs[f]; end
            loc.magnetic_field.Bx.=B0[1].+h.*dB[1]; loc.magnetic_field.By.=B0[2].+h.*dB[2]; loc.magnetic_field.Bz.=B0[3].+h.*dB[3]
            v=distributed_regularization_penalty(distributed,spec,dx,dy,context).total
            for f in ADJ_FIELDS; getfield(loc,f).=base[f]; end
            loc.magnetic_field.Bx.=B0[1]; loc.magnetic_field.By.=B0[2]; loc.magnetic_field.Bz.=B0[3]
            v
        end
        pairing=sum(sum(getfield(bar,f).*dirs[f]) for f in ADJ_FIELDS)+
            sum(bar.magnetic_field.Bx.*dB[1])+sum(bar.magnetic_field.By.*dB[2])+sum(bar.magnetic_field.Bz.*dB[3])
        @test adjoint_fd_error(pen,pairing)<1e-7
    end

    @testset "HSEMHS equilibrium VJP ($kind, B $magnetic)" for kind in (:log10tau500,:log10m,:z_m),magnetic in (false,true)
        kind===:z_m && magnetic && continue                         # column HSE on the height grid does not use B
        atmosphere=equilibrium_atmosphere(kind;magnetic=magnetic)
        distributed=distribute_atmosphere(Float64,atmosphere,context)
        mode=kind===:z_m ? :column_hse_z : kind===:log10m ? :column_hse_logm : :column_hse_ltau
        cfg=EquilibriumConfig("HSEMHS",:column_hse,mode,274.0,HSEMHS_TABLE,:reject_trial,column_options(kind),nothing)
        backend=prepare_equilibrium(cfg,distributed,context)
        set_equilibrium_record!(backend,true)
        @test reconstruct_equilibrium!(backend,distributed,context).valid
        a=distributed.local_atmosphere
        bar=AtmosphereCotangent(a)
        for f in (:pgas,:rho,:ne,:chi500); getfield(bar,f).=randn(rng,size(a.temperature))./getfield(a,f); end
        kind===:z_m || (bar.z.=randn(rng,size(a.temperature))./1e5)
        status=equilibrium_vjp!(bar,backend,distributed,context)
        @test status===:exact
        dT=a.temperature.*0.01.*randn(rng,size(a.temperature))
        dB=magnetic ? (1e-3 .* randn(rng,size(a.temperature)),1e-3 .* randn(rng,size(a.temperature)),1e-3 .* randn(rng,size(a.temperature))) : nothing
        T0=copy(a.temperature); B0=magnetic ? (copy(a.magnetic_field.Bx),copy(a.magnetic_field.By),copy(a.magnetic_field.Bz)) : nothing
        function pair(h)
            a.temperature.=T0.+h.*dT
            if magnetic
                a.magnetic_field.Bx.=B0[1].+h.*dB[1]; a.magnetic_field.By.=B0[2].+h.*dB[2]; a.magnetic_field.Bz.=B0[3].+h.*dB[3]
            end
            reconstruct_equilibrium!(backend,distributed,context).valid || error("invalid trial")
            v=sum(sum(getfield(bar,f).*getfield(a,f)) for f in (:pgas,:rho,:ne,:chi500))+(kind===:z_m ? 0.0 : sum(bar.z.*a.z))
            a.temperature.=T0
            magnetic && (a.magnetic_field.Bx.=B0[1]; a.magnetic_field.By.=B0[2]; a.magnetic_field.Bz.=B0[3])
            v
        end
        pairing=sum(bar.temperature.*dT)+(magnetic ? sum(bar.magnetic_field.Bx.*dB[1])+sum(bar.magnetic_field.By.*dB[2])+
            sum(bar.magnetic_field.Bz.*dB[3]) : 0.0)
        @test adjoint_fd_error(pair,pairing)<1e-6
        close_equilibrium!(backend)
    end

    @testset "objective gradient: the equilibrium adjoint against finite differences ($kind)" for kind in (:log10tau500,:z_m)
        nz,nx,ny=24,4,3
        axis=kind===:z_m ? collect(range(1.5e6,-4.0e5;length=nz)) : collect(range(-6.0,1.2;length=nz))
        grid=Grid3D(VerticalAxis(kind,axis),4.0e4 .* collect(0:nx-1),4.0e4 .* collect(0:ny-1))
        T0=[(kind===:z_m ? 6000.0+1500.0*tanh(axis[k]/4.0e5) : 4600.0+1500.0*exp10(min(axis[k],0.5)/2))+20.0*i-15.0*j
            for k in 1:nz,i in 1:nx,j in 1:ny]
        shape=(nz,nx,ny)
        atmosphere=Atmosphere3D(grid,T0,fill(400.0,shape),fill(-300.0,shape),fill(200.0,shape),fill(800.0,shape))
        distributed=distribute_atmosphere(Float64,atmosphere,context)
        wavelength=collect(range(656.245e-9,656.315e-9,length=5))
        workspace=HybridForwardWorkspace(Float64,distributed,wavelength,StokesSet(:I),Dict(:H=>2))
        synthesizer=MixedIntensitySynthesizer([OpacityContributor(TabulatedOpacityModel(fill(2e-5,nz,5),fill(2e8,nz,5))),
            OpacityContributor(FFNOTransition(:H,656.28e-9,1,2,0.64,1.6735575e-27,1e8),workspace.populations[:H])])
        populations=CompositeDistributedPopulationModel(Dict(:H=>LocalDistributedPopulationModel(AdjointTwoLevelPopulation(1e12),2)))
        mode=kind===:z_m ? :column_hse_z : :column_hse_ltau
        backend=prepare_equilibrium(EquilibriumConfig("HSEMHS",:column_hse,mode,274.0,HSEMHS_TABLE,:reject_trial,
            kind===:z_m ? H.ColumnZOptions(anchor=:bottom,P_ref=1.2e4) : H.ColumnTauOptions(align="none"),nothing),distributed,context)
        model=HybridForwardModel(populations,NonPRD(),synthesizer,IdentityObservation(),backend,CapabilityManifest())
        nodes=kind===:z_m ? [1.0e6,0.0] : [-4.0,0.5]
        layout=ControlMapLayout(ControlMapSpec(:temperature,NodeField(reshape([5200.0,6000.0],2,1,1),nodes,kind);
            lower=3500.0,upper=9000.0,scale=1000.0))
        regularization=RegularizationSpec(vertical=VerticalRegularizationSpec((1,0,0,0,0,0,1),1e-5,ntuple(_->1.0,7);
                length_scale_m=kind===:z_m ? 1e5 : nothing),
            horizontal=Dict(:temperature=>1e-6),scales=Dict(:temperature=>1000.0,:pgas_boundary=>1.0),horizontal_order=1)
        data=forward!(workspace,model,distributed,context).spectrum.data
        observation=ObservationCube(SpectralCube(data.*(1 .+ 0.01 .* reshape(sin.(1:length(data)),size(data))),wavelength,StokesSet(:I)),
            fill(maximum(abs,data)*1e-3,size(data)),ones(size(data)))
        problem=DistributedInversionProblem(model,workspace,distributed,observation,regularization,4.0e4,4.0e4,context;
            control_layout=layout,reference_id="adjoint-test-$kind")
        parameters=initial_parameters(layout).+[150.0,-100.0]
        adjoint=objective_gradient!(HybridAdjointObjectiveGradient(),problem,layout,parameters,context)
        fdeq=objective_gradient!(HybridAdjointObjectiveGradient(equilibrium=:finite_difference,force_balance_step=1e-5),
            problem,layout,parameters,context)
        oracle=objective_gradient!(FiniteDifferenceObjectiveGradient(step=1e-5),problem,layout,parameters,context)
        rel(a,b)=norm(a.gradient-b.gradient)/max(norm(b.gradient),eps())
        @info "objective gradient" kind adjoint_vs_fd_equilibrium=rel(adjoint,fdeq) adjoint_vs_oracle=rel(adjoint,oracle)
        @test norm(oracle.gradient)>0
        @test rel(adjoint,fdeq)<1e-5
        @test rel(adjoint,oracle)<1e-4
        # the first gradient at the initial parameters, with the factory's prepared (unrecorded) equilibrium
        initial=initial_parameters(layout)
        apply_control_maps!(distributed,layout,initial)
        model.prepared_equilibrium[]=forward!(workspace,model,distributed,context).equilibrium
        prepared=DistributedInversionProblem(model,workspace,distributed,observation,regularization,4.0e4,4.0e4,context;
            control_layout=layout,reference_id="adjoint-prepared-$kind")
        @test prepared.prepared_initial_available[]
        first=objective_gradient!(HybridAdjointObjectiveGradient(),prepared,layout,initial,context)
        @test rel(first,objective_gradient!(FiniteDifferenceObjectiveGradient(step=1e-5),prepared,layout,initial,context))<1e-4
        close_equilibrium!(backend)
    end
end
end
