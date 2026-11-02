using Test
using FFNOInversion
using Serialization

# Inclined view on a native height grid across ranks (docs/viewing-angle/README.md §6, V8):
# `mpiexec -n R julia --project test/mpi_view_worker.jl OUT` (R = 1, 2, 3; x slabs, NX = 11 gives uneven splits and
# shifts that cross several ranks). Rank 0 serializes the gathered spectrum, the objective gradient and the HSEMHS
# τ_LOS to OUT_n<R>.jls, so the driver can compare rank counts.

include(joinpath(@__DIR__,"test_equilibrium.jl"))
const H=FFNOInversion.HSEMHS
const TABLE=get(ENV,"HSEMHS_EOS_TABLE","/mn/stornext/d9/data/harshm/MHS/HSEMHS/cache/eos_table_fullz.h5")

struct MPIViewPopulation <: AbstractPopulationModel end
function FFNOInversion.predict_populations!(out,::MPIViewPopulation,a::Atmosphere3D,cache=nothing)
    @views out[:,:,:,1].=1e10 .* a.temperature./5000; @views out[:,:,:,2].=1e9 .* a.temperature./5000; out
end
function FFNOInversion.population_vjp!(feature_bar,z_bar,::MPIViewPopulation,a,population_bar)
    fill!(feature_bar,0); fill!(z_bar,0)
    @views feature_bar[1,:,:,:].=(1e10 .* population_bar[:,:,:,1].+1e9 .* population_bar[:,:,:,2])./5000
    feature_bar,z_bar
end

out=ARGS[1]
context=initialize_parallel(options=ParallelOptions(enabled=true,decomposition=:x_slabs,threads_per_rank=Threads.nthreads()))
R=context.size
# t ≈ (1.21, 0.56): up to ~30 cells over the column; VIEW_MU=1 runs the same problem vertically (comparison)
view=ViewGeometry(mu=parse(Float64,get(ENV,"VIEW_MU","0.6")),azimuth_deg=25.0,geometry=:inclined)
nz,nx,ny=20,11,6
z=collect(range(1.0e6,-2.0e5;length=nz))
grid=Grid3D(VerticalAxis(:z_m,z),4.0e4 .* collect(0:nx-1),5.0e4 .* collect(0:ny-1),view)
root_atmosphere=isroot(context) ? begin
    T=[5200.0+1300.0*tanh(-z[k]/3e5)+250.0*sin(2π*(i-1)/nx)*cos(2π*(j-1)/ny) for k in 1:nz,i in 1:nx,j in 1:ny]
    v(s)=[s*sin(0.3k+0.9i-0.5j) for k in 1:nz,i in 1:nx,j in 1:ny]
    Atmosphere3D(grid,T,v(1.5e3),v(-1e3),v(2e3),fill(900.0,nz,nx,ny);z=repeat(reshape(z,nz,1,1),1,nx,ny))
end : nothing
results=Dict{Symbol,Any}()
ts=@testset "inclined native-z footprint n=$R" begin
    distributed=distribute_atmosphere(Float64,root_atmosphere,context)
    wavelength=collect(range(656.245e-9,656.315e-9,length=5))
    workspace=HybridForwardWorkspace(Float64,distributed,wavelength,StokesSet(:I),Dict(:H=>2))
    @test (workspace.footprint!==nothing)==is_slanted(view)
    line=FFNOTransition(:H,656.28e-9,1,2,0.64,1.6735575e-27,1e8)
    synthesizer=MixedIntensitySynthesizer([OpacityContributor(TabulatedOpacityModel(fill(2e-6,nz,5),fill(2e2,nz,5))),
        OpacityContributor(line,workspace.populations[:H])])
    model=HybridForwardModel(CompositeDistributedPopulationModel(Dict(:H=>LocalDistributedPopulationModel(MPIViewPopulation(),2))),
        NonPRD(),synthesizer,IdentityObservation(),TestHydrostaticEquilibrium(),CapabilityManifest())
    nodes=[1.0e6,-2.0e5]
    layout=ControlMapLayout(ControlMapSpec(:temperature,NodeField(reshape([5200.0,6300.0],2,1,1),nodes,:z_m);
            lower=3500.0,upper=8000.0,scale=1000.0),
        ControlMapSpec(:vlos,NodeField(reshape([0.0,0.0],2,1,1),nodes,:z_m);lower=-5e3,upper=5e3,scale=1000.0))
    result=forward!(workspace,model,distributed,context)
    data=result.spectrum.data
    sigma=1e-3*allreduce_max(maximum(data),context)                  # global, so the objective is rank independent
    observation=ObservationCube(SpectralCube(data.*1.01,wavelength,StokesSet(:I)),fill(sigma,size(data)),
        ones(size(data)))
    regularization=RegularizationSpec(vertical=VerticalRegularizationSpec((0,0,0,0,0,0,0),0.0,ntuple(_->1.0,7)),
        horizontal=Dict(:vlos=>1e-6),scales=Dict(:vlos=>1000.0),horizontal_order=1)
    problem=DistributedInversionProblem(model,workspace,distributed,observation,regularization,4e4,5e4,context;
        control_layout=layout,reference_id="mpi-view")
    parameters=initial_parameters(layout).+[150.0,-100.0,300.0,-200.0]
    gradient=objective_gradient!(HybridAdjointObjectiveGradient(force_balance_step=2e-5),problem,layout,parameters,context)
    spectrum=gather_spectrum(workspace.output,distributed,context)
    @test all(isfinite,gradient.gradient)
    oracle=objective_gradient!(FiniteDifferenceObjectiveGradient(step=2e-5),problem,layout,parameters,context)
    relative=sum(abs2,gradient.gradient.-oracle.gradient)^0.5/sum(abs2,oracle.gradient)^0.5
    @test relative<5e-3
    isroot(context) && (results[:spectrum]=spectrum.data; results[:gradient]=gradient.gradient;
        results[:objective]=[gradient.evaluation.components.total,gradient.evaluation.components.regularization])
    isroot(context) && println("MPI_VIEW n=$R adjoint vs FD relative $relative objective $(gradient.evaluation.components.total) "*
        "regularization $(gradient.evaluation.components.regularization)")

    # HSEMHS column HSE on the height grid with the view: τ along the slanted rays at the grid points
    if isfile(TABLE)
        d=distribute_atmosphere(Float64,root_atmosphere,context)
        backend=prepare_equilibrium(EquilibriumConfig("HSEMHS",:column_hse,:column_hse_z,274.0,TABLE,:reject_trial,
            H.ColumnZOptions(anchor=:top,P_ref=0.3),nothing),d,context)
        outcome=reconstruct_equilibrium!(backend,d,context)
        @test outcome.valid
        a=gather_atmosphere(d,context)
        isroot(context) && (results[:logtau500]=a.logtau500; results[:pgas]=a.pgas)
        close_equilibrium!(backend)
    end
end
# a failing @testset throws on every rank before this line, so reaching it means the rank-local tests passed
isroot(context) && open(io->serialize(io,(results=results,passed=true)),"$(out)_n$(R).jls","w")
finalize_parallel!(context)
