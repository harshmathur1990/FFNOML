using FFNOInversion
using Serialization

struct MPIPhase6Synthesizer <: AbstractSynthesizer end
function FFNOInversion.synthesize!(cube::SpectralCube{T},::MPIPhase6Synthesizer,::NonPRD,
        atmosphere::Atmosphere3D,populations,cache=nothing) where T
    nz,nx,ny=size(atmosphere.temperature)
    for i in 1:nx,j in 1:ny
        cube.data[1,1,i,j]=atmosphere.temperature[1,i,j]/T(5000)
        cube.data[2,1,i,j]=atmosphere.temperature[nz,i,j]/T(5000)
        cube.data[3,1,i,j]=one(T)+atmosphere.vz[1,i,j]/T(5000)
        cube.data[4,1,i,j]=one(T)+atmosphere.vz[nz,i,j]/T(5000)
    end
    cube
end

struct MPIPhase6VJP <: AbstractObjectiveGradient end
function FFNOInversion.objective_gradient!(::MPIPhase6VJP,
        problem::DistributedInversionProblem,layout::ControlMapLayout{T},
        parameters::AbstractVector,context::ParallelContext) where T
    evaluation=evaluate_objective!(problem,layout,parameters,context)
    synthetic=problem.workspace.output.data; observation=problem.observation
    spectrum_bar=similar(synthetic)
    @. spectrum_bar=(T(2)/problem.active_residual_count)*observation.inversion_weights^2*
        (synthetic-observation.spectrum.data)/observation.sigma^2
    atmosphere=problem.distributed.local_atmosphere; nz=size(atmosphere.temperature,1)
    temperature_bar=zeros(T,size(atmosphere.temperature)); vz_bar=zeros(T,size(atmosphere.vz))
    for i in axes(temperature_bar,2),j in axes(temperature_bar,3)
        temperature_bar[1,i,j]+=spectrum_bar[1,1,i,j]/T(5000)
        temperature_bar[nz,i,j]+=spectrum_bar[2,1,i,j]/T(5000)
        vz_bar[1,i,j]+=spectrum_bar[3,1,i,j]/T(5000)
        vz_bar[nz,i,j]+=spectrum_bar[4,1,i,j]/T(5000)
    end
    gradient=zeros(T,layout.parameter_count)
    accumulate_control_vjp!(gradient,layout,:temperature,temperature_bar,problem.distributed,context)
    accumulate_control_vjp!(gradient,layout,:vz,vz_bar,problem.distributed,context)
    ObjectiveGradientEvaluation(evaluation,gradient,1)
end

struct MPIRootPopulationVJP <: AbstractPopulationModel end
function FFNOInversion.predict_populations!(out,::MPIRootPopulationVJP,atmosphere,cache=nothing)
    @views out[:,:,:,1].=atmosphere.temperature
    @views out[:,:,:,2].=2 .* atmosphere.temperature
    out
end
function FFNOInversion.population_vjp!(feature_bar,z_bar,::MPIRootPopulationVJP,atmosphere,population_bar)
    fill!(feature_bar,0); fill!(z_bar,0)
    @views feature_bar[1,:,:,:].=population_bar[:,:,:,1].+2 .* population_bar[:,:,:,2]
    feature_bar,z_bar
end

function phase6_layout(temperature,vz)
    temp=NodeField(reshape(Float64.(temperature),2,2,1),[-5.0,-1.0])
    velocity=NodeField(reshape(Float64.(vz),2,1,1),[-5.0,-1.0])
    ControlMapLayout(ControlMapSpec(:temperature,temp;lower=3000.0,upper=8000.0,scale=1000.0),
        ControlMapSpec(:vz,velocity;lower=-3000.0,upper=3000.0,scale=1000.0))
end

context=initialize_parallel(options=ParallelOptions(enabled=true,threads_per_rank=Threads.nthreads()))
try
    nx,ny=5,4
    grid=Grid3D([-5.0,-3.0,-1.0],collect(0.0:50e3:(nx-1)*50e3),collect(0.0:50e3:(ny-1)*50e3))
    root_atmosphere=if isroot(context)
        shape=(3,nx,ny); zeros3=zeros(shape)
        Atmosphere3D(grid,fill(5500.0,shape),copy(zeros3),copy(zeros3),copy(zeros3),copy(zeros3))
    else
        nothing
    end
    distributed=distribute_atmosphere(Float64,root_atmosphere,context); root_atmosphere=nothing
    wavelength=[500.0e-9,500.1e-9,500.2e-9,500.3e-9]
    force=ForceBalanceOptions(max_iterations=4,relative_tolerance=5.0,force_tolerance=5.0,
        height_tolerance_m=1e9,relaxation=0.5,pressure_sweeps=4)
    model=HybridForwardModel(LocalDistributedPopulationModel(MockPopulationModel(1e10),1),NonPRD(),
        MPIPhase6Synthesizer(),IdentityObservation(),IdealGasEOS(),ReferenceOpacity500(kappa_m2_kg=0.02),
        HE3DBoundaryState(1e-10,1.0,:top),force,CapabilityManifest())
    workspace=HybridForwardWorkspace(Float64,distributed,wavelength,StokesSet(:I),1)
    truth_layout=phase6_layout([5000,6000,5500,6500],[-1000,1000])
    apply_control_maps!(distributed,truth_layout,initial_parameters(truth_layout))
    truth=forward!(workspace,model,distributed,context)

    # Exercise the production gather -> rank-0 VJP -> scatter route. Only rank
    # zero owns the model, while every rank supplies and receives tile arrays.
    root_vjp=RootDistributedPopulationModel(isroot(context) ? MPIRootPopulationVJP() : nothing,2)
    vjp_populations=zeros(Float64,size(distributed.local_atmosphere.temperature)...,2)
    predict_distributed_populations!(vjp_populations,root_vjp,distributed,context)
    population_bar=similar(vjp_populations); @views population_bar[:,:,:,1].=1; @views population_bar[:,:,:,2].=2
    atmosphere_bar=AtmosphereCotangent(distributed.local_atmosphere)
    predict_distributed_populations_vjp!(atmosphere_bar,root_vjp,distributed,population_bar,context)
    all(atmosphere_bar.temperature.==5) || error("rank-0 population VJP returned an incorrect temperature cotangent")
    all(iszero,atmosphere_bar.z) || error("rank-0 population VJP returned an unexpected z cotangent")
    isroot(context) && println("MPI_PHASE6_ROOT_POPULATION_VJP_OK ranks=$(context.size)")

    observation=ObservationCube(SpectralCube(copy(truth.spectrum.data),wavelength,StokesSet(:I)),
        fill(0.02,size(truth.spectrum.data)),ones(size(truth.spectrum.data)))
    regularization=RegularizationSpec(vertical=VerticalRegularizationSpec(ntuple(_->0,7),0.0,ntuple(_->1.0,7)),
        horizontal=Dict{Symbol,Float64}(),scales=Dict{Symbol,Float64}(),horizontal_order=1)
    problem=DistributedInversionProblem(model,workspace,distributed,observation,regularization,50e3,50e3,context)
    initial_layout=phase6_layout(fill(4500.0,4),[0.0,0.0])
    iterations=parse(Int,get(ENV,"PHASE6_MAX_ITERATIONS","8"))
    checkpoint_path=get(ENV,"PHASE6_CHECKPOINT_PATH","")
    restart=get(ENV,"PHASE6_RESTART","0")=="1"
    result=lbfgs_invert!(problem,initial_layout,MPIPhase6VJP(),context;restart=restart,
        options=LBFGSSolverOptions(max_iterations=iterations,history_length=4,
            gradient_tolerance=1e-12,objective_tolerance=0.0,checkpoint_path=checkpoint_path))
    if isroot(context)
        result_path=get(ENV,"PHASE6_RESULT_PATH","")
        if !isempty(result_path)
            open(result_path,"w") do io
                serialize(io,(parameters=result.state.parameters,gradient=result.state.gradient,
                    objective=result.state.objective,history=result.state.history,
                    forward_evaluations=result.state.forward_evaluations,
                    iteration=result.state.iteration,termination=result.state.termination))
            end
        end
        println("MPI_PHASE6_OK ranks=$(context.size) iteration=$(result.state.iteration) objective=$(result.state.objective)")
    end
finally
    finalize_parallel!(context)
end
