using FFNOInversion
using Serialization

struct MPIPhase5Synthesizer <: AbstractSynthesizer end
function FFNOInversion.synthesize!(cube::SpectralCube{T},::MPIPhase5Synthesizer,::NonPRD,
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

function layout(temperature,vz)
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
    else nothing end
    distributed=distribute_atmosphere(Float64,root_atmosphere,context); root_atmosphere=nothing
    wavelength=[500.0e-9,500.1e-9,500.2e-9,500.3e-9]
    force=ForceBalanceOptions(max_iterations=4,relative_tolerance=5.0,force_tolerance=5.0,
        height_tolerance_m=1e9,relaxation=0.5,pressure_sweeps=4)
    model=HybridForwardModel(LocalDistributedPopulationModel(MockPopulationModel(1e10),1),NonPRD(),
        MPIPhase5Synthesizer(),IdentityObservation(),IdealGasEOS(),ReferenceOpacity500(kappa_m2_kg=0.02),
        HE3DBoundaryState(1e-10,1.0,:top),force,CapabilityManifest())
    workspace=HybridForwardWorkspace(Float64,distributed,wavelength,StokesSet(:I),1)
    truth_layout=layout([5000,6000,5500,6500],[-1000,1000])
    apply_control_maps!(distributed,truth_layout,initial_parameters(truth_layout))
    truth=forward!(workspace,model,distributed,context)
    observation=ObservationCube(SpectralCube(copy(truth.spectrum.data),wavelength,StokesSet(:I)),
        fill(0.02,size(truth.spectrum.data)),ones(size(truth.spectrum.data)))
    regularization=RegularizationSpec(vertical=VerticalRegularizationSpec(ntuple(_->0,7),0.0,ntuple(_->1.0,7)),
        horizontal=Dict{Symbol,Float64}(),scales=Dict{Symbol,Float64}(),horizontal_order=1)
    problem=DistributedInversionProblem(model,workspace,distributed,observation,regularization,50e3,50e3,context)
    initial_layout=layout(fill(4500.0,4),[0.0,0.0])
    iterations=parse(Int,get(ENV,"PHASE5_MAX_ITERATIONS","8"))
    checkpoint_path=get(ENV,"PHASE5_CHECKPOINT_PATH","")
    restart=get(ENV,"PHASE5_RESTART","0")=="1"
    result=prototype_invert!(problem,initial_layout,context;restart=restart,
        options=PrototypeSolverOptions(max_iterations=iterations,initial_step=0.5,
            improvement_tolerance=1e-12,checkpoint_path=checkpoint_path))
    if isroot(context)
        result_path=get(ENV,"PHASE5_RESULT_PATH","")
        if !isempty(result_path)
            open(result_path,"w") do io
                serialize(io,(parameters=result.state.parameters,objective=result.state.objective,
                    history=result.state.history,evaluations=result.state.evaluations,iteration=result.state.iteration))
            end
        end
        println("MPI_PHASE5_OK ranks=$(context.size) iteration=$(result.state.iteration) objective=$(result.state.objective)")
    end
finally
    finalize_parallel!(context)
end
