"""Canonical two-file scientific input after selecting one time plane."""
struct InversionInputBundle{A,O,P}
    atmosphere::A
    observation::O
    pressure_top::P
end

"""Application factory kept separate from the two scientific input files.

`population_levels` is an integer or species-to-level dictionary. `build_model`
is called on every rank as
`build_model(config, distributed, workspace, local_pressure_top, context)`.
Only rank 0 should construct a live Python FFNO model; other ranks place
`nothing` in `RootDistributedPopulationModel`.
"""
struct InversionModelFactory{L,B}
    population_levels::L
    build_model::B
end

struct InversionRunResult{S,Y,A,P,O,V}
    solver::S
    synthesis::Y
    atmosphere::A
    populations::P
    objective::O
    provenance::V
end

function _read_dataset(file,path)
    haskey(file,path) || throw(ArgumentError("HDF5 dataset not found: $path"))
    read(file[path])
end

function _time_slice(value,time_index,without_time_ndims,name)
    if ndims(value)==without_time_ndims
        return Array(value)
    elseif ndims(value)==without_time_ndims+1
        1<=time_index<=size(value,1) || throw(ArgumentError(
            "$name time index $time_index is outside 1:$(size(value,1))"))
        return Array(selectdim(value,1,time_index))
    end
    throw(DimensionMismatch("$name needs $without_time_ndims dimensions, optionally preceded by time"))
end

_zyx_to_zxy(value)=permutedims(value,(1,3,2))
_syx_to_xys(value)=permutedims(value,(2,1))
_slyx_to_lsxy(value)=permutedims(value,(2,1,4,3))

"""Read the configured observation and initial-atmosphere HDF5 files.

File axes follow the user contract: atmosphere `(time,z,y,x)` and observation
`(time,Stokes,wavelength,y,x)`. Internal axes are `(z,x,y)` and
`(wavelength,Stokes,x,y)`.
"""
function read_inversion_inputs(config::RunConfig{T}) where T
    atmosphere_data=h5open(config.atmosphere.file,"r") do file
        logtau_raw=_read_dataset(file,config.atmosphere.logtau500_dataset)
        logtau=ndims(logtau_raw)==1 ? vec(logtau_raw) : vec(_time_slice(logtau_raw,config.time_index,1,"logtau500"))
        temperature=_zyx_to_zxy(_time_slice(_read_dataset(file,config.atmosphere.temperature_dataset),
            config.time_index,3,"temperature"))
        vx=_zyx_to_zxy(_time_slice(_read_dataset(file,config.atmosphere.vx_dataset),config.time_index,3,"vx"))
        vy=_zyx_to_zxy(_time_slice(_read_dataset(file,config.atmosphere.vy_dataset),config.time_index,3,"vy"))
        vz=_zyx_to_zxy(_time_slice(_read_dataset(file,config.atmosphere.vz_dataset),config.time_index,3,"vz"))
        size(temperature)==size(vx)==size(vy)==size(vz) || throw(DimensionMismatch(
            "initial-atmosphere parameter arrays differ"))
        nz,nx,ny=size(temperature); length(logtau)==nz || throw(DimensionMismatch(
            "logtau500 length differs from atmospheric depth"))
        pressure=if config.atmosphere.pressure_top isa String
            raw=_read_dataset(file,config.atmosphere.pressure_top)
            selected=ndims(raw)==0 ? fill(T(raw[]),ny,nx) : _time_slice(raw,config.time_index,2,"pressure_top")
            T.(_syx_to_xys(selected))
        else
            fill(T(config.atmosphere.pressure_top),nx,ny)
        end
        magnetic=if config.atmosphere.magnetic_datasets===nothing
            nothing
        else
            names=config.atmosphere.magnetic_datasets
            arrays=map(name->T.(_zyx_to_zxy(_time_slice(_read_dataset(file,name),
                config.time_index,3,"magnetic field"))),names)
            MagneticField3D(arrays...)
        end
        x=T.(0:nx-1).*config.synthesis.dx_m; y=T.(0:ny-1).*config.synthesis.dy_m
        grid=Grid3D(T.(logtau),x,y)
        atmosphere=Atmosphere3D(grid,T.(temperature),T.(vx),T.(vy),T.(vz),zeros(T,nz,nx,ny);
            magnetic_field=magnetic)
        (atmosphere=atmosphere,pressure=pressure)
    end
    observation=h5open(config.observed.file,"r") do file
        intensity=T.(_slyx_to_lsxy(_time_slice(_read_dataset(file,config.observed.intensity_dataset),
            config.time_index,4,"observed intensity")))
        sigma=T.(_slyx_to_lsxy(_time_slice(_read_dataset(file,config.observed.sigma_dataset),
            config.time_index,4,"observed sigma")))
        wavelength_weights=T.(_time_slice(_read_dataset(file,config.weights.wavelength_dataset),
            config.time_index,2,"wavelength weights"))
        spatial_weights=T.(_syx_to_xys(_time_slice(_read_dataset(file,config.weights.spatial_dataset),
            config.time_index,2,"spatial weights")))
        size(intensity)==size(sigma) || throw(DimensionMismatch("intensity and sigma shapes differ"))
        size(intensity,1)==length(config.synthesis.wavelength_m) || throw(DimensionMismatch(
            "observed wavelength count differs from configured synthesis grid"))
        size(intensity,2)==length(config.stokes.components) || throw(DimensionMismatch(
            "observed Stokes count differs from configuration"))
        weights=build_inversion_weights(wavelength_weights,spatial_weights)
        size(weights)==size(intensity) || throw(DimensionMismatch(
            "wavelength/spatial weights do not span the observed cube"))
        ObservationCube(SpectralCube(intensity,T.(config.synthesis.wavelength_m),config.stokes),sigma,weights)
    end
    size(observation.spectrum.data)[3:4]==size(atmosphere_data.atmosphere.temperature)[2:3] ||
        throw(DimensionMismatch("observation and initial atmosphere spatial grids differ"))
    InversionInputBundle(atmosphere_data.atmosphere,observation,atmosphere_data.pressure)
end

function _with_time_zyx(value)
    zyx=permutedims(value,(1,3,2)); reshape(zyx,(1,size(zyx)...))
end

function _with_time_slyx(value)
    slyx=permutedims(value,(2,1,4,3)); reshape(slyx,(1,size(slyx)...))
end

"""Write exactly the configured synthesis and atmosphere products on rank 0."""
function write_inversion_outputs(config::RunConfig,result::InversionRunResult)
    result.synthesis===nothing && return nothing
    mkpath(dirname(abspath(config.outputs.synthesis_file)))
    mkpath(dirname(abspath(config.outputs.atmosphere_file)))
    h5open(config.outputs.synthesis_file,"w") do file
        file["intensity"]=_with_time_slyx(result.synthesis.data)
        file["wavelength_m"]=result.synthesis.wavelength_m
        file["objective_total"]=result.objective.components.total
        file["objective_data"]=result.objective.components.data
        file["objective_regularization"]=result.objective.components.regularization
        attributes(file)["axis_order"]="time,stokes,wavelength,y,x"
    end
    atmosphere=result.atmosphere
    h5open(config.outputs.atmosphere_file,"w") do file
        file["logtau_500"]=atmosphere.grid.log_tau500
        for name in (:temperature,:vx,:vy,:vz,:vturb,:pgas,:rho,:ne,:z)
            value=getfield(atmosphere,name); value===nothing || (file[string(name)]=_with_time_zyx(value))
        end
        if atmosphere.magnetic_field!==nothing
            for name in (:Bx,:By,:Bz)
                file[string(name)]=_with_time_zyx(getfield(atmosphere.magnetic_field,name))
            end
        end
        if result.populations isa AbstractDict
            group=create_group(file,"populations")
            for (species,value) in result.populations
                group[string(species)]=permutedims(value,(4,1,3,2))
            end
        elseif result.populations!==nothing
            file["populations"]=permutedims(result.populations,(4,1,3,2))
        end
        attributes(file)["axis_order"]="time,z,y,x; populations=level,z,y,x"
        attributes(file)["solver_termination"]=hasproperty(result.solver.state,:termination) ?
            string(result.solver.state.termination) : (result.solver.state.converged ? "converged" : "maximum_iterations")
    end
    (synthesis=config.outputs.synthesis_file,atmosphere=config.outputs.atmosphere_file)
end

_factory_levels(factory::InversionModelFactory,config)=factory.population_levels isa Function ?
    factory.population_levels(config) : factory.population_levels

function _gather_populations(populations,distributed,context)
    grid=distributed.global_grid; tile=distributed.tile; nz=length(grid.log_tau500)
    nx=length(grid.x); ny=length(grid.y)
    gather_one(value,tag)=begin
        packed=permutedims(value,(1,4,2,3))
        global_packed=gather_field(_field(packed,tile,(nz,size(value,4),nx,ny)),context;tag=tag)
        isroot(context) ? permutedims(global_packed,(1,3,4,2)) : nothing
    end
    if populations isa AbstractDict
        gathered=Dict{Symbol,Any}()
        for (offset,species) in enumerate(sort!(collect(keys(populations));by=string))
            value=gather_one(populations[species],780+offset)
            isroot(context) && (gathered[species]=value)
        end
        return isroot(context) ? gathered : nothing
    end
    gather_one(populations,779)
end

"""Execute the sole MPI/hybrid two-input/two-output inversion route."""
function run_inversion!(config::RunConfig,root_inputs,
        factory::InversionModelFactory,context::ParallelContext;
        gradient_backend::AbstractObjectiveGradient=HybridAdjointObjectiveGradient(),restart=false)
    metadata=mpi_broadcast(if isroot(context)
        root_inputs isa InversionInputBundle || throw(ArgumentError("rank 0 must provide InversionInputBundle"))
        (atmosphere_shape=size(root_inputs.atmosphere.temperature),
            observation_shape=size(root_inputs.observation.spectrum.data))
    else nothing end,context)
    distributed=distribute_atmosphere(Float64,isroot(context) ? root_inputs.atmosphere : nothing,context)
    pressure_field=distribute_field(Float64,isroot(context) ? root_inputs.pressure_top : nothing,
        (metadata.atmosphere_shape[2],metadata.atmosphere_shape[3]),context;tag=760).values
    levels=_factory_levels(factory,config)
    workspace=HybridForwardWorkspace(Float64,distributed,config.synthesis.wavelength_m,config.stokes,levels)
    model=factory.build_model(config,distributed,workspace,pressure_field,context)
    model isa HybridForwardModel || throw(ArgumentError("model factory must return HybridForwardModel"))
    local_observation=distribute_observation(Float64,isroot(context) ? root_inputs.observation : nothing,
        metadata.observation_shape,config.synthesis.wavelength_m,config.stokes,context)
    layout=mpi_broadcast(isroot(context) ? build_control_layout(root_inputs.atmosphere,config.controls) : nothing,context)
    problem=DistributedInversionProblem(model,workspace,distributed,local_observation,
        config.regularization,config.synthesis.dx_m,config.synthesis.dy_m,context)
    solver=if config.solver isa LBFGSSolverOptions
        lbfgs_invert!(problem,layout,gradient_backend,context;options=config.solver,restart=restart)
    else
        restart ? prototype_invert!(problem,layout,context;options=config.solver,restart=true) :
            prototype_invert!(problem,layout,context;options=config.solver)
    end
    final_parameters=solver.state.parameters
    objective=evaluate_objective!(problem,layout,final_parameters,context)
    synthesis=gather_spectrum(workspace.output,distributed,context)
    atmosphere=gather_atmosphere(distributed,context)
    populations=_gather_populations(workspace.populations,distributed,context)
    provenance=parallel_provenance(context,distributed.tile;capabilities=["intensity","non_prd",
        "matrix_free_vjp","lbfgs"])
    InversionRunResult(solver,synthesis,atmosphere,populations,objective,provenance)
end

function run_inversion_files!(config_path::AbstractString,factory::InversionModelFactory;
        gradient_backend::AbstractObjectiveGradient=HybridAdjointObjectiveGradient(),restart=false)
    config=load_config(config_path); context=initialize_parallel(options=config.parallel)
    try
        inputs=isroot(context) ? read_inversion_inputs(config) : nothing
        result=run_inversion!(config,inputs,factory,context;gradient_backend=gradient_backend,restart=restart)
        isroot(context) && write_inversion_outputs(config,result)
        result
    finally
        finalize_parallel!(context)
    end
end
