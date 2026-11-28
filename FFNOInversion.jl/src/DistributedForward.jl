"""Rank-owned atmosphere together with its immutable global coordinate metadata."""
struct DistributedAtmosphere{T<:AbstractFloat,A<:Atmosphere3D{T}}
    local_atmosphere::A
    global_grid::Grid3D{T}
    tile::Tile2D
end

_field(a,tile,global_shape) = DistributedField(a,global_shape,tile)

function distribute_atmosphere(::Type{T},root_atmosphere,context::ParallelContext) where T<:AbstractFloat
    metadata=mpi_broadcast(if isroot(context)
        root_atmosphere===nothing && throw(ArgumentError("root must supply the initial atmosphere"))
        (grid=root_atmosphere.grid,shape=size(root_atmosphere.temperature),
         magnetic=root_atmosphere.magnetic_field!==nothing,pgas=root_atmosphere.pgas!==nothing,
         rho=root_atmosphere.rho!==nothing,ne=root_atmosphere.ne!==nothing,z=root_atmosphere.z!==nothing,
         logtau500=root_atmosphere.logtau500!==nothing,chi500=root_atmosphere.chi500!==nothing,
         h_neutral=root_atmosphere.h_neutral!==nothing,h_proton=root_atmosphere.h_proton!==nothing)
    else nothing end,context)
    shape=metadata.shape; tile=local_tile(context,shape[2],shape[3])
    getroot(name)=isroot(context) ? getfield(root_atmosphere,name) : nothing
    temperature=distribute_field(T,getroot(:temperature),shape,context;tag=301).values
    vx=distribute_field(T,getroot(:vx),shape,context;tag=302).values
    vy=distribute_field(T,getroot(:vy),shape,context;tag=303).values
    vz=distribute_field(T,getroot(:vz),shape,context;tag=304).values
    vturb=distribute_field(T,getroot(:vturb),shape,context;tag=305).values
    optional(name,flag,tag)=flag ? distribute_field(T,getroot(name),shape,context;tag=tag).values : nothing
    pgas=optional(:pgas,metadata.pgas,306); rho=optional(:rho,metadata.rho,307)
    ne=optional(:ne,metadata.ne,308); z=optional(:z,metadata.z,309)
    logtau500=optional(:logtau500,metadata.logtau500,313); chi500=optional(:chi500,metadata.chi500,314)
    h_neutral=optional(:h_neutral,metadata.h_neutral,315); h_proton=optional(:h_proton,metadata.h_proton,316)
    B=if metadata.magnetic
        rootB=isroot(context) ? root_atmosphere.magnetic_field : nothing
        bx=distribute_field(T,isroot(context) ? rootB.Bx : nothing,shape,context;tag=310).values
        by=distribute_field(T,isroot(context) ? rootB.By : nothing,shape,context;tag=311).values
        bz=distribute_field(T,isroot(context) ? rootB.Bz : nothing,shape,context;tag=312).values
        MagneticField3D(bx,by,bz)
    else nothing end
    local_grid=Grid3D(metadata.grid.vertical,copy(metadata.grid.x[tile.xrange]),copy(metadata.grid.y[tile.yrange]),
        metadata.grid.view)
    local_atmos=Atmosphere3D(local_grid,temperature,vx,vy,vz,vturb;magnetic_field=B,pgas=pgas,rho=rho,ne=ne,z=z,
        logtau500=logtau500,chi500=chi500,h_neutral=h_neutral,h_proton=h_proton)
    DistributedAtmosphere(local_atmos,metadata.grid,tile)
end

"""Collect final output state on rank 0 only."""
function gather_atmosphere(distributed::DistributedAtmosphere,context::ParallelContext)
    a=distributed.local_atmosphere; g=distributed.global_grid
    shape=(nlevels(g),length(g.x),length(g.y)); tile=distributed.tile
    gather(value,tag)=gather_field(_field(value,tile,shape),context;tag=tag)
    temperature=gather(a.temperature,321); vx=gather(a.vx,322); vy=gather(a.vy,323)
    vz=gather(a.vz,324); vturb=gather(a.vturb,325)
    pgas=a.pgas===nothing ? nothing : gather(a.pgas,326)
    rho=a.rho===nothing ? nothing : gather(a.rho,327)
    ne=a.ne===nothing ? nothing : gather(a.ne,328)
    z=a.z===nothing ? nothing : gather(a.z,329)
    logtau500=a.logtau500===nothing ? nothing : gather(a.logtau500,333)
    chi500=a.chi500===nothing ? nothing : gather(a.chi500,334)
    h_neutral=a.h_neutral===nothing ? nothing : gather(a.h_neutral,335)
    h_proton=a.h_proton===nothing ? nothing : gather(a.h_proton,336)
    B=if a.magnetic_field===nothing
        nothing
    else
        bx=gather(a.magnetic_field.Bx,330); by=gather(a.magnetic_field.By,331); bz=gather(a.magnetic_field.Bz,332)
        isroot(context) ? MagneticField3D(bx,by,bz) : nothing
    end
    isroot(context) ? Atmosphere3D(g,temperature,vx,vy,vz,vturb;magnetic_field=B,pgas=pgas,rho=rho,ne=ne,z=z,
        logtau500=logtau500,chi500=chi500,h_neutral=h_neutral,h_proton=h_proton) : nothing
end

abstract type AbstractDistributedPopulationModel end
struct LocalDistributedPopulationModel{M<:AbstractPopulationModel} <: AbstractDistributedPopulationModel
    model::M; levels::Int
end
struct RootDistributedPopulationModel{M} <: AbstractDistributedPopulationModel
    root_model::M; levels::Int
end
struct CompositeDistributedPopulationModel{M<:AbstractDict} <: AbstractDistributedPopulationModel
    models::M
    function CompositeDistributedPopulationModel(models::M) where {M<:AbstractDict}
        isempty(models) && throw(ArgumentError("composite population model cannot be empty"))
        all(value->value isa AbstractDistributedPopulationModel,values(models)) || throw(ArgumentError(
            "every composite population entry must be a distributed population model"))
        new{M}(models)
    end
end

freeze_population_mesh!(model::LocalDistributedPopulationModel,frozen::Bool=true;readapt::Bool=false)=
    (freeze_population_mesh!(model.model,frozen;readapt=readapt); model)
freeze_population_mesh!(model::RootDistributedPopulationModel,frozen::Bool=true;readapt::Bool=false)=
    (freeze_population_mesh!(model.root_model,frozen;readapt=readapt); model)
freeze_population_mesh!(model::CompositeDistributedPopulationModel,frozen::Bool=true;readapt::Bool=false)=
    (foreach(m->freeze_population_mesh!(m,frozen;readapt=readapt),values(model.models)); model)

function predict_distributed_populations!(out,backend::LocalDistributedPopulationModel,distributed,context)
    predict_populations!(out,backend.model,distributed.local_atmosphere)
end

function predict_distributed_populations!(out,backend::RootDistributedPopulationModel,distributed,context)
    global_atmosphere=gather_atmosphere(distributed,context)
    shape=(nlevels(distributed.global_grid),length(distributed.global_grid.x),length(distributed.global_grid.y))
    coordinator=RootGPUCoordinator(() -> begin
        backend.root_model===nothing && error("rank 0 has no FFNO population model")
        populations=zeros(eltype(out),shape...,backend.levels)
        predict_populations!(populations,backend.root_model,global_atmosphere)
        populations
    end;launcher_rank=context.root)
    global_populations=launch_gpu!(coordinator,context)
    packed=isroot(context) ? permutedims(global_populations,(1,4,2,3)) : nothing
    field=distribute_field(eltype(out),packed,(shape[1],backend.levels,shape[2],shape[3]),context;tag=350)
    out.=permutedims(field.values,(1,3,4,2)); out
end

"""Species request order: alphabetical unless `FFNO_POPULATION_MODEL_ORDER` (comma-separated, naming every
species once, e.g. `H,CA`) overrides it for diagnostics. Each request is collective, so the variable must be
identical on all ranks."""
function _population_model_order(models)
    names=sort!(collect(keys(models));by=string)
    raw=strip(get(ENV,"FFNO_POPULATION_MODEL_ORDER",""))
    isempty(raw) && return names
    requested=[Symbol(strip(name)) for name in split(raw,',')]
    sort(requested;by=string)==names || throw(ArgumentError(
        "FFNO_POPULATION_MODEL_ORDER=$raw must name each population model once: $(join(names,','))"))
    requested
end

function predict_distributed_populations!(out::AbstractDict,backend::CompositeDistributedPopulationModel,
        distributed,context)
    Set(keys(out))==Set(keys(backend.models)) || throw(ArgumentError(
        "population workspace species differ from composite model"))
    for species in _population_model_order(backend.models)
        predict_distributed_populations!(out[species],backend.models[species],distributed,context)
    end
    out
end

"""
Inclined view on a native height grid (docs/viewing-angle/README.md §6): the synthesis runs on the footprint
atmosphere, whose columns are the rays x = ξ + t z, built every forward by shifting the Cartesian state level by level
(`footprint_state!`). `populations` holds the Cartesian FFNO prediction; the workspace's own `populations` arrays,
which the synthesis contributors reference, then hold the footprint populations.
"""
mutable struct FootprintState{A,P}
    atmosphere::A
    populations::P
end

mutable struct HybridForwardWorkspace{T<:AbstractFloat,P}
    populations::P
    intrinsic::SpectralCube{T,Array{T,4}}
    output::SpectralCube{T,Array{T,4}}
    synthesis_cache::ThreadedSynthesisCache{T}
    footprint::Union{Nothing,FootprintState}
end
HybridForwardWorkspace(populations,intrinsic::SpectralCube,output::SpectralCube,cache::ThreadedSynthesisCache)=
    HybridForwardWorkspace(populations,intrinsic,output,cache,nothing)

"""True when the synthesis needs footprint columns: an inclined view on a native height grid."""
needs_footprint(grid::Grid3D)=vertical_coordinate(grid)===:z_m && is_slanted(grid.view)

"""Populations the FFNO predicts into: the Cartesian ones (equal to `workspace.populations` unless footprint)."""
cartesian_populations(workspace::HybridForwardWorkspace)=
    workspace.footprint===nothing ? workspace.populations : workspace.footprint.populations
"""Atmosphere the synthesis reads: the footprint atmosphere when there is one."""
synthesis_atmosphere(workspace::HybridForwardWorkspace,distributed)=
    workspace.footprint===nothing ? distributed.local_atmosphere : workspace.footprint.atmosphere

function _footprint_state(distributed::DistributedAtmosphere,populations)
    needs_footprint(distributed.global_grid) || return nothing
    a=distributed.local_atmosphere; g=a.grid
    grid=Grid3D(g.vertical,g.x,g.y,footprint_view(g.view))
    f()=fill!(similar(a.temperature),one(eltype(a.temperature)))   # filled by footprint_state! before every synthesis
    atmosphere=Atmosphere3D(grid,copy(a.temperature),copy(a.vx),copy(a.vy),copy(a.vz),copy(a.vturb);
        pgas=f(),rho=f(),ne=f(),z=copy(a.z),
        h_neutral=a.h_neutral===nothing ? nothing : copy(a.h_neutral),h_proton=a.h_proton===nothing ? nothing : copy(a.h_proton))
    FootprintState(atmosphere,deepcopy(populations))
end

"""Wall-clock breakdown for one complete distributed forward evaluation."""
struct HybridForwardTimings
    equilibrium_seconds::Float64
    populations_seconds::Float64
    synthesis_seconds::Float64
    observation_seconds::Float64
    total_seconds::Float64
end

function HybridForwardWorkspace(::Type{T},distributed::DistributedAtmosphere,wavelength,stokes,levels) where T
    a=distributed.local_atmosphere; nz,nx,ny=size(a.temperature); nλ=length(wavelength)
    cube()=SpectralCube(zeros(T,nλ,length(stokes.components),nx,ny),T.(wavelength),stokes)
    populations=zeros(T,nz,nx,ny,levels)
    HybridForwardWorkspace(populations,cube(),cube(),ThreadedSynthesisCache(T,nz,nλ),
        _footprint_state(distributed,populations))
end

function HybridForwardWorkspace(::Type{T},distributed::DistributedAtmosphere,wavelength,stokes,
        levels::AbstractDict) where T
    a=distributed.local_atmosphere; nz,nx,ny=size(a.temperature); nλ=length(wavelength)
    cube()=SpectralCube(zeros(T,nλ,length(stokes.components),nx,ny),T.(wavelength),stokes)
    populations=Dict{Symbol,Array{T,4}}(Symbol(species)=>zeros(T,nz,nx,ny,Int(count))
        for (species,count) in levels)
    HybridForwardWorkspace(populations,cube(),cube(),ThreadedSynthesisCache(T,nz,nλ),
        _footprint_state(distributed,populations))
end

# ---------------------------------------------------------------------------------------------------------------------
# Footprint state (inclined view, native z)
# ---------------------------------------------------------------------------------------------------------------------

# positive fields are shifted in log, velocities linearly (design §6)
const _FOOTPRINT_LOG_FIELDS=(:temperature,:pgas,:rho,:ne)
const _FOOTPRINT_LINEAR_FIELDS=(:vx,:vy,:vz,:vturb)

_population_dict(populations::AbstractDict)=populations
_population_dict(populations::AbstractArray)=Dict(:_single=>populations)

"""Global Cartesian populations on the root rank (species => (nz, nx, ny, levels)); `nothing` elsewhere."""
function _gather_population_dict(populations,distributed,context;tag)
    grid=distributed.global_grid; shape=(nlevels(grid),length(grid.x),length(grid.y))
    out=Dict{Symbol,Any}()
    for (offset,species) in enumerate(sort!(collect(keys(populations));by=string))
        value=populations[species]; L=size(value,4)
        packed=gather_field(_field(permutedims(value,(1,4,2,3)),distributed.tile,(shape[1],L,shape[2],shape[3])),
            context;tag=tag+offset)
        isroot(context) && (out[species]=permutedims(packed,(1,3,4,2)))
    end
    isroot(context) ? out : nothing
end

"""Scatter global (nz, nx, ny, levels) arrays from the root into the rank-local arrays of `destination`."""
function _scatter_population_dict!(destination,root_values,distributed,context;tag)
    grid=distributed.global_grid; shape=(nlevels(grid),length(grid.x),length(grid.y))
    for (offset,species) in enumerate(sort!(collect(keys(destination));by=string))
        local_value=destination[species]; L=size(local_value,4)
        packed=isroot(context) ? permutedims(root_values[species],(1,4,2,3)) : nothing
        field=distribute_field(eltype(local_value),packed,(shape[1],L,shape[2],shape[3]),context;tag=tag+offset)
        local_value.=permutedims(field.values,(1,3,4,2))
    end
    destination
end

"""Shift `A` (nz, nx, ny) by the plan, in log when `logarithmic`."""
_footprint_shift(A,plan,logarithmic)=logarithmic ? exp.(shear_levels!(similar(A),log.(A),plan)) :
    shear_levels!(similar(A),A,plan)

"""
Build the footprint atmosphere and populations from the Cartesian state (collective): gathered on the root rank,
shifted there by +t z per grid level, scattered back (design §6). The heights are the grid's, so `z` is unchanged.
"""
function footprint_state!(workspace::HybridForwardWorkspace,distributed::DistributedAtmosphere,context::ParallelContext)
    fp=workspace.footprint; fp===nothing && return nothing
    grid=distributed.global_grid; shape=(nlevels(grid),length(grid.x),length(grid.y))
    global_atmosphere=gather_atmosphere(distributed,context)
    global_populations=_gather_population_dict(_population_dict(fp.populations),distributed,context;tag=900)
    plan=isroot(context) ? ShearPlan(grid,native_depth(grid),1) : nothing
    for (i,name) in enumerate((_FOOTPRINT_LOG_FIELDS...,_FOOTPRINT_LINEAR_FIELDS...))
        destination=getfield(fp.atmosphere,name)
        root_values=isroot(context) ? _footprint_shift(getfield(global_atmosphere,name),plan,name in _FOOTPRINT_LOG_FIELDS) : nothing
        destination.=distribute_field(eltype(destination),root_values,shape,context;tag=920+i).values
    end
    # a given atmosphere's hydrogen, which only the footprint's continuum and broadening read
    for (i,name) in enumerate((:h_neutral,:h_proton))
        destination=getfield(fp.atmosphere,name); destination===nothing && continue
        root_values=isroot(context) ? _footprint_shift(getfield(global_atmosphere,name),plan,false) : nothing
        destination.=distribute_field(eltype(destination),root_values,shape,context;tag=930+i).values
    end
    shifted=isroot(context) ? Dict(species=>begin
            out=similar(value)
            for l in axes(value,4); out[:,:,:,l].=_footprint_shift(value[:,:,:,l],plan,true); end
            out
        end for (species,value) in global_populations) : nothing
    _scatter_population_dict!(_population_dict(workspace.populations),shifted,distributed,context;tag=940)
    fp.atmosphere
end

"""Distributed forward model. `equilibrium` is an `AbstractEquilibriumBackend`
(HSEMHS in production); `prepared_equilibrium` optionally holds the outcome of an
equilibrium and population evaluation already done on the current state, reused
once by `forward!(...; reuse_prepared=true)`."""
struct HybridForwardModel{P,R,S,O,E,Q}
    populations::P; redistribution::R; synthesizer::S; observation::O
    equilibrium::E; capabilities::CapabilityManifest
    prepared_equilibrium::Q
end

function HybridForwardModel(populations,redistribution,synthesizer,observation,equilibrium,
        capabilities::CapabilityManifest;prepared_equilibrium=nothing)
    HybridForwardModel(populations,redistribution,synthesizer,observation,equilibrium,capabilities,
        Ref{Any}(prepared_equilibrium))
end

function _distributed_observation!(output,intrinsic,model::IdentityObservation,distributed,context)
    copyto!(output.data,intrinsic.data); output
end

function _convolve_padded_owned(padded,kernel,axis,lx,ly)
    radius=(length(kernel)-1)÷2; lead=(size(padded,1),size(padded,2)); out=zeros(eltype(padded),lead...,lx,ly)
    for l in axes(out,1),s in axes(out,2),i in 1:lx,j in 1:ly
        acc=zero(eltype(out))
        for o in -radius:radius
            ii=axis==3 ? i+radius+o : i+radius
            jj=axis==4 ? j+radius+o : j+radius
            acc+=kernel[o+radius+1]*padded[l,s,ii,jj]
        end
        out[l,s,i,j]=acc
    end
    out
end

function _distributed_observation!(output,intrinsic,model::GaussianPSFObservation,distributed,context)
    T=eltype(intrinsic.data)
    spectral=model.spectral_fwhm_m==0 ? copy(intrinsic.data) : _convolve_axis(intrinsic.data,
        _kernel(T,T(model.spectral_fwhm_m),_uniform_spacing(intrinsic.wavelength_m)),1)
    tile=distributed.tile; global_shape=(size(spectral,1),size(spectral,2),length(distributed.global_grid.x),length(distributed.global_grid.y))
    kx=_kernel(T,T(model.spatial_fwhm_x_m),T(model.dx_m)); rx=(length(kx)-1)÷2
    xdata=rx==0 ? spectral : _convolve_padded_owned(exchange_halos(_field(spectral,tile,global_shape),context,rx),kx,3,size(spectral,3),size(spectral,4))
    ky=_kernel(T,T(model.spatial_fwhm_y_m),T(model.dy_m)); ry=(length(ky)-1)÷2
    ydata=ry==0 ? xdata : _convolve_padded_owned(exchange_halos(_field(xdata,tile,global_shape),context,ry),ky,4,size(xdata,3),size(xdata,4))
    copyto!(output.data,ydata); output
end

# The run log is in sections: a header at the margin ("Iteration 3") and that section's lines below it after a tab.
const _LOG_INDENT=Ref("")

"""Root-rank progress line for a forward stage (stdout, flushed so a running job shows where it is)."""
_forward_stage(context,message)=_log_line(context,"FORWARD stage=$message")

"""A root-rank line of the run log, indented under the current section."""
function _log_line(context,message)
    isroot(context) || return nothing
    println(stdout,_LOG_INDENT[],message); flush(stdout); nothing
end

"""Start a run-log section: `title` at the margin on rank 0, and the lines that follow it indented by a tab."""
function _log_section(context,title)
    _LOG_INDENT[]=""; _log_line(context,title); _LOG_INDENT[]="\t"; nothing
end
_end_log_section()=(_LOG_INDENT[]=""; nothing)

"""One collective forward evaluation: equilibrium, populations, synthesis,
observation. If the equilibrium is invalid (identical verdict on every rank) the
populations, synthesis and observation are skipped, the workspace keeps its
previous spectrum and populations, and `valid = false` is returned with the
equilibrium outcome; the caller decides whether that rejects a trial or aborts."""
function forward!(workspace::HybridForwardWorkspace,model::HybridForwardModel,
                  distributed::DistributedAtmosphere,context::ParallelContext;reuse_prepared::Bool=false)
    total_start=time_ns()
    validate_capabilities(model.capabilities,model.redistribution,workspace.output.stokes)
    prepared=reuse_prepared ? model.prepared_equilibrium[] : nothing
    model.prepared_equilibrium[]=nothing
    outcome,equilibrium_seconds,populations_seconds=if prepared===nothing
        stage_start=time_ns()
        equilibrium=reconstruct_equilibrium!(model.equilibrium,distributed,context)
        equilibrium_elapsed=(time_ns()-stage_start)/1e9
        if !equilibrium.valid
            timings=HybridForwardTimings(equilibrium_elapsed,0.0,0.0,0.0,(time_ns()-total_start)/1e9)
            return (spectrum=workspace.output,equilibrium=equilibrium,timings=timings,valid=false)
        end
        stage_start=time_ns()
        _forward_stage(context,"populations begin")
        predict_distributed_populations!(cartesian_populations(workspace),model.populations,distributed,context)
        populations_elapsed=(time_ns()-stage_start)/1e9
        _forward_stage(context,"populations done seconds=$(round(populations_elapsed;digits=1))")
        (equilibrium,equilibrium_elapsed,populations_elapsed)
    else
        isroot(context) && (println(stdout,"FORWARD event=reuse_prepared_equilibrium_and_populations"); flush(stdout))
        (prepared,0.0,0.0)
    end
    stage_start=time_ns()
    if workspace.footprint!==nothing
        _forward_stage(context,"footprint begin")
        footprint_state!(workspace,distributed,context)
    end
    _forward_stage(context,"synthesis begin integration=$(_axis_integration(distributed.local_atmosphere) ? vertical_coordinate(distributed.local_atmosphere.grid) : "z")")
    Base.ScopedValues.with(SYNTHESIS_PROGRESS=>isroot(context)) do
        synthesize!(workspace.intrinsic,model.synthesizer,model.redistribution,synthesis_atmosphere(workspace,distributed),
            workspace.populations,workspace.synthesis_cache)
    end
    synthesis_seconds=(time_ns()-stage_start)/1e9
    _forward_stage(context,"synthesis done seconds=$(round(synthesis_seconds;digits=1))")
    stage_start=time_ns()
    _distributed_observation!(workspace.output,workspace.intrinsic,model.observation,distributed,context)
    observation_seconds=(time_ns()-stage_start)/1e9
    _forward_stage(context,"observation done seconds=$(round(observation_seconds;digits=1))")
    timings=HybridForwardTimings(equilibrium_seconds,populations_seconds,synthesis_seconds,
        observation_seconds,(time_ns()-total_start)/1e9)
    (spectrum=workspace.output,equilibrium=outcome,timings=timings,valid=true)
end

_array_bytes(value)=value isa AbstractArray ? sizeof(eltype(value))*length(value) : 0

"""Audit array ownership in the distributed scientific hot path.

The report intentionally excludes final root gathers and the transient global
atmosphere/population staging on the GPU launcher rank.
"""
function distributed_memory_report(distributed::DistributedAtmosphere,workspace::HybridForwardWorkspace,
        context::ParallelContext)
    a=distributed.local_atmosphere
    atmosphere_bytes=sum(_array_bytes(getfield(a,name)) for name in
        (:temperature,:vx,:vy,:vz,:vturb,:pgas,:rho,:ne,:z))
    if a.magnetic_field!==nothing
        atmosphere_bytes+=sum(_array_bytes(getfield(a.magnetic_field,name)) for name in (:Bx,:By,:Bz))
    end
    population_bytes=workspace.populations isa AbstractDict ?
        sum(_array_bytes(value) for value in values(workspace.populations)) : _array_bytes(workspace.populations)
    workspace_bytes=population_bytes+_array_bytes(workspace.intrinsic.data)+
        _array_bytes(workspace.output.data)+sum(_array_bytes(ws.extinction)+_array_bytes(ws.emissivity)
            for ws in workspace.synthesis_cache.workspaces)
    local_entry=Dict{String,Any}(
        "rank"=>context.rank,
        "tile_shape"=>[length(distributed.tile.xrange),length(distributed.tile.yrange)],
        "atmosphere_bytes"=>atmosphere_bytes,
        "workspace_bytes"=>workspace_bytes,
        "owned_bytes"=>atmosphere_bytes+workspace_bytes)
    ranks=context.enabled ? (_assert_mpi_thread(context); MPI.gather(local_entry,context.comm;root=context.root)) : [local_entry]
    isroot(context) || return nothing
    Dict{String,Any}(
        "scope"=>"scientific hot path; excludes root final gathers and transient GPU staging",
        "global_spatial_shape"=>[distributed.tile.global_nx,distributed.tile.global_ny],
        "rank_count"=>context.size,
        "rank_owned"=>ranks,
        "maximum_owned_bytes"=>maximum(entry["owned_bytes"] for entry in ranks),
        "sum_owned_bytes"=>sum(entry["owned_bytes"] for entry in ranks))
end

function gather_spectrum(cube::SpectralCube,distributed::DistributedAtmosphere,context::ParallelContext)
    shape=(size(cube.data,1),size(cube.data,2),length(distributed.global_grid.x),length(distributed.global_grid.y))
    values=gather_field(_field(cube.data,distributed.tile,shape),context;tag=360)
    isroot(context) ? SpectralCube(values,cube.wavelength_m,cube.stokes) : nothing
end

function distribute_observation(::Type{T},root_observation,global_shape::NTuple{4,Int},
                                wavelength,stokes,context::ParallelContext) where T<:AbstractFloat
    spectrum=distribute_field(T,isroot(context) ? root_observation.spectrum.data : nothing,global_shape,context;tag=370).values
    sigma=distribute_field(T,isroot(context) ? root_observation.sigma : nothing,global_shape,context;tag=371).values
    weights=distribute_field(T,isroot(context) ? root_observation.inversion_weights : nothing,global_shape,context;tag=372).values
    ObservationCube(SpectralCube(spectrum,T.(wavelength),stokes),sigma,weights)
end

function distributed_chi2(synthetic::SpectralCube,observation::ObservationCube,context::ParallelContext)
    size(synthetic.data)==size(observation.spectrum.data) || throw(DimensionMismatch("local observation and synthesis differ"))
    local_chi2=zero(eltype(synthetic.data))
    for i in eachindex(synthetic.data)
        r=observation.inversion_weights[i]*(synthetic.data[i]-observation.spectrum.data[i])/observation.sigma[i]
        local_chi2+=r*r
    end
    allreduce_sum(local_chi2,context)
end

function distributed_regularization_penalty(distributed::DistributedAtmosphere,spec::RegularizationSpec,dx,dy,context)
    a=distributed.local_atmosphere; terms=Dict{Symbol,Float64}()
    for (i,var) in enumerate(VERTICAL_PARAMETER_ORDER)
        typ=spec.vertical.types[i]; typ==0 && continue
        values=_atmospheric_variable(a,var); scale=get(spec.scales,var,one(eltype(values)))
        localpen=var===:pgas_boundary && typ==1 ? _mean_square(values./scale.-1) : _vertical_penalty(values./scale,_vertical_spacing_axis(a.grid,spec),typ)
        weight=length(values); globalpen=allreduce_sum(localpen*weight,context)/allreduce_sum(weight,context)
        terms[var]=spec.vertical.regularize*spec.vertical.weights[i]*globalpen
    end
    # Horizontal terms use a root-free global reduction over uniquely owned forward differences.
    for (var,weight) in spec.horizontal
        values=_atmospheric_variable(a,var)./get(spec.scales,var,one(eltype(a.temperature)))
        order=spec.horizontal_order; halo=exchange_halos(_field(values,distributed.tile,
            (size(values,1),length(distributed.global_grid.x),length(distributed.global_grid.y))),context,order)
        sx=sy=0.0; cx=cy=0; nxg=length(distributed.global_grid.x); nyg=length(distributed.global_grid.y)
        for k in axes(values,1),i in axes(values,2),j in axes(values,3)
            gi=first(distributed.tile.xrange)+i-1; gj=first(distributed.tile.yrange)+j-1
            if gi<=nxg-order
                d=order==1 ? (halo[k,i+order+1,j+order]-halo[k,i+order,j+order])/dx :
                    (halo[k,i+order+2,j+order]-2halo[k,i+order+1,j+order]+halo[k,i+order,j+order])/dx^2
                sx+=d*d; cx+=1
            end
            if gj<=nyg-order
                d=order==1 ? (halo[k,i+order,j+order+1]-halo[k,i+order,j+order])/dy :
                    (halo[k,i+order,j+order+2]-2halo[k,i+order,j+order+1]+halo[k,i+order,j+order])/dy^2
                sy+=d*d; cy+=1
            end
        end
        gx=allreduce_sum(sx,context)/max(allreduce_sum(cx,context),1)
        gy=allreduce_sum(sy,context)/max(allreduce_sum(cy,context),1)
        terms[var]=get(terms,var,0.0)+weight*(gx+gy)
    end
    (total=sum(values(terms);init=0.0),terms=terms)
end

"""Gradient of `_vertical_penalty(a, axis, typ)` with respect to `a`."""
function _vertical_penalty_grad(a,axis,typ)
    g=zero(a)
    (typ==0 || isempty(a)) && return g
    typ==3 && return 2 .* a ./ length(a)
    typ==2 && return 2 .* (a .- sum(a,dims=1)./size(a,1)) ./ length(a)
    axis===nothing && throw(ArgumentError(
        "vertical regularization types 1 and 4 on a height grid need regularization.vertical.length_scale_m"))
    size(a,1)>=2 || return g
    spacing=reshape(diff(axis),:,1,1)
    slope=diff(a,dims=1)./spacing
    sbar=if typ==1
        2 .* slope ./ length(slope)
    else
        size(a,1)>=3 || return g
        dc=reshape(diff((axis[1:end-1].+axis[2:end])./2),:,1,1)
        curv=diff(slope,dims=1)./dc
        cb=2 .* curv ./ length(curv) ./ dc
        s=zero(slope); s[1:end-1,:,:].-=cb; s[2:end,:,:].+=cb; s
    end
    sb=sbar./spacing
    g[1:end-1,:,:].-=sb; g[2:end,:,:].+=sb
    g
end

"""Adds ∂/∂field of the cotangent `g` of the regularized variable `variable` (as `_atmospheric_variable` forms it) to `bar`."""
function _add_variable_cotangent!(bar,atmosphere,variable::Symbol,g)
    if variable===:temperature || variable in (:vx,:vy,:vz,:vturb,:pgas,:rho,:ne,:z)
        getfield(bar,variable).+=g
    elseif variable===:vlos
        v=atmosphere.grid.view
        if is_vertical(v)
            bar.vz.+=g
        else
            n=view_direction(v); bar.vx.+=n[1].*g; bar.vy.+=n[2].*g; bar.vz.+=n[3].*g
        end
    elseif variable===:pgas_boundary
        @views bar.pgas[1:1,:,:].+=g
    elseif variable in (:Bx,:By,:Bz)
        getfield(bar.magnetic_field,variable).+=g
    elseif variable in (:B,:inc,:azi)
        B=atmosphere.magnetic_field; mb=bar.magnetic_field
        @inbounds for i in eachindex(g)
            bx=B.Bx[i]; by=B.By[i]; bz=B.Bz[i]; b2=bx^2+by^2+bz^2; b=sqrt(b2)
            if variable===:B
                b>0 || continue
                mb.Bx[i]+=g[i]*bx/b; mb.By[i]+=g[i]*by/b; mb.Bz[i]+=g[i]*bz/b
            elseif variable===:inc                     # inc = acos(Bz/|B|)
                (b>0 && abs(bz/b)<1) || continue
                w=-g[i]/sqrt(1-(bz/b)^2)/(b2*b)
                mb.Bx[i]+=w*(-bz*bx); mb.By[i]+=w*(-bz*by); mb.Bz[i]+=w*(bx^2+by^2)
            else                                        # azi = atan(By, Bx)
                h2=bx^2+by^2; h2>0 || continue
                mb.Bx[i]+=-g[i]*by/h2; mb.By[i]+=g[i]*bx/h2
            end
        end
    else
        throw(ArgumentError("unsupported regularized variable $variable"))
    end
    bar
end

"""
    distributed_regularization_vjp!(bar, distributed, spec, dx, dy, context) -> bar

Adds the gradient of `distributed_regularization_penalty(...).total` with respect to the atmosphere's fields to the
`AtmosphereCotangent` bar (temperature, velocities, vturb, B, and pgas for `pgas_boundary` or a horizontal pgas term).
Collective (the horizontal terms send their halo cotangents back to their owners).
"""
function distributed_regularization_vjp!(bar,distributed::DistributedAtmosphere,spec::RegularizationSpec,dx,dy,context)
    a=distributed.local_atmosphere
    for (i,var) in enumerate(VERTICAL_PARAMETER_ORDER)
        typ=spec.vertical.types[i]; typ==0 && continue
        values=_atmospheric_variable(a,var); scale=get(spec.scales,var,one(eltype(values)))
        factor=spec.vertical.regularize*spec.vertical.weights[i]*length(values)/allreduce_sum(length(values),context)
        g=var===:pgas_boundary && typ==1 ? 2 .* (values./scale.-1)./length(values) :
            _vertical_penalty_grad(values./scale,_vertical_spacing_axis(a.grid,spec),typ)
        _add_variable_cotangent!(bar,a,var,(factor/scale).*g)
    end
    nxg=length(distributed.global_grid.x); nyg=length(distributed.global_grid.y)
    for (var,weight) in spec.horizontal
        scale=get(spec.scales,var,one(eltype(a.temperature)))
        values=_atmospheric_variable(a,var)./scale
        order=spec.horizontal_order
        halo=exchange_halos(_field(values,distributed.tile,(size(values,1),nxg,nyg)),context,order)
        cx=cy=0
        for k in axes(values,1),i in axes(values,2),j in axes(values,3)
            gi=first(distributed.tile.xrange)+i-1; gj=first(distributed.tile.yrange)+j-1
            gi<=nxg-order && (cx+=1); gj<=nyg-order && (cy+=1)
        end
        Cx=max(allreduce_sum(cx,context),1); Cy=max(allreduce_sum(cy,context),1)
        hb=zero(halo); o=order
        for k in axes(values,1),i in axes(values,2),j in axes(values,3)
            gi=first(distributed.tile.xrange)+i-1; gj=first(distributed.tile.yrange)+j-1
            if gi<=nxg-o
                if o==1
                    d=(halo[k,i+2,j+1]-halo[k,i+1,j+1])/dx; w=weight*2d/Cx/dx
                    hb[k,i+2,j+1]+=w; hb[k,i+1,j+1]-=w
                else
                    d=(halo[k,i+4,j+2]-2halo[k,i+3,j+2]+halo[k,i+2,j+2])/dx^2; w=weight*2d/Cx/dx^2
                    hb[k,i+4,j+2]+=w; hb[k,i+3,j+2]-=2w; hb[k,i+2,j+2]+=w
                end
            end
            if gj<=nyg-o
                if o==1
                    d=(halo[k,i+1,j+2]-halo[k,i+1,j+1])/dy; w=weight*2d/Cy/dy
                    hb[k,i+1,j+2]+=w; hb[k,i+1,j+1]-=w
                else
                    d=(halo[k,i+2,j+4]-2halo[k,i+2,j+3]+halo[k,i+2,j+2])/dy^2; w=weight*2d/Cy/dy^2
                    hb[k,i+2,j+4]+=w; hb[k,i+2,j+3]-=2w; hb[k,i+2,j+2]+=w
                end
            end
        end
        _add_variable_cotangent!(bar,a,var,exchange_halos_vjp(hb,distributed.tile,context,order)./scale)
    end
    bar
end
