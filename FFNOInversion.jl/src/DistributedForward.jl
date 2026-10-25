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
         logtau500=root_atmosphere.logtau500!==nothing)
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
    logtau500=optional(:logtau500,metadata.logtau500,313)
    B=if metadata.magnetic
        rootB=isroot(context) ? root_atmosphere.magnetic_field : nothing
        bx=distribute_field(T,isroot(context) ? rootB.Bx : nothing,shape,context;tag=310).values
        by=distribute_field(T,isroot(context) ? rootB.By : nothing,shape,context;tag=311).values
        bz=distribute_field(T,isroot(context) ? rootB.Bz : nothing,shape,context;tag=312).values
        MagneticField3D(bx,by,bz)
    else nothing end
    local_grid=Grid3D(metadata.grid.vertical,copy(metadata.grid.x[tile.xrange]),copy(metadata.grid.y[tile.yrange]))
    local_atmos=Atmosphere3D(local_grid,temperature,vx,vy,vz,vturb;magnetic_field=B,pgas=pgas,rho=rho,ne=ne,z=z,
        logtau500=logtau500)
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
    B=if a.magnetic_field===nothing
        nothing
    else
        bx=gather(a.magnetic_field.Bx,330); by=gather(a.magnetic_field.By,331); bz=gather(a.magnetic_field.Bz,332)
        isroot(context) ? MagneticField3D(bx,by,bz) : nothing
    end
    isroot(context) ? Atmosphere3D(g,temperature,vx,vy,vz,vturb;magnetic_field=B,pgas=pgas,rho=rho,ne=ne,z=z,
        logtau500=logtau500) : nothing
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

mutable struct HybridForwardWorkspace{T<:AbstractFloat,P}
    populations::P
    intrinsic::SpectralCube{T,Array{T,4}}
    output::SpectralCube{T,Array{T,4}}
    synthesis_cache::ThreadedSynthesisCache{T}
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
    HybridForwardWorkspace(zeros(T,nz,nx,ny,levels),cube(),cube(),ThreadedSynthesisCache(T,nz,nλ))
end

function HybridForwardWorkspace(::Type{T},distributed::DistributedAtmosphere,wavelength,stokes,
        levels::AbstractDict) where T
    a=distributed.local_atmosphere; nz,nx,ny=size(a.temperature); nλ=length(wavelength)
    cube()=SpectralCube(zeros(T,nλ,length(stokes.components),nx,ny),T.(wavelength),stokes)
    populations=Dict{Symbol,Array{T,4}}(Symbol(species)=>zeros(T,nz,nx,ny,Int(count))
        for (species,count) in levels)
    HybridForwardWorkspace(populations,cube(),cube(),ThreadedSynthesisCache(T,nz,nλ))
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

"""Root-rank progress line for a forward stage (stdout, flushed so a running job shows where it is)."""
function _forward_stage(context,message)
    isroot(context) || return nothing
    println(stdout,"FORWARD stage=$message"); flush(stdout); nothing
end

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
        _forward_stage(context,"equilibrium begin")
        equilibrium=reconstruct_equilibrium!(model.equilibrium,distributed,context)
        equilibrium_elapsed=(time_ns()-stage_start)/1e9
        _forward_stage(context,"equilibrium done seconds=$(round(equilibrium_elapsed;digits=1)) valid=$(equilibrium.valid)")
        if !equilibrium.valid
            timings=HybridForwardTimings(equilibrium_elapsed,0.0,0.0,0.0,(time_ns()-total_start)/1e9)
            return (spectrum=workspace.output,equilibrium=equilibrium,timings=timings,valid=false)
        end
        stage_start=time_ns()
        _forward_stage(context,"populations begin")
        predict_distributed_populations!(workspace.populations,model.populations,distributed,context)
        populations_elapsed=(time_ns()-stage_start)/1e9
        _forward_stage(context,"populations done seconds=$(round(populations_elapsed;digits=1))")
        (equilibrium,equilibrium_elapsed,populations_elapsed)
    else
        isroot(context) && (println(stdout,"FORWARD event=reuse_prepared_equilibrium_and_populations"); flush(stdout))
        (prepared,0.0,0.0)
    end
    stage_start=time_ns()
    _forward_stage(context,"synthesis begin")
    Base.ScopedValues.with(SYNTHESIS_PROGRESS=>isroot(context)) do
        synthesize!(workspace.intrinsic,model.synthesizer,model.redistribution,distributed.local_atmosphere,
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
