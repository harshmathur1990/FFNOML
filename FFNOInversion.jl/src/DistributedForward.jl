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
         rho=root_atmosphere.rho!==nothing,ne=root_atmosphere.ne!==nothing,z=root_atmosphere.z!==nothing)
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
    B=if metadata.magnetic
        rootB=isroot(context) ? root_atmosphere.magnetic_field : nothing
        bx=distribute_field(T,isroot(context) ? rootB.Bx : nothing,shape,context;tag=310).values
        by=distribute_field(T,isroot(context) ? rootB.By : nothing,shape,context;tag=311).values
        bz=distribute_field(T,isroot(context) ? rootB.Bz : nothing,shape,context;tag=312).values
        MagneticField3D(bx,by,bz)
    else nothing end
    local_grid=Grid3D(copy(metadata.grid.log_tau500),copy(metadata.grid.x[tile.xrange]),copy(metadata.grid.y[tile.yrange]))
    local_atmos=Atmosphere3D(local_grid,temperature,vx,vy,vz,vturb;magnetic_field=B,pgas=pgas,rho=rho,ne=ne,z=z)
    DistributedAtmosphere(local_atmos,metadata.grid,tile)
end

"""Collect final output state on rank 0 only."""
function gather_atmosphere(distributed::DistributedAtmosphere,context::ParallelContext)
    a=distributed.local_atmosphere; g=distributed.global_grid
    shape=(length(g.log_tau500),length(g.x),length(g.y)); tile=distributed.tile
    gather(value,tag)=gather_field(_field(value,tile,shape),context;tag=tag)
    temperature=gather(a.temperature,321); vx=gather(a.vx,322); vy=gather(a.vy,323)
    vz=gather(a.vz,324); vturb=gather(a.vturb,325)
    pgas=a.pgas===nothing ? nothing : gather(a.pgas,326)
    rho=a.rho===nothing ? nothing : gather(a.rho,327)
    ne=a.ne===nothing ? nothing : gather(a.ne,328)
    z=a.z===nothing ? nothing : gather(a.z,329)
    B=if a.magnetic_field===nothing
        nothing
    else
        bx=gather(a.magnetic_field.Bx,330); by=gather(a.magnetic_field.By,331); bz=gather(a.magnetic_field.Bz,332)
        isroot(context) ? MagneticField3D(bx,by,bz) : nothing
    end
    isroot(context) ? Atmosphere3D(g,temperature,vx,vy,vz,vturb;magnetic_field=B,pgas=pgas,rho=rho,ne=ne,z=z) : nothing
end

function _local_boundary(boundary::HE3DBoundaryState,tile::Tile2D)
    cut(value)=if value isa Number
        value
    elseif size(value)==(length(tile.xrange),length(tile.yrange))
        copy(value)
    else
        copy(@view value[tile.xrange,tile.yrange])
    end
    HE3DBoundaryState(cut(boundary.rho0),cut(boundary.p0),boundary.boundary)
end

function _global_zcoord(z,context,global_pixels)
    local_sum=vec(dropdims(sum(z,dims=(2,3)),dims=(2,3)))
    context.enabled && MPI.Allreduce!(local_sum,MPI.SUM,context.comm)
    local_sum./global_pixels
end

@inline function _horizontal_derivative(padded,k,i,j,global_i,coords,axis,width)
    n=length(coords); n==1 && return zero(eltype(padded))
    if axis==2
        global_i==1 && return (padded[k,width+i+1,width+j]-padded[k,width+i,width+j])/(coords[2]-coords[1])
        global_i==n && return (padded[k,width+i,width+j]-padded[k,width+i-1,width+j])/(coords[n]-coords[n-1])
        return (padded[k,width+i+1,width+j]-padded[k,width+i-1,width+j])/(coords[global_i+1]-coords[global_i-1])
    end
    global_i==1 && return (padded[k,width+i,width+j+1]-padded[k,width+i,width+j])/(coords[2]-coords[1])
    global_i==n && return (padded[k,width+i,width+j]-padded[k,width+i,width+j-1])/(coords[n]-coords[n-1])
    (padded[k,width+i,width+j+1]-padded[k,width+i,width+j-1])/(coords[global_i+1]-coords[global_i-1])
end

function _distributed_lorentz!(fx,fy,fz,distributed::DistributedAtmosphere,z,context)
    a=distributed.local_atmosphere; B=a.magnetic_field; tile=distributed.tile; g=distributed.global_grid
    bx=exchange_halos(_field(B.Bx,tile,(size(B.Bx,1),length(g.x),length(g.y))),context,1)
    by=exchange_halos(_field(B.By,tile,(size(B.By,1),length(g.x),length(g.y))),context,1)
    bz=exchange_halos(_field(B.Bz,tile,(size(B.Bz,1),length(g.x),length(g.y))),context,1)
    zcoord=_global_zcoord(z,context,length(g.x)*length(g.y))
    for k in axes(fx,1),i in axes(fx,2),j in axes(fx,3)
        gi=first(tile.xrange)+i-1; gj=first(tile.yrange)+j-1
        dbzdy=_horizontal_derivative(bz,k,i,j,gj,g.y,3,1)
        dbydz=_derivative(B.By,k,i,j,zcoord,1); dbxdz=_derivative(B.Bx,k,i,j,zcoord,1)
        dbzdx=_horizontal_derivative(bz,k,i,j,gi,g.x,2,1)
        dbydx=_horizontal_derivative(by,k,i,j,gi,g.x,2,1)
        dbxdy=_horizontal_derivative(bx,k,i,j,gj,g.y,3,1)
        jx=(dbzdy-dbydz)/MU0; jy=(dbxdz-dbzdx)/MU0; jz=(dbydx-dbxdy)/MU0
        fx[k,i,j]=jy*B.Bz[k,i,j]-jz*B.By[k,i,j]
        fy[k,i,j]=jz*B.Bx[k,i,j]-jx*B.Bz[k,i,j]
        fz[k,i,j]=jx*B.By[k,i,j]-jy*B.Bx[k,i,j]
    end
end

function _distributed_pressure_relax!(p,fx,fy,fz,rho,z,distributed,pboundary,order,g,context,sweeps)
    tile=distributed.tile; grid=distributed.global_grid; nz,nx,ny=size(p); boundary_k=order[1]
    zcoord=_global_zcoord(z,context,length(grid.x)*length(grid.y)); targetz=fz.-rho.*g
    global_shape=(nz,length(grid.x),length(grid.y)); next=similar(p)
    fxh=exchange_halos(_field(fx,tile,global_shape),context,1)
    fyh=exchange_halos(_field(fy,tile,global_shape),context,1)
    for _ in 1:sweeps
        ph=exchange_halos(_field(p,tile,global_shape),context,1)
        copyto!(next,p); @views next[boundary_k,:,:].=pboundary
        Threads.@threads :static for column in 1:nx*ny
            i=(column-1)%nx+1; j=(column-1)÷nx+1
            gi=first(tile.xrange)+i-1; gj=first(tile.yrange)+j-1
            for k in 1:nz
                k==boundary_k && continue
                total=0.0; count=0
                if gi>1
                    total+=ph[k,i,j+1]+(fxh[k,i,j+1]+fx[k,i,j])*(grid.x[gi]-grid.x[gi-1])/2; count+=1
                end
                if gi<length(grid.x)
                    total+=ph[k,i+2,j+1]-(fxh[k,i+2,j+1]+fx[k,i,j])*(grid.x[gi+1]-grid.x[gi])/2; count+=1
                end
                if gj>1
                    total+=ph[k,i+1,j]+(fyh[k,i+1,j]+fy[k,i,j])*(grid.y[gj]-grid.y[gj-1])/2; count+=1
                end
                if gj<length(grid.y)
                    total+=ph[k,i+1,j+2]-(fyh[k,i+1,j+2]+fy[k,i,j])*(grid.y[gj+1]-grid.y[gj])/2; count+=1
                end
                k>1 && (total+=p[k-1,i,j]+(targetz[k-1,i,j]+targetz[k,i,j])*(zcoord[k]-zcoord[k-1])/2; count+=1)
                k<nz && (total+=p[k+1,i,j]-(targetz[k+1,i,j]+targetz[k,i,j])*(zcoord[k+1]-zcoord[k])/2; count+=1)
                next[k,i,j]=max(total/count,eps(eltype(p)))
            end
        end
        p,next=next,p
    end
    @views p[boundary_k,:,:].=pboundary
    p
end

function _distributed_force_residual(p,rho,z,fx,fy,fz,distributed,context,g)
    tile=distributed.tile; grid=distributed.global_grid; shape=(size(p,1),length(grid.x),length(grid.y))
    ph=exchange_halos(_field(p,tile,shape),context,1)
    scale_local=max(maximum(abs.(rho.*g)),maximum(sqrt.(fx.^2 .+ fy.^2 .+ fz.^2)))
    scale=allreduce_max(scale_local,context)+eps(eltype(p)); worst=zero(eltype(p))
    for k in axes(p,1),i in axes(p,2),j in axes(p,3)
        gi=first(tile.xrange)+i-1; gj=first(tile.yrange)+j-1
        rx=_horizontal_derivative(ph,k,i,j,gi,grid.x,2,1)-fx[k,i,j]
        ry=_horizontal_derivative(ph,k,i,j,gj,grid.y,3,1)-fy[k,i,j]
        rz=_derivative(p,k,i,j,vec(@view(z[:,i,j])),1)+rho[k,i,j]*g-fz[k,i,j]
        worst=max(worst,sqrt(rx^2+ry^2+rz^2)/scale)
    end
    allreduce_max(worst,context)
end

function reconstruct_force_balance_distributed!(distributed::DistributedAtmosphere{Float64},boundary::HE3DBoundaryState,
        eos::AbstractEOS,opacity::AbstractOpacity500,context::ParallelContext;options=ForceBalanceOptions())
    a=distributed.local_atmosphere; local_boundary=_local_boundary(boundary,distributed.tile)
    shape=size(a.temperature); nz,nx,ny=shape; tau=10.0.^a.grid.log_tau500
    top_to_bottom=tau[1]<tau[end] ? collect(1:nz) : collect(nz:-1:1)
    order=local_boundary.boundary==:top ? top_to_bottom : reverse(top_to_bottom)
    pboundary=_boundary_map(local_boundary.p0,nx,ny,:pressure); rhoboundary=_boundary_map(local_boundary.rho0,nx,ny,:density)
    p=a.pgas===nothing ? repeat(reshape(pboundary,1,nx,ny),nz,1,1) : copy(a.pgas)
    rho=a.rho===nothing ? repeat(reshape(rhoboundary,1,nx,ny),nz,1,1) : copy(a.rho)
    ne=a.ne===nothing ? similar(rho) : copy(a.ne); z=a.z===nothing ? zeros(shape) : copy(a.z)
    kappa=similar(rho); pnew=similar(p); rhonew=similar(rho); nenew=similar(ne); znew=similar(z)
    fx=zeros(shape); fy=zeros(shape); fz=zeros(shape); mode=select_force_balance(a)
    pchange=rchange=fres=zchange=Inf; lorentzmax=0.0
    for iteration in 1:options.max_iterations
        thermodynamics!(rhonew,nenew,eos,a.temperature,p)
        iteration==1 && (@views rhonew[order[1],:,:].=rhoboundary)
        opacity500!(kappa,opacity,a.temperature,p,rhonew,nenew)
        _height_from_tau!(znew,kappa,rhonew,tau,top_to_bottom)
        mode isa MHSMode ? _distributed_lorentz!(fx,fy,fz,distributed,znew,context) : (fill!(fx,0);fill!(fy,0);fill!(fz,0))
        lorentzmax=allreduce_max(maximum(sqrt.(fx.^2 .+ fy.^2 .+ fz.^2)),context)
        _pressure_from_force!(pnew,rhonew,znew,fz,pboundary,order,options.gravity_m_s2)
        pnew=_distributed_pressure_relax!(pnew,fx,fy,fz,rhonew,znew,distributed,pboundary,order,
            options.gravity_m_s2,context,options.pressure_sweeps)
        pchange=allreduce_max(_relative_change(pnew,p),context); rchange=allreduce_max(_relative_change(rhonew,rho),context)
        zchange=allreduce_max(maximum(abs.(znew.-z)),context)
        fres=_distributed_force_residual(pnew,rhonew,znew,fx,fy,fz,distributed,context,options.gravity_m_s2)
        @. p=options.relaxation*pnew+(1-options.relaxation)*p
        @. rho=options.relaxation*rhonew+(1-options.relaxation)*rho
        copyto!(ne,nenew); copyto!(z,znew)
        if pchange<=options.relative_tolerance && rchange<=options.relative_tolerance && fres<=options.force_tolerance && zchange<=options.height_tolerance_m
            a.pgas=p; a.rho=rho; a.ne=ne; a.z=z
            return ForceBalanceDiagnostics(mode isa HE3DMode ? :HE3D : :MHS,iteration,true,pchange,rchange,fres,zchange,0.0,0.0,lorentzmax)
        end
    end
    throw(ErrorException("distributed force balance did not converge (dP=$pchange, drho=$rchange, dz=$zchange)"))
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
    shape=(length(distributed.global_grid.log_tau500),length(distributed.global_grid.x),length(distributed.global_grid.y))
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

function predict_distributed_populations!(out::AbstractDict,backend::CompositeDistributedPopulationModel,
        distributed,context)
    Set(keys(out))==Set(keys(backend.models)) || throw(ArgumentError(
        "population workspace species differ from composite model"))
    for species in sort!(collect(keys(backend.models));by=string)
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
    force_balance_seconds::Float64
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

struct HybridForwardModel{P,R,S,O,E,K,B,F}
    populations::P; redistribution::R; synthesizer::S; observation::O
    eos::E; opacity::K; boundary::B; force_options::F; capabilities::CapabilityManifest
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

function forward!(workspace::HybridForwardWorkspace,model::HybridForwardModel,
                  distributed::DistributedAtmosphere,context::ParallelContext)
    total_start=time_ns()
    validate_capabilities(model.capabilities,model.redistribution,workspace.output.stokes)
    stage_start=time_ns()
    diagnostics=reconstruct_force_balance_distributed!(distributed,model.boundary,model.eos,model.opacity,context;
        options=model.force_options)
    force_seconds=(time_ns()-stage_start)/1e9
    stage_start=time_ns()
    predict_distributed_populations!(workspace.populations,model.populations,distributed,context)
    populations_seconds=(time_ns()-stage_start)/1e9
    stage_start=time_ns()
    synthesize!(workspace.intrinsic,model.synthesizer,model.redistribution,distributed.local_atmosphere,
        workspace.populations,workspace.synthesis_cache)
    synthesis_seconds=(time_ns()-stage_start)/1e9
    stage_start=time_ns()
    _distributed_observation!(workspace.output,workspace.intrinsic,model.observation,distributed,context)
    observation_seconds=(time_ns()-stage_start)/1e9
    timings=HybridForwardTimings(force_seconds,populations_seconds,synthesis_seconds,
        observation_seconds,(time_ns()-total_start)/1e9)
    (spectrum=workspace.output,force_balance=diagnostics,timings=timings)
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
        localpen=var===:pgas_boundary && typ==1 ? _mean_square(values./scale.-1) : _vertical_penalty(values./scale,a.grid.log_tau500,typ)
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
