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

@inline function _distributed_physical_derivatives(a,ah,z,zh,grid,k,i,j,gi,gj)
    aτ=_derivative(a,k,i,j,grid.log_tau500,1)
    zτ=_derivative(z,k,i,j,grid.log_tau500,1)
    abs(zτ)>eps(eltype(z)) || throw(ErrorException(
        "degenerate optical-depth mapping at depth=$k global_x=$gi global_y=$gj"))
    ax=_horizontal_derivative(ah,k,i,j,gi,grid.x,2,1)
    ay=_horizontal_derivative(ah,k,i,j,gj,grid.y,3,1)
    zx=_horizontal_derivative(zh,k,i,j,gi,grid.x,2,1)
    zy=_horizontal_derivative(zh,k,i,j,gj,grid.y,3,1)
    (ax-zx*aτ/zτ,ay-zy*aτ/zτ,aτ/zτ)
end

function _distributed_lorentz!(fx,fy,fz,distributed::DistributedAtmosphere,z,context)
    a=distributed.local_atmosphere; B=a.magnetic_field; tile=distributed.tile; g=distributed.global_grid
    bx=exchange_halos(_field(B.Bx,tile,(size(B.Bx,1),length(g.x),length(g.y))),context,1)
    by=exchange_halos(_field(B.By,tile,(size(B.By,1),length(g.x),length(g.y))),context,1)
    bz=exchange_halos(_field(B.Bz,tile,(size(B.Bz,1),length(g.x),length(g.y))),context,1)
    zh=exchange_halos(_field(z,tile,(size(z,1),length(g.x),length(g.y))),context,1)
    nx,ny=size(fx,2),size(fx,3)
    _distributed_local_stage!("3-D Lorentz derivatives",context) do
        Threads.@threads :static for column in 1:nx*ny
            i=(column-1)%nx+1; j=(column-1)÷nx+1
            gi=first(tile.xrange)+i-1; gj=first(tile.yrange)+j-1
            for k in axes(fx,1)
                dbxdx,dbxdy,dbxdz=_distributed_physical_derivatives(B.Bx,bx,z,zh,g,k,i,j,gi,gj)
                dbydx,dbydy,dbydz=_distributed_physical_derivatives(B.By,by,z,zh,g,k,i,j,gi,gj)
                dbzdx,dbzdy,dbzdz=_distributed_physical_derivatives(B.Bz,bz,z,zh,g,k,i,j,gi,gj)
                jx=(dbzdy-dbydz)/MU0; jy=(dbxdz-dbzdx)/MU0; jz=(dbydx-dbxdy)/MU0
                fx[k,i,j]=jy*B.Bz[k,i,j]-jz*B.By[k,i,j]
                fy[k,i,j]=jz*B.Bx[k,i,j]-jx*B.Bz[k,i,j]
                fz[k,i,j]=jx*B.By[k,i,j]-jy*B.Bx[k,i,j]
            end
        end
    end
end

function _distributed_pressure_relax!(p,fx,fy,fz,rho,z,distributed,pboundary,order,g,context,
        sweeps,horizontal_weight=1.0;max_sweeps=sweeps,tolerance=0.0,sweeps_used=nothing,
        converged_ref=nothing)
    max_sweeps>=sweeps>0 || throw(ArgumentError(
        "pressure sweep limits must satisfy max >= minimum > 0"))
    tile=distributed.tile; grid=distributed.global_grid; nz,nx,ny=size(p); boundary_k=order[1]
    targetz=fz.-rho.*g
    global_shape=(nz,length(grid.x),length(grid.y)); next=similar(p)
    fxh=exchange_halos(_field(fx,tile,global_shape),context,1)
    fyh=exchange_halos(_field(fy,tile,global_shape),context,1)
    qzh=exchange_halos(_field(targetz,tile,global_shape),context,1)
    zh=exchange_halos(_field(z,tile,global_shape),context,1)
    used=0; inner_converged=false
    for sweep in 1:max_sweeps
        ph=exchange_halos(_field(p,tile,global_shape),context,1)
        copyto!(next,p); @views next[boundary_k,:,:].=pboundary
        Threads.@threads :static for column in 1:nx*ny
            i=(column-1)%nx+1; j=(column-1)÷nx+1
            gi=first(tile.xrange)+i-1; gj=first(tile.yrange)+j-1
            for k in 1:nz
                k==boundary_k && continue
                total=0.0; weight=0.0
                if gi>1
                    dx=grid.x[gi]-grid.x[gi-1]; dz=z[k,i,j]-zh[k,i,j+1]
                    estimate=ph[k,i,j+1]+(fxh[k,i,j+1]+fx[k,i,j])*dx/2+
                        (qzh[k,i,j+1]+targetz[k,i,j])*dz/2
                    total+=horizontal_weight*estimate; weight+=horizontal_weight
                end
                if gi<length(grid.x)
                    dx=grid.x[gi+1]-grid.x[gi]; dz=zh[k,i+2,j+1]-z[k,i,j]
                    estimate=ph[k,i+2,j+1]-(fxh[k,i+2,j+1]+fx[k,i,j])*dx/2-
                        (qzh[k,i+2,j+1]+targetz[k,i,j])*dz/2
                    total+=horizontal_weight*estimate; weight+=horizontal_weight
                end
                if gj>1
                    dy=grid.y[gj]-grid.y[gj-1]; dz=z[k,i,j]-zh[k,i+1,j]
                    estimate=ph[k,i+1,j]+(fyh[k,i+1,j]+fy[k,i,j])*dy/2+
                        (qzh[k,i+1,j]+targetz[k,i,j])*dz/2
                    total+=horizontal_weight*estimate; weight+=horizontal_weight
                end
                if gj<length(grid.y)
                    dy=grid.y[gj+1]-grid.y[gj]; dz=zh[k,i+1,j+2]-z[k,i,j]
                    estimate=ph[k,i+1,j+2]-(fyh[k,i+1,j+2]+fy[k,i,j])*dy/2-
                        (qzh[k,i+1,j+2]+targetz[k,i,j])*dz/2
                    total+=horizontal_weight*estimate; weight+=horizontal_weight
                end
                if k>1
                    dz=z[k,i,j]-z[k-1,i,j]
                    total+=p[k-1,i,j]+(targetz[k-1,i,j]+targetz[k,i,j])*dz/2; weight+=1
                end
                if k<nz
                    dz=z[k+1,i,j]-z[k,i,j]
                    total+=p[k+1,i,j]-(targetz[k+1,i,j]+targetz[k,i,j])*dz/2; weight+=1
                end
                next[k,i,j]=max(total/weight,eps(eltype(p)))
            end
        end
        local_change=maximum(abs.(next.-p)./max.(abs.(p),one(eltype(p))))
        p,next=next,p; used=sweep
        if sweep>=sweeps && tolerance>0 && (sweep%sweeps==0 || sweep==max_sweeps)
            if allreduce_max(local_change,context)<=tolerance
                inner_converged=true; break
            end
        end
    end
    @views p[boundary_k,:,:].=pboundary
    sweeps_used===nothing || (sweeps_used[]=used)
    converged_ref===nothing || (converged_ref[]=inner_converged)
    p
end

function _distributed_force_residual(p,rho,z,fx,fy,fz,distributed,context,g)
    tile=distributed.tile; grid=distributed.global_grid; shape=(size(p,1),length(grid.x),length(grid.y))
    ph=exchange_halos(_field(p,tile,shape),context,1)
    fxh=exchange_halos(_field(fx,tile,shape),context,1)
    fyh=exchange_halos(_field(fy,tile,shape),context,1)
    qz=fz.-rho.*g
    qzh=exchange_halos(_field(qz,tile,shape),context,1)
    zh=exchange_halos(_field(z,tile,shape),context,1)
    scale_local=max(abs(g)*maximum(abs,rho),maximum(sqrt.(fx.^2 .+ fy.^2 .+ fz.^2)))
    scale=allreduce_max(scale_local,context)+eps(eltype(p)); worst=zero(eltype(p))
    nx,ny=size(p,2),size(p,3); column_worst=zeros(eltype(p),nx*ny)
    Threads.@threads :static for column in 1:nx*ny
        i=(column-1)%nx+1; j=(column-1)÷nx+1
        gi=first(tile.xrange)+i-1; gj=first(tile.yrange)+j-1
        local_worst=zero(eltype(p))
        for k in axes(p,1)
            if gi<length(grid.x)
                dx=grid.x[gi+1]-grid.x[gi]; dz=zh[k,i+2,j+1]-z[k,i,j]; ds=hypot(dx,dz)
                mismatch=ph[k,i+2,j+1]-p[k,i,j]-(fxh[k,i+2,j+1]+fx[k,i,j])*dx/2-
                    (qzh[k,i+2,j+1]+qz[k,i,j])*dz/2
                local_worst=max(local_worst,abs(mismatch)/(ds*scale))
            end
            if gj<length(grid.y)
                dy=grid.y[gj+1]-grid.y[gj]; dz=zh[k,i+1,j+2]-z[k,i,j]; ds=hypot(dy,dz)
                mismatch=ph[k,i+1,j+2]-p[k,i,j]-(fyh[k,i+1,j+2]+fy[k,i,j])*dy/2-
                    (qzh[k,i+1,j+2]+qz[k,i,j])*dz/2
                local_worst=max(local_worst,abs(mismatch)/(ds*scale))
            end
            if k<size(p,1)
                dz=z[k+1,i,j]-z[k,i,j]
                mismatch=p[k+1,i,j]-p[k,i,j]-(qz[k+1,i,j]+qz[k,i,j])*dz/2
                local_worst=max(local_worst,abs(mismatch)/(abs(dz)*scale))
            end
        end
        column_worst[column]=local_worst
    end
    worst=maximum(column_worst)
    allreduce_max(worst,context)
end

function _force_balance_log(context::ParallelContext,event::AbstractString;values...)
    isroot(context) || return nothing
    print(stdout,"FORCE_BALANCE event=",event)
    for (key,value) in pairs(values)
        print(stdout,' ',key,'=',value)
    end
    println(stdout); flush(stdout)
    nothing
end

function _validate_distributed_force_state(stage,context,arrays...)
    local_invalid=any(array->!all(isfinite,array),arrays)
    invalid_ranks=allreduce_sum(local_invalid ? 1 : 0,context)
    invalid_ranks==0 || throw(ErrorException(
        "$stage produced non-finite values on $invalid_ranks MPI rank(s)"))
    nothing
end

"""Run a rank-local stage and turn any exception into one collective failure."""
function _distributed_local_stage!(f,stage,context)
    local_error=nothing
    try
        f()
    catch error
        local_error=sprint(showerror,error)
    end
    failed_ranks=allreduce_sum(local_error===nothing ? 0 : 1,context)
    failed_ranks==0 && return nothing
    detail=local_error===nothing ? "another rank failed" : local_error
    throw(ErrorException("$stage failed on $failed_ranks MPI rank(s); $detail"))
end

function _distributed_bootstrap_hse_1d!(p,rho,ne,z,kappa,pnew,rhonew,nenew,znew,
        temperature,eos,opacity,tau,top_to_bottom,order,pboundary,options,context)
    zero_force=zeros(eltype(p),size(p)); pchange=rchange=zchange=Inf
    recent_metrics=Float64[]
    _force_balance_log(context,"bootstrap_start";mode="HSE1D",
        max_iterations=options.bootstrap_iterations)
    for iteration in 1:options.bootstrap_iterations
        iteration_start=time_ns()
        _distributed_local_stage!("1-D HSE EOS",context) do
            thermodynamics!(rhonew,nenew,eos,temperature,p)
        end
        _distributed_local_stage!("1-D HSE opacity",context) do
            opacity500!(kappa,opacity,temperature,p,rhonew,nenew)
        end
        _distributed_local_stage!("1-D HSE height",context) do
            _height_from_tau!(znew,kappa,rhonew,tau,top_to_bottom)
        end
        _distributed_local_stage!("1-D HSE pressure",context) do
            _pressure_from_force!(pnew,rhonew,znew,zero_force,pboundary,order,
                options.gravity_m_s2)
        end
        _validate_distributed_force_state("1-D HSE bootstrap",context,
            pnew,rhonew,nenew,znew)
        pchange=allreduce_max(_relative_change(pnew,p),context)
        rchange=allreduce_max(_relative_change(rhonew,rho),context)
        zchange=allreduce_max(maximum(abs.(znew.-z)),context)
        @. p=options.relaxation*pnew+(1-options.relaxation)*p
        @. rho=options.relaxation*rhonew+(1-options.relaxation)*rho
        copyto!(ne,nenew); copyto!(z,znew)
        push!(recent_metrics,_bootstrap_change_metric(pchange,rchange,zchange,options))
        length(recent_metrics)>20 && popfirst!(recent_metrics)
        _force_balance_log(context,"bootstrap_iteration";iteration=iteration,dP=pchange,
            drho=rchange,dz_m=zchange,
            total_s=round((time_ns()-iteration_start)/1e9;digits=3))
        if pchange<=options.relative_tolerance && rchange<=options.relative_tolerance &&
                zchange<=options.height_tolerance_m
            _force_balance_log(context,"bootstrap_converged";iteration=iteration)
            return iteration
        end
    end
    local_valid=_valid_hse_1d_state(p,rho,ne,z,kappa,top_to_bottom)
    invalid_ranks=allreduce_sum(local_valid ? 0 : 1,context)
    final_metric=last(recent_metrics)
    nondivergent=isfinite(final_metric) &&
        final_metric<=options.backtrack_growth_limit*minimum(recent_metrics)
    divergent_ranks=allreduce_sum(nondivergent ? 0 : 1,context)
    if invalid_ranks==0 && divergent_ranks==0
        _force_balance_log(context,"bootstrap_accepted_unconverged";
            iterations=options.bootstrap_iterations,dP=pchange,drho=rchange,dz_m=zchange)
        return options.bootstrap_iterations
    end
    _force_balance_log(context,"bootstrap_failed";iterations=options.bootstrap_iterations,
        invalid_ranks=invalid_ranks,divergent_ranks=divergent_ranks,
        dP=pchange,drho=rchange,dz_m=zchange)
    throw(ErrorException("distributed 1-D HSE bootstrap was invalid or diverging " *
        "(invalid_ranks=$invalid_ranks, divergent_ranks=$divergent_ranks, " *
        "dP=$pchange, drho=$rchange, dz=$zchange)"))
end

function reconstruct_force_balance_distributed!(distributed::DistributedAtmosphere{Float64},boundary::HE3DBoundaryState,
        eos::AbstractEOS,opacity::AbstractOpacity500,context::ParallelContext;options=ForceBalanceOptions())
    options.max_iterations>0 || throw(ArgumentError("max_iterations must be positive"))
    options.pressure_sweeps>0 || throw(ArgumentError("pressure_sweeps must be positive"))
    options.pressure_max_sweeps>=options.pressure_sweeps || throw(ArgumentError(
        "pressure_max_sweeps must be at least pressure_sweeps"))
    options.pressure_tolerance>0 || throw(ArgumentError("pressure_tolerance must be positive"))
    options.bootstrap_iterations>0 || throw(ArgumentError("bootstrap_iterations must be positive"))
    options.continuation_iterations>0 || throw(ArgumentError("continuation_iterations must be positive"))
    options.max_backtracks>=0 || throw(ArgumentError("max_backtracks must be non-negative"))
    options.backtrack_growth_limit>=1 || throw(ArgumentError(
        "backtrack_growth_limit must be at least one"))
    0<options.minimum_relaxation<=options.relaxation || throw(ArgumentError(
        "minimum_relaxation must lie in (0, relaxation]"))
    options.lateral_boundary===:force_neumann || throw(ArgumentError("unsupported lateral boundary"))
    options.bottom_boundary===:force_neumann || throw(ArgumentError("unsupported bottom boundary"))
    0<options.relaxation<=1 || throw(ArgumentError("relaxation must lie in (0,1]"))
    a=distributed.local_atmosphere; local_boundary=_local_boundary(boundary,distributed.tile)
    shape=size(a.temperature); nz,nx,ny=shape; tau=10.0.^a.grid.log_tau500
    top_to_bottom=tau[1]<tau[end] ? collect(1:nz) : collect(nz:-1:1)
    order=local_boundary.boundary==:top ? top_to_bottom : reverse(top_to_bottom)
    pboundary=_boundary_map(local_boundary.p0,nx,ny,:pressure); rhoboundary=_boundary_map(local_boundary.rho0,nx,ny,:density)
    p=a.pgas===nothing ? repeat(reshape(pboundary,1,nx,ny),nz,1,1) : copy(a.pgas)
    rho=a.rho===nothing ? repeat(reshape(rhoboundary,1,nx,ny),nz,1,1) : copy(a.rho)
    ne=a.ne===nothing ? similar(rho) : copy(a.ne); z=a.z===nothing ? zeros(shape) : copy(a.z)
    kappa=similar(rho); pnew=similar(p); rhonew=similar(rho); nenew=similar(ne); znew=similar(z)
    ptrial=similar(p); rhotrial=similar(rho); netrial=similar(ne); ztrial=similar(z)
    fx=zeros(shape); fy=zeros(shape); fz=zeros(shape); mode=select_force_balance(a)
    fxtrial=zeros(shape); fytrial=zeros(shape); fztrial=zeros(shape)
    pchange=rchange=fres=zchange=Inf; lorentzmax=0.0; iterations=0; converged=false
    _force_balance_log(context,"start";mode=mode isa HE3DMode ? "HE3D" : "MHS",
        global_shape="$(length(distributed.global_grid.log_tau500))x$(length(distributed.global_grid.x))x$(length(distributed.global_grid.y))",
        local_shape="$(nz)x$(nx)x$(ny)",threads=Threads.nthreads(),
        max_iterations=options.max_iterations,pressure_sweeps=options.pressure_sweeps,
        pressure_max_sweeps=options.pressure_max_sweeps,
        pressure_tolerance=options.pressure_tolerance,
        bootstrap_iterations=options.bootstrap_iterations,
        continuation_iterations=options.continuation_iterations,
        max_backtracks=options.max_backtracks,
        lateral_boundary=options.lateral_boundary,bottom_boundary=options.bottom_boundary)
    _distributed_bootstrap_hse_1d!(p,rho,ne,z,kappa,pnew,rhonew,nenew,znew,
        a.temperature,eos,opacity,tau,top_to_bottom,order,pboundary,options,context)
    if mode isa MHSMode
        _distributed_lorentz!(fx,fy,fz,distributed,z,context)
        lorentzmax=allreduce_max(maximum(sqrt.(fx.^2 .+ fy.^2 .+ fz.^2)),context)
    end
    initial_3d_residual=_distributed_force_residual(p,rho,z,fx,fy,fz,distributed,
        context,options.gravity_m_s2)
    continuation_needed=initial_3d_residual>options.force_tolerance
    _force_balance_log(context,"bootstrap_complete";initial_3d_force=initial_3d_residual,
        continuation=continuation_needed)
    best_state=(copy(p),copy(rho),copy(ne),copy(z)); fallback_reason=nothing
    best_residual=initial_3d_residual; best_pchange=best_rchange=best_zchange=0.0
    best_lorentzmax=lorentzmax
    current_residual=initial_3d_residual
    for iteration in 1:options.max_iterations
            iterations=iteration
            iteration_start=time_ns(); stage_start=time_ns()
            _distributed_local_stage!("3-D EOS",context) do
                thermodynamics!(rhonew,nenew,eos,a.temperature,p)
            end
            eos_seconds=(time_ns()-stage_start)/1e9; stage_start=time_ns()
            _distributed_local_stage!("3-D opacity",context) do
                opacity500!(kappa,opacity,a.temperature,p,rhonew,nenew)
            end
            opacity_seconds=(time_ns()-stage_start)/1e9; stage_start=time_ns()
            _distributed_local_stage!("3-D height",context) do
                _height_from_tau!(znew,kappa,rhonew,tau,top_to_bottom)
            end
            _validate_distributed_force_state("3-D height",context,znew,rhonew,nenew)
            height_seconds=(time_ns()-stage_start)/1e9; stage_start=time_ns()
            if mode isa MHSMode
                _distributed_lorentz!(fx,fy,fz,distributed,znew,context)
                lorentzmax=allreduce_max(maximum(sqrt.(fx.^2 .+ fy.^2 .+ fz.^2)),context)
            end
            force_seconds=(time_ns()-stage_start)/1e9; stage_start=time_ns()
            _distributed_local_stage!("3-D vertical pressure seed",context) do
                _pressure_from_force!(pnew,rhonew,znew,fz,pboundary,order,
                    options.gravity_m_s2)
            end
            horizontal_weight=continuation_needed ?
                min(1.0,iteration/options.continuation_iterations) : 1.0
            inner_sweeps=Ref(0); inner_converged=Ref(false)
            pnew=_distributed_pressure_relax!(pnew,fx,fy,fz,rhonew,znew,distributed,pboundary,order,
                options.gravity_m_s2,context,options.pressure_sweeps,horizontal_weight;
                max_sweeps=options.pressure_max_sweeps,tolerance=options.pressure_tolerance,
                sweeps_used=inner_sweeps,converged_ref=inner_converged)
            _validate_distributed_force_state("3-D force relaxation",context,
                pnew,rhonew,nenew,znew,fx,fy,fz)
            pressure_seconds=(time_ns()-stage_start)/1e9; stage_start=time_ns()
            accepted=false; alpha=options.relaxation; accepted_backtracks=0
            for backtrack in 0:options.max_backtracks
                alpha<options.minimum_relaxation && break
                _blend_state!(ptrial,pnew,p,alpha); _blend_state!(rhotrial,rhonew,rho,alpha)
                _blend_state!(netrial,nenew,ne,alpha); _blend_state!(ztrial,znew,z,alpha)
                _validate_distributed_force_state("3-D backtracking trial",context,
                    ptrial,rhotrial,netrial,ztrial)
                if mode isa MHSMode
                    _distributed_lorentz!(fxtrial,fytrial,fztrial,distributed,ztrial,context)
                else
                    fill!(fxtrial,0); fill!(fytrial,0); fill!(fztrial,0)
                end
                trial_residual=_distributed_force_residual(ptrial,rhotrial,ztrial,
                    fxtrial,fytrial,fztrial,distributed,context,options.gravity_m_s2)
                threshold=max(options.force_tolerance,
                    options.backtrack_growth_limit*current_residual)
                if isfinite(trial_residual) && trial_residual<=threshold
                    pchange=allreduce_max(_relative_change(ptrial,p),context)
                    rchange=allreduce_max(_relative_change(rhotrial,rho),context)
                    zchange=allreduce_max(maximum(abs.(ztrial.-z)),context)
                    fres=trial_residual; accepted_backtracks=backtrack
                    copyto!(p,ptrial); copyto!(rho,rhotrial); copyto!(ne,netrial); copyto!(z,ztrial)
                    copyto!(fx,fxtrial); copyto!(fy,fytrial); copyto!(fz,fztrial)
                    current_residual=fres; accepted=true; break
                end
                _force_balance_log(context,"backtrack";iteration=iteration,attempt=backtrack+1,
                    relaxation=alpha,trial_force=trial_residual,previous_force=current_residual)
                alpha/=2
            end
            if !accepted
                fallback_reason="all backtracking trials increased the force residual from $current_residual"
                _force_balance_log(context,"step_rejected";iteration=iteration,
                    previous_force=current_residual)
                break
            end
            residual_seconds=(time_ns()-stage_start)/1e9
            if fres<best_residual
                best_residual=fres; best_pchange=pchange; best_rchange=rchange
                best_zchange=zchange
                best_lorentzmax=mode isa MHSMode ? allreduce_max(
                    maximum(sqrt.(fxtrial.^2 .+ fytrial.^2 .+ fztrial.^2)),context) : 0.0
                copyto!(best_state[1],p); copyto!(best_state[2],rho)
                copyto!(best_state[3],ne); copyto!(best_state[4],z)
            end
            _force_balance_log(context,"iteration";iteration=iteration,dP=pchange,drho=rchange,
                force=fres,dz_m=zchange,horizontal_weight=horizontal_weight,
                relaxation=alpha,backtracks=accepted_backtracks,inner_sweeps=inner_sweeps[],
                inner_converged=inner_converged[],
                eos_s=round(eos_seconds;digits=3),
                opacity_s=round(opacity_seconds;digits=3),height_s=round(height_seconds;digits=3),
                force_s=round(force_seconds;digits=3),pressure_s=round(pressure_seconds;digits=3),
                residual_s=round(residual_seconds;digits=3),total_s=round((time_ns()-iteration_start)/1e9;digits=3))
            if horizontal_weight==1 && pchange<=options.relative_tolerance &&
                    rchange<=options.relative_tolerance && fres<=options.force_tolerance &&
                    zchange<=options.height_tolerance_m
                converged=true
                _force_balance_log(context,"converged";iteration=iteration)
                break
            end
    end
    absolute_improvement=max(0.0,initial_3d_residual-best_residual)
    improvement=absolute_improvement/max(initial_3d_residual,eps(Float64))
    accepted_3d=best_residual<initial_3d_residual*(1-sqrt(eps(Float64)))
    copyto!(p,best_state[1]); copyto!(rho,best_state[2])
    copyto!(ne,best_state[3]); copyto!(z,best_state[4])
    pchange=best_pchange; rchange=best_rchange; zchange=best_zchange
    fres=best_residual; lorentzmax=best_lorentzmax
    a.pgas=p; a.rho=rho; a.ne=ne; a.z=z
    if accepted_3d
        _force_balance_log(context,converged ? "accepted_converged_3d" :
            "accepted_best_unconverged_3d";iterations=iterations,dP=pchange,drho=rchange,
            force=fres,dz_m=zchange,initial_force=initial_3d_residual,
            absolute_improvement=absolute_improvement,relative_improvement=improvement,
            reason=converged ? "converged" : something(fallback_reason,"iteration limit reached"))
    else
        _force_balance_log(context,"restored_hse1d_no_improvement";iterations=iterations,
            force=fres,initial_force=initial_3d_residual,absolute_improvement=absolute_improvement,
            relative_improvement=improvement,
            reason=something(fallback_reason,"no improving 3-D iterate"))
    end
    ForceBalanceDiagnostics(mode isa HE3DMode ? :HE3D : :MHS,iterations,converged,
        pchange,rchange,fres,zchange,0.0,0.0,lorentzmax,initial_3d_residual,
        best_residual,absolute_improvement,improvement,accepted_3d)
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

struct HybridForwardModel{P,R,S,O,E,K,B,F,Q}
    populations::P; redistribution::R; synthesizer::S; observation::O
    eos::E; opacity::K; boundary::B; force_options::F; capabilities::CapabilityManifest
    prepared_force_balance::Q
end

function HybridForwardModel(populations,redistribution,synthesizer,observation,eos,opacity,
        boundary,force_options,capabilities::CapabilityManifest;prepared_force_balance=nothing)
    HybridForwardModel(populations,redistribution,synthesizer,observation,eos,opacity,boundary,
        force_options,capabilities,Ref{Any}(prepared_force_balance))
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
                  distributed::DistributedAtmosphere,context::ParallelContext;reuse_prepared::Bool=false)
    total_start=time_ns()
    validate_capabilities(model.capabilities,model.redistribution,workspace.output.stokes)
    prepared=reuse_prepared ? model.prepared_force_balance[] : nothing
    model.prepared_force_balance[]=nothing
    diagnostics,force_seconds,populations_seconds=if prepared===nothing
        stage_start=time_ns()
        force=reconstruct_force_balance_distributed!(distributed,model.boundary,model.eos,model.opacity,context;
            options=model.force_options)
        force_elapsed=(time_ns()-stage_start)/1e9
        stage_start=time_ns()
        predict_distributed_populations!(workspace.populations,model.populations,distributed,context)
        (force,force_elapsed,(time_ns()-stage_start)/1e9)
    else
        isroot(context) && (println(stdout,"FORWARD event=reuse_prepared_force_balance_and_populations"); flush(stdout))
        (prepared,0.0,0.0)
    end
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
