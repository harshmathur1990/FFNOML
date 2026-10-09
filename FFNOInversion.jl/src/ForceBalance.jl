abstract type AbstractForceBalanceMode end
struct HE3DMode <: AbstractForceBalanceMode end
struct MHSMode <: AbstractForceBalanceMode end

"""Select HE3D only when B is truly absent; never manufacture a zero B field."""
select_force_balance(atmosphere::Atmosphere3D) = isnothing(atmosphere.magnetic_field) ? HE3DMode() : MHSMode()

struct ForceBalanceOptions{T<:AbstractFloat}
    gravity_m_s2::T
    max_iterations::Int
    relative_tolerance::T
    force_tolerance::T
    height_tolerance_m::T
    relaxation::T
    pressure_sweeps::Int
    pressure_max_sweeps::Int
    pressure_tolerance::T
    bootstrap_iterations::Int
    continuation_iterations::Int
    max_backtracks::Int
    backtrack_growth_limit::T
    minimum_relaxation::T
    lateral_boundary::Symbol
    bottom_boundary::Symbol
end
function ForceBalanceOptions(;gravity_m_s2=274.0,max_iterations=100,relative_tolerance=1e-6,
                             force_tolerance=1e-5,height_tolerance_m=1e-3,relaxation=0.7,
                             pressure_sweeps=10,pressure_max_sweeps=200,
                             pressure_tolerance=1e-6,bootstrap_iterations=100,
                             continuation_iterations=10,max_backtracks=8,
                             backtrack_growth_limit=1.25,minimum_relaxation=1/256,
                             lateral_boundary=:force_neumann,
                             bottom_boundary=:force_neumann)
    values=promote(gravity_m_s2,relative_tolerance,force_tolerance,height_tolerance_m,
        relaxation,pressure_tolerance,backtrack_growth_limit,minimum_relaxation)
    lateral_boundary===:force_neumann || throw(ArgumentError(
        "lateral_boundary must be :force_neumann"))
    bottom_boundary===:force_neumann || throw(ArgumentError(
        "bottom_boundary must be :force_neumann"))
    ForceBalanceOptions(values[1],max_iterations,values[2],values[3],values[4],values[5],
        pressure_sweeps,pressure_max_sweeps,values[6],bootstrap_iterations,
        continuation_iterations,max_backtracks,values[7],values[8],lateral_boundary,bottom_boundary)
end

struct ForceBalanceDiagnostics{T<:AbstractFloat}
    mode::Symbol; iterations::Int; converged::Bool
    pressure_change::T; density_change::T; force_residual::T; height_change_m::T
    temperature_remap_error::T; magnetic_remap_error::T; lorentz_max_n_m3::T
end

const MU0 = 4pi*1e-7

@inline function _derivative(a,k,i,j,coord,dim)
    n=size(a,dim); n==1 && return zero(eltype(a)); q=dim==1 ? k : dim==2 ? i : j
    q0,q1=q==1 ? (1,2) : q==n ? (n-1,n) : (q-1,q+1)
    v0=dim==1 ? a[q0,i,j] : dim==2 ? a[k,q0,j] : a[k,i,q0]
    v1=dim==1 ? a[q1,i,j] : dim==2 ? a[k,q1,j] : a[k,i,q1]
    (v1-v0)/(coord[q1]-coord[q0])
end

@inline function _physical_derivatives(a,z,grid,k,i,j)
    aτ=_derivative(a,k,i,j,grid.log_tau500,1)
    zτ=_derivative(z,k,i,j,grid.log_tau500,1)
    abs(zτ)>eps(eltype(z)) || throw(ErrorException(
        "degenerate optical-depth mapping at depth=$k x=$i y=$j"))
    ax=_derivative(a,k,i,j,grid.x,2); ay=_derivative(a,k,i,j,grid.y,3)
    zx=_derivative(z,k,i,j,grid.x,2); zy=_derivative(z,k,i,j,grid.y,3)
    (ax-zx*aτ/zτ,ay-zy*aτ/zτ,aτ/zτ)
end

"""Compute current density and Lorentz force on the corrugated optical-depth grid."""
function lorentz_force!(fx,fy,fz,B::MagneticField3D,grid::Grid3D,z)
    size(fx)==size(fy)==size(fz)==size(B.Bx) || throw(DimensionMismatch("Lorentz arrays differ"))
    nx,ny=size(fx,2),size(fx,3)
    Threads.@threads :static for column in 1:nx*ny
        i=(column-1)%nx+1; j=(column-1)÷nx+1
        for k in axes(fx,1)
            dbxdx,dbxdy,dbxdz=_physical_derivatives(B.Bx,z,grid,k,i,j)
            dbydx,dbydy,dbydz=_physical_derivatives(B.By,z,grid,k,i,j)
            dbzdx,dbzdy,dbzdz=_physical_derivatives(B.Bz,z,grid,k,i,j)
            jx=(dbzdy-dbydz)/MU0
            jy=(dbxdz-dbzdx)/MU0
            jz=(dbydx-dbxdy)/MU0
            fx[k,i,j]=jy*B.Bz[k,i,j]-jz*B.By[k,i,j]
            fy[k,i,j]=jz*B.Bx[k,i,j]-jx*B.Bz[k,i,j]
            fz[k,i,j]=jx*B.By[k,i,j]-jy*B.Bx[k,i,j]
        end
    end
    fx,fy,fz
end

function _boundary_map(value,nx,ny,name)
    value isa Number && return fill(Float64(value),nx,ny)
    size(value)==(nx,ny) || throw(DimensionMismatch("$name boundary must be scalar or (nx,ny)"))
    Float64.(value)
end

function _height_from_tau!(z,kappa,rho,tau,order)
    fill!(z,0)
    nx,ny=size(z,2),size(z,3)
    Threads.@threads :static for column in 1:nx*ny
        i=(column-1)%nx+1; j=(column-1)÷nx+1
        for q in 2:length(order)
            k0,k1=order[q-1],order[q]; dtau=abs(tau[k1]-tau[k0])
            extinction=(kappa[k0,i,j]*rho[k0,i,j]+kappa[k1,i,j]*rho[k1,i,j])/2
            z[k1,i,j]=z[k0,i,j]-dtau/extinction
        end
    end
    z
end

function _pressure_from_force!(p,rho,z,fz,pboundary,order,g)
    @views p[order[1],:,:].=pboundary
    nx,ny=size(p,2),size(p,3)
    Threads.@threads :static for column in 1:nx*ny
        i=(column-1)%nx+1; j=(column-1)÷nx+1
        for q in 2:length(order)
            k0,k1=order[q-1],order[q]
            ds=abs(z[k1,i,j]-z[k0,i,j])
            p[k1,i,j]=p[k0,i,j]+(g*(rho[k0,i,j]+rho[k1,i,j])/2-(fz[k0,i,j]+fz[k1,i,j])/2)*ds
            p[k1,i,j]>0 || throw(ErrorException("non-positive pressure at depth=$k1 x=$i y=$j"))
        end
    end
end

"""Relax scalar pressure against physical edge integrals of the target force.

Each sweep averages pressure estimates propagated from all neighboring cells.
For a horizontal edge the actual corrugation `Δz` is retained, so the discrete
constraint is `ΔP = q⋅Δr`, not a derivative at fixed optical depth.
"""
function _relax_pressure_3d!(p,fx,fy,fz,rho,z,grid,pboundary,order,g,sweeps,
        horizontal_weight=1.0;max_sweeps=sweeps,tolerance=0.0,sweeps_used=nothing,
        converged_ref=nothing)
    max_sweeps>=sweeps>0 || throw(ArgumentError("pressure sweep limits must satisfy max >= minimum > 0"))
    output=p; boundary_k=order[1]
    targetz=fz.-rho.*g; next=similar(p)
    used=0; inner_converged=false
    for sweep in 1:max_sweeps
        @views p[boundary_k,:,:].=pboundary
        copyto!(next,p); @views next[boundary_k,:,:].=pboundary
        nx,ny=size(p,2),size(p,3)
        Threads.@threads :static for column in 1:nx*ny
            i=(column-1)%nx+1; j=(column-1)÷nx+1
            for k in axes(p,1)
                k==boundary_k && continue
                total=0.0; weight=0.0
                if i>1
                    dx=grid.x[i]-grid.x[i-1]; dz=z[k,i,j]-z[k,i-1,j]
                    estimate=p[k,i-1,j]+(fx[k,i-1,j]+fx[k,i,j])*dx/2+
                        (targetz[k,i-1,j]+targetz[k,i,j])*dz/2
                    total+=horizontal_weight*estimate; weight+=horizontal_weight
                end
                if i<nx
                    dx=grid.x[i+1]-grid.x[i]; dz=z[k,i+1,j]-z[k,i,j]
                    estimate=p[k,i+1,j]-(fx[k,i+1,j]+fx[k,i,j])*dx/2-
                        (targetz[k,i+1,j]+targetz[k,i,j])*dz/2
                    total+=horizontal_weight*estimate; weight+=horizontal_weight
                end
                if j>1
                    dy=grid.y[j]-grid.y[j-1]; dz=z[k,i,j]-z[k,i,j-1]
                    estimate=p[k,i,j-1]+(fy[k,i,j-1]+fy[k,i,j])*dy/2+
                        (targetz[k,i,j-1]+targetz[k,i,j])*dz/2
                    total+=horizontal_weight*estimate; weight+=horizontal_weight
                end
                if j<ny
                    dy=grid.y[j+1]-grid.y[j]; dz=z[k,i,j+1]-z[k,i,j]
                    estimate=p[k,i,j+1]-(fy[k,i,j+1]+fy[k,i,j])*dy/2-
                        (targetz[k,i,j+1]+targetz[k,i,j])*dz/2
                    total+=horizontal_weight*estimate; weight+=horizontal_weight
                end
                if k>1
                    dz=z[k,i,j]-z[k-1,i,j]
                    total+=p[k-1,i,j]+(targetz[k-1,i,j]+targetz[k,i,j])*dz/2; weight+=1
                end
                if k<size(p,1)
                    dz=z[k+1,i,j]-z[k,i,j]
                    total+=p[k+1,i,j]-(targetz[k+1,i,j]+targetz[k,i,j])*dz/2; weight+=1
                end
                next[k,i,j]=max(total/weight,eps(eltype(p)))
            end
        end
        change=maximum(abs.(next.-p)./max.(abs.(p),one(eltype(p))))
        p,next=next,p; used=sweep
        if sweep>=sweeps && tolerance>0 && change<=tolerance
            inner_converged=true; break
        end
    end
    p===output || copyto!(output,p)
    p=output
    @views p[boundary_k,:,:].=pboundary
    sweeps_used===nothing || (sweeps_used[]=used)
    converged_ref===nothing || (converged_ref[]=inner_converged)
    p
end

_relative_change(a,b)=maximum(abs.(a.-b)./max.(abs.(b),eps(eltype(b))))

function _blend_state!(out,candidate,current,alpha)
    @. out=alpha*candidate+(1-alpha)*current
    out
end

function _validate_force_state(stage,arrays...)
    for array in arrays
        all(isfinite,array) || throw(ErrorException("$stage produced non-finite values"))
    end
    nothing
end

"""Build a finite column-wise hydrostatic initial state before 3-D relaxation."""
function _bootstrap_hse_1d!(p,rho,ne,z,kappa,pnew,rhonew,nenew,znew,temperature,
        eos,opacity,tau,top_to_bottom,order,pboundary,options)
    zero_force=zeros(eltype(p),size(p))
    pchange=rchange=zchange=Inf
    for iteration in 1:options.bootstrap_iterations
        thermodynamics!(rhonew,nenew,eos,temperature,p)
        opacity500!(kappa,opacity,temperature,p,rhonew,nenew)
        _height_from_tau!(znew,kappa,rhonew,tau,top_to_bottom)
        _pressure_from_force!(pnew,rhonew,znew,zero_force,pboundary,order,
            options.gravity_m_s2)
        _validate_force_state("1-D HSE bootstrap",pnew,rhonew,nenew,znew)
        pchange=_relative_change(pnew,p)
        rchange=_relative_change(rhonew,rho)
        zchange=maximum(abs.(znew.-z))
        @. p=options.relaxation*pnew+(1-options.relaxation)*p
        @. rho=options.relaxation*rhonew+(1-options.relaxation)*rho
        copyto!(ne,nenew); copyto!(z,znew)
        if pchange<=options.relative_tolerance && rchange<=options.relative_tolerance &&
                zchange<=options.height_tolerance_m
            return iteration,pchange,rchange,zchange
        end
    end
    throw(ErrorException("1-D HSE bootstrap did not converge after $(options.bootstrap_iterations) iterations (dP=$pchange, drho=$rchange, dz=$zchange)"))
end

function _force_residual(p,rho,z,fx,fy,fz,grid,g)
    scale=max(abs(g)*maximum(abs,rho),maximum(sqrt.(fx.^2 .+ fy.^2 .+ fz.^2)))+eps(eltype(p))
    qz=fz.-rho.*g
    nx,ny=size(p,2),size(p,3); column_worst=zeros(eltype(p),nx*ny)
    Threads.@threads :static for column in 1:nx*ny
        i=(column-1)%nx+1; j=(column-1)÷nx+1
        local_worst=zero(eltype(p))
        for k in axes(p,1)
            if i<nx
                dx=grid.x[i+1]-grid.x[i]; dz=z[k,i+1,j]-z[k,i,j]; ds=hypot(dx,dz)
                mismatch=p[k,i+1,j]-p[k,i,j]-(fx[k,i+1,j]+fx[k,i,j])*dx/2-
                    (qz[k,i+1,j]+qz[k,i,j])*dz/2
                local_worst=max(local_worst,abs(mismatch)/(ds*scale))
            end
            if j<ny
                dy=grid.y[j+1]-grid.y[j]; dz=z[k,i,j+1]-z[k,i,j]; ds=hypot(dy,dz)
                mismatch=p[k,i,j+1]-p[k,i,j]-(fy[k,i,j+1]+fy[k,i,j])*dy/2-
                    (qz[k,i,j+1]+qz[k,i,j])*dz/2
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
    maximum(column_worst)
end

"""Iteratively reconstruct `Pgas`, `rho`, `ne`, and corrugated `z` on fixed log-tau points."""
function reconstruct_force_balance!(atmosphere::Atmosphere3D{Float64},boundary::HE3DBoundaryState,
                                    eos::AbstractEOS,opacity::AbstractOpacity500;
                                    options::ForceBalanceOptions=ForceBalanceOptions())
    boundary.boundary in (:top,:bottom) || throw(ArgumentError("Phase 1 reconstruction supports top or bottom boundaries"))
    options.max_iterations>0 || throw(ArgumentError("max_iterations must be positive"))
    options.pressure_sweeps>0 || throw(ArgumentError("pressure_sweeps must be positive"))
    options.pressure_max_sweeps>=options.pressure_sweeps || throw(ArgumentError(
        "pressure_max_sweeps must be at least pressure_sweeps"))
    options.pressure_tolerance>0 || throw(ArgumentError("pressure_tolerance must be positive"))
    options.bootstrap_iterations>0 || throw(ArgumentError("bootstrap_iterations must be positive"))
    options.continuation_iterations>0 || throw(ArgumentError("continuation_iterations must be positive"))
    options.max_backtracks>=0 || throw(ArgumentError("max_backtracks must be non-negative"))
    options.backtrack_growth_limit>=1 || throw(ArgumentError("backtrack_growth_limit must be at least one"))
    0<options.minimum_relaxation<=options.relaxation || throw(ArgumentError(
        "minimum_relaxation must lie in (0, relaxation]"))
    options.lateral_boundary===:force_neumann || throw(ArgumentError("unsupported lateral boundary"))
    options.bottom_boundary===:force_neumann || throw(ArgumentError("unsupported bottom boundary"))
    0<options.relaxation<=1 || throw(ArgumentError("relaxation must lie in (0,1]"))
    shape=size(atmosphere.temperature); nz,nx,ny=shape; tau=10.0.^atmosphere.grid.log_tau500
    top_to_bottom=tau[1]<tau[end] ? collect(1:nz) : collect(nz:-1:1)
    order=boundary.boundary==:top ? top_to_bottom : reverse(top_to_bottom)
    pboundary=_boundary_map(boundary.p0,nx,ny,:pressure); rho_boundary=_boundary_map(boundary.rho0,nx,ny,:density)
    p=atmosphere.pgas===nothing ? repeat(reshape(pboundary,1,nx,ny),nz,1,1) : copy(atmosphere.pgas)
    rho=atmosphere.rho===nothing ? repeat(reshape(rho_boundary,1,nx,ny),nz,1,1) : copy(atmosphere.rho)
    ne=atmosphere.ne===nothing ? similar(rho) : copy(atmosphere.ne); z=atmosphere.z===nothing ? zeros(Float64,shape) : copy(atmosphere.z)
    kappa=similar(rho); pnew=similar(p); rhonew=similar(rho); nenew=similar(ne); znew=similar(z)
    ptrial=similar(p); rhotrial=similar(rho); netrial=similar(ne); ztrial=similar(z)
    fx=zeros(shape); fy=zeros(shape); fz=zeros(shape); mode=select_force_balance(atmosphere)
    fxtrial=zeros(shape); fytrial=zeros(shape); fztrial=zeros(shape)
    pchange=rchange=fres=zchange=Inf; lorentzmax=0.0; converged=false; iterations=0
    _bootstrap_hse_1d!(p,rho,ne,z,kappa,pnew,rhonew,nenew,znew,
        atmosphere.temperature,eos,opacity,tau,top_to_bottom,order,pboundary,options)
    if mode isa MHSMode
        lorentz_force!(fx,fy,fz,atmosphere.magnetic_field,atmosphere.grid,z)
        lorentzmax=maximum(sqrt.(fx.^2 .+ fy.^2 .+ fz.^2))
    end
    initial_3d_residual=_force_residual(p,rho,z,fx,fy,fz,atmosphere.grid,
        options.gravity_m_s2)
    continuation_needed=initial_3d_residual>options.force_tolerance
    bootstrap_state=(copy(p),copy(rho),copy(ne),copy(z)); fallback_reason=nothing
    current_residual=initial_3d_residual
    for iteration in 1:options.max_iterations
        iterations=iteration; thermodynamics!(rhonew,nenew,eos,atmosphere.temperature,p)
        opacity500!(kappa,opacity,atmosphere.temperature,p,rhonew,nenew)
        _height_from_tau!(znew,kappa,rhonew,tau,top_to_bottom)
        if mode isa MHSMode
            lorentz_force!(fx,fy,fz,atmosphere.magnetic_field,atmosphere.grid,znew)
            lorentzmax=maximum(sqrt.(fx.^2 .+ fy.^2 .+ fz.^2))
        end
        _pressure_from_force!(pnew,rhonew,znew,fz,pboundary,order,options.gravity_m_s2)
        horizontal_weight=continuation_needed ?
            min(1.0,iteration/options.continuation_iterations) : 1.0
        inner_sweeps=Ref(0); inner_converged=Ref(false)
        _relax_pressure_3d!(pnew,fx,fy,fz,rhonew,znew,atmosphere.grid,pboundary,order,
            options.gravity_m_s2,options.pressure_sweeps,horizontal_weight;
            max_sweeps=options.pressure_max_sweeps,tolerance=options.pressure_tolerance,
            sweeps_used=inner_sweeps,converged_ref=inner_converged)
        _validate_force_state("3-D force relaxation",pnew,rhonew,nenew,znew,fx,fy,fz)
        accepted=false; alpha=options.relaxation
        for backtrack in 0:options.max_backtracks
            alpha<options.minimum_relaxation && break
            _blend_state!(ptrial,pnew,p,alpha); _blend_state!(rhotrial,rhonew,rho,alpha)
            _blend_state!(netrial,nenew,ne,alpha); _blend_state!(ztrial,znew,z,alpha)
            _validate_force_state("3-D backtracking trial",ptrial,rhotrial,netrial,ztrial)
            if mode isa MHSMode
                lorentz_force!(fxtrial,fytrial,fztrial,atmosphere.magnetic_field,
                    atmosphere.grid,ztrial)
            else
                fill!(fxtrial,0); fill!(fytrial,0); fill!(fztrial,0)
            end
            trial_residual=_force_residual(ptrial,rhotrial,ztrial,fxtrial,fytrial,
                fztrial,atmosphere.grid,options.gravity_m_s2)
            threshold=max(options.force_tolerance,
                options.backtrack_growth_limit*current_residual)
            if isfinite(trial_residual) && trial_residual<=threshold
                pchange=_relative_change(ptrial,p); rchange=_relative_change(rhotrial,rho)
                zchange=maximum(abs.(ztrial.-z)); fres=trial_residual
                copyto!(p,ptrial); copyto!(rho,rhotrial); copyto!(ne,netrial); copyto!(z,ztrial)
                copyto!(fx,fxtrial); copyto!(fy,fytrial); copyto!(fz,fztrial)
                current_residual=fres; accepted=true; break
            end
            alpha/=2
        end
        if !accepted
            fallback_reason="all backtracking trials increased the force residual from $current_residual"
            break
        end
        if horizontal_weight==1 && pchange<=options.relative_tolerance &&
                rchange<=options.relative_tolerance && fres<=options.force_tolerance &&
                zchange<=options.height_tolerance_m
            converged=true; break
        end
    end
    degraded=!converged && (fallback_reason!==nothing ||
        fres>max(options.force_tolerance,initial_3d_residual))
    if degraded && mode isa HE3DMode
        copyto!(p,bootstrap_state[1]); copyto!(rho,bootstrap_state[2])
        copyto!(ne,bootstrap_state[3]); copyto!(z,bootstrap_state[4])
        pchange=rchange=zchange=0.0; fres=initial_3d_residual
        @warn "3-D HE relaxation degraded; using the converged 1-D HSE bootstrap" reason=something(fallback_reason,"3-D residual increased") force=fres
    end
    if !converged && mode isa MHSMode
        throw(ErrorException("force balance did not converge after $(options.max_iterations) iterations (dP=$pchange, drho=$rchange, force=$fres, dz=$zchange)"))
    elseif !converged && !degraded
        @warn "HE3D reached its iteration limit; accepting the 3-D relaxed atmosphere" iterations=options.max_iterations dP=pchange drho=rchange force=fres dz_m=zchange
    end
    atmosphere.pgas=p; atmosphere.rho=rho; atmosphere.ne=ne; atmosphere.z=z
    ForceBalanceDiagnostics(mode isa HE3DMode ? :HE3D : :MHS,iterations,converged,pchange,rchange,fres,zchange,0.0,0.0,lorentzmax)
end
