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
    lateral_boundary::Symbol
    bottom_boundary::Symbol
end
function ForceBalanceOptions(;gravity_m_s2=274.0,max_iterations=100,relative_tolerance=1e-6,
                             force_tolerance=1e-5,height_tolerance_m=1e-3,relaxation=0.7,
                             pressure_sweeps=10,lateral_boundary=:force_neumann,
                             bottom_boundary=:force_neumann)
    values=promote(gravity_m_s2,relative_tolerance,force_tolerance,height_tolerance_m,relaxation)
    lateral_boundary===:force_neumann || throw(ArgumentError(
        "lateral_boundary must be :force_neumann"))
    bottom_boundary===:force_neumann || throw(ArgumentError(
        "bottom_boundary must be :force_neumann"))
    ForceBalanceOptions(values[1],max_iterations,values[2],values[3],values[4],values[5],
        pressure_sweeps,lateral_boundary,bottom_boundary)
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
function _relax_pressure_3d!(p,fx,fy,fz,rho,z,grid,pboundary,order,g,sweeps)
    output=p; boundary_k=order[1]
    targetz=fz.-rho.*g; next=similar(p)
    for _ in 1:sweeps
        @views p[boundary_k,:,:].=pboundary
        copyto!(next,p); @views next[boundary_k,:,:].=pboundary
        for k in axes(p,1),i in axes(p,2),j in axes(p,3)
            k==boundary_k && continue
            total=0.0; count=0
            if i>1
                dx=grid.x[i]-grid.x[i-1]; dz=z[k,i,j]-z[k,i-1,j]
                total+=p[k,i-1,j]+(fx[k,i-1,j]+fx[k,i,j])*dx/2+
                    (targetz[k,i-1,j]+targetz[k,i,j])*dz/2; count+=1
            end
            if i<size(p,2)
                dx=grid.x[i+1]-grid.x[i]; dz=z[k,i+1,j]-z[k,i,j]
                total+=p[k,i+1,j]-(fx[k,i+1,j]+fx[k,i,j])*dx/2-
                    (targetz[k,i+1,j]+targetz[k,i,j])*dz/2; count+=1
            end
            if j>1
                dy=grid.y[j]-grid.y[j-1]; dz=z[k,i,j]-z[k,i,j-1]
                total+=p[k,i,j-1]+(fy[k,i,j-1]+fy[k,i,j])*dy/2+
                    (targetz[k,i,j-1]+targetz[k,i,j])*dz/2; count+=1
            end
            if j<size(p,3)
                dy=grid.y[j+1]-grid.y[j]; dz=z[k,i,j+1]-z[k,i,j]
                total+=p[k,i,j+1]-(fy[k,i,j+1]+fy[k,i,j])*dy/2-
                    (targetz[k,i,j+1]+targetz[k,i,j])*dz/2; count+=1
            end
            if k>1
                dz=z[k,i,j]-z[k-1,i,j]
                total+=p[k-1,i,j]+(targetz[k-1,i,j]+targetz[k,i,j])*dz/2; count+=1
            end
            if k<size(p,1)
                dz=z[k+1,i,j]-z[k,i,j]
                total+=p[k+1,i,j]-(targetz[k+1,i,j]+targetz[k,i,j])*dz/2; count+=1
            end
            next[k,i,j]=max(total/count,eps(eltype(p)))
        end
        p,next=next,p
    end
    p===output || copyto!(output,p)
    p=output
    @views p[boundary_k,:,:].=pboundary
    p
end

_relative_change(a,b)=maximum(abs.(a.-b)./max.(abs.(b),eps(eltype(b))))

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
    fx=zeros(shape); fy=zeros(shape); fz=zeros(shape); mode=select_force_balance(atmosphere)
    pchange=rchange=fres=zchange=Inf; lorentzmax=0.0; converged=false; iterations=0
    for iteration in 1:options.max_iterations
        iterations=iteration; thermodynamics!(rhonew,nenew,eos,atmosphere.temperature,p)
        opacity500!(kappa,opacity,atmosphere.temperature,p,rhonew,nenew)
        _height_from_tau!(znew,kappa,rhonew,tau,top_to_bottom)
        if mode isa MHSMode
            lorentz_force!(fx,fy,fz,atmosphere.magnetic_field,atmosphere.grid,znew)
            lorentzmax=maximum(sqrt.(fx.^2 .+ fy.^2 .+ fz.^2))
        end
        _pressure_from_force!(pnew,rhonew,znew,fz,pboundary,order,options.gravity_m_s2)
        _relax_pressure_3d!(pnew,fx,fy,fz,rhonew,znew,atmosphere.grid,pboundary,order,
                           options.gravity_m_s2,options.pressure_sweeps)
        pchange=_relative_change(pnew,p); rchange=_relative_change(rhonew,rho); zchange=maximum(abs.(znew.-z))
        fres=_force_residual(pnew,rhonew,znew,fx,fy,fz,atmosphere.grid,options.gravity_m_s2)
        @. p=options.relaxation*pnew+(1-options.relaxation)*p
        @. rho=options.relaxation*rhonew+(1-options.relaxation)*rho
        copyto!(ne,nenew); copyto!(z,znew)
        if pchange<=options.relative_tolerance && rchange<=options.relative_tolerance && fres<=options.force_tolerance && zchange<=options.height_tolerance_m
            converged=true; break
        end
    end
    if !converged && mode isa MHSMode
        throw(ErrorException("force balance did not converge after $(options.max_iterations) iterations (dP=$pchange, drho=$rchange, force=$fres, dz=$zchange)"))
    elseif !converged
        @warn "HE3D reached its iteration limit; accepting the 3-D relaxed atmosphere" iterations=options.max_iterations dP=pchange drho=rchange force=fres dz_m=zchange
    end
    atmosphere.pgas=p; atmosphere.rho=rho; atmosphere.ne=ne; atmosphere.z=z
    ForceBalanceDiagnostics(mode isa HE3DMode ? :HE3D : :MHS,iterations,converged,pchange,rchange,fres,zchange,0.0,0.0,lorentzmax)
end
