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
function ForceBalanceOptions(;gravity_m_s2=274.0,max_iterations=500,relative_tolerance=1e-6,
                             force_tolerance=1e-5,height_tolerance_m=1e-3,relaxation=0.7,
                             pressure_sweeps=10,pressure_max_sweeps=200,
                             pressure_tolerance=1e-6,bootstrap_iterations=1000,
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
    initial_force_residual::T; best_force_residual::T
    absolute_force_improvement::T; relative_force_improvement::T; accepted_3d::Bool
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

"""Return adjacent samples and the linear weight for `logtau500 = 0`.

The returned weight is not clamped: grids that do not span zero use the two
nearest endpoint samples for linear extrapolation.  Production atmospheres
normally span zero, in which case this is ordinary interpolation.
"""
function _zero_logtau_bracket(logtau)
    n=length(logtau)
    n>=2 || throw(ArgumentError("at least two optical-depth samples are required to define z(logtau500=0)"))
    increasing=logtau[end]>logtau[1]
    decreasing=logtau[end]<logtau[1]
    (increasing || decreasing) || throw(ArgumentError("logtau500 must be monotonic"))
    if increasing
        k1=clamp(searchsortedfirst(logtau,zero(eltype(logtau))),2,n)
        k0=k1-1
    else
        reversed=reverse(logtau)
        q1=clamp(searchsortedfirst(reversed,zero(eltype(logtau))),2,n)
        q0=q1-1
        k0=n-q0+1
        k1=n-q1+1
    end
    denominator=logtau[k1]-logtau[k0]
    iszero(denominator) && throw(ArgumentError("logtau500 samples must be distinct"))
    weight=-logtau[k0]/denominator
    k0,k1,weight
end

function _height_from_tau!(z,kappa,rho,tau,order)
    fill!(z,0)
    nx,ny=size(z,2),size(z,3)
    logtau=log10.(tau)
    reference_k0,reference_k1,reference_weight=_zero_logtau_bracket(logtau)
    Threads.@threads :static for column in 1:nx*ny
        i=(column-1)%nx+1; j=(column-1)÷nx+1
        for q in 2:length(order)
            k0,k1=order[q-1],order[q]; dtau=abs(tau[k1]-tau[k0])
            extinction=(kappa[k0,i,j]*rho[k0,i,j]+kappa[k1,i,j]*rho[k1,i,j])/2
            z[k1,i,j]=z[k0,i,j]-dtau/extinction
        end
        reference_height=(1-reference_weight)*z[reference_k0,i,j]+
            reference_weight*z[reference_k1,i,j]
        for k in axes(z,1)
            z[k,i,j]-=reference_height
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

@inline function _x_force_integral(fx,qz,z,grid,k,i,j)
    dx=grid.x[i+1]-grid.x[i]; dz=z[k,i+1,j]-z[k,i,j]
    (fx[k,i,j]+fx[k,i+1,j])*dx/2+(qz[k,i,j]+qz[k,i+1,j])*dz/2
end

@inline function _y_force_integral(fy,qz,z,grid,k,i,j)
    dy=grid.y[j+1]-grid.y[j]; dz=z[k,i,j+1]-z[k,i,j]
    (fy[k,i,j]+fy[k,i,j+1])*dy/2+(qz[k,i,j]+qz[k,i,j+1])*dz/2
end

@inline function _z_force_integral(qz,z,k,i,j)
    dz=z[k+1,i,j]-z[k,i,j]
    (qz[k,i,j]+qz[k+1,i,j])*dz/2
end

"""Discrete curl diagnostic for the requested force on physical grid faces.

Returns the maximum and RMS normalized circulation. Zero means that the edge
force is locally compatible with a scalar pressure. Coordinates in `grid`,
including the configured physical `dx` and `dy`, enter every horizontal edge.
"""
function _force_integrability(fx,fy,fz,rho,z,grid,g)
    qz=fz.-rho.*g; nz,nx,ny=size(rho)
    sums=zeros(Float64,nx*ny); counts=zeros(Int,nx*ny); maxima=zeros(Float64,nx*ny)
    Threads.@threads :static for column in 1:nx*ny
        i=(column-1)%nx+1; j=(column-1)÷nx+1
        local_sum=0.0; local_count=0; local_max=0.0
        if i<nx && j<ny
            for k in 1:nz
                edges=(_x_force_integral(fx,qz,z,grid,k,i,j),
                    _y_force_integral(fy,qz,z,grid,k,i+1,j),
                    _x_force_integral(fx,qz,z,grid,k,i,j+1),
                    _y_force_integral(fy,qz,z,grid,k,i,j))
                circulation=edges[1]+edges[2]-edges[3]-edges[4]
                ratio=abs(circulation)/(sum(abs,edges)+eps(Float64))
                local_sum+=ratio^2; local_count+=1; local_max=max(local_max,ratio)
            end
        end
        if i<nx
            for k in 1:nz-1
                edges=(_x_force_integral(fx,qz,z,grid,k,i,j),
                    _z_force_integral(qz,z,k,i+1,j),
                    _x_force_integral(fx,qz,z,grid,k+1,i,j),
                    _z_force_integral(qz,z,k,i,j))
                circulation=edges[1]+edges[2]-edges[3]-edges[4]
                ratio=abs(circulation)/(sum(abs,edges)+eps(Float64))
                local_sum+=ratio^2; local_count+=1; local_max=max(local_max,ratio)
            end
        end
        if j<ny
            for k in 1:nz-1
                edges=(_y_force_integral(fy,qz,z,grid,k,i,j),
                    _z_force_integral(qz,z,k,i,j+1),
                    _y_force_integral(fy,qz,z,grid,k+1,i,j),
                    _z_force_integral(qz,z,k,i,j))
                circulation=edges[1]+edges[2]-edges[3]-edges[4]
                ratio=abs(circulation)/(sum(abs,edges)+eps(Float64))
                local_sum+=ratio^2; local_count+=1; local_max=max(local_max,ratio)
            end
        end
        sums[column]=local_sum; counts[column]=local_count; maxima[column]=local_max
    end
    total_count=sum(counts)
    (maximum=max(maximum(maxima),0.0),rms=sqrt(sum(sums)/max(total_count,1)),
        faces=total_count)
end

function _pressure_poisson_system!(rhs,diagonal,fx,fy,fz,rho,z,grid,pboundary,
        boundary_k,g,horizontal_weight)
    fill!(rhs,0); fill!(diagonal,0); qz=fz.-rho.*g
    nz,nx,ny=size(rhs)
    Threads.@threads :static for column in 1:nx*ny
        i=(column-1)%nx+1; j=(column-1)÷nx+1
        for k in 1:nz
            k==boundary_k && continue
            source=0.0; degree=0.0
            if i>1
                dx=grid.x[i]-grid.x[i-1]; dz=z[k,i,j]-z[k,i-1,j]
                weight=horizontal_weight/(dx^2+dz^2)
                edge=(fx[k,i-1,j]+fx[k,i,j])*dx/2+(qz[k,i-1,j]+qz[k,i,j])*dz/2
                degree+=weight; source+=weight*edge
            end
            if i<nx
                dx=grid.x[i+1]-grid.x[i]; dz=z[k,i+1,j]-z[k,i,j]
                weight=horizontal_weight/(dx^2+dz^2)
                edge=(fx[k,i,j]+fx[k,i+1,j])*dx/2+(qz[k,i,j]+qz[k,i+1,j])*dz/2
                degree+=weight; source-=weight*edge
            end
            if j>1
                dy=grid.y[j]-grid.y[j-1]; dz=z[k,i,j]-z[k,i,j-1]
                weight=horizontal_weight/(dy^2+dz^2)
                edge=(fy[k,i,j-1]+fy[k,i,j])*dy/2+(qz[k,i,j-1]+qz[k,i,j])*dz/2
                degree+=weight; source+=weight*edge
            end
            if j<ny
                dy=grid.y[j+1]-grid.y[j]; dz=z[k,i,j+1]-z[k,i,j]
                weight=horizontal_weight/(dy^2+dz^2)
                edge=(fy[k,i,j]+fy[k,i,j+1])*dy/2+(qz[k,i,j]+qz[k,i,j+1])*dz/2
                degree+=weight; source-=weight*edge
            end
            if k>1
                dz=z[k,i,j]-z[k-1,i,j]; weight=inv(dz^2)
                edge=(qz[k-1,i,j]+qz[k,i,j])*dz/2
                degree+=weight; source+=weight*edge
                k-1==boundary_k && (source+=weight*pboundary[i,j])
            end
            if k<nz
                dz=z[k+1,i,j]-z[k,i,j]; weight=inv(dz^2)
                edge=(qz[k,i,j]+qz[k+1,i,j])*dz/2
                degree+=weight; source-=weight*edge
                k+1==boundary_k && (source+=weight*pboundary[i,j])
            end
            rhs[k,i,j]=source; diagonal[k,i,j]=degree
        end
    end
    @views rhs[boundary_k,:,:].=0; @views diagonal[boundary_k,:,:].=1
    rhs,diagonal
end

function _apply_pressure_laplacian!(out,x,z,grid,boundary_k,horizontal_weight)
    fill!(out,0); nz,nx,ny=size(x)
    Threads.@threads :static for column in 1:nx*ny
        i=(column-1)%nx+1; j=(column-1)÷nx+1
        for k in 1:nz
            k==boundary_k && continue
            value=0.0; degree=0.0
            if i>1
                ds2=(grid.x[i]-grid.x[i-1])^2+(z[k,i,j]-z[k,i-1,j])^2
                weight=horizontal_weight/ds2; degree+=weight; value-=weight*x[k,i-1,j]
            end
            if i<nx
                ds2=(grid.x[i+1]-grid.x[i])^2+(z[k,i+1,j]-z[k,i,j])^2
                weight=horizontal_weight/ds2; degree+=weight; value-=weight*x[k,i+1,j]
            end
            if j>1
                ds2=(grid.y[j]-grid.y[j-1])^2+(z[k,i,j]-z[k,i,j-1])^2
                weight=horizontal_weight/ds2; degree+=weight; value-=weight*x[k,i,j-1]
            end
            if j<ny
                ds2=(grid.y[j+1]-grid.y[j])^2+(z[k,i,j+1]-z[k,i,j])^2
                weight=horizontal_weight/ds2; degree+=weight; value-=weight*x[k,i,j+1]
            end
            if k>1
                weight=inv((z[k,i,j]-z[k-1,i,j])^2); degree+=weight
                k-1==boundary_k || (value-=weight*x[k-1,i,j])
            end
            if k<nz
                weight=inv((z[k+1,i,j]-z[k,i,j])^2); degree+=weight
                k+1==boundary_k || (value-=weight*x[k+1,i,j])
            end
            out[k,i,j]=degree*x[k,i,j]+value
        end
    end
    out
end

"""Solve the physical-edge least-squares pressure projection with PCG."""
function _solve_pressure_poisson_3d!(p,fx,fy,fz,rho,z,grid,pboundary,order,g;
        horizontal_weight=1.0,max_iterations=1000,tolerance=1e-8)
    boundary_k=order[1]; rhs=similar(p); diagonal=similar(p)
    _pressure_poisson_system!(rhs,diagonal,fx,fy,fz,rho,z,grid,pboundary,
        boundary_k,g,horizontal_weight)
    @views p[boundary_k,:,:].=pboundary
    applied=similar(p); residual=similar(p); preconditioned=similar(p)
    direction=similar(p); product=similar(p)
    _apply_pressure_laplacian!(applied,p,z,grid,boundary_k,horizontal_weight)
    @. residual=rhs-applied
    @views residual[boundary_k,:,:].=0
    @. preconditioned=residual/diagonal
    copyto!(direction,preconditioned)
    rz=dot(vec(residual),vec(preconditioned))
    rhs_norm=max(norm(vec(rhs)),eps(Float64)); relative=norm(vec(residual))/rhs_norm
    relative<=tolerance && return (iterations=0,residual=relative,converged=true)
    for iteration in 1:max_iterations
        _apply_pressure_laplacian!(product,direction,z,grid,boundary_k,horizontal_weight)
        denominator=dot(vec(direction),vec(product))
        isfinite(denominator) && denominator>0 || return (
            iterations=iteration-1,residual=relative,converged=false)
        alpha=rz/denominator
        @. p=p+alpha*direction
        @. residual=residual-alpha*product
        @views p[boundary_k,:,:].=pboundary
        @views residual[boundary_k,:,:].=0
        relative=norm(vec(residual))/rhs_norm
        relative<=tolerance && return (iterations=iteration,residual=relative,converged=true)
        @. preconditioned=residual/diagonal
        rz_new=dot(vec(residual),vec(preconditioned))
        beta=rz_new/rz
        @. direction=preconditioned+beta*direction
        @views direction[boundary_k,:,:].=0
        rz=rz_new
    end
    (iterations=max_iterations,residual=relative,converged=false)
end

"""Normalized physical-edge least-squares objective used by the Poisson projection."""
function _pressure_projection_residual(p,fx,fy,fz,rho,z,grid,g;
        horizontal_weight=1.0)
    qz=fz.-rho.*g; nz,nx,ny=size(p)
    numerators=zeros(Float64,nx*ny); denominators=zeros(Float64,nx*ny)
    Threads.@threads :static for column in 1:nx*ny
        i=(column-1)%nx+1; j=(column-1)÷nx+1
        numerator=0.0; denominator=0.0
        if i<nx
            for k in 1:nz
                dx=grid.x[i+1]-grid.x[i]; dz=z[k,i+1,j]-z[k,i,j]
                weight=horizontal_weight/(dx^2+dz^2)
                edge=_x_force_integral(fx,qz,z,grid,k,i,j)
                mismatch=p[k,i+1,j]-p[k,i,j]-edge
                numerator+=weight*mismatch^2; denominator+=weight*edge^2
            end
        end
        if j<ny
            for k in 1:nz
                dy=grid.y[j+1]-grid.y[j]; dz=z[k,i,j+1]-z[k,i,j]
                weight=horizontal_weight/(dy^2+dz^2)
                edge=_y_force_integral(fy,qz,z,grid,k,i,j)
                mismatch=p[k,i,j+1]-p[k,i,j]-edge
                numerator+=weight*mismatch^2; denominator+=weight*edge^2
            end
        end
        for k in 1:nz-1
            dz=z[k+1,i,j]-z[k,i,j]; weight=inv(dz^2)
            edge=_z_force_integral(qz,z,k,i,j)
            mismatch=p[k+1,i,j]-p[k,i,j]-edge
            numerator+=weight*mismatch^2; denominator+=weight*edge^2
        end
        numerators[column]=numerator; denominators[column]=denominator
    end
    sqrt(sum(numerators)/max(sum(denominators),eps(Float64)))
end

"""Bound-constrained pressure projection using preconditioned projected gradients.

All free cells obey `p >= pressure_floor`; the selected boundary plane remains
fixed to `pboundary`. This is a diagnostic/robust fallback for incompatible
force fields, where the unconstrained Poisson optimum can have negative gas
pressure.
"""
function _solve_pressure_poisson_positive_3d!(p,fx,fy,fz,rho,z,grid,pboundary,
        order,g;horizontal_weight=1.0,max_iterations=2000,tolerance=1e-8,
        pressure_floor=minimum(pboundary))
    pressure_floor>0 || throw(ArgumentError("pressure_floor must be positive"))
    boundary_k=order[1]; rhs=similar(p); diagonal=similar(p)
    _pressure_poisson_system!(rhs,diagonal,fx,fy,fz,rho,z,grid,pboundary,
        boundary_k,g,horizontal_weight)
    @. p=max(p,pressure_floor)
    @views p[boundary_k,:,:].=pboundary
    applied=similar(p); residual=similar(p); preconditioned=similar(p)
    direction=similar(p); product=similar(p)
    rhs_norm=max(norm(vec(rhs)),eps(Float64)); relative=Inf; rz=0.0
    restart=true; free_cells=length(p)-size(p,2)*size(p,3)
    for iteration in 1:max_iterations
        _apply_pressure_laplacian!(applied,p,z,grid,boundary_k,horizontal_weight)
        @. residual=rhs-applied
        @. residual=ifelse(p<=pressure_floor*(1+1e-12) && residual<0,0.0,residual)
        @views residual[boundary_k,:,:].=0
        relative=norm(vec(residual))/rhs_norm
        if relative<=tolerance
            active=count(index->index[1]!=boundary_k &&
                p[index]<=pressure_floor*(1+1e-10),CartesianIndices(p))
            return (iterations=iteration-1,residual=relative,
                converged=true,active_fraction=active/free_cells)
        end
        @. preconditioned=residual/diagonal
        rz_new=dot(vec(residual),vec(preconditioned))
        if restart
            copyto!(direction,preconditioned)
        else
            beta=rz_new/rz
            @. direction=preconditioned+beta*direction
        end
        @views direction[boundary_k,:,:].=0
        _apply_pressure_laplacian!(product,direction,z,grid,boundary_k,horizontal_weight)
        denominator=dot(vec(direction),vec(product))
        if !(isfinite(denominator) && denominator>0 && isfinite(rz_new) && rz_new>0)
            active=count(index->index[1]!=boundary_k &&
                p[index]<=pressure_floor*(1+1e-10),CartesianIndices(p))
            return (iterations=iteration-1,residual=relative,converged=false,
                active_fraction=active/free_cells)
        end
        alpha=rz_new/denominator; bound_hit=false
        for index in CartesianIndices(p)
            index[1]==boundary_k && continue
            if direction[index]<0
                limit=(p[index]-pressure_floor)/(-direction[index])
                if limit<alpha
                    alpha=max(limit,0.0); bound_hit=true
                end
            end
        end
        if alpha==0
            restart=true; rz=rz_new
            @. direction=ifelse(p<=pressure_floor*(1+1e-12) && direction<0,0.0,direction)
            continue
        end
        @. p=max(pressure_floor,p+alpha*direction)
        @views p[boundary_k,:,:].=pboundary
        restart=bound_hit; rz=rz_new
    end
    active=count(index->index[1]!=boundary_k &&
        p[index]<=pressure_floor*(1+1e-10),CartesianIndices(p))
    (iterations=max_iterations,residual=relative,converged=false,
        active_fraction=active/free_cells)
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

function _valid_hse_1d_state(p,rho,ne,z,kappa,top_to_bottom)
    all(>(zero(eltype(p))),p) || return false
    all(>(zero(eltype(rho))),rho) || return false
    all(>(zero(eltype(ne))),ne) || return false
    all(>(zero(eltype(kappa))),kappa) || return false
    all(isfinite,z) || return false
    for q in 2:length(top_to_bottom)
        upper,lower=top_to_bottom[q-1],top_to_bottom[q]
        all(@view(z[lower,:,:]) .< @view(z[upper,:,:])) || return false
    end
    true
end

@inline function _bootstrap_change_metric(pchange,rchange,zchange,options)
    max(pchange/options.relative_tolerance,rchange/options.relative_tolerance,
        zchange/options.height_tolerance_m)
end

"""Build a finite column-wise hydrostatic initial state before 3-D relaxation."""
function _bootstrap_hse_1d!(p,rho,ne,z,kappa,pnew,rhonew,nenew,znew,temperature,
        eos,opacity,tau,top_to_bottom,order,pboundary,options)
    zero_force=zeros(eltype(p),size(p))
    pchange=rchange=zchange=Inf
    recent_metrics=Float64[]
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
        push!(recent_metrics,_bootstrap_change_metric(pchange,rchange,zchange,options))
        length(recent_metrics)>20 && popfirst!(recent_metrics)
        if pchange<=options.relative_tolerance && rchange<=options.relative_tolerance &&
                zchange<=options.height_tolerance_m
            return iteration,pchange,rchange,zchange
        end
    end
    state_valid=_valid_hse_1d_state(p,rho,ne,z,kappa,top_to_bottom)
    final_metric=last(recent_metrics)
    nondivergent=isfinite(final_metric) &&
        final_metric<=options.backtrack_growth_limit*minimum(recent_metrics)
    if state_valid && nondivergent
        @warn "1-D HSE reached its iteration limit; accepting the finite, positive, monotonic, non-divergent state" iterations=options.bootstrap_iterations dP=pchange drho=rchange dz_m=zchange
        return options.bootstrap_iterations,pchange,rchange,zchange
    end
    throw(ErrorException("1-D HSE bootstrap was invalid or diverging after $(options.bootstrap_iterations) iterations (valid=$state_valid, nondivergent=$nondivergent, dP=$pchange, drho=$rchange, dz=$zchange)"))
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
    best_state=(copy(p),copy(rho),copy(ne),copy(z)); fallback_reason=nothing
    best_residual=initial_3d_residual; best_pchange=best_rchange=best_zchange=0.0
    best_lorentzmax=lorentzmax
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
        if fres<best_residual
            best_residual=fres; best_pchange=pchange; best_rchange=rchange
            best_zchange=zchange
            best_lorentzmax=mode isa MHSMode ?
                maximum(sqrt.(fxtrial.^2 .+ fytrial.^2 .+ fztrial.^2)) : 0.0
            copyto!(best_state[1],p); copyto!(best_state[2],rho)
            copyto!(best_state[3],ne); copyto!(best_state[4],z)
        end
        if horizontal_weight==1 && pchange<=options.relative_tolerance &&
                rchange<=options.relative_tolerance && fres<=options.force_tolerance &&
                zchange<=options.height_tolerance_m
            converged=true; break
        end
    end
    absolute_improvement=max(0.0,initial_3d_residual-best_residual)
    improvement=absolute_improvement/max(initial_3d_residual,eps(Float64))
    accepted_3d=best_residual<initial_3d_residual*(1-sqrt(eps(Float64)))
    copyto!(p,best_state[1]); copyto!(rho,best_state[2])
    copyto!(ne,best_state[3]); copyto!(z,best_state[4])
    pchange=best_pchange; rchange=best_rchange; zchange=best_zchange
    fres=best_residual; lorentzmax=best_lorentzmax
    if accepted_3d && !converged
        @warn "accepting the best non-divergent 3-D force-balance improvement" iterations=iterations initial_force=initial_3d_residual best_force=best_residual relative_improvement=improvement reason=something(fallback_reason,"iteration limit reached")
    elseif !accepted_3d
        @warn "3-D force balance did not improve on the 1-D baseline; restoring 1-D HSE" iterations=iterations initial_force=initial_3d_residual reason=something(fallback_reason,"no improving 3-D iterate")
    end
    atmosphere.pgas=p; atmosphere.rho=rho; atmosphere.ne=ne; atmosphere.z=z
    ForceBalanceDiagnostics(mode isa HE3DMode ? :HE3D : :MHS,iterations,converged,
        pchange,rchange,fres,zchange,0.0,0.0,lorentzmax,initial_3d_residual,
        best_residual,absolute_improvement,improvement,accepted_3d)
end
