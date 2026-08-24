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
end
function ForceBalanceOptions(;gravity_m_s2=274.0,max_iterations=100,relative_tolerance=1e-6,
                             force_tolerance=1e-5,height_tolerance_m=1e-3,relaxation=0.7,
                             pressure_sweeps=100)
    values=promote(gravity_m_s2,relative_tolerance,force_tolerance,height_tolerance_m,relaxation)
    ForceBalanceOptions(values[1],max_iterations,values[2],values[3],values[4],values[5],pressure_sweeps)
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

"""Compute current density and Lorentz force using second-order centered interiors and one-sided boundaries."""
function lorentz_force!(fx,fy,fz,B::MagneticField3D,grid::Grid3D,z)
    size(fx)==size(fy)==size(fz)==size(B.Bx) || throw(DimensionMismatch("Lorentz arrays differ"))
    zcoord=vec(dropdims(sum(z,dims=(2,3))./(size(z,2)*size(z,3)),dims=(2,3)))
    for k in axes(fx,1),i in axes(fx,2),j in axes(fx,3)
        jx=(_derivative(B.Bz,k,i,j,grid.y,3)-_derivative(B.By,k,i,j,zcoord,1))/MU0
        jy=(_derivative(B.Bx,k,i,j,zcoord,1)-_derivative(B.Bz,k,i,j,grid.x,2))/MU0
        jz=(_derivative(B.By,k,i,j,grid.x,2)-_derivative(B.Bx,k,i,j,grid.y,3))/MU0
        fx[k,i,j]=jy*B.Bz[k,i,j]-jz*B.By[k,i,j]
        fy[k,i,j]=jz*B.Bx[k,i,j]-jx*B.Bz[k,i,j]
        fz[k,i,j]=jx*B.By[k,i,j]-jy*B.Bx[k,i,j]
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
    for q in 2:length(order)
        k0,k1=order[q-1],order[q]; dtau=abs(tau[k1]-tau[k0])
        for i in axes(z,2),j in axes(z,3)
            extinction=(kappa[k0,i,j]*rho[k0,i,j]+kappa[k1,i,j]*rho[k1,i,j])/2
            z[k1,i,j]=z[k0,i,j]-dtau/extinction
        end
    end
    z
end

function _pressure_from_force!(p,rho,z,fz,pboundary,order,g)
    @views p[order[1],:,:].=pboundary
    for q in 2:length(order)
        k0,k1=order[q-1],order[q]
        for i in axes(p,2),j in axes(p,3)
            ds=abs(z[k1,i,j]-z[k0,i,j])
            p[k1,i,j]=p[k0,i,j]+(g*(rho[k0,i,j]+rho[k1,i,j])/2-(fz[k0,i,j]+fz[k1,i,j])/2)*ds
            p[k1,i,j]>0 || throw(ErrorException("non-positive pressure at depth=$k1 x=$i y=$j"))
        end
    end
end

"""Relax scalar pressure against all three components of the target force.

Each sweep averages pressure estimates propagated from all neighboring cells.
This is the normal-equation iteration for a discrete least-squares gradient
problem and therefore couples every `(x,y)` column.
"""
function _relax_pressure_3d!(p,fx,fy,fz,rho,z,grid,pboundary,order,g,sweeps)
    boundary_k=order[1]; zcoord=vec(dropdims(sum(z,dims=(2,3))./(size(z,2)*size(z,3)),dims=(2,3)))
    targetz=fz.-rho.*g
    for _ in 1:sweeps
        @views p[boundary_k,:,:].=pboundary
        for k in axes(p,1),i in axes(p,2),j in axes(p,3)
            k==boundary_k && continue
            total=0.0; count=0
            if i>1
                total+=p[k,i-1,j]+(fx[k,i-1,j]+fx[k,i,j])*(grid.x[i]-grid.x[i-1])/2; count+=1
            end
            if i<size(p,2)
                total+=p[k,i+1,j]-(fx[k,i+1,j]+fx[k,i,j])*(grid.x[i+1]-grid.x[i])/2; count+=1
            end
            if j>1
                total+=p[k,i,j-1]+(fy[k,i,j-1]+fy[k,i,j])*(grid.y[j]-grid.y[j-1])/2; count+=1
            end
            if j<size(p,3)
                total+=p[k,i,j+1]-(fy[k,i,j+1]+fy[k,i,j])*(grid.y[j+1]-grid.y[j])/2; count+=1
            end
            if k>1
                total+=p[k-1,i,j]+(targetz[k-1,i,j]+targetz[k,i,j])*(zcoord[k]-zcoord[k-1])/2; count+=1
            end
            if k<size(p,1)
                total+=p[k+1,i,j]-(targetz[k+1,i,j]+targetz[k,i,j])*(zcoord[k+1]-zcoord[k])/2; count+=1
            end
            p[k,i,j]=max(total/count,eps(eltype(p)))
        end
    end
    @views p[boundary_k,:,:].=pboundary
    p
end

_relative_change(a,b)=maximum(abs.(a.-b)./max.(abs.(b),eps(eltype(b))))

function _force_residual(p,rho,z,fx,fy,fz,grid,g)
    scale=max(maximum(abs.(rho.*g)),maximum(sqrt.(fx.^2 .+ fy.^2 .+ fz.^2)))+eps(eltype(p)); worst=zero(eltype(p))
    for k in axes(p,1),i in axes(p,2),j in axes(p,3)
        rx=_derivative(p,k,i,j,grid.x,2)-fx[k,i,j]
        ry=_derivative(p,k,i,j,grid.y,3)-fy[k,i,j]
        rz=_derivative(p,k,i,j,vec(@view(z[:,i,j])),1)+rho[k,i,j]*g-fz[k,i,j]
        worst=max(worst,sqrt(rx^2+ry^2+rz^2)/scale)
    end
    worst
end

"""Iteratively reconstruct `Pgas`, `rho`, `ne`, and corrugated `z` on fixed log-tau points."""
function reconstruct_force_balance!(atmosphere::Atmosphere3D{Float64},boundary::HE3DBoundaryState,
                                    eos::AbstractEOS,opacity::AbstractOpacity500;
                                    options::ForceBalanceOptions=ForceBalanceOptions())
    boundary.boundary in (:top,:bottom) || throw(ArgumentError("Phase 1 reconstruction supports top or bottom boundaries"))
    options.max_iterations>0 || throw(ArgumentError("max_iterations must be positive"))
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
        iteration==1 && (@views rhonew[order[1],:,:].=rho_boundary)
        opacity500!(kappa,opacity,atmosphere.temperature,p,rhonew,nenew)
        _height_from_tau!(znew,kappa,rhonew,tau,top_to_bottom)
        if mode isa MHSMode
            lorentz_force!(fx,fy,fz,atmosphere.magnetic_field,atmosphere.grid,znew)
            lorentzmax=maximum(sqrt.(fx.^2 .+ fy.^2 .+ fz.^2))
        else
            fill!(fx,0); fill!(fy,0); fill!(fz,0)
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
    converged || throw(ErrorException("force balance did not converge after $(options.max_iterations) iterations (dP=$pchange, drho=$rchange, force=$fres, dz=$zchange)"))
    atmosphere.pgas=p; atmosphere.rho=rho; atmosphere.ne=ne; atmosphere.z=z
    ForceBalanceDiagnostics(mode isa HE3DMode ? :HE3D : :MHS,iterations,true,pchange,rchange,fres,zchange,0.0,0.0,lorentzmax)
end
