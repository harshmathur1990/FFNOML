"""
FFNO populations for a native log-tau atmosphere are predicted on a height mesh.

The FFNO checkpoints were trained on simulation height grids: one set of heights shared by every column, 12-30 km
spacing through the photosphere and chromosphere and bottoms 1-2.5 Mm below the surface. Its vertical layers are
1-D convolutions over the grid index and its horizontal Fourier layers mix columns at the same index, so the
95-point log-tau column (spacing 30 km to several Mm, a different height at every point of every column) is far
outside that: H n2 is ~100x off in the chromosphere with the simulation's own T, v, ne and rho.

`MeshPopulationModel` maps each column from its log-tau points (HSE heights `z`) onto a fixed mesh, calls the
wrapped model there and interpolates the populations back:
  * inside the column: log T, log ne, log rho log-linear in z, velocities linear;
  * above its top: the top values;
  * below its bottom (log tau500 ~ 1.4, 0.1-0.3 Mm deep): hydrostatic equilibrium with d ln T/d ln P = `nabla`
    (the near-adiabatic convection zone), rho and ne from the HSEMHS EOS table, velocities held;
  * back: populations log-linear in z.
The VJP is the transpose of these maps (finite differences for the three bottom values of the deep extension).
"""
struct PopulationMeshConfig{T<:AbstractFloat}
    bottom_m::T
    top_m::T
    spacing_m::T
    uniform_top_m::T
    stretch::T
    max_spacing_m::T
    nabla::T
    function PopulationMeshConfig{T}(bottom_m,top_m,spacing_m,uniform_top_m,stretch,max_spacing_m,nabla) where T
        bottom_m<uniform_top_m<=top_m || throw(ArgumentError("population mesh needs bottom_m < uniform_top_m <= top_m"))
        0<spacing_m<=max_spacing_m || throw(ArgumentError("population mesh needs 0 < spacing_m <= max_spacing_m"))
        stretch>=1 || throw(ArgumentError("population mesh stretch must be >= 1"))
        0<nabla<1 || throw(ArgumentError("population mesh nabla must lie in (0, 1)"))
        new{T}(bottom_m,top_m,spacing_m,uniform_top_m,stretch,max_spacing_m,nabla)
    end
end
"""Defaults inside the range of the training grids (bottoms -1 to -2.5 Mm, spacing 12-30 km stretching to
100-160 km in the corona, tops up to 14.4 Mm)."""
PopulationMeshConfig(;bottom_m=-1.0e6,top_m=1.4e7,spacing_m=2.0e4,uniform_top_m=4.5e6,stretch=1.01,
    max_spacing_m=1.0e5,nabla=0.13)=PopulationMeshConfig{Float64}(Float64.((bottom_m,top_m,spacing_m,uniform_top_m,
    stretch,max_spacing_m,nabla))...)

"""Mesh heights [m], top first: `spacing_m` up to `uniform_top_m`, then growing by `stretch` per point up to
`max_spacing_m`."""
function population_mesh(c::PopulationMeshConfig{T}) where T
    z=T[c.bottom_m]; dz=c.spacing_m
    while z[end]<c.top_m
        z[end]>=c.uniform_top_m && (dz=min(dz*c.stretch,c.max_spacing_m))
        c.top_m-z[end]<1.5dz ? push!(z,c.top_m) : push!(z,z[end]+dz)
    end
    reverse!(z)
end

mutable struct MeshPopulationModel{M<:AbstractPopulationModel,E,T<:AbstractFloat} <: AbstractPopulationModel
    inner::M
    mesh::Vector{T}
    eos::E
    gravity::T
    nabla::T
    last_features::Any        # mesh features and populations of the last prediction (reused by the VJP)
    last_populations::Any
end
function MeshPopulationModel(inner::AbstractPopulationModel,config::PopulationMeshConfig,eos,gravity)
    gravity>0 || throw(ArgumentError("population mesh gravity must be positive"))
    MeshPopulationModel(inner,population_mesh(config),eos,Float64(gravity),config.nabla,nothing,nothing)
end

_mesh_active(atmosphere)=vertical_coordinate(atmosphere.grid)===:log10tau500

const _LN10=log(10.0)

"""ln P with ln rho(ln T, ln P) = `lnrho` (Newton on the HSEMHS table)."""
function _mesh_lnP(eos,lnT,lnrho)
    lnP=lnrho+log(1.380649e-23*exp(lnT)/eos.mass_per_H)
    for _ in 1:50
        r,d,_=HSEMHS.lnrho_dY(eos,lnT,lnP)
        isfinite(r)&&isfinite(d)&&d>0 || error("population mesh: EOS table has no state at T=$(exp(lnT)) K, rho=$(exp(lnrho))")
        step=(lnrho-r)/d; lnP+=step
        abs(step)<1e-12 && return lnP
    end
    lnP
end

"""Deep extension below a column bottom (z_b, T_b, rho_b): hydrostatic, ln T = ln T_b + nabla (ln P - ln P_b).
Writes (ln T, log10 ne, log10 rho) at the heights `zs` (decreasing, all below z_b) into `out` (3, length(zs))."""
function _mesh_deep!(out,eos,g,nabla,zb,lnTb,lnrhob,zs)
    lnPb=_mesh_lnP(eos,lnTb,lnrhob); lnP=lnPb; z=zb
    rate(lp)=(r=HSEMHS.state(eos,lnTb+nabla*(lp-lnPb),lp)[1]; isfinite(r) || error(
        "population mesh: deep extension left the EOS table"); exp(r)*g/exp(lp))   # d ln P / d depth
    for (m,target) in enumerate(zs)
        while z>target
            h=min(2.0e4,z-target)
            k1=rate(lnP); k2=rate(lnP+h/2*k1); k3=rate(lnP+h/2*k2); k4=rate(lnP+h*k3)
            lnP+=h/6*(k1+2k2+2k3+k4); z-=h
        end
        lnT=lnTb+nabla*(lnP-lnPb); r,n,_,_=HSEMHS.state(eos,lnT,lnP)
        out[1,m]=lnT; out[2,m]=n/_LN10; out[3,m]=r/_LN10
    end
    out
end

"""Interpolation variables of one log-tau column: (ln T, vx, vy, vz, log10 ne, log10 rho) by point."""
function _mesh_column_variables(atmosphere,x,y)
    nz=size(atmosphere.temperature,1); a=Matrix{Float64}(undef,6,nz)
    for k in 1:nz
        a[1,k]=log(atmosphere.temperature[k,x,y]); a[2,k]=atmosphere.vx[k,x,y]; a[3,k]=atmosphere.vy[k,x,y]
        a[4,k]=atmosphere.vz[k,x,y]; a[5,k]=log10(atmosphere.ne[k,x,y]); a[6,k]=log10(atmosphere.rho[k,x,y])
    end
    a
end

"""Mesh point m of a column: (:top,), (:inside,k,w) between points k and k+1 at weight w, or (:deep,)."""
function _mesh_locate(zc,zeta,k)
    zeta>=zc[1] && return :top,1,0.0
    zeta<zc[end] && return :deep,length(zc),0.0
    while zc[k+1]>zeta; k+=1; end
    :inside,k,(zc[k]-zeta)/(zc[k]-zc[k+1])
end

"""Mesh features `(6, nmesh)` (T, vx, vy, vz, log10 ne, log10 rho) of column (x, y)."""
function _mesh_column!(f,model,atmosphere,x,y)
    a=_mesh_column_variables(atmosphere,x,y); zc=@view atmosphere.z[:,x,y]; mesh=model.mesh
    k=1; deep=Int[]
    for m in eachindex(mesh)
        kind,k,w=_mesh_locate(zc,mesh[m],k)
        kind===:deep && (push!(deep,m); continue)
        for c in 1:6; f[c,m]=kind===:top ? a[c,1] : (1-w)*a[c,k]+w*a[c,k+1]; end
        f[1,m]=exp(f[1,m])
    end
    if !isempty(deep)
        nz=length(zc); d=_mesh_deep!(Matrix{Float64}(undef,3,length(deep)),model.eos,model.gravity,model.nabla,
            zc[nz],a[1,nz],a[6,nz]*_LN10,mesh[deep])
        for (i,m) in enumerate(deep)
            f[1,m]=exp(d[1,i]); f[2,m]=a[2,nz]; f[3,m]=a[3,nz]; f[4,m]=a[4,nz]; f[5,m]=d[2,i]; f[6,m]=d[3,i]
        end
    end
    f
end

function _mesh_features(model,atmosphere)
    nz,nx,ny=size(atmosphere.temperature); f=Array{Float64}(undef,6,length(model.mesh),nx,ny)
    Threads.@threads for x in 1:nx
        for y in 1:ny; _mesh_column!(@view(f[:,:,x,y]),model,atmosphere,x,y); end
    end
    f
end

function _mesh_atmosphere(model,atmosphere,f)
    T=eltype(atmosphere.temperature); nm=length(model.mesh); nx,ny=size(atmosphere.temperature)[2:3]
    grid=Grid3D(VerticalAxis(:z_m,T.(model.mesh)),atmosphere.grid.x,atmosphere.grid.y)
    c(i)=T.(f[i,:,:,:]); z=repeat(reshape(T.(model.mesh),nm,1,1),1,nx,ny)
    Atmosphere3D(grid,c(1),c(2),c(3),c(4),zeros(T,nm,nx,ny);rho=T.(exp10.(f[6,:,:,:])),ne=T.(exp10.(f[5,:,:,:])),z=z)
end

"""Mesh index pair and weight for a height: mesh[j] >= zeta >= mesh[j+1] (clamped to the mesh ends)."""
function _mesh_bracket(mesh,zeta)
    zeta>=mesh[1] && return 1,1,0.0
    zeta<=mesh[end] && return length(mesh),length(mesh),0.0
    j=searchsortedlast(mesh,zeta;rev=true)
    j,j+1,(mesh[j]-zeta)/(mesh[j]-mesh[j+1])
end

function predict_populations!(out::AbstractArray{T,4},model::MeshPopulationModel,atmosphere::Atmosphere3D,
        cache=nothing) where T
    _mesh_active(atmosphere) || return predict_populations!(out,model.inner,atmosphere,cache)
    size(out)[1:3]==size(atmosphere.temperature) || throw(DimensionMismatch("population shape differs from atmosphere"))
    atmosphere.z===nothing && throw(ArgumentError("population mesh needs the equilibrium heights z"))
    f=_mesh_features(model,atmosphere); mesh_atmosphere=_mesh_atmosphere(model,atmosphere,f)
    nz,nx,ny=size(atmosphere.temperature); levels=size(out,4)
    pm=zeros(T,length(model.mesh),nx,ny,levels)
    predict_populations!(pm,model.inner,mesh_atmosphere)
    model.last_features=f; model.last_populations=pm
    Threads.@threads for x in 1:nx
        for y in 1:ny, k in 1:nz
            j0,j1,u=_mesh_bracket(model.mesh,atmosphere.z[k,x,y])
            for l in 1:levels
                out[k,x,y,l]=exp((1-u)*log(pm[j0,x,y,l])+u*log(pm[j1,x,y,l]))
            end
        end
    end
    out
end

function population_vjp!(feature_bar::AbstractArray{T,4},z_bar::AbstractArray{T,3},model::MeshPopulationModel,
        atmosphere::Atmosphere3D,population_bar::AbstractArray{T,4}) where T
    _mesh_active(atmosphere) || return population_vjp!(feature_bar,z_bar,model.inner,atmosphere,population_bar)
    nz,nx,ny=size(atmosphere.temperature); levels=size(population_bar,4); mesh=model.mesh; nm=length(mesh)
    f=_mesh_features(model,atmosphere); mesh_atmosphere=_mesh_atmosphere(model,atmosphere,f)
    pm=if model.last_features!==nothing && size(model.last_features)==size(f) && model.last_features==f &&
            size(model.last_populations,4)==levels
        model.last_populations
    else
        p=zeros(T,nm,nx,ny,levels); predict_populations!(p,model.inner,mesh_atmosphere); p
    end
    fill!(feature_bar,zero(T)); fill!(z_bar,zero(T))
    # back-map: ln p(z_k) = (1-u) ln pm[j0] + u ln pm[j1]
    pm_bar=zeros(T,nm,nx,ny,levels)
    Threads.@threads for x in 1:nx
        for y in 1:ny, k in 1:nz
            zk=atmosphere.z[k,x,y]; j0,j1,u=_mesh_bracket(mesh,zk)
            for l in 1:levels
                p=exp((1-u)*log(pm[j0,x,y,l])+u*log(pm[j1,x,y,l])); g=population_bar[k,x,y,l]*p
                pm_bar[j0,x,y,l]+=g*(1-u)/pm[j0,x,y,l]; pm_bar[j1,x,y,l]+=g*u/pm[j1,x,y,l]
                j0!=j1 && (z_bar[k,x,y]-=g*(log(pm[j1,x,y,l])-log(pm[j0,x,y,l]))/(mesh[j0]-mesh[j1]))
            end
        end
    end
    fm_bar=zeros(T,6,nm,nx,ny); zm_bar=zeros(T,nm,nx,ny)
    population_vjp!(fm_bar,zm_bar,model.inner,mesh_atmosphere,pm_bar)        # mesh heights are fixed: zm_bar unused
    Threads.@threads for x in 1:nx
        for y in 1:ny
            _mesh_column_vjp!(@view(feature_bar[:,:,x,y]),@view(z_bar[:,x,y]),model,atmosphere,x,y,
                @view(f[:,:,x,y]),@view(fm_bar[:,:,x,y]))
        end
    end
    feature_bar,z_bar
end

"""Transpose of `_mesh_column!` for column (x, y): mesh-feature cotangents `fb` (6, nmesh) to the column's
feature cotangents (6, nz) and height cotangents."""
function _mesh_column_vjp!(feature_bar,z_bar,model,atmosphere,x,y,f,fb)
    a=_mesh_column_variables(atmosphere,x,y); zc=@view atmosphere.z[:,x,y]; mesh=model.mesh; nz=length(zc)
    a_bar=zeros(6,nz); k=1; deep=Int[]
    for m in eachindex(mesh)
        kind,k,w=_mesh_locate(zc,mesh[m],k)
        kind===:deep && (push!(deep,m); continue)
        G=ntuple(c->fb[c,m]*(c==1 ? f[1,m] : 1.0),6)      # cotangents of the interpolated variables
        if kind===:top
            for c in 1:6; a_bar[c,1]+=G[c]; end
        else
            w_bar=0.0
            for c in 1:6
                a_bar[c,k]+=G[c]*(1-w); a_bar[c,k+1]+=G[c]*w; w_bar+=G[c]*(a[c,k+1]-a[c,k])
            end
            d=zc[k]-zc[k+1]; z_bar[k]+=w_bar*(1-w)/d; z_bar[k+1]+=w_bar*w/d
        end
    end
    if !isempty(deep)
        zs=mesh[deep]; zb=zc[nz]; lnT=a[1,nz]; lnrho=a[6,nz]*_LN10
        ext(dz,dT,dr)=_mesh_deep!(Matrix{Float64}(undef,3,length(deep)),model.eos,model.gravity,model.nabla,
            zb+dz,lnT+dT,lnrho+dr,zs)
        project(d)=sum(fb[1,m]*exp(d[1,i])+fb[5,m]*d[2,i]+fb[6,m]*d[3,i] for (i,m) in enumerate(deep))
        hT=1e-6; hz=1.0
        dT=(project(ext(0.0,hT,0.0))-project(ext(0.0,-hT,0.0)))/(2hT)          # per d ln T_b
        dr=(project(ext(0.0,0.0,hT))-project(ext(0.0,0.0,-hT)))/(2hT)          # per d ln rho_b
        dz=(project(ext(hz,0.0,0.0))-project(ext(-hz,0.0,0.0)))/(2hz)
        a_bar[1,nz]+=dT; a_bar[6,nz]+=dr*_LN10; z_bar[nz]+=dz
        for m in deep, c in 2:4; a_bar[c,nz]+=fb[c,m]; end
    end
    for k in 1:nz
        feature_bar[1,k]+=a_bar[1,k]/atmosphere.temperature[k,x,y]
        for c in 2:6; feature_bar[c,k]+=a_bar[c,k]; end
    end
    feature_bar
end

"""Population-mesh settings of a run, or `nothing` when FFNO runs on the native grid (native z, or disabled)."""
function population_mesh_options(config)
    config.population_mesh===nothing && return nothing
    (config=config.population_mesh,eos_table_file=config.equilibrium.eos_table_file,
     gravity=config.equilibrium.gravity_m_s2)
end

"""Wrap `model` for prediction on the population mesh (`options` from `population_mesh_options`; `eos` the loaded
HSEMHS table, shared between species)."""
wrap_population_mesh(model,::Nothing,eos=nothing)=model
wrap_population_mesh(model,options,eos=HSEMHS.load_table(options.eos_table_file))=
    MeshPopulationModel(model,options.config,eos,options.gravity)

_unwrap_population_model(model)=model
_unwrap_population_model(model::MeshPopulationModel)=_unwrap_population_model(model.inner)
