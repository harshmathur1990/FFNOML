"""Control values on vertical nodes of one native coordinate kind.

`nodes` may be given in either monotonic order and are in the unit of `kind`
(`:z_m` metres or `:log10tau500` dex). Expanding onto a grid of another kind
throws: node positions are never reinterpreted in another coordinate.
"""
struct NodeField{T<:AbstractFloat,A<:AbstractArray{T,3}}
    values::A                 # (nnode,ncx,ncy)
    nodes::Vector{T}
    kind::Symbol
    function NodeField(values::A, nodes::AbstractVector{T}, kind::Symbol=:log10tau500) where {T<:AbstractFloat,A<:AbstractArray{T,3}}
        unit = vertical_unit(kind)
        size(values,1) == length(nodes) || throw(DimensionMismatch("node depth axis mismatch"))
        all(isfinite, values) || throw(ArgumentError("node values must be finite"))
        all(isfinite, nodes) || throw(ArgumentError("node positions [$unit] must be finite"))
        d = diff(nodes)
        (all(>(zero(T)), d) || all(<(zero(T)), d)) || throw(ArgumentError("node positions [$unit] must be strictly monotonic"))
        new{T,A}(values, collect(nodes), kind)
    end
end

function _check_node_kind(field::NodeField, grid::Grid3D)
    kind = vertical_coordinate(grid)
    field.kind === kind || throw(ArgumentError(
        "nodes are on :$(field.kind) [$(vertical_unit(field.kind))] but the grid's native coordinate is :$kind [$(vertical_unit(kind))]"))
    nothing
end

function _bracket(grid, x)
    asc = grid[end] > grid[1]
    g = asc ? grid : reverse(grid)
    xx = clamp(x, g[1], g[end])
    hi = clamp(searchsortedfirst(g, xx), 2, length(g))
    lo = hi - 1
    if asc
        return lo, hi, (xx-g[lo])/(g[hi]-g[lo])
    end
    ilo, ihi = length(grid)-hi+1, length(grid)-lo+1
    return ilo, ihi, (x-grid[ilo])/(grid[ihi]-grid[ilo])
end

"""Trilinearly expand `(node,control-x,control-y)` values to `(nz,nx,ny)`."""
function expand_nodes(field::NodeField{T}, grid::Grid3D{T}) where T
    _check_node_kind(field, grid)
    nnode, ncx, ncy = size(field.values)
    depth = native_depth(grid)
    out = Array{T}(undef, length(depth), length(grid.x), length(grid.y))
    cx = ncx == 1 ? T[zero(T)] : collect(range(zero(T), one(T), length=ncx))
    cy = ncy == 1 ? T[zero(T)] : collect(range(zero(T), one(T), length=ncy))
    xn = length(grid.x) == 1 ? T[zero(T)] : collect(range(zero(T), one(T), length=length(grid.x)))
    yn = length(grid.y) == 1 ? T[zero(T)] : collect(range(zero(T), one(T), length=length(grid.y)))
    for iz in eachindex(depth), ix in eachindex(xn), iy in eachindex(yn)
        z0,z1,wz = _bracket(field.nodes, depth[iz])
        x0,x1,wx = ncx == 1 ? (1,1,zero(T)) : _bracket(cx, xn[ix])
        y0,y1,wy = ncy == 1 ? (1,1,zero(T)) : _bracket(cy, yn[iy])
        v0 = (1-wx)*((1-wy)*field.values[z0,x0,y0] + wy*field.values[z0,x0,y1]) + wx*((1-wy)*field.values[z0,x1,y0] + wy*field.values[z0,x1,y1])
        v1 = (1-wx)*((1-wy)*field.values[z1,x0,y0] + wy*field.values[z1,x0,y1]) + wx*((1-wy)*field.values[z1,x1,y0] + wy*field.values[z1,x1,y1])
        out[iz,ix,iy] = (1-wz)*v0 + wz*v1
    end
    out
end

"""Expand node maps directly onto one MPI-owned tile using global normalized coordinates."""
function expand_nodes(field::NodeField{T},grid::Grid3D{T},tile::Tile2D) where T
    _check_node_kind(field,grid)
    nnode,ncx,ncy=size(field.values); depth=native_depth(grid); nz=length(depth)
    out=Array{T}(undef,nz,length(tile.xrange),length(tile.yrange))
    cx=ncx==1 ? T[zero(T)] : collect(range(zero(T),one(T),length=ncx))
    cy=ncy==1 ? T[zero(T)] : collect(range(zero(T),one(T),length=ncy))
    xn=length(grid.x)==1 ? T[zero(T)] : collect(range(zero(T),one(T),length=length(grid.x)))
    yn=length(grid.y)==1 ? T[zero(T)] : collect(range(zero(T),one(T),length=length(grid.y)))
    for iz in 1:nz,(ii,ix) in enumerate(tile.xrange),(jj,iy) in enumerate(tile.yrange)
        z0,z1,wz=_bracket(field.nodes,depth[iz])
        x0,x1,wx=ncx==1 ? (1,1,zero(T)) : _bracket(cx,xn[ix])
        y0,y1,wy=ncy==1 ? (1,1,zero(T)) : _bracket(cy,yn[iy])
        v0=(1-wx)*((1-wy)*field.values[z0,x0,y0]+wy*field.values[z0,x0,y1])+wx*((1-wy)*field.values[z0,x1,y0]+wy*field.values[z0,x1,y1])
        v1=(1-wx)*((1-wy)*field.values[z1,x0,y0]+wy*field.values[z1,x0,y1])+wx*((1-wy)*field.values[z1,x1,y0]+wy*field.values[z1,x1,y1])
        out[iz,ii,jj]=(1-wz)*v0+wz*v1
    end
    out
end
