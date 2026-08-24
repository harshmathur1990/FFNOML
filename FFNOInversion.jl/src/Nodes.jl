struct NodeField{T<:AbstractFloat,A<:AbstractArray{T,3}}
    values::A                 # (nnode,ncx,ncy)
    log_tau_nodes::Vector{T}
    function NodeField(values::A, log_tau_nodes::AbstractVector{T}) where {T<:AbstractFloat,A<:AbstractArray{T,3}}
        size(values,1) == length(log_tau_nodes) || throw(DimensionMismatch("node depth axis mismatch"))
        all(isfinite, values) || throw(ArgumentError("node values must be finite"))
        d = diff(log_tau_nodes)
        (all(>(zero(T)), d) || all(<(zero(T)), d)) || throw(ArgumentError("node depths must be strictly monotonic"))
        new{T,A}(values, collect(log_tau_nodes))
    end
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
    nnode, ncx, ncy = size(field.values)
    out = Array{T}(undef, length(grid.log_tau500), length(grid.x), length(grid.y))
    cx = ncx == 1 ? T[zero(T)] : collect(range(zero(T), one(T), length=ncx))
    cy = ncy == 1 ? T[zero(T)] : collect(range(zero(T), one(T), length=ncy))
    xn = length(grid.x) == 1 ? T[zero(T)] : collect(range(zero(T), one(T), length=length(grid.x)))
    yn = length(grid.y) == 1 ? T[zero(T)] : collect(range(zero(T), one(T), length=length(grid.y)))
    for iz in eachindex(grid.log_tau500), ix in eachindex(xn), iy in eachindex(yn)
        z0,z1,wz = _bracket(field.log_tau_nodes, grid.log_tau500[iz])
        x0,x1,wx = ncx == 1 ? (1,1,zero(T)) : _bracket(cx, xn[ix])
        y0,y1,wy = ncy == 1 ? (1,1,zero(T)) : _bracket(cy, yn[iy])
        v0 = (1-wx)*((1-wy)*field.values[z0,x0,y0] + wy*field.values[z0,x0,y1]) + wx*((1-wy)*field.values[z0,x1,y0] + wy*field.values[z0,x1,y1])
        v1 = (1-wx)*((1-wy)*field.values[z1,x0,y0] + wy*field.values[z1,x0,y1]) + wx*((1-wy)*field.values[z1,x1,y0] + wy*field.values[z1,x1,y1])
        out[iz,ix,iy] = (1-wz)*v0 + wz*v1
    end
    out
end
