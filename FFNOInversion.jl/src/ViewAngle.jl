"""
Viewing angle (docs/viewing-angle/README.md): the line-of-sight velocity, the slanted path length and the periodic shear
between Cartesian columns and the rays of an inclined view.

Every function here is the identity, or reads the same array as before, for a vertical view (μ = 1), so μ = 1 runs are
unchanged bit for bit.
"""

const VIEW_Z_REF_M = 0.0              # ray origin: x = ξ + t_x (z − z_ref); the run's height origin (HSEMHS gauge)
const VIEW_DOPPLER_SIGN = "positive v_los toward the observer: line centre at lambda0*(1 - v_los/c), blue shift"

"""Line-of-sight velocity n·v at point (k, x, y); exactly `vz` for a vertical view."""
@inline function los_velocity(atmosphere, k, x, y)
    v = atmosphere.grid.view
    is_vertical(v) && return atmosphere.vz[k, x, y]
    n = view_direction(v)
    oftype(atmosphere.vz[k, x, y], n[1] * atmosphere.vx[k, x, y] + n[2] * atmosphere.vy[k, x, y] + n[3] * atmosphere.vz[k, x, y])
end

"""n·v on the whole grid (a new array; `vz` itself is returned for a vertical view, not a copy)."""
function los_velocity_field(atmosphere)
    v = atmosphere.grid.view
    is_vertical(v) && return atmosphere.vz
    n = view_direction(v)
    eltype(atmosphere.vz).(n[1] .* atmosphere.vx .+ n[2] .* atmosphere.vy .+ n[3] .* atmosphere.vz)
end

"""Adjoint of `los_velocity`: adds `value` = ∂/∂v_los at (k, x, y) to the velocity cotangents of `bar`."""
@inline function add_los_cotangent!(bar, atmosphere, k, x, y, value)
    v = atmosphere.grid.view
    if is_vertical(v)
        bar.vz[k, x, y] += value
    else
        n = view_direction(v)
        bar.vx[k, x, y] += n[1] * value; bar.vy[k, x, y] += n[2] * value; bar.vz[k, x, y] += n[3] * value
    end
    bar
end

"""Path-length factor 1/μ of the geometric (native-z) formal solution; 1 for a vertical view."""
path_factor(v::ViewGeometry) = is_vertical(v) ? 1.0 : 1 / v.mu

"""File attributes recording the view (design §8)."""
function view_attributes(v::ViewGeometry, columns::AbstractString)
    t = slant(v)
    Dict{String,Any}("view_mu" => v.mu, "view_azimuth_deg" => v.azimuth_deg, "view_geometry" => String(v.geometry),
        "view_tx" => t[1], "view_ty" => t[2], "view_z_ref_m" => VIEW_Z_REF_M, "view_doppler_sign" => VIEW_DOPPLER_SIGN,
        "view_columns" => columns)
end

"""Columns of an atmosphere: rays for native log τ, native log m and footprint atmospheres, Cartesian for native z."""
view_columns(grid) = vertical_coordinate(grid) !== :z_m || grid.view.footprint ? "rays" : "cartesian"

"""HSEMHS `ViewOptions` of a view."""
hsemhs_view(v::ViewGeometry) = HSEMHS.ViewOptions(mu = v.mu, azimuth_deg = v.azimuth_deg, geometry = v.geometry)

"""
The view recorded in a file's attributes, or `nothing` for a file without them (legacy: a disk-centre file).
"""
function read_view_attributes(attrs)
    haskey(attrs, "view_mu") || return nothing
    value(x) = x isa HDF5.Attribute ? read(x) : x          # HDF5 attribute dicts read on indexing; plain dicts too
    ViewGeometry(Float64(value(attrs["view_mu"])), Float64(value(attrs["view_azimuth_deg"])),
        Symbol(value(attrs["view_geometry"])))
end

"""
Check that a native log-τ or log-m input file was built for the configured view (design §8): at μ < 1 the file must carry the
same μ, azimuth and geometry; a file without view attributes is a disk-centre file and is accepted only at μ = 1.
Native-z inputs are Cartesian and need no tag.
"""
function check_input_view(file_view, view::ViewGeometry, kind::Symbol, path)
    kind === :z_m && return nothing
    if file_view === nothing
        is_vertical(view) || throw(ArgumentError(
            "$path has no view attributes (a disk-centre, vertical file) but [view] mu = $(view.mu): a native $kind " *
            "input at μ < 1 must hold its axis along the line of sight; build it for this view and tag it with view_mu, " *
            "view_azimuth_deg and view_geometry"))
        return nothing
    end
    same = file_view.mu == view.mu && (is_vertical(view) ||
        (file_view.azimuth_deg == view.azimuth_deg && file_view.geometry === view.geometry))
    same || throw(ArgumentError("$path was built for view mu=$(file_view.mu), azimuth=$(file_view.azimuth_deg), " *
        "geometry=$(file_view.geometry); the configuration asks for mu=$(view.mu), azimuth=$(view.azimuth_deg), " *
        "geometry=$(view.geometry)"))
    nothing
end

# ---------------------------------------------------------------------------------------------------------------------
# Periodic shear between Cartesian columns and footprint rays
# ---------------------------------------------------------------------------------------------------------------------

"""
Per-level shift of a periodic bilinear shear: level k is read at the fractional cell offset
s·(t_x (z_k − z_ref)/dx, t_y (z_k − z_ref)/dy), HSEMHS `ViewTau.shear_levels` convention:
    out[k, i, j] = A(k, i − 1 + f_x, j − 1 + f_y)   (0-based fractional indices, periodic).
s = +1 maps Cartesian values to the footprint rays (ray (ξ, η) at height z sits at ξ + t z); s = −1 maps back.
The four weights of a level are the same for every cell, so they are computed once.
"""
struct ShearPlan
    i0::Vector{Int}
    j0::Vector{Int}
    wx::Vector{Float64}
    wy::Vector{Float64}
end

function ShearPlan(z::AbstractVector, dx::Real, dy::Real, t::NTuple{2,Float64}, s::Real)
    nz = length(z); i0 = zeros(Int, nz); j0 = zeros(Int, nz); wx = zeros(nz); wy = zeros(nz)
    for k in 1:nz
        fx = s * t[1] * (z[k] - VIEW_Z_REF_M) / dx; fy = s * t[2] * (z[k] - VIEW_Z_REF_M) / dy
        i0[k] = floor(Int, fx); j0[k] = floor(Int, fy); wx[k] = fx - i0[k]; wy[k] = fy - j0[k]
    end
    ShearPlan(i0, j0, wx, wy)
end

"""Uniform horizontal spacing of a grid axis (the shear needs a regular periodic grid)."""
function _shear_spacing(values, name)
    length(values) >= 2 || throw(ArgumentError("an inclined view needs at least two points along $name"))
    d = values[2] - values[1]
    all(abs(diff(values)[i] - d) <= 1e-6 * abs(d) for i in 1:length(values)-1) || throw(ArgumentError(
        "an inclined view needs a uniform $name spacing"))
    Float64(d)
end

"""Shear plan of a grid's levels for its view (`s` = +1 Cartesian → rays, −1 rays → Cartesian) at heights `z`."""
function ShearPlan(grid::Grid3D, z::AbstractVector, s::Real)
    t = slant(grid.view)
    ShearPlan(z, t[1] == 0 ? 1.0 : _shear_spacing(grid.x, "x"), t[2] == 0 ? 1.0 : _shear_spacing(grid.y, "y"), t, s)
end

"""`out = shear(A)` level by level (arrays (nz, nx, ny); `out` must not alias `A`)."""
function shear_levels!(out::AbstractArray{<:Real,3}, A::AbstractArray{<:Real,3}, plan::ShearPlan)
    nz, nx, ny = size(A); size(out) == size(A) || throw(DimensionMismatch("shear output shape differs"))
    length(plan.i0) == nz || throw(DimensionMismatch("shear plan has $(length(plan.i0)) levels; the array has $nz"))
    Threads.@threads for j in 1:ny
        @inbounds for i in 1:nx, k in 1:nz
            ia = mod(i - 1 + plan.i0[k], nx) + 1; ib = mod(ia, nx) + 1
            ja = mod(j - 1 + plan.j0[k], ny) + 1; jb = mod(ja, ny) + 1
            wx = plan.wx[k]; wy = plan.wy[k]
            out[k, i, j] = (1 - wy) * ((1 - wx) * A[k, ia, ja] + wx * A[k, ib, ja]) +
                           wy * ((1 - wx) * A[k, ia, jb] + wx * A[k, ib, jb])
        end
    end
    out
end

"""Transpose of `shear_levels!`: adds the scatter of `out_bar` with the same weights to `A_bar` (not the inverse shear)."""
function shear_levels_transpose!(A_bar::AbstractArray{<:Real,3}, out_bar::AbstractArray{<:Real,3}, plan::ShearPlan)
    nz, nx, ny = size(out_bar); size(A_bar) == size(out_bar) || throw(DimensionMismatch("shear cotangent shape differs"))
    # scatter by level: each level's plane is independent, so threads over levels never collide
    Threads.@threads for k in 1:nz
        wx = plan.wx[k]; wy = plan.wy[k]
        @inbounds for j in 1:ny, i in 1:nx
            ia = mod(i - 1 + plan.i0[k], nx) + 1; ib = mod(ia, nx) + 1
            ja = mod(j - 1 + plan.j0[k], ny) + 1; jb = mod(ja, ny) + 1
            g = out_bar[k, i, j]
            A_bar[k, ia, ja] += (1 - wy) * (1 - wx) * g; A_bar[k, ib, ja] += (1 - wy) * wx * g
            A_bar[k, ia, jb] += wy * (1 - wx) * g; A_bar[k, ib, jb] += wy * wx * g
        end
    end
    A_bar
end

"""Horizontal travel of the rays over the heights `z` [m] and whether it exceeds the box (repeated periodic wraps)."""
function view_travel(grid::Grid3D, z::AbstractVector)
    t = slant(grid.view); span = maximum(z) - minimum(z)
    travel = span * hypot(t...)
    lx = length(grid.x) * (length(grid.x) > 1 ? grid.x[2] - grid.x[1] : 0.0)
    ly = length(grid.y) * (length(grid.y) > 1 ? grid.y[2] - grid.y[1] : 0.0)
    wraps = abs(t[1]) * span > lx || abs(t[2]) * span > ly
    (travel_m = travel, wraps = wraps)
end

"""Warn (once per call) when the rays travel more than a box length over `z`, whatever μ (design §7)."""
function warn_view_travel(grid::Grid3D, z::AbstractVector, label)
    is_slanted(grid.view) || return nothing
    tr = view_travel(grid, z)
    tr.wraps && @warn "inclined view: $label rays travel $(round(tr.travel_m / 1e3; digits=1)) km horizontally, more " *
        "than the box: the periodic shear wraps them around more than once" maxlog=1
    grid.view.mu < 0.2 && @warn "inclined view at mu = $(grid.view.mu) < 0.2: rays are nearly horizontal" maxlog=1
    nothing
end
