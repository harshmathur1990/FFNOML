const VALID_STOKES = (:I, :Q, :U, :V)

const VERTICAL_KINDS = (:z_m, :log10tau500)

"""Unit of a vertical coordinate kind, for messages and file attributes."""
vertical_unit(kind::Symbol) = kind === :z_m ? "m" : kind === :log10tau500 ? "dex" :
    throw(ArgumentError("unknown vertical coordinate kind :$kind; expected one of $(VERTICAL_KINDS)"))

"""Native vertical coordinate of a grid, stored top first (index 1 is the top).

`:z_m` is geometrical height in metres, positive upward, so its values strictly
decrease; `:log10tau500` is log10 of the 500 nm optical depth, so its values
strictly increase. Inputs in the other order are reversed, together with every
field, by the readers that hold those fields; this type never reorders.
"""
struct VerticalAxis{T<:AbstractFloat}
    kind::Symbol
    values::Vector{T}
    function VerticalAxis(kind::Symbol, values::AbstractVector{T}) where {T<:AbstractFloat}
        unit = vertical_unit(kind)
        length(values) >= 2 || throw(ArgumentError("vertical axis :$kind needs at least two levels"))
        all(isfinite, values) || throw(ArgumentError("vertical axis :$kind [$unit] must be finite"))
        d = diff(values)
        if kind === :z_m
            all(<(zero(T)), d) || throw(ArgumentError(
                "height axis [m] must strictly decrease (top first); reverse the axis and every field together, or read through read_inversion_inputs, which normalizes"))
        else
            all(>(zero(T)), d) || throw(ArgumentError(
                "log10tau500 axis [dex] must strictly increase (top first); reverse the axis and every field together, or read through read_inversion_inputs, which normalizes"))
        end
        new{T}(kind, collect(values))
    end
end

"""Three-dimensional inversion grid: a native vertical axis plus horizontal pixel
coordinates. x and y must be finite and strictly increasing (one point allowed)."""
struct Grid3D{T<:AbstractFloat}
    vertical::VerticalAxis{T}
    x::Vector{T}
    y::Vector{T}
    function Grid3D(vertical::VerticalAxis{T}, x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
        length(x) >= 1 && length(y) >= 1 || throw(ArgumentError("x and y cannot be empty"))
        all(isfinite, x) && all(isfinite, y) || throw(ArgumentError("grid values must be finite"))
        all(>(zero(T)), diff(x)) || throw(ArgumentError("x must be strictly increasing; decreasing horizontal axes are not normalized"))
        all(>(zero(T)), diff(y)) || throw(ArgumentError("y must be strictly increasing; decreasing horizontal axes are not normalized"))
        new{T}(vertical, collect(x), collect(y))
    end
end

"""Legacy constructor: a log10tau500 axis, which must already be top first (increasing)."""
Grid3D(log_tau500::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat} =
    Grid3D(VerticalAxis(:log10tau500, log_tau500), x, y)

"""Kind of the grid's native vertical coordinate (`:z_m` or `:log10tau500`)."""
vertical_coordinate(grid::Grid3D) = grid.vertical.kind
"""Values of the native vertical coordinate, top first, in its own unit."""
native_depth(grid::Grid3D) = grid.vertical.values
"""Number of vertical levels."""
nlevels(grid::Grid3D) = length(grid.vertical.values)

function _axis_of_kind(grid::Grid3D, kind::Symbol, name)
    actual = vertical_coordinate(grid)
    actual === kind || throw(ArgumentError(
        "$name requested, but the grid's native vertical coordinate is :$actual [$(vertical_unit(actual))]"))
    native_depth(grid)
end
"""The native log10tau500 axis [dex]; throws on a height grid."""
log_tau500_axis(grid::Grid3D) = _axis_of_kind(grid, :log10tau500, "log10tau500 axis [dex]")
"""The native height axis [m]; throws on an optical-depth grid."""
z_axis(grid::Grid3D) = _axis_of_kind(grid, :z_m, "height axis [m]")

struct MagneticField3D{T<:AbstractFloat,A<:AbstractArray{T,3}}
    Bx::A
    By::A
    Bz::A
    function MagneticField3D(Bx::A, By::A, Bz::A) where {T<:AbstractFloat,A<:AbstractArray{T,3}}
        size(Bx) == size(By) == size(Bz) || throw(DimensionMismatch("Bx, By and Bz shapes differ"))
        all(isfinite, Bx) && all(isfinite, By) && all(isfinite, Bz) || throw(ArgumentError("B must be finite"))
        new{T,A}(Bx, By, Bz)
    end
end

"""Atmosphere arrays use `(nz,nx,ny)`, vertical index 1 at the top. SI units: K, m/s, Pa, kg/m^3, m^-3, m, tesla.

`z` (m) and `logtau500` are full 3-D coordinate fields. The one matching the
grid's native axis equals that axis broadcast over columns (see
`check_coordinate_consistency`); the other is derived and may vary by column.
"""
mutable struct Atmosphere3D{T<:AbstractFloat,A<:AbstractArray{T,3},BM}
    grid::Grid3D{T}
    temperature::A
    vx::A
    vy::A
    vz::A
    vturb::A
    magnetic_field::BM
    pgas::Union{Nothing,A}
    rho::Union{Nothing,A}
    ne::Union{Nothing,A}
    z::Union{Nothing,A}
    logtau500::Union{Nothing,A}
end

function Atmosphere3D(grid::Grid3D{T}, temperature::A, vx::A, vy::A, vz::A, vturb::A;
                      magnetic_field=nothing, pgas=nothing, rho=nothing, ne=nothing, z=nothing,
                      logtau500=nothing) where {T<:AbstractFloat,A<:AbstractArray{T,3}}
    expected = (nlevels(grid), length(grid.x), length(grid.y))
    for (name, value) in ((:temperature,temperature),(:vx,vx),(:vy,vy),(:vz,vz),(:vturb,vturb))
        size(value) == expected || throw(DimensionMismatch("$name has size $(size(value)); expected $expected"))
        all(isfinite, value) || throw(ArgumentError("$name must be finite"))
    end
    minimum(temperature) > 0 || throw(ArgumentError("temperature must be positive"))
    magnetic_field === nothing || size(magnetic_field.Bx) == expected || throw(DimensionMismatch("magnetic field shape differs from grid"))
    for (name, value) in ((:pgas,pgas),(:rho,rho),(:ne,ne),(:z,z),(:logtau500,logtau500))
        value === nothing && continue
        size(value) == expected || throw(DimensionMismatch("$name shape differs from grid"))
        all(isfinite, value) || throw(ArgumentError("$name must be finite"))
    end
    Atmosphere3D{T,A,typeof(magnetic_field)}(grid, temperature, vx, vy, vz, vturb, magnetic_field, pgas, rho, ne, z, logtau500)
end

"""Throw unless the 3-D coordinate field of the native kind (`z` for a height
grid, `logtau500` for an optical-depth grid) is absent or equals the native axis
broadcast over every column."""
function check_coordinate_consistency(atmosphere::Atmosphere3D)
    grid = atmosphere.grid; kind = vertical_coordinate(grid); axis = native_depth(grid)
    field = kind === :z_m ? atmosphere.z : atmosphere.logtau500
    field === nothing && return true
    for k in axes(field,1)
        all(==(axis[k]), @view field[k,:,:]) || throw(ArgumentError(
            "native $(kind) field differs from the grid axis at level $k; the native coordinate cannot vary by column"))
    end
    true
end

struct HE3DBoundaryState{T<:AbstractFloat,A}
    rho0::A
    p0::A
    boundary::Symbol
    function HE3DBoundaryState(rho0::A, p0::A, boundary::Symbol=:top) where {T<:AbstractFloat,A<:Union{T,AbstractArray{T}}}
        boundary in (:top, :bottom, :full) || throw(ArgumentError("boundary must be :top, :bottom or :full"))
        all(>(zero(T)), rho0) && all(>(zero(T)), p0) || throw(ArgumentError("rho0 and p0 must be positive"))
        new{T,A}(rho0, p0, boundary)
    end
end

struct StokesSet
    components::Tuple{Vararg{Symbol}}
    function StokesSet(components::Tuple{Vararg{Symbol}})
        isempty(components) && throw(ArgumentError("at least one Stokes component is required"))
        all(c -> c in VALID_STOKES, components) || throw(ArgumentError("valid Stokes components are I,Q,U,V"))
        length(unique(components)) == length(components) || throw(ArgumentError("duplicate Stokes component"))
        order = map(c -> something(findfirst(==(c), VALID_STOKES), 0), components)
        issorted(order) || throw(ArgumentError("Stokes components must follow I,Q,U,V order"))
        new(components)
    end
end
StokesSet(component::Symbol) = StokesSet((component,))

struct SpectralCube{T<:AbstractFloat,A<:AbstractArray{T,4}}
    data::A
    wavelength_m::Vector{T}
    stokes::StokesSet
    function SpectralCube(data::A, wavelength_m::AbstractVector{T}, stokes::StokesSet) where {T<:AbstractFloat,A<:AbstractArray{T,4}}
        size(data,1) == length(wavelength_m) || throw(DimensionMismatch("wavelength axis mismatch"))
        size(data,2) == length(stokes.components) || throw(DimensionMismatch("Stokes axis mismatch"))
        all(isfinite, data) && all(isfinite, wavelength_m) || throw(ArgumentError("spectral cube must be finite"))
        new{T,A}(data, collect(wavelength_m), stokes)
    end
end

struct ObservationCube{T<:AbstractFloat,A<:AbstractArray{T,4}}
    spectrum::SpectralCube{T,A}
    sigma::A
    inversion_weights::A
    function ObservationCube(spectrum::SpectralCube{T,A}, sigma::A, inversion_weights::A) where {T<:AbstractFloat,A<:AbstractArray{T,4}}
        size(sigma) == size(inversion_weights) == size(spectrum.data) || throw(DimensionMismatch("observation arrays differ"))
        all(>(zero(T)), sigma) || throw(ArgumentError("sigma must be positive"))
        all(>=(zero(T)), inversion_weights) || throw(ArgumentError("inversion weights must be non-negative"))
        new{T,A}(spectrum, sigma, inversion_weights)
    end
end
ObservationCube(spectrum::SpectralCube{T,A}, sigma::A) where {T<:AbstractFloat,A<:AbstractArray{T,4}} = ObservationCube(spectrum,sigma,ones(T,size(sigma)))

Base.@kwdef struct CapabilityManifest
    prd::Bool = false
    stokes::Tuple{Vararg{Symbol}} = (:I,)
    schema_version::VersionNumber = v"1.0.0"
end
