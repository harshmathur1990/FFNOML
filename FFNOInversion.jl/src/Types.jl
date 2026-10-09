const VALID_STOKES = (:I, :Q, :U, :V)

"""Three-dimensional inversion grid. `log_tau500` must be strictly monotonic."""
struct Grid3D{T<:AbstractFloat}
    log_tau500::Vector{T}
    x::Vector{T}
    y::Vector{T}
    function Grid3D(log_tau500::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
        length(log_tau500) >= 2 || throw(ArgumentError("log_tau500 needs at least two points"))
        length(x) >= 1 && length(y) >= 1 || throw(ArgumentError("x and y cannot be empty"))
        all(isfinite, log_tau500) && all(isfinite, x) && all(isfinite, y) || throw(ArgumentError("grid values must be finite"))
        d = diff(log_tau500)
        (all(>(zero(T)), d) || all(<(zero(T)), d)) || throw(ArgumentError("log_tau500 must be strictly monotonic"))
        new{T}(collect(log_tau500), collect(x), collect(y))
    end
end

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

"""Atmosphere arrays use `(nz,nx,ny)`. SI units: K, m/s, Pa, kg/m^3, m^-3, m, tesla."""
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
end

function Atmosphere3D(grid::Grid3D{T}, temperature::A, vx::A, vy::A, vz::A, vturb::A;
                      magnetic_field=nothing, pgas=nothing, rho=nothing, ne=nothing, z=nothing) where {T<:AbstractFloat,A<:AbstractArray{T,3}}
    expected = (length(grid.log_tau500), length(grid.x), length(grid.y))
    for (name, value) in ((:temperature,temperature),(:vx,vx),(:vy,vy),(:vz,vz),(:vturb,vturb))
        size(value) == expected || throw(DimensionMismatch("$name has size $(size(value)); expected $expected"))
        all(isfinite, value) || throw(ArgumentError("$name must be finite"))
    end
    minimum(temperature) > 0 || throw(ArgumentError("temperature must be positive"))
    magnetic_field === nothing || size(magnetic_field.Bx) == expected || throw(DimensionMismatch("magnetic field shape differs from grid"))
    for (name, value) in ((:pgas,pgas),(:rho,rho),(:ne,ne),(:z,z))
        value === nothing && continue
        size(value) == expected || throw(DimensionMismatch("$name shape differs from grid"))
        all(isfinite, value) || throw(ArgumentError("$name must be finite"))
    end
    Atmosphere3D{T,A,typeof(magnetic_field)}(grid, temperature, vx, vy, vz, vturb, magnetic_field, pgas, rho, ne, z)
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
