abstract type AbstractObservationModel end
function apply_observation! end

struct IdentityObservation <: AbstractObservationModel end

"""Separable Gaussian spectral and spatial degradation on the full synthesis grid.

FWHM values are physical: wavelength in m and spatial widths in m. This operator
does not sample or resize. Chi-square participation is controlled only by weights.
"""
struct GaussianPSFObservation{T<:AbstractFloat} <: AbstractObservationModel
    spectral_fwhm_m::T
    spatial_fwhm_x_m::T
    spatial_fwhm_y_m::T
    dx_m::T
    dy_m::T
    function GaussianPSFObservation(spectral_fwhm_m::T, spatial_fwhm_x_m::T, spatial_fwhm_y_m::T,
                                    dx_m::T, dy_m::T) where T<:AbstractFloat
        all(>=(zero(T)),(spectral_fwhm_m,spatial_fwhm_x_m,spatial_fwhm_y_m)) || throw(ArgumentError("PSF FWHM values must be non-negative"))
        dx_m > 0 && dy_m > 0 || throw(ArgumentError("dx_m and dy_m must be positive"))
        new{T}(spectral_fwhm_m,spatial_fwhm_x_m,spatial_fwhm_y_m,dx_m,dy_m)
    end
end

function _kernel(T, fwhm, spacing)
    fwhm == 0 && return T[one(T)]
    sigma = fwhm / (T(2)*sqrt(T(2)*log(T(2))) * spacing)
    radius = max(1,ceil(Int,4*sigma))
    k = T[exp(-T(i*i)/(T(2)*sigma*sigma)) for i in -radius:radius]
    k ./ sum(k)
end

function _uniform_spacing(wavelength)
    length(wavelength) == 1 && return one(eltype(wavelength))
    d = diff(wavelength); reference = sum(d)/length(d)
    maximum(abs.(d .- reference)) <= max(abs(reference)*1e-6,eps(eltype(wavelength))*10) ||
        throw(ArgumentError("Gaussian spectral convolution currently requires a uniform synthesis wavelength grid"))
    abs(reference)
end

function _convolve_axis(input::AbstractArray{T,4}, kernel, axis::Int) where T
    length(kernel) == 1 && return copy(input)
    output = similar(input); radius = (length(kernel)-1) ÷ 2
    for index in CartesianIndices(input)
        coord = Tuple(index); acc = zero(T); norm = zero(T)
        for offset in -radius:radius
            pos = coord[axis] + offset
            1 <= pos <= size(input,axis) || continue
            source = CartesianIndex(ntuple(d -> d == axis ? pos : coord[d],4))
            weight = kernel[offset+radius+1]
            acc += weight*input[source]; norm += weight
        end
        output[index] = acc/norm
    end
    output
end

function apply_observation!(output::SpectralCube{T}, model::GaussianPSFObservation,
                            intrinsic::SpectralCube{T}, cache=nothing) where T
    size(output.data) == size(intrinsic.data) || throw(DimensionMismatch("PSF output must retain the full synthesis-grid shape"))
    output.stokes.components == intrinsic.stokes.components || throw(DimensionMismatch("Stokes sets differ"))
    output.wavelength_m == intrinsic.wavelength_m || throw(DimensionMismatch("wavelength grids differ"))
    degraded = model.spectral_fwhm_m==0 ? copy(intrinsic.data) : _convolve_axis(intrinsic.data,
        _kernel(T,T(model.spectral_fwhm_m),_uniform_spacing(intrinsic.wavelength_m)),1)
    degraded = _convolve_axis(degraded,_kernel(T,T(model.spatial_fwhm_x_m),T(model.dx_m)),3)
    degraded = _convolve_axis(degraded,_kernel(T,T(model.spatial_fwhm_y_m),T(model.dy_m)),4)
    copyto!(output.data,degraded)
    output
end

function apply_observation!(output::SpectralCube, ::IdentityObservation, intrinsic::SpectralCube, cache=nothing)
    output.stokes.components == intrinsic.stokes.components || throw(DimensionMismatch("Stokes sets differ"))
    output.wavelength_m == intrinsic.wavelength_m || throw(DimensionMismatch("wavelength grids differ"))
    size(output.data) == size(intrinsic.data) || throw(DimensionMismatch("spectral cube shapes differ"))
    copyto!(output.data,intrinsic.data)
    output
end

"""Construct `(lambda,Stokes,x,y)` chi-square weights.

`wavelength_stokes_weights` has shape `(n_lambda,n_stokes)`. A zero column
disables that Stokes component. `spatial_weights` has shape `(nx,ny)`.
"""
function build_inversion_weights(wavelength_stokes_weights::AbstractMatrix{T},
                                 spatial_weights::AbstractMatrix{T}) where T<:AbstractFloat
    all(>=(zero(T)),wavelength_stokes_weights) && all(>=(zero(T)),spatial_weights) ||
        throw(ArgumentError("inversion weights must be non-negative"))
    nlambda,nstokes = size(wavelength_stokes_weights)
    nx,ny = size(spatial_weights)
    out = Array{T}(undef,nlambda,nstokes,nx,ny)
    for il in 1:nlambda, is in 1:nstokes, ix in 1:nx, iy in 1:ny
        out[il,is,ix,iy] = wavelength_stokes_weights[il,is]*spatial_weights[ix,iy]
    end
    out
end
