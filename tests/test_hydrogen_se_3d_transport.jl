module HydrogenSE3DTransportTests

using Test

# HydrogenSE.jl only needs the concrete atmosphere type when the top-level SE
# update is called.  A minimal definition keeps these transport-unit tests
# independent of the external Muspel installation.
struct Atmosphere3D
    nx::Int
    ny::Int
    nz::Int
    x
    y
    z
    temperature
    velocity_x
    velocity_y
    velocity_z
    electron_density
    hydrogen1_density
    proton_density
end

include(joinpath(@__DIR__, "..", "HydrogenSE.jl"))

@testset "3D periodic interpolation" begin
    axis = [0.0, 1.0, 2.0]
    plane = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    @test hse_bilinear_periodic(plane, axis, axis, 0.5, 0.5) ≈ 3.0
    @test hse_bilinear_periodic(plane, axis, axis, 3.0, 0.0) ≈ plane[1, 1]
    @test hse_bilinear_periodic(plane, axis, axis, -1.0, 0.0) ≈ plane[1, 3]
end

@testset "homogeneous atmosphere preserves horizontal symmetry" begin
    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0, 2.0]
    extinction = fill(0.7, 3, 2, 3)
    source = fill(4.2, 3, 2, 3)
    intensity = similar(source)
    hse_formal_direction_3d!(
        intensity, x, y, z, extinction, source, 0.4, 0.3, sqrt(0.75),
    )
    for k in axes(intensity, 1)
        @test all(@view(intensity[k, :, :]) .≈ intensity[k, 1, 1])
    end
    @test all(intensity .≈ 4.2)
end

@testset "inclined ray transports between horizontal cells" begin
    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    extinction = fill(1.0, 2, 2, 3)
    source = zeros(2, 2, 3)
    source[1, 1, 1] = 10.0
    intensity = similar(source)
    hse_formal_direction_3d!(
        intensity, x, y, z, extinction, source, inv(sqrt(2)), 0.0,
        inv(sqrt(2)),
    )
    # Between z planes the ray moves by +1 in x, so radiation entering at
    # x=0 appears at x=1 rather than remaining in its original column.
    @test intensity[2, 1, 2] > 0
    @test intensity[2, 1, 2] > intensity[2, 1, 1]
end

@testset "3D angular quadrature and zero-scattering limit" begin
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0, 2.0]
    extinction = fill(0.5, 3, 2, 2)
    true_emissivity = fill(1.5, 3, 2, 2)
    scattering = zeros(3, 2, 2)
    arrays = [zeros(3, 2, 2) for _ in 1:4]
    mean_intensity, source, trial, intensity = arrays
    iterations, residual = hse_formal_mean_intensity_scattering_3d!(
        mean_intensity,
        x,
        y,
        z,
        extinction,
        true_emissivity,
        scattering,
        [0.5],
        [1.0],
        [π / 4, 3π / 4, 5π / 4, 7π / 4],
        fill(0.25, 4),
        source,
        trial,
        intensity,
    )
    @test iterations == 1
    @test residual == 0.0
    @test all(isfinite, mean_intensity)
    @test all(mean_intensity .>= 0)
    for k in axes(mean_intensity, 1)
        @test all(@view(mean_intensity[k, :, :]) .≈ mean_intensity[k, 1, 1])
    end
end

end
