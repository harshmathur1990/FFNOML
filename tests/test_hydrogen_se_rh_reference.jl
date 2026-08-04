using Test

# Load definitions without starting the Forward.jl driver.
empty!(ARGS)
push!(ARGS, "--help")
redirect_stdout(devnull) do
    include(joinpath(@__DIR__, "..", "Forward.jl"))
end

@testset "RH background-continuum algebra" begin
    wavelength = 500.0
    temperature = 6500.0
    electron_density = 2.0e16
    neutral_hydrogen = 8.0e19
    proton_hydrogen = 2.0e18

    # RH/background.c keeps absorption and scattering separate, and
    # RH/rhf1d/formal.c adds sca_c * J to the emissivity in the formal solve.
    expected_absorption, expected_scattering = Muspel.α_cont_no_itp(
        wavelength,
        temperature,
        electron_density,
        neutral_hydrogen,
        proton_hydrogen,
    )
    absorption, thermal_emissivity, scattering = hse_background_continuum(
        wavelength,
        temperature,
        electron_density,
        neutral_hydrogen,
        proton_hydrogen,
    )
    planck = Muspel.blackbody_λ(wavelength, temperature)
    @test absorption == expected_absorption
    @test scattering == expected_scattering
    @test thermal_emissivity == absorption * planck

    trial_j = 0.37 * planck
    @test (thermal_emissivity + scattering * trial_j) / (absorption + scattering) ==
          (absorption * planck + scattering * trial_j) / (absorption + scattering)
    @test thermal_emissivity != (absorption + scattering) * planck || iszero(scattering)

    background = hse_background_continuum_data()
    metal_absorption, metal_emissivity, metal_scattering = hse_background_continuum(
        wavelength,
        temperature,
        electron_density,
        neutral_hydrogen,
        proton_hydrogen,
        background,
    )
    @test metal_absorption >= absorption
    @test metal_emissivity == metal_absorption * planck
    @test metal_scattering == scattering
end

@testset "RH scattering fixed point" begin
    nz = 5
    x = [0.0]
    y = [0.0]
    z = collect(range(0.0, 8.0e5; length=nz))
    extinction = fill(2.0e-6, nz, 1, 1)
    scattering = fill(3.0e-7, nz, 1, 1)
    true_emissivity = reshape(
        collect(range(1.0e-3, 1.8e-3; length=nz)), nz, 1, 1,
    ) .* extinction
    mus = [0.211324865405187, 0.788675134594813]
    weights = [0.5, 0.5]
    azimuths = [π / 4, 3π / 4, 5π / 4, 7π / 4]
    azimuth_weights = fill(0.25, 4)
    arrays = [zeros(nz, 1, 1) for _ in 1:4]
    mean_intensity, source, trial, intensity = arrays
    check_j = zeros(nz, 1, 1)

    iterations, residual = hse_formal_mean_intensity_scattering_3d!(
        mean_intensity,
        x,
        y,
        z,
        extinction,
        true_emissivity,
        scattering,
        mus,
        weights,
        azimuths,
        azimuth_weights,
        source,
        trial,
        intensity;
        max_iterations=200,
        tolerance=1e-8,
    )
    @. source = (true_emissivity + scattering * mean_intensity) / extinction
    hse_formal_mean_intensity_3d!(
        check_j,
        x,
        y,
        z,
        extinction,
        source,
        mus,
        weights,
        azimuths,
        azimuth_weights,
        intensity,
    )
    @test iterations > 1
    @test residual <= 1e-8
    @test hse_maximum_relative_radiation_change(check_j, mean_intensity) <= 1e-8

    fill!(scattering, 0.0)
    iterations, residual = hse_formal_mean_intensity_scattering_3d!(
        mean_intensity,
        x,
        y,
        z,
        extinction,
        true_emissivity,
        scattering,
        mus,
        weights,
        azimuths,
        azimuth_weights,
        source,
        trial,
        intensity,
    )
    @test iterations == 1
    @test residual == 0.0
end

@testset "RH statistical-equilibrium matrix orientation" begin
    # Julia rates[i,j] is P(i -> j). hse_solve_rate_matrix transposes this to
    # RH's Gamma[i,j] = P(j -> i) convention before imposing conservation.
    rates = [0.0 2.0 0.5; 1.0 0.0 3.0; 0.25 4.0 0.0]
    total_population = 7.0e19
    solution = hse_solve_rate_matrix(rates, [6.0, 2.0, 1.0], total_population)
    @test all(>=(0), solution)
    @test sum(solution) ≈ total_population rtol=1e-14
    @test hse_rate_residual(rates, solution) <= 1e-12
end
