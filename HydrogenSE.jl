"""
Hydrogen statistical-equilibrium correction used by Forward.jl.

This is a 1.5D, complete-redistribution solver: each vertical column is
independent, but all depths in a column are coupled by the formal solution.
All three velocity components enter bound-bound rates through their projection
onto an azimuthal ray quadrature; the rays still sample only the local column.
It computes bound-bound and bound-free radiative rates, CE/CI/Omega electron
collision rates, and solves the stationary hydrogen rate equations subject to
hydrogen-particle conservation. Charge conservation remains in Forward.jl,
where non-hydrogen ions are evaluated in LTE.

This is intentionally a separate option from the inexpensive charge-only
fixed-point iteration. It is not a full 3D MALI solver and does not implement
PRD; PRD transitions are treated in CRD.
"""

const HSE_H_PLANCK = 6.62607015e-34
const HSE_C_LIGHT = 2.99792458e8
const HSE_K_BOLTZMANN = 1.380649e-23
const HSE_M_ELECTRON = 9.1093837139e-31
const HSE_R_BOHR = 5.29177210544e-11
const HSE_E_RYDBERG = 2.1798723611035e-18
const HSE_SCATTERING_MAX_ITERATIONS = 50
const HSE_SCATTERING_TOLERANCE = 1e-4


struct HydrogenSECollision
    lower::Int
    upper::Int
    kind::Symbol
    temperature::Vector{Float64}
    coefficient::Vector{Float64}
end


struct HSEBackgroundContinuumData{A,I}
    atoms::A
    bound_free_interpolants::I
end


function hse_background_continuum_data()
    # Atomic models have their level count in the concrete type parameter. RH's
    # background set is heterogeneous, so retain Muspel's intended abstract
    # AtomicModel element type rather than allowing a union-typed comprehension.
    atoms = Muspel.AtomicModel[]
    for atom_file in default_background_atom_files()
        atom = Muspel.read_atom(atom_file)
        atom.element == :H || push!(atoms, atom)
    end
    interpolants = Muspel.get_atoms_bf_interpolant(atoms)
    return HSEBackgroundContinuumData(atoms, interpolants)
end


function hse_level_index(atom, label::AbstractString)
    index = findfirst(==(label), atom.label)
    index === nothing && error("Hydrogen atom level label not found: $(label)")
    return index
end


function hse_transition_indices(atom, transition, levels)
    first_label = levels[transition[1]]["label"]
    second_label = levels[transition[2]]["label"]
    first_index = hse_level_index(atom, first_label)
    second_index = hse_level_index(atom, second_label)
    return min(first_index, second_index), max(first_index, second_index)
end


function load_hydrogen_se_collisions(atom_file::String, atom)
    data = Muspel.YAML.load_file(atom_file)
    levels = data["atomic_levels"]
    result = HydrogenSECollision[]

    for transition_data in get(data, "collisional", [])
        lower, upper = hse_transition_indices(atom, transition_data["transition"], levels)
        for rate_data in transition_data["data"]
            kind = Symbol(uppercase(rate_data["type"]))
            kind in (:CE, :CI, :OMEGA) || error(
                "Unsupported hydrogen collision type $(rate_data["type"]) in $(atom_file). " *
                "The hydrogen-se-1p5d solver currently supports CE, CI, and Omega."
            )
            temperature = Float64.(rate_data["temperature"]["value"])
            coefficient = Float64.(rate_data["data"]["value"])
            length(temperature) == length(coefficient) || error(
                "Collision temperature and coefficient grids differ for levels $(lower), $(upper)"
            )
            push!(
                result,
                HydrogenSECollision(lower, upper, kind, temperature, coefficient),
            )
        end
    end

    isempty(result) && error("No hydrogen collision rates found in $(atom_file)")
    return result
end


function hse_interp_clamped(x::Real, grid, values)
    x <= grid[1] && return values[1]
    x >= grid[end] && return values[end]
    upper = searchsortedfirst(grid, x)
    lower = upper - 1
    fraction = (x - grid[lower]) / (grid[upper] - grid[lower])
    return muladd(fraction, values[upper] - values[lower], values[lower])
end


function hse_trapezoid_weights(grid)
    length(grid) >= 2 || error("A radiative transition needs at least two wavelengths")
    weights = zeros(Float64, length(grid))
    weights[1] = (grid[2] - grid[1]) / 2
    weights[end] = (grid[end] - grid[end - 1]) / 2
    for index in 2:length(grid)-1
        weights[index] = (grid[index + 1] - grid[index - 1]) / 2
    end
    return abs.(weights)
end


function hse_subsample_grid(values, stride::Int)
    stride > 0 || error("Hydrogen SE wavelength stride must be positive")
    stride == 1 && return Float64.(values)
    indices = collect(1:stride:length(values))
    indices[end] == length(values) || push!(indices, length(values))
    length(indices) == 1 && push!(indices, length(values))
    return Float64.(values[unique(indices)])
end


function hse_formal_mean_intensity!(
    mean_intensity,
    z,
    extinction,
    source,
    mus,
    weights,
    downward,
    upward,
    ray_extinction,
)
    fill!(mean_intensity, 0.0)
    for ray in eachindex(mus)
        ray_extinction .= extinction ./ mus[ray]
        Muspel.piecewise_1D_linear!(
            z,
            ray_extinction,
            source,
            downward;
            to_end=true,
            initial_condition=:zero,
        )
        Muspel.piecewise_1D_linear!(
            z,
            ray_extinction,
            source,
            upward;
            to_end=false,
            initial_condition=:source,
        )
        @. mean_intensity += 0.5 * weights[ray] * (downward + upward)
    end
    return mean_intensity
end


function hse_formal_direction!(
    intensity,
    z,
    extinction,
    source,
    mu,
    ray_extinction;
    to_end::Bool,
)
    ray_extinction .= extinction ./ mu
    Muspel.piecewise_1D_linear!(
        z,
        ray_extinction,
        source,
        intensity;
        to_end=to_end,
        initial_condition=to_end ? :zero : :source,
    )
    return intensity
end


function hse_background_continuum(
    λ,
    temperature,
    ne,
    neutral_h,
    proton_h,
    background_data::Union{Nothing,HSEBackgroundContinuumData}=nothing,
)
    absorption, scattering = Muspel.α_cont_no_itp(
        Float64(λ),
        Float64(temperature),
        Float64(ne),
        Float64(neutral_h),
        Float64(proton_h),
    )

    # RH background.c treats LTE metal bound-free opacity as true absorption.
    # Muspel's no-interpolation continuum deliberately omits it, so add the
    # same LTE contribution per total hydrogen nucleus when reference atomic
    # data have been prepared by hse_background_continuum_data().
    if background_data !== nothing
        absorption += Muspel.σH_atoms_bf(
            background_data.bound_free_interpolants,
            background_data.atoms,
            Float64(λ),
            Float64(temperature),
            Float64(ne),
        ) * Float64(neutral_h + proton_h)
    end

    thermal_emissivity = absorption * Muspel.blackbody_λ(
        Float64(λ),
        Float64(temperature),
    )
    # Match RH exactly at the transfer-equation level:
    #   χ_c = χ_abs + χ_scat
    #   η_c = η_thermal
    #   η_scat = χ_scat J_old
    # The χ_scat J term is added and iterated by the formal-solution helpers;
    # scattering is not thermalized into χ_scat Bλ.
    return absorption, thermal_emissivity, scattering
end


function hse_maximum_relative_radiation_change(new_values, old_values)
    scale_floor = max(maximum(abs, new_values), maximum(abs, old_values), 1e-30) * 1e-12
    change = 0.0
    @inbounds for index in eachindex(new_values, old_values)
        denominator = max(abs(new_values[index]), abs(old_values[index]), scale_floor)
        change = max(change, abs(new_values[index] - old_values[index]) / denominator)
    end
    return change
end


function hse_formal_mean_intensity_scattering!(
    mean_intensity,
    z,
    extinction,
    true_emissivity,
    scattering,
    mus,
    weights,
    source,
    trial_mean_intensity,
    downward,
    upward,
    ray_extinction;
    max_iterations::Int=HSE_SCATTERING_MAX_ITERATIONS,
    tolerance::Real=HSE_SCATTERING_TOLERANCE,
)
    max_iterations > 0 || error("Background-scattering max iterations must be positive")
    tolerance > 0 || error("Background-scattering tolerance must be positive")
    @. mean_intensity = true_emissivity / max(extinction - scattering, 1e-30)

    # With no scattering this is one ordinary formal solution, not a fixed-point
    # problem. Besides avoiding a redundant pass, this is a useful exact limit
    # for comparison with RH's sca_c == 0 path.
    if all(iszero, scattering)
        @. source = true_emissivity / extinction
        hse_formal_mean_intensity!(
            mean_intensity,
            z,
            extinction,
            source,
            mus,
            weights,
            downward,
            upward,
            ray_extinction,
        )
        return 1, 0.0
    end
    residual = Inf
    iterations = 0
    for iteration in 1:max_iterations
        iterations = iteration
        @. source = (true_emissivity + scattering * mean_intensity) / extinction
        hse_formal_mean_intensity!(
            trial_mean_intensity,
            z,
            extinction,
            source,
            mus,
            weights,
            downward,
            upward,
            ray_extinction,
        )
        residual = hse_maximum_relative_radiation_change(
            trial_mean_intensity,
            mean_intensity,
        )
        mean_intensity .= trial_mean_intensity
        residual <= tolerance && break
    end
    residual <= tolerance || error(
        "Background-scattering iteration did not converge: residual=$(residual), " *
        "iterations=$(iterations), tolerance=$(tolerance)"
    )
    return iterations, residual
end


function hse_line_indices(atom, line)
    lower = findfirst(==(line.label_lo), atom.label)
    upper = findfirst(==(line.label_up), atom.label)
    if lower === nothing
        lower = argmin(abs.(atom.χ .- line.χlo))
    end
    if upper === nothing
        upper = argmin(abs.(atom.χ .- line.χup))
    end
    return lower, upper
end


function hse_add_bound_bound_rates!(
    rates,
    atom,
    z,
    temperature,
    ne,
    neutral_h,
    proton_h,
    velocity_x,
    velocity_y,
    velocity_z,
    populations,
    voigt_itp,
    mus,
    ray_weights,
    azimuths,
    azimuth_weights,
    background_data;
    wavelength_stride::Int,
)
    nz = length(z)
    extinction = zeros(Float64, nz)
    emissivity = zeros(Float64, nz)
    source = zeros(Float64, nz)
    downward = zeros(Float64, nz)
    upward = zeros(Float64, nz)
    ray_extinction = zeros(Float64, nz)
    profile = zeros(Float64, nz)
    background_absorption = zeros(Float64, nz)
    background_emissivity = zeros(Float64, nz)
    background_scattering = zeros(Float64, nz)
    profile_integral = zeros(Float64, nz)
    j_integral = zeros(Float64, nz)
    wavelength_profile_integral = zeros(Float64, nz)
    wavelength_j_integral = zeros(Float64, nz)
    mean_intensity = zeros(Float64, nz)
    trial_mean_intensity = zeros(Float64, nz)
    doppler_width = zeros(Float64, nz)
    broadening = zeros(Float64, nz)
    max_scattering_iterations = 0
    max_scattering_residual = 0.0

    for line in atom.lines
        lower, upper = hse_line_indices(atom, line)
        wavelengths = hse_subsample_grid(line.λ, wavelength_stride)
        wavelength_weights = hse_trapezoid_weights(wavelengths)
        fill!(profile_integral, 0.0)
        fill!(j_integral, 0.0)

        for depth in 1:nz
            doppler_width[depth] = Muspel.doppler_width(
                line.λ0,
                line.mass,
                temperature[depth],
            )
            broadening[depth] = Muspel.calc_broadening(
                line.γ,
                temperature[depth],
                ne[depth],
                neutral_h[depth],
            )
        end

        gamma_energy = HSE_H_PLANCK * HSE_C_LIGHT / (4π * line.λ0 * 1e-9)
        for wavelength_index in eachindex(wavelengths)
            λ = wavelengths[wavelength_index]
            for depth in 1:nz
                background_absorption[depth], background_emissivity[depth],
                background_scattering[depth] =
                    hse_background_continuum(
                    λ,
                    temperature[depth],
                    ne[depth],
                    neutral_h[depth],
                    proton_h[depth],
                    background_data,
                )
                mean_intensity[depth] = Muspel.blackbody_λ(λ, temperature[depth])
            end

            scattering_residual = Inf
            scattering_iterations = 0
            for scattering_iteration in 1:HSE_SCATTERING_MAX_ITERATIONS
                scattering_iterations = scattering_iteration
                fill!(trial_mean_intensity, 0.0)
                fill!(wavelength_profile_integral, 0.0)
                fill!(wavelength_j_integral, 0.0)

                for ray in eachindex(mus)
                    mu = mus[ray]
                    transverse = sqrt(max(0.0, 1 - mu^2))
                    for azimuth_index in eachindex(azimuths)
                        azimuth = azimuths[azimuth_index]
                        horizontal_x = transverse * cos(azimuth)
                        horizontal_y = transverse * sin(azimuth)
                        angular_weight = 0.5 * ray_weights[ray] *
                                         azimuth_weights[azimuth_index]

                        # The two directions are antipodal. Muspel's convention
                        # uses a negative projection for integration toward the
                        # end of the height array and a positive projection for
                        # integration toward its beginning.
                        for direction in (-1.0, 1.0)
                            for depth in 1:nz
                                projected_velocity = direction * (
                                    mu * velocity_z[depth] +
                                    horizontal_x * velocity_x[depth] +
                                    horizontal_y * velocity_y[depth]
                                )
                                damping = Muspel.damping(
                                    broadening[depth],
                                    λ,
                                    doppler_width[depth],
                                )
                                profile_velocity = (
                                    λ - line.λ0 +
                                    line.λ0 * projected_velocity / HSE_C_LIGHT
                                ) / doppler_width[depth]
                                profile[depth] = real(
                                    voigt_itp(damping, abs(profile_velocity))
                                ) / (sqrt(π) * doppler_width[depth])
                                line_factor = gamma_energy * profile[depth]
                                line_extinction = line_factor * (
                                    populations[depth, lower] * line.Blu -
                                    populations[depth, upper] * line.Bul
                                ) * 1e9
                                line_emissivity = line_factor *
                                                  populations[depth, upper] *
                                                  line.Aul * 1e-3
                                extinction[depth] = max(
                                    background_absorption[depth] +
                                    background_scattering[depth] + line_extinction,
                                    1e-30,
                                )
                                emissivity[depth] = max(
                                    background_emissivity[depth] + line_emissivity,
                                    0.0,
                                )
                                source[depth] = (
                                    emissivity[depth] +
                                    background_scattering[depth] * mean_intensity[depth]
                                ) / extinction[depth]
                            end

                            intensity = direction < 0 ? downward : upward
                            hse_formal_direction!(
                                intensity,
                                z,
                                extinction,
                                source,
                                mu,
                                ray_extinction;
                                to_end=direction < 0,
                            )
                            integration_weight = wavelength_weights[wavelength_index] *
                                                 angular_weight
                            @. trial_mean_intensity += angular_weight * intensity
                            @. wavelength_profile_integral += integration_weight * profile
                            @. wavelength_j_integral += integration_weight * profile * intensity
                        end
                    end
                end

                scattering_residual = hse_maximum_relative_radiation_change(
                    trial_mean_intensity,
                    mean_intensity,
                )
                mean_intensity .= trial_mean_intensity
                scattering_residual <= HSE_SCATTERING_TOLERANCE && break
            end
            scattering_residual <= HSE_SCATTERING_TOLERANCE || error(
                "Line background-scattering iteration did not converge for λ=$(λ) nm: " *
                "residual=$(scattering_residual), iterations=$(scattering_iterations)"
            )
            profile_integral .+= wavelength_profile_integral
            j_integral .+= wavelength_j_integral
            max_scattering_iterations = max(
                max_scattering_iterations,
                scattering_iterations,
            )
            max_scattering_residual = max(max_scattering_residual, scattering_residual)
        end

        for depth in 1:nz
            jbar_per_nm = j_integral[depth] / max(profile_integral[depth], eps(Float64))
            jbar_per_m = jbar_per_nm * 1e12 # kW m^-2 nm^-1 -> W m^-3
            rates[lower, upper, depth] += line.Blu * jbar_per_m
            rates[upper, lower, depth] += line.Aul + line.Bul * jbar_per_m
        end
    end
    return max_scattering_iterations, max_scattering_residual
end


function hse_cross_section(continuum, wavelength)
    wavelengths = continuum.λ
    wavelength < minimum(wavelengths) && return 0.0
    wavelength > maximum(wavelengths) && return 0.0
    return hse_interp_clamped(wavelength, wavelengths, continuum.σ)
end


function hse_add_bound_free_rates!(
    rates,
    atom,
    z,
    temperature,
    ne,
    neutral_h,
    proton_h,
    populations,
    nstar,
    mus,
    ray_weights,
    background_data;
    wavelength_stride::Int,
)
    isempty(atom.continua) && return 0, 0.0
    all_wavelengths = sort(unique(vcat([Float64.(continuum.λ) for continuum in atom.continua]...)))
    wavelengths = hse_subsample_grid(all_wavelengths, wavelength_stride)
    wavelength_weights = hse_trapezoid_weights(wavelengths)
    nz = length(z)
    extinction = zeros(Float64, nz)
    emissivity = zeros(Float64, nz)
    source = zeros(Float64, nz)
    mean_intensity = zeros(Float64, nz)
    trial_mean_intensity = zeros(Float64, nz)
    scattering = zeros(Float64, nz)
    downward = zeros(Float64, nz)
    upward = zeros(Float64, nz)
    ray_extinction = zeros(Float64, nz)
    max_scattering_iterations = 0
    max_scattering_residual = 0.0

    for wavelength_index in eachindex(wavelengths)
        λ = wavelengths[wavelength_index]
        λ_m = λ * 1e-9
        planck_numerator = 2 * HSE_H_PLANCK * HSE_C_LIGHT^2 / λ_m^5
        cross_sections = [hse_cross_section(continuum, λ) for continuum in atom.continua]
        all(iszero, cross_sections) && continue

        for depth in 1:nz
            background_absorption, background_emissivity, background_scattering =
                hse_background_continuum(
                λ,
                temperature[depth],
                ne[depth],
                neutral_h[depth],
                proton_h[depth],
                background_data,
            )
            total_extinction = background_absorption + background_scattering
            total_emissivity = background_emissivity
            scattering[depth] = background_scattering
            exponential = exp(-HSE_H_PLANCK * HSE_C_LIGHT /
                              (λ_m * HSE_K_BOLTZMANN * temperature[depth]))

            for (continuum_index, continuum) in enumerate(atom.continua)
                cross_section = cross_sections[continuum_index]
                cross_section == 0 && continue
                lower = continuum.lo
                upper = continuum.up
                gij = nstar[depth, lower] / max(nstar[depth, upper], eps(Float64)) * exponential
                total_extinction += cross_section * (
                    populations[depth, lower] - populations[depth, upper] * gij
                )
                total_emissivity += populations[depth, upper] * gij * cross_section *
                                    planck_numerator * 1e-12
            end
            extinction[depth] = max(total_extinction, 1e-30)
            emissivity[depth] = max(total_emissivity, 0.0)
            source[depth] = emissivity[depth] / extinction[depth]
        end

        scattering_iterations, scattering_residual =
            hse_formal_mean_intensity_scattering!(
            mean_intensity,
            z,
            extinction,
            emissivity,
            scattering,
            mus,
            ray_weights,
            source,
            trial_mean_intensity,
            downward,
            upward,
            ray_extinction,
        )
        max_scattering_iterations = max(max_scattering_iterations, scattering_iterations)
        max_scattering_residual = max(max_scattering_residual, scattering_residual)
        integration_width_m = wavelength_weights[wavelength_index] * 1e-9
        for depth in 1:nz
            j_per_m = mean_intensity[depth] * 1e12
            exponential = exp(-HSE_H_PLANCK * HSE_C_LIGHT /
                              (λ_m * HSE_K_BOLTZMANN * temperature[depth]))
            photon_factor = 4π * λ_m * integration_width_m /
                            (HSE_H_PLANCK * HSE_C_LIGHT)
            for (continuum_index, continuum) in enumerate(atom.continua)
                cross_section = cross_sections[continuum_index]
                cross_section == 0 && continue
                lower = continuum.lo
                upper = continuum.up
                gij = nstar[depth, lower] / max(nstar[depth, upper], eps(Float64)) * exponential
                rates[lower, upper, depth] += cross_section * j_per_m * photon_factor
                rates[upper, lower, depth] += cross_section * gij *
                                              (planck_numerator + j_per_m) * photon_factor
            end
        end
    end
    return max_scattering_iterations, max_scattering_residual
end


function hse_add_collisional_rates!(rates, collisions, atom, temperature, ne, nstar)
    omega_constant = HSE_E_RYDBERG / sqrt(HSE_M_ELECTRON) * π * HSE_R_BOHR^2 *
                     sqrt(8 / (π * HSE_K_BOLTZMANN))
    for collision in collisions
        lower = collision.lower
        upper = collision.upper
        energy_difference = atom.χ[upper] - atom.χ[lower]
        for depth in eachindex(temperature)
            coefficient = max(
                hse_interp_clamped(
                    temperature[depth],
                    collision.temperature,
                    collision.coefficient,
                ),
                0.0,
            )
            if collision.kind == :CE
                downward = coefficient * ne[depth] * atom.g[lower] / atom.g[upper] *
                           sqrt(temperature[depth])
                upward = downward * nstar[depth, upper] /
                         max(nstar[depth, lower], eps(Float64))
            elseif collision.kind == :CI
                upward = coefficient * ne[depth] * sqrt(temperature[depth]) *
                         exp(-energy_difference / (HSE_K_BOLTZMANN * temperature[depth]))
                downward = upward * nstar[depth, lower] /
                           max(nstar[depth, upper], eps(Float64))
            else # Omega
                downward = omega_constant * ne[depth] * coefficient /
                           (atom.g[upper] * sqrt(temperature[depth]))
                upward = downward * nstar[depth, upper] /
                         max(nstar[depth, lower], eps(Float64))
            end
            rates[lower, upper, depth] += upward
            rates[upper, lower, depth] += downward
        end
    end
    return rates
end


function hse_solve_rate_matrix(rates, initial_population, total_hydrogen)
    # This file stores rates[i,j] = P(i -> j). RH stores the transposed
    # convention Gamma[i,j] = P(j -> i), so the matrix assembled below is
    # exactly RH's Gamma before its particle-conservation row is substituted.
    nlevels = size(rates, 1)
    matrix = zeros(Float64, nlevels, nlevels)
    rhs = zeros(Float64, nlevels)
    for level in 1:nlevels
        for other in 1:nlevels
            level == other && continue
            matrix[level, other] = rates[other, level]
        end
        matrix[level, level] = -sum(@view rates[level, :])
    end

    conservation_row = argmax(initial_population)
    matrix[conservation_row, :] .= 1.0
    rhs[conservation_row] = total_hydrogen
    solution = matrix \ rhs
    all(isfinite, solution) || error("Hydrogen SE rate matrix produced non-finite populations")
    minimum(solution) >= -1e-8 * total_hydrogen || error(
        "Hydrogen SE rate matrix produced a materially negative population: $(minimum(solution))"
    )
    solution .= max.(solution, 0.0)
    solution .*= total_hydrogen / max(sum(solution), eps(Float64))
    return solution
end


function hse_rate_residual(rates, populations)
    nlevels = size(rates, 1)
    maximum_residual = 0.0
    for level in 1:nlevels
        incoming = 0.0
        outgoing = 0.0
        for other in 1:nlevels
            level == other && continue
            incoming += populations[other] * rates[other, level]
            outgoing += populations[level] * rates[level, other]
        end
        scale = max(abs(incoming), abs(outgoing), eps(Float64))
        maximum_residual = max(maximum_residual, abs(incoming - outgoing) / scale)
    end
    return maximum_residual
end


function hydrogen_se_update_1p5d(
    atmos::Atmosphere3D,
    nlte_h,
    atom,
    atom_file::String,
    voigt_itp;
    relaxation::Real=1.0,
    wavelength_stride::Int=1,
    mus=(0.211324865405187, 0.788675134594813),
    ray_weights=(0.5, 0.5),
    azimuths=(π / 4, 3π / 4, 5π / 4, 7π / 4),
    azimuth_weights=(0.25, 0.25, 0.25, 0.25),
    background_data=nothing,
)
    0 < relaxation <= 1 || error("Hydrogen SE relaxation must be in (0, 1]")
    length(mus) == length(ray_weights) || error("Hydrogen SE ray arrays differ in length")
    length(azimuths) == length(azimuth_weights) || error(
        "Hydrogen SE azimuth arrays differ in length"
    )
    all(mu -> 0 < mu <= 1, mus) || error("Hydrogen SE ray cosines must be in (0, 1]")
    all(>=(0), ray_weights) || error("Hydrogen SE ray weights must be non-negative")
    all(>=(0), azimuth_weights) || error(
        "Hydrogen SE azimuth weights must be non-negative"
    )
    isapprox(sum(ray_weights), 1.0; atol=1e-12) || error("Hydrogen SE ray weights must sum to one")
    isapprox(sum(azimuth_weights), 1.0; atol=1e-12) || error(
        "Hydrogen SE azimuth weights must sum to one"
    )
    size(nlte_h)[1:3] == size(atmos.temperature) || error(
        "Hydrogen SE population and atmosphere shapes differ"
    )

    collisions = load_hydrogen_se_collisions(atom_file, atom)
    background_data === nothing && (background_data = hse_background_continuum_data())
    mus_float = Float64.(mus)
    ray_weights_float = Float64.(ray_weights)
    azimuths_float = Float64.(azimuths)
    azimuth_weights_float = Float64.(azimuth_weights)
    corrected = similar(nlte_h)
    residuals = zeros(Float64, atmos.ny, atmos.nx)
    scattering_iterations = zeros(Int, atmos.ny, atmos.nx)
    scattering_residuals = zeros(Float64, atmos.ny, atmos.nx)
    z = Float64.(atmos.z)
    total_hydrogen = atmos.hydrogen1_density .+ atmos.proton_density

    column_count = atmos.nx * atmos.ny
    # Flattening exposes every independent 1.5D column to Julia's scheduler,
    # even when an MPI rank owns fewer x positions than Julia threads.
    Threads.@threads for column in 1:column_count
        x = div(column - 1, atmos.ny) + 1
        y = mod(column - 1, atmos.ny) + 1
        temperature = Float64.(@view atmos.temperature[:, y, x])
        ne = Float64.(@view atmos.electron_density[:, y, x])
        velocity_x = Float64.(@view atmos.velocity_x[:, y, x])
        velocity_y = Float64.(@view atmos.velocity_y[:, y, x])
        velocity_z = Float64.(@view atmos.velocity_z[:, y, x])
        populations = Float64.(@view nlte_h[:, y, x, :])
        n_h_total = Float64.(@view total_hydrogen[:, y, x])
        neutral_h = vec(sum(populations[:, atom.stage .== 1]; dims=2))
        proton_h = vec(sum(populations[:, atom.stage .> 1]; dims=2))
        nstar = zeros(Float64, atmos.nz, atom.nlevels)
        for depth in 1:atmos.nz
            nstar[depth, :] .= Muspel.saha_boltzmann(
                atom,
                temperature[depth],
                ne[depth],
                1.0,
            )
        end

        rates = zeros(Float64, atom.nlevels, atom.nlevels, atmos.nz)
        bb_scattering_iterations, bb_scattering_residual = hse_add_bound_bound_rates!(
            rates,
            atom,
            z,
            temperature,
            ne,
            neutral_h,
            proton_h,
            velocity_x,
            velocity_y,
            velocity_z,
            populations,
            voigt_itp,
            mus_float,
            ray_weights_float,
            azimuths_float,
            azimuth_weights_float,
            background_data;
            wavelength_stride=wavelength_stride,
        )
        bf_scattering_iterations, bf_scattering_residual = hse_add_bound_free_rates!(
            rates,
            atom,
            z,
            temperature,
            ne,
            neutral_h,
            proton_h,
            populations,
            nstar,
            mus_float,
            ray_weights_float,
            background_data;
            wavelength_stride=wavelength_stride,
        )
        hse_add_collisional_rates!(rates, collisions, atom, temperature, ne, nstar)

        column_residual = 0.0
        for depth in 1:atmos.nz
            solved = hse_solve_rate_matrix(
                @view(rates[:, :, depth]),
                @view(populations[depth, :]),
                n_h_total[depth],
            )
            solved .= (1 - relaxation) .* populations[depth, :] .+ relaxation .* solved
            solved .*= n_h_total[depth] / max(sum(solved), eps(Float64))
            corrected[depth, y, x, :] .= solved
            column_residual = max(
                column_residual,
                hse_rate_residual(@view(rates[:, :, depth]), solved),
            )
        end
        residuals[y, x] = column_residual
        scattering_iterations[y, x] = max(
            bb_scattering_iterations,
            bf_scattering_iterations,
        )
        scattering_residuals[y, x] = max(
            bb_scattering_residual,
            bf_scattering_residual,
        )
    end

    return corrected, maximum(residuals), (
        max_iterations=maximum(scattering_iterations),
        max_residual=maximum(scattering_residuals),
        tolerance=HSE_SCATTERING_TOLERANCE,
    )
end
