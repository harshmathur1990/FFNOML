"""
Hydrogen statistical-equilibrium correction used by Forward.jl.

This is a 3D, complete-redistribution solver.  The formal solution follows
inclined characteristics through the full Cartesian atmosphere, with periodic
horizontal boundaries and zero/thermalized upper/lower vertical boundaries.
All three velocity components enter bound-bound rates through their projection
onto the 3D angular quadrature.
It computes bound-bound and bound-free radiative rates, CE/CI/Omega electron
collision rates, and solves the stationary hydrogen rate equations subject to
hydrogen-particle conservation. Charge conservation remains in Forward.jl,
where non-hydrogen ions are evaluated in LTE.

This is intentionally a separate option from the inexpensive charge-only
fixed-point iteration.  It uses ordinary Lambda iteration for background
scattering and does not implement PRD; PRD transitions are treated in CRD.
"""

const HSE_H_PLANCK = 6.62607015e-34
const HSE_C_LIGHT = 2.99792458e8
const HSE_K_BOLTZMANN = 1.380649e-23
const HSE_M_ELECTRON = 9.1093837139e-31
const HSE_R_BOHR = 5.29177210544e-11
const HSE_E_RYDBERG = 2.1798723611035e-18
const HSE_SCATTERING_MAX_ITERATIONS = 50
const HSE_SCATTERING_TOLERANCE = 1e-4
const HSE_NUMBER_DENSITY_UNIT = Muspel.Unitful.uparse("m^-3")
const HSE_WAVELENGTH_UNIT = Muspel.Unitful.uparse("nm")
const HSE_INVERSE_LENGTH_UNIT = Muspel.Unitful.uparse("m^-1")


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
                "The hydrogen-se-3d solver currently supports CE, CI, and Omega."
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


function hse_periodic_bracket(axis, coordinate)
    n = length(axis)
    n > 0 || error("A horizontal Hydrogen SE coordinate axis is empty")
    n == 1 && return 1, 1, 0.0
    all(diff(axis) .> 0) || error("Hydrogen SE horizontal axes must be increasing")

    # The atmosphere coordinates are cell centres.  The missing interval from
    # the last centre back to the first therefore has the representative grid
    # spacing at the two periodic edges.
    edge_spacing = 0.5 * ((axis[2] - axis[1]) + (axis[end] - axis[end - 1]))
    period = axis[end] - axis[1] + edge_spacing
    wrapped = mod(coordinate - axis[1], period) + axis[1]
    if wrapped >= axis[end]
        fraction = (wrapped - axis[end]) / (axis[1] + period - axis[end])
        return n, 1, fraction
    end
    lower = searchsortedlast(axis, wrapped)
    lower = clamp(lower, 1, n - 1)
    fraction = (wrapped - axis[lower]) / (axis[lower + 1] - axis[lower])
    return lower, lower + 1, fraction
end


function hse_bilinear_periodic(plane, x, y, x_coordinate, y_coordinate)
    x0, x1, tx = hse_periodic_bracket(x, x_coordinate)
    y0, y1, ty = hse_periodic_bracket(y, y_coordinate)
    lower = muladd(tx, plane[y0, x1] - plane[y0, x0], plane[y0, x0])
    upper = muladd(tx, plane[y1, x1] - plane[y1, x0], plane[y1, x0])
    return muladd(ty, upper - lower, lower)
end


function hse_linear_characteristic(intensity_upwind, source_upwind, source_local, optical_depth)
    optical_depth <= 0 && return intensity_upwind
    if optical_depth < 1e-3
        # Series forms avoid cancellation in the linear-source weights.
        dt = optical_depth
        attenuation = exp(-dt)
        weight_local = dt / 2 - dt^2 / 6 + dt^3 / 24
        weight_upwind = dt / 2 - dt^2 / 3 + dt^3 / 8
    else
        dt = optical_depth
        attenuation = exp(-dt)
        weight_local = (dt - 1 + attenuation) / dt
        weight_upwind = (1 - (dt + 1) * attenuation) / dt
    end
    return intensity_upwind * attenuation +
           weight_upwind * source_upwind + weight_local * source_local
end


"""Solve one 3D ordinate using horizontal-plane characteristics.

The horizontal footprint on the preceding z plane is bilinearly interpolated
with periodic x/y wrapping.  This is the long-horizontal-characteristic case
used by 3D short-characteristic solvers when an inclined ray crosses one or
more vertical cell walls before reaching the preceding horizontal plane.
"""
function hse_formal_direction_3d!(
    intensity,
    x,
    y,
    z,
    extinction,
    source,
    mux,
    muy,
    muz,
)
    abs(muz) > eps(Float64) || error("Hydrogen SE rays may not be exactly horizontal")
    size(intensity) == size(extinction) == size(source) || error(
        "Hydrogen SE 3D formal-solution arrays differ in shape"
    )
    nz, ny, nx = size(extinction)
    (length(x), length(y), length(z)) == (nx, ny, nz) || error(
        "Hydrogen SE coordinates and atmosphere shape differ"
    )
    all(diff(z) .!= 0) && (all(diff(z) .> 0) || all(diff(z) .< 0)) || error(
        "Hydrogen SE vertical coordinates must be strictly monotonic"
    )

    step = muz * (z[end] - z[1]) > 0 ? 1 : -1
    first_plane = step > 0 ? 1 : nz
    last_plane = step > 0 ? nz : 1
    # z is a geometrical height: upward rays enter through a thermalized lower
    # boundary, downward rays through a dark upper boundary.
    if muz > 0
        @views intensity[first_plane, :, :] .= source[first_plane, :, :]
    else
        @views fill!(intensity[first_plane, :, :], 0.0)
    end

    for k in (first_plane + step):step:last_plane
        previous = k - step
        dz = z[k] - z[previous]
        distance = abs(dz / muz)
        x_shift = mux / muz * dz
        y_shift = muy / muz * dz
        @views begin
            previous_intensity = intensity[previous, :, :]
            previous_extinction = extinction[previous, :, :]
            previous_source = source[previous, :, :]
            for ix in 1:nx, iy in 1:ny
                upstream_x = x[ix] - x_shift
                upstream_y = y[iy] - y_shift
                intensity_upwind = hse_bilinear_periodic(
                    previous_intensity, x, y, upstream_x, upstream_y,
                )
                extinction_upwind = hse_bilinear_periodic(
                    previous_extinction, x, y, upstream_x, upstream_y,
                )
                source_upwind = hse_bilinear_periodic(
                    previous_source, x, y, upstream_x, upstream_y,
                )
                optical_depth = 0.5 * (
                    extinction_upwind + extinction[k, iy, ix]
                ) * distance
                intensity[k, iy, ix] = hse_linear_characteristic(
                    intensity_upwind,
                    source_upwind,
                    source[k, iy, ix],
                    optical_depth,
                )
            end
        end
    end
    return intensity
end


function hse_formal_mean_intensity_3d!(
    mean_intensity,
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
    fill!(mean_intensity, 0.0)
    for ray in eachindex(mus), azimuth_index in eachindex(azimuths)
        mu = mus[ray]
        transverse = sqrt(max(0.0, 1 - mu^2))
        mux = transverse * cos(azimuths[azimuth_index])
        muy = transverse * sin(azimuths[azimuth_index])
        angular_weight = 0.5 * weights[ray] * azimuth_weights[azimuth_index]
        for direction in (-1.0, 1.0)
            hse_formal_direction_3d!(
                intensity, x, y, z, extinction, source,
                direction * mux, direction * muy, direction * mu,
            )
            @. mean_intensity += angular_weight * intensity
        end
    end
    return mean_intensity
end


function hse_background_continuum(
    λ,
    temperature,
    ne,
    neutral_h,
    proton_h,
    background_data::Union{Nothing,HSEBackgroundContinuumData}=nothing,
)
    continuum = Muspel.α_cont_no_itp(
        Float64(λ),
        Float64(temperature),
        Float64(ne),
        Float64(neutral_h),
        Float64(proton_h),
    )
    absorption, scattering = if continuum isa Tuple
        # Muspel versions newer than 0.2.5 return thermal absorption and
        # scattering separately.
        continuum
    else
        # Muspel 0.2.5 returns their sum.  Recover the same Thomson and neutral-
        # hydrogen Rayleigh terms used internally so scattering remains
        # explicit in the RH-style source-function iteration below.
        scattering_quantity =
            Muspel.α_thomson(Float64(ne) * HSE_NUMBER_DENSITY_UNIT) +
            Muspel.α_rayleigh_h(
                Float64(λ) * HSE_WAVELENGTH_UNIT,
                Float64(neutral_h) * HSE_NUMBER_DENSITY_UNIT,
            )
        scattering_value = Muspel.Unitful.ustrip(
            Muspel.Unitful.uconvert(HSE_INVERSE_LENGTH_UNIT, scattering_quantity),
        )
        Float64(continuum) - scattering_value, scattering_value
    end

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


function hse_formal_mean_intensity_scattering_3d!(
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
    trial_mean_intensity,
    intensity;
    max_iterations::Int=HSE_SCATTERING_MAX_ITERATIONS,
    tolerance::Real=HSE_SCATTERING_TOLERANCE,
)
    max_iterations > 0 || error("Background-scattering max iterations must be positive")
    tolerance > 0 || error("Background-scattering tolerance must be positive")
    @. mean_intensity = true_emissivity / max(extinction - scattering, 1e-30)

    if all(iszero, scattering)
        @. source = true_emissivity / extinction
        hse_formal_mean_intensity_3d!(
            mean_intensity, x, y, z, extinction, source,
            mus, weights, azimuths, azimuth_weights, intensity,
        )
        return 1, 0.0
    end

    residual = Inf
    iterations = 0
    for iteration in 1:max_iterations
        iterations = iteration
        @. source = (true_emissivity + scattering * mean_intensity) / extinction
        hse_formal_mean_intensity_3d!(
            trial_mean_intensity, x, y, z, extinction, source,
            mus, weights, azimuths, azimuth_weights, intensity,
        )
        residual = hse_maximum_relative_radiation_change(
            trial_mean_intensity,
            mean_intensity,
        )
        mean_intensity .= trial_mean_intensity
        residual <= tolerance && break
    end
    residual <= tolerance || error(
        "3D background-scattering iteration did not converge: residual=$(residual), " *
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


function hse_cross_section(continuum, wavelength)
    wavelengths = continuum.λ
    wavelength < minimum(wavelengths) && return 0.0
    wavelength > maximum(wavelengths) && return 0.0
    return hse_interp_clamped(wavelength, wavelengths, continuum.σ)
end


function hse_add_bound_bound_rates_3d!(
    rates,
    atom,
    x,
    y,
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
    volume_shape = size(temperature)
    work() = zeros(Float64, volume_shape)
    extinction = work()
    emissivity = work()
    source = work()
    intensity = work()
    profile = work()
    background_absorption = work()
    background_emissivity = work()
    background_scattering = work()
    profile_integral = work()
    j_integral = work()
    wavelength_profile_integral = work()
    wavelength_j_integral = work()
    mean_intensity = work()
    trial_mean_intensity = work()
    doppler_width = work()
    broadening = work()
    max_scattering_iterations = 0
    max_scattering_residual = 0.0

    for line in atom.lines
        lower, upper = hse_line_indices(atom, line)
        wavelengths = hse_subsample_grid(line.λ, wavelength_stride)
        wavelength_weights = hse_trapezoid_weights(wavelengths)
        fill!(profile_integral, 0.0)
        fill!(j_integral, 0.0)
        for cell in CartesianIndices(temperature)
            doppler_width[cell] = Muspel.doppler_width(
                line.λ0, line.mass, temperature[cell],
            )
            broadening[cell] = Muspel.calc_broadening(
                line.γ, temperature[cell], ne[cell], neutral_h[cell],
            )
        end

        gamma_energy = HSE_H_PLANCK * HSE_C_LIGHT / (4π * line.λ0 * 1e-9)
        for wavelength_index in eachindex(wavelengths)
            λ = wavelengths[wavelength_index]
            for cell in CartesianIndices(temperature)
                background_absorption[cell], background_emissivity[cell],
                background_scattering[cell] = hse_background_continuum(
                    λ, temperature[cell], ne[cell], neutral_h[cell], proton_h[cell],
                    background_data,
                )
                mean_intensity[cell] = Muspel.blackbody_λ(λ, temperature[cell])
            end

            scattering_residual = Inf
            scattering_iterations = 0
            for scattering_iteration in 1:HSE_SCATTERING_MAX_ITERATIONS
                scattering_iterations = scattering_iteration
                fill!(trial_mean_intensity, 0.0)
                fill!(wavelength_profile_integral, 0.0)
                fill!(wavelength_j_integral, 0.0)

                for ray in eachindex(mus), azimuth_index in eachindex(azimuths)
                    mu = mus[ray]
                    transverse = sqrt(max(0.0, 1 - mu^2))
                    horizontal_x = transverse * cos(azimuths[azimuth_index])
                    horizontal_y = transverse * sin(azimuths[azimuth_index])
                    angular_weight = 0.5 * ray_weights[ray] *
                                     azimuth_weights[azimuth_index]
                    for direction in (-1.0, 1.0)
                        mux = direction * horizontal_x
                        muy = direction * horizontal_y
                        muz = direction * mu
                        for cell in CartesianIndices(temperature)
                            projected_velocity = mux * velocity_x[cell] +
                                                 muy * velocity_y[cell] +
                                                 muz * velocity_z[cell]
                            damping = Muspel.damping(
                                broadening[cell], λ, doppler_width[cell],
                            )
                            profile_velocity = (
                                λ - line.λ0 +
                                line.λ0 * projected_velocity / HSE_C_LIGHT
                            ) / doppler_width[cell]
                            profile[cell] = real(
                                voigt_itp(damping, abs(profile_velocity))
                            ) / (sqrt(π) * doppler_width[cell])
                            line_factor = gamma_energy * profile[cell]
                            line_extinction = line_factor * (
                                populations[cell, lower] * line.Blu -
                                populations[cell, upper] * line.Bul
                            ) * 1e9
                            line_emissivity = line_factor * populations[cell, upper] *
                                              line.Aul * 1e-3
                            extinction[cell] = max(
                                background_absorption[cell] +
                                background_scattering[cell] + line_extinction,
                                1e-30,
                            )
                            emissivity[cell] = max(
                                background_emissivity[cell] + line_emissivity,
                                0.0,
                            )
                            source[cell] = (
                                emissivity[cell] +
                                background_scattering[cell] * mean_intensity[cell]
                            ) / extinction[cell]
                        end
                        hse_formal_direction_3d!(
                            intensity, x, y, z, extinction, source, mux, muy, muz,
                        )
                        integration_weight = wavelength_weights[wavelength_index] *
                                             angular_weight
                        @. trial_mean_intensity += angular_weight * intensity
                        @. wavelength_profile_integral += integration_weight * profile
                        @. wavelength_j_integral += integration_weight * profile * intensity
                    end
                end

                scattering_residual = hse_maximum_relative_radiation_change(
                    trial_mean_intensity, mean_intensity,
                )
                mean_intensity .= trial_mean_intensity
                scattering_residual <= HSE_SCATTERING_TOLERANCE && break
            end
            scattering_residual <= HSE_SCATTERING_TOLERANCE || error(
                "3D line background-scattering iteration did not converge for λ=$(λ) nm: " *
                "residual=$(scattering_residual), iterations=$(scattering_iterations)"
            )
            profile_integral .+= wavelength_profile_integral
            j_integral .+= wavelength_j_integral
            max_scattering_iterations = max(
                max_scattering_iterations, scattering_iterations,
            )
            max_scattering_residual = max(max_scattering_residual, scattering_residual)
        end

        for cell in CartesianIndices(temperature)
            jbar_per_nm = j_integral[cell] /
                          max(profile_integral[cell], eps(Float64))
            jbar_per_m = jbar_per_nm * 1e12
            k, iy, ix = Tuple(cell)
            rates[lower, upper, k, iy, ix] += line.Blu * jbar_per_m
            rates[upper, lower, k, iy, ix] += line.Aul + line.Bul * jbar_per_m
        end
    end
    return max_scattering_iterations, max_scattering_residual
end


function hse_add_bound_free_rates_3d!(
    rates,
    atom,
    x,
    y,
    z,
    temperature,
    ne,
    neutral_h,
    proton_h,
    populations,
    nstar,
    mus,
    ray_weights,
    azimuths,
    azimuth_weights,
    background_data;
    wavelength_stride::Int,
)
    isempty(atom.continua) && return 0, 0.0
    all_wavelengths = sort(unique(vcat(
        [Float64.(continuum.λ) for continuum in atom.continua]...,
    )))
    wavelengths = hse_subsample_grid(all_wavelengths, wavelength_stride)
    wavelength_weights = hse_trapezoid_weights(wavelengths)
    volume_shape = size(temperature)
    work() = zeros(Float64, volume_shape)
    extinction = work()
    emissivity = work()
    source = work()
    mean_intensity = work()
    trial_mean_intensity = work()
    scattering = work()
    intensity = work()
    max_scattering_iterations = 0
    max_scattering_residual = 0.0

    for wavelength_index in eachindex(wavelengths)
        λ = wavelengths[wavelength_index]
        λ_m = λ * 1e-9
        planck_numerator = 2 * HSE_H_PLANCK * HSE_C_LIGHT^2 / λ_m^5
        cross_sections = [hse_cross_section(continuum, λ) for continuum in atom.continua]
        all(iszero, cross_sections) && continue

        for cell in CartesianIndices(temperature)
            background_absorption, background_emissivity, background_scattering =
                hse_background_continuum(
                    λ, temperature[cell], ne[cell], neutral_h[cell], proton_h[cell],
                    background_data,
                )
            total_extinction = background_absorption + background_scattering
            total_emissivity = background_emissivity
            scattering[cell] = background_scattering
            exponential = exp(
                -HSE_H_PLANCK * HSE_C_LIGHT /
                (λ_m * HSE_K_BOLTZMANN * temperature[cell]),
            )
            for (continuum_index, continuum) in enumerate(atom.continua)
                cross_section = cross_sections[continuum_index]
                cross_section == 0 && continue
                lower = continuum.lo
                upper = continuum.up
                gij = nstar[cell, lower] /
                      max(nstar[cell, upper], eps(Float64)) * exponential
                total_extinction += cross_section * (
                    populations[cell, lower] - populations[cell, upper] * gij
                )
                total_emissivity += populations[cell, upper] * gij * cross_section *
                                    planck_numerator * 1e-12
            end
            extinction[cell] = max(total_extinction, 1e-30)
            emissivity[cell] = max(total_emissivity, 0.0)
        end

        scattering_iterations, scattering_residual =
            hse_formal_mean_intensity_scattering_3d!(
                mean_intensity, x, y, z, extinction, emissivity, scattering,
                mus, ray_weights, azimuths, azimuth_weights,
                source, trial_mean_intensity, intensity,
            )
        max_scattering_iterations = max(max_scattering_iterations, scattering_iterations)
        max_scattering_residual = max(max_scattering_residual, scattering_residual)
        integration_width_m = wavelength_weights[wavelength_index] * 1e-9
        for cell in CartesianIndices(temperature)
            j_per_m = mean_intensity[cell] * 1e12
            exponential = exp(
                -HSE_H_PLANCK * HSE_C_LIGHT /
                (λ_m * HSE_K_BOLTZMANN * temperature[cell]),
            )
            photon_factor = 4π * λ_m * integration_width_m /
                            (HSE_H_PLANCK * HSE_C_LIGHT)
            k, iy, ix = Tuple(cell)
            for (continuum_index, continuum) in enumerate(atom.continua)
                cross_section = cross_sections[continuum_index]
                cross_section == 0 && continue
                lower = continuum.lo
                upper = continuum.up
                gij = nstar[cell, lower] /
                      max(nstar[cell, upper], eps(Float64)) * exponential
                rates[lower, upper, k, iy, ix] +=
                    cross_section * j_per_m * photon_factor
                rates[upper, lower, k, iy, ix] += cross_section * gij *
                    (planck_numerator + j_per_m) * photon_factor
            end
        end
    end
    return max_scattering_iterations, max_scattering_residual
end


function hse_add_collisional_rates_3d!(rates, collisions, atom, temperature, ne, nstar)
    omega_constant = HSE_E_RYDBERG / sqrt(HSE_M_ELECTRON) * π * HSE_R_BOHR^2 *
                     sqrt(8 / (π * HSE_K_BOLTZMANN))
    for collision in collisions
        lower = collision.lower
        upper = collision.upper
        energy_difference = atom.χ[upper] - atom.χ[lower]
        for cell in CartesianIndices(temperature)
            coefficient = max(
                hse_interp_clamped(
                    temperature[cell], collision.temperature, collision.coefficient,
                ),
                0.0,
            )
            if collision.kind == :CE
                downward = coefficient * ne[cell] * atom.g[lower] / atom.g[upper] *
                           sqrt(temperature[cell])
                upward = downward * nstar[cell, upper] /
                         max(nstar[cell, lower], eps(Float64))
            elseif collision.kind == :CI
                upward = coefficient * ne[cell] * sqrt(temperature[cell]) *
                         exp(-energy_difference /
                             (HSE_K_BOLTZMANN * temperature[cell]))
                downward = upward * nstar[cell, lower] /
                           max(nstar[cell, upper], eps(Float64))
            else
                downward = omega_constant * ne[cell] * coefficient /
                           (atom.g[upper] * sqrt(temperature[cell]))
                upward = downward * nstar[cell, upper] /
                         max(nstar[cell, lower], eps(Float64))
            end
            k, iy, ix = Tuple(cell)
            rates[lower, upper, k, iy, ix] += upward
            rates[upper, lower, k, iy, ix] += downward
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


function hydrogen_se_update_3d(
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
    wavelength_stride > 0 || error("Hydrogen SE wavelength stride must be positive")
    length(mus) == length(ray_weights) || error("Hydrogen SE ray arrays differ in length")
    length(azimuths) == length(azimuth_weights) || error(
        "Hydrogen SE azimuth arrays differ in length"
    )
    all(mu -> 0 < mu <= 1, mus) || error("Hydrogen SE ray cosines must be in (0, 1]")
    all(>=(0), ray_weights) || error("Hydrogen SE ray weights must be non-negative")
    all(>=(0), azimuth_weights) || error(
        "Hydrogen SE azimuth weights must be non-negative"
    )
    isapprox(sum(ray_weights), 1.0; atol=1e-12) || error(
        "Hydrogen SE ray weights must sum to one"
    )
    isapprox(sum(azimuth_weights), 1.0; atol=1e-12) || error(
        "Hydrogen SE azimuth weights must sum to one"
    )
    size(nlte_h)[1:3] == size(atmos.temperature) || error(
        "Hydrogen SE population and atmosphere shapes differ"
    )
    size(nlte_h, 4) == atom.nlevels || error(
        "Hydrogen SE population level count differs from the atomic model"
    )

    collisions = load_hydrogen_se_collisions(atom_file, atom)
    background_data === nothing && (background_data = hse_background_continuum_data())
    mus_float = Float64.(mus)
    ray_weights_float = Float64.(ray_weights)
    azimuths_float = Float64.(azimuths)
    azimuth_weights_float = Float64.(azimuth_weights)
    x = Float64.(atmos.x)
    y = Float64.(atmos.y)
    z = Float64.(atmos.z)
    temperature = Float64.(atmos.temperature)
    ne = Float64.(atmos.electron_density)
    velocity_x = Float64.(atmos.velocity_x)
    velocity_y = Float64.(atmos.velocity_y)
    velocity_z = Float64.(atmos.velocity_z)
    populations = Float64.(nlte_h)
    total_hydrogen = Float64.(atmos.hydrogen1_density .+ atmos.proton_density)
    neutral_h = dropdims(
        sum(populations[:, :, :, atom.stage .== 1]; dims=4);
        dims=4,
    )
    proton_h = dropdims(
        sum(populations[:, :, :, atom.stage .> 1]; dims=4);
        dims=4,
    )
    nstar = zeros(Float64, atmos.nz, atmos.ny, atmos.nx, atom.nlevels)
    for cell in CartesianIndices(temperature)
        nstar[cell, :] .= Muspel.saha_boltzmann(
            atom, temperature[cell], ne[cell], 1.0,
        )
    end

    rates = zeros(
        Float64, atom.nlevels, atom.nlevels, atmos.nz, atmos.ny, atmos.nx,
    )
    bb_scattering_iterations, bb_scattering_residual =
        hse_add_bound_bound_rates_3d!(
            rates, atom, x, y, z, temperature, ne, neutral_h, proton_h,
            velocity_x, velocity_y, velocity_z, populations, voigt_itp,
            mus_float, ray_weights_float, azimuths_float, azimuth_weights_float,
            background_data;
            wavelength_stride=wavelength_stride,
        )
    bf_scattering_iterations, bf_scattering_residual =
        hse_add_bound_free_rates_3d!(
            rates, atom, x, y, z, temperature, ne, neutral_h, proton_h,
            populations, nstar, mus_float, ray_weights_float,
            azimuths_float, azimuth_weights_float, background_data;
            wavelength_stride=wavelength_stride,
        )
    hse_add_collisional_rates_3d!(rates, collisions, atom, temperature, ne, nstar)

    corrected = similar(nlte_h)
    residuals = zeros(Float64, size(temperature))
    cells = collect(CartesianIndices(temperature))
    Threads.@threads for cell_index in eachindex(cells)
        cell = cells[cell_index]
        k, iy, ix = Tuple(cell)
        local_rates = @view rates[:, :, k, iy, ix]
        initial = @view populations[k, iy, ix, :]
        solved = hse_solve_rate_matrix(local_rates, initial, total_hydrogen[cell])
        solved .= (1 - relaxation) .* initial .+ relaxation .* solved
        solved .*= total_hydrogen[cell] / max(sum(solved), eps(Float64))
        corrected[k, iy, ix, :] .= solved
        residuals[cell] = hse_rate_residual(local_rates, solved)
    end

    return corrected, maximum(residuals), (
        max_iterations=max(bb_scattering_iterations, bf_scattering_iterations),
        max_residual=max(bb_scattering_residual, bf_scattering_residual),
        tolerance=HSE_SCATTERING_TOLERANCE,
    )
end
