# Offline test of enforcing conservation on FFNO populations: for every species alike, n_i -> n_i * (A n_H) / sum(n_i), with
# n_H = rho / mass_per_H from the run's own atmosphere and A from the atom file. Same synthesis (H-alpha + Fe I, Ca II + Si I, vturb).
# Nothing in FFNOInversion changes. Writes falc_<run>_norm_synthesis.h5 (same layout as the runs) for compare_rh.py; populations too.
# usage (hyades): julia --project=FFNOInversion.jl response_run/falc/falc_mass_norm.jl run...
using FFNOInversion, HDF5, Libdl, Printf
import Muspel, HSEMHS
const R = joinpath(@__DIR__); const A = joinpath(R, "..", "inputs")
mass_per_H = HSEMHS.load_table("/mn/stornext/d9/data/harshm/hsemhs_env112/work/e2e/inputs/eos_table_fullz.h5").mass_per_H
@printf("mass_per_H = %.4e kg (%.3f amu)\n", mass_per_H, mass_per_H / 1.66053906660e-27)
abund = Dict(:H => 1.0, :CA => 10^(6.34 - 12))
eos = WittmannEOS(joinpath(A, "wittmann", "libwitt_ffno.$(Libdl.dlext)"), joinpath(A, "pf_Kurucz.input"))
sources = ((SpectralRegionConfig(655.8e-9, 0.002e-9, 701, 1.0, :none, nothing, [SpectralSourceConfig(:ffno, nothing), SpectralSourceConfig(:kurucz_lte, joinpath(R, "kurucz_halpha_blend.list"))])),
           (SpectralRegionConfig(853.4e-9, 0.002e-9, 601, 1.0, :none, nothing, [SpectralSourceConfig(:ffno, nothing), SpectralSourceConfig(:kurucz_lte, joinpath(R, "kurucz_8542_blend.list"))])))
scale = output_intensity_scale(collect(sources), :si)
for run in ARGS
    f = h5open(joinpath(R, "falc_$(run)_atmosphere.h5"))
    field(n) = (v = read(f[n])[1, :, :, :]; permutedims(v[end-81:end, :, :], (1, 3, 2)))   # (z, x, y) FALC's 82 levels
    temperature, ne, rho, z, vturb, pgas = field("temperature"), field("ne"), field("rho"), field("z"), field("vturb"), field("pgas")
    pop(s) = (v = read(f["populations/$s"]); permutedims(v[:, end-81:end, :, :], (2, 4, 3, 1)))    # HDF5.jl (level, z, y, x)
    pops = Dict(:H => pop("H"), :CA => pop("CA")); close(f)
    for (s, p) in pops
        total = sum(p; dims=4)
        p .*= reshape(abund[s] .* rho ./ mass_per_H, size(rho)..., 1) ./ total
    end
    nz, nx, ny = size(temperature); zero3 = zeros(nz, nx, ny)
    grid = Grid3D(VerticalAxis(:z_m, z[:, 3, 3]), 48e3 .* (0.0:4.0), 48e3 .* (0.0:4.0))
    atmosphere = FFNOInversion.Atmosphere3D(grid, temperature, copy(zero3), copy(zero3), copy(zero3), vturb; pgas=pgas, rho=rho, ne=ne, z=z)
    setup = build_synthesis_setup((regions=collect(sources),), atmosphere, pops, eos;
        atom_files=Dict(:H => joinpath(A, "atoms", "atom.h6_tiago2.yaml"), :CA => joinpath(A, "atoms", "atom.ca2.yaml")))
    wave = vcat(setup[1].wavelength_air_m, setup[2].wavelength_air_m)
    c = SpectralCube(zeros(length(wave), 1, nx, ny), wave, StokesSet(:I))
    synthesize!(c, setup, NonPRD(), atmosphere, pops, ThreadedSynthesisCache(Float64, nz, length(wave)))
    h5open(joinpath(R, "falc_$(run)_norm_synthesis.h5"), "w") do o
        o["wavelength_m"] = wave; o["intensity"] = FFNOInversion._with_time_slyx(c.data .* scale)
    end
    h5open(joinpath(R, "falc_$(run)_norm_atmosphere.h5"), "w") do o
        for (s, p) in pops; o["populations/$s"] = permutedims(p, (4, 1, 3, 2)); end
    end
    println("done ", run)
end
