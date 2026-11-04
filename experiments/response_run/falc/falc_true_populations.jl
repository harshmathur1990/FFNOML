# Control for the FALC runs: the same FFNOInversion synthesis (Muspel line + continuum, Wittmann Kurucz blend) with FALC's own
# NLTE H populations and ne from FALC_82_5x5.hdf5 instead of FFNO's, on the file's heights. H-alpha + Fe I only (the file
# has no Ca populations). Isolates the FFNO's share of the error; RH's populations, no HSEMHS.
# usage (hyades, after ~/loadgcc.sh): julia --project=../FFNOInversion.jl response_run/falc/falc_true_populations.jl
using FFNOInversion, HDF5, Libdl, Printf
import Muspel
const R = "/mn/stornext/d9/data/harshm/FFNOML/response_run"
file(name) = h5read(joinpath(R, "falc", "FALC_82_5x5.hdf5"), name)
# HDF5.jl reverses h5py's order: scalars (z, y, x, t), populations (z, y, x, level, t)
T(a) = Float64.(a[:, :, :, 1])
temperature = T(file("temperature")); ne = T(file("electron_density")); vturb = T(file("velocity_turbulent")); vz = T(file("velocity_z"))
nz, nx, ny = size(temperature); z = Float64.(file("z")[:, 1])
hp = permutedims(Float64.(file("hydrogen_populations")[:, :, :, :, 1]), (1, 2, 3, 4))
size(hp) == (nz, nx, ny, 6) || error("populations shape $(size(hp))")
zero3 = zeros(nz, nx, ny); grid = Grid3D(VerticalAxis(:z_m, z), 48e3 .* (0.0:4.0), 48e3 .* (0.0:4.0))
pgas = (dropdims(sum(hp; dims=4); dims=4) .* 1.1 .+ ne) .* 1.380649e-23 .* temperature
atmosphere = FFNOInversion.Atmosphere3D(grid, temperature, copy(zero3), copy(zero3), vz, vturb;
    pgas=pgas, rho=pgas .* 0, ne=ne, z=repeat(reshape(z, :, 1, 1), 1, nx, ny))
eos = WittmannEOS(joinpath(R, "inputs", "wittmann", "libwitt_ffno.$(Libdl.dlext)"), joinpath(R, "inputs", "pf_Kurucz.input"))
ffno = SpectralSourceConfig(:ffno, nothing)
fe = SpectralSourceConfig(:kurucz_lte, joinpath(R, "falc", "kurucz_halpha_blend.list"))
region = SpectralRegionConfig(655.8e-9, 0.002e-9, 701, 1.0, :none, nothing, [ffno, fe])
setup = build_synthesis_setup((regions=[region],), atmosphere, Dict(:H => hp), eos;
    atom_files=Dict(:H => joinpath(R, "inputs", "atoms", "atom.h6_tiago2.yaml")))
wave = setup[1].wavelength_air_m
cube = SpectralCube(zeros(length(wave), 1, nx, ny), wave, StokesSet(:I))
synthesize!(cube, setup, NonPRD(), atmosphere, Dict(:H => hp), ThreadedSynthesisCache(Float64, nz, length(wave)))
h5open(joinpath(R, "falc", "falc_true_populations.h5"), "w") do f
    f["wavelength_air_m"] = wave; f["intensity"] = cube.data[:, 1, 3, 3]
    attributes(f["intensity"])["units"] = FFNOInversion.SYNTHESIS_INTENSITY_UNITS
end
I = cube.data[:, 1, 3, 3]; m = argmin(I)
@printf("true populations: edges %.2f %.2f, min %.2f at %.3f nm, core/edge %.3f (%s)\n", I[1], I[end], I[m], wave[m] * 1e9, I[m] / I[1], FFNOInversion.SYNTHESIS_INTENSITY_UNITS)
