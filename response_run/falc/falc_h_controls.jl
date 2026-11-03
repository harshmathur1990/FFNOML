# H-alpha with RH's own H populations in FFNOInversion's solver, FALC heights and ne, with vturb: what is left is ours, not the FFNO's.
using FFNOInversion, HDF5, Libdl, Printf
import Muspel
const R = joinpath(@__DIR__); const A = joinpath(R, "..", "inputs")
falc(name) = h5read(joinpath(R, "FALC_82_5x5.hdf5"), name); T3(a) = Float64.(a[:, :, :, 1])
temperature, ne, vturb = T3(falc("temperature")), T3(falc("electron_density")), T3(falc("velocity_turbulent")); nz, nx, ny = size(temperature)
z = Float64.(falc("z")[:, 1]); zero3 = zeros(nz, nx, ny); rh = permutedims(h5read(joinpath(R, "falc_rh.h5"), "H_n"), (2, 1))
H = zeros(nz, nx, ny, 6); for y in 1:ny, x in 1:nx; H[:, x, y, :] .= rh; end
pgas = (dropdims(sum(H; dims=4); dims=4) .* 1.1 .+ ne) .* 1.380649e-23 .* temperature
atmosphere = FFNOInversion.Atmosphere3D(Grid3D(VerticalAxis(:z_m, z), 48e3 .* (0.0:4.0), 48e3 .* (0.0:4.0)), temperature, copy(zero3), copy(zero3), copy(zero3), vturb;
    pgas=pgas, rho=pgas .* 0, ne=ne, z=repeat(reshape(z, :, 1, 1), 1, nx, ny))
eos = WittmannEOS(joinpath(A, "wittmann", "libwitt_ffno.$(Libdl.dlext)"), joinpath(A, "pf_Kurucz.input"))
region = SpectralRegionConfig(655.8e-9, 0.002e-9, 701, 1.0, :none, nothing, [SpectralSourceConfig(:ffno, :H, :halpha, nothing), SpectralSourceConfig(:kurucz_lte, nothing, nothing, joinpath(R, "kurucz_halpha_blend.list"))])
pops = Dict(:H => H); setup = build_synthesis_setup((regions=[region],), atmosphere, pops, eos; atom_files=Dict(:H => joinpath(A, "atoms", "atom.h6_tiago2.yaml")))
wave = setup[1].wavelength_air_m; c = SpectralCube(zeros(length(wave), 1, nx, ny), wave, StokesSet(:I))
synthesize!(c, setup, NonPRD(), atmosphere, pops, ThreadedSynthesisCache(Float64, nz, length(wave)))
I = c.data[:, 1, 3, 3] .* output_intensity_scale([region], :si) .* 1e8; w = wave .* 1e9; m = argmin(I)
@printf("RH H in our solver: edges %.3f %.3f core %.3f at %.3f; FWHM-ish width at I=2: %.3f nm\n", I[1], I[end], I[m], w[m], count(<(2.0), I) * 0.002)
h5open(joinpath(R, "falc_h_controls.h5"), "w") do f; f["wavelength_air_nm"] = w; f["intensity"] = I; end
