# RH's own H and Ca II populations (pops.H.out, pops.CA.out, in falc_rh.h5) and FALC's ne in FFNOInversion's solver, in place of the
# FFNO's: H-alpha + Fe I and Ca II + Si I, default EOS continuum, with vturb. What differs from RH's spectrum then is our solver and
# atoms, not the FFNO. Writes falc_rhpops_synthesis.h5 (same layout as the runs).
# usage (hyades): julia --project=../FFNOInversion.jl response_run/falc/falc_rh_populations.jl
using FFNOInversion, HDF5, Libdl, Printf
import Muspel
const R = joinpath(@__DIR__); const A = joinpath(R, "..", "inputs")
falc(name) = h5read(joinpath(R, "FALC_82_5x5.hdf5"), name); T3(a) = Float64.(a[:, :, :, 1])
temperature, ne, vturb = T3(falc("temperature")), T3(falc("electron_density")), T3(falc("velocity_turbulent")); nz, nx, ny = size(temperature)
z = Float64.(falc("z")[:, 1]); zero3 = zeros(nz, nx, ny)
function populations(name)
    p = permutedims(h5read(joinpath(R, "falc_rh.h5"), name), (2, 1)); out = zeros(nz, nx, ny, 6)
    for y in 1:ny, x in 1:nx; out[:, x, y, :] .= p; end
    out
end
pops = Dict(:H => populations("H_n"), :CA => populations("CA_n"))
pgas = (dropdims(sum(pops[:H]; dims=4); dims=4) .* 1.1 .+ ne) .* 1.380649e-23 .* temperature
atmosphere = FFNOInversion.Atmosphere3D(Grid3D(VerticalAxis(:z_m, z), 48e3 .* (0.0:4.0), 48e3 .* (0.0:4.0)), temperature, copy(zero3), copy(zero3), copy(zero3), vturb;
    pgas=pgas, rho=pgas .* 0, ne=ne, z=repeat(reshape(z, :, 1, 1), 1, nx, ny))
eos = WittmannEOS(joinpath(A, "wittmann", "libwitt_ffno.$(Libdl.dlext)"), joinpath(A, "pf_Kurucz.input"))
ffno = SpectralSourceConfig(:ffno, nothing); kurucz(f) = SpectralSourceConfig(:kurucz_lte, joinpath(R, f))
regions = [SpectralRegionConfig(655.8e-9, 0.002e-9, 701, 1.0, :none, nothing, [ffno, kurucz("kurucz_halpha_blend.list")]),
           SpectralRegionConfig(853.4e-9, 0.002e-9, 601, 1.0, :none, nothing, [ffno, kurucz("kurucz_8542_blend.list")])]
setup = build_synthesis_setup((regions=regions,), atmosphere, pops, eos; atom_files=Dict(:H => joinpath(A, "atoms", "atom.h6_tiago2.yaml"), :CA => joinpath(A, "atoms", "atom.ca2.yaml")))
wave = vcat(setup[1].wavelength_air_m, setup[2].wavelength_air_m)
c = SpectralCube(zeros(length(wave), 1, nx, ny), wave, StokesSet(:I))
synthesize!(c, setup, NonPRD(), atmosphere, pops, ThreadedSynthesisCache(Float64, nz, length(wave)))
h5open(joinpath(R, "falc_rhpops_synthesis.h5"), "w") do f
    f["wavelength_m"] = wave; f["intensity"] = FFNOInversion._with_time_slyx(c.data .* output_intensity_scale(regions, :si))
end
println("done")
