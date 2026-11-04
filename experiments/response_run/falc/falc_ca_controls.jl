# Why the FFNO Ca II 854.2 nm differs from RH on FALC: the same FFNOInversion synthesis on FALC's heights and ne (with vturb) with
# H and Ca populations from RH or the FFNO (z route with the corona, falc_zext), and with atom.ca2's Unsold broadening or RH's ABO
# (sigma 291, alpha 0.275). Writes falc_ca_controls.h5; prints against RH.
# usage (hyades): julia --project=../FFNOInversion.jl response_run/falc/falc_ca_controls.jl
using FFNOInversion, HDF5, Libdl, Printf
import Muspel
const R = joinpath(@__DIR__); const A = joinpath(R, "..", "inputs")
falc(name) = h5read(joinpath(R, "FALC_82_5x5.hdf5"), name)
T3(a) = Float64.(a[:, :, :, 1])
temperature = T3(falc("temperature")); ne = T3(falc("electron_density")); vturb = T3(falc("velocity_turbulent")); nz, nx, ny = size(temperature)
z = Float64.(falc("z")[:, 1]); zero3 = zeros(nz, nx, ny)
# RH outputs (depth, level) as h5py wrote them read back reversed: HDF5.jl gives (level, depth)
rh(name) = permutedims(h5read(joinpath(R, "falc_rh.h5"), name), (2, 1))
cube(p) = (out = zeros(nz, nx, ny, 6); for y in 1:ny, x in 1:nx; out[:, x, y, :] .= p; end; out)
rhH, rhCa = cube(rh("H_n")), cube(rh("CA_n"))
function ffno(species)    # zext atmosphere file: populations (level, z, y, x) in HDF5.jl order; FALC's levels are the last 82
    p = h5read(joinpath(R, "falc_zext_atmosphere.h5"), "populations/$species")
    cube(permutedims(p[:, end-nz+1:end, 3, 3], (2, 1)))
end
fH, fCa = ffno("H"), ffno("CA")
pgas = (dropdims(sum(rhH; dims=4); dims=4) .* 1.1 .+ ne) .* 1.380649e-23 .* temperature
grid = Grid3D(VerticalAxis(:z_m, z), 48e3 .* (0.0:4.0), 48e3 .* (0.0:4.0))
atmosphere = FFNOInversion.Atmosphere3D(grid, temperature, copy(zero3), copy(zero3), copy(zero3), vturb;
    pgas=pgas, rho=pgas .* 0, ne=ne, z=repeat(reshape(z, :, 1, 1), 1, nx, ny))
eos = WittmannEOS(joinpath(A, "wittmann", "libwitt_ffno.$(Libdl.dlext)"), joinpath(A, "pf_Kurucz.input"))
unsold = joinpath(A, "atoms", "atom.ca2.yaml")
text = read(unsold, String)
abo = joinpath(mktempdir(), "atom.ca2.abo.yaml")
write(abo, replace(text, r"(transition: \[lev5, lev3\].*?)\{type: VanderWaals_Unsold[^\n]*"s => s"\1{type: VanderWaals_ABO, σ: {value: 291.0}, α: {value: 0.275}}"))
occursin("VanderWaals_ABO", read(abo, String)) || error("ABO substitution failed")
si = SpectralSourceConfig(:kurucz_lte, joinpath(R, "kurucz_8542_blend.list"))
ca = SpectralSourceConfig(:ffno, nothing)
region = SpectralRegionConfig(853.4e-9, 0.002e-9, 601, 1.0, :none, nothing, [ca, si])
cases = (("RH H, RH Ca, Unsold", rhH, rhCa, unsold), ("RH H, RH Ca, ABO", rhH, rhCa, abo),
         ("FFNO H, RH Ca, ABO", fH, rhCa, abo), ("RH H, FFNO Ca, ABO", rhH, fCa, abo), ("FFNO H, FFNO Ca, ABO", fH, fCa, abo))
scale = output_intensity_scale([region], :si)
results = Dict{String,Vector{Float64}}()
for (name, H, Ca, atom) in cases
    pops = Dict(:H => H, :CA => Ca)
    setup = build_synthesis_setup((regions=[region],), atmosphere, pops, eos; atom_files=Dict(:H => joinpath(A, "atoms", "atom.h6_tiago2.yaml"), :CA => atom))
    wave = setup[1].wavelength_air_m
    c = SpectralCube(zeros(length(wave), 1, nx, ny), wave, StokesSet(:I))
    synthesize!(c, setup, NonPRD(), atmosphere, pops, ThreadedSynthesisCache(Float64, nz, length(wave)))
    I = c.data[:, 1, 3, 3] .* scale .* 1e8; results[name] = I
    w = wave .* 1e9; at(x) = argmin(abs.(w .- x)); m = argmin(I)
    @printf("%-24s edges %.3f %.3f | I(854.0) %.3f I(854.4) %.3f | core %.3f at %.3f\n", name, I[1], I[end], I[at(854.0)], I[at(854.4)], I[m], w[m])
end
h5open(joinpath(R, "falc_ca_controls.h5"), "w") do f
    f["wavelength_air_nm"] = collect(853.4 .+ 0.002 .* (0:600))
    for (k, v) in results; f[k] = v; end
end
