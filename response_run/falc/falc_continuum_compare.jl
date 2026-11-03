# This code's opacity and source function on FALC with RH's populations and FALC's ne (falc_rh_populations.jl's atmosphere), at the
# wavelengths where RH's solveray wrote chi and S (compare_continuum_rh.py). Writes falc_continuum_compare.h5.
using FFNOInversion, HDF5, Libdl
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
air = [657.1996859357996, 655.9996873653901, 853.3994585203535, 854.5994571707921]
vac = FFNOInversion._air_to_vacuum_nm.(air) .* 1e-9
model = KuruczLTEModel(KuruczLine{Float64}[], eos; include_continuum=true)
atoms = Dict(:H => joinpath(A, "atoms", "atom.h6_tiago2.yaml"), :CA => joinpath(A, "atoms", "atom.ca2.yaml"))
ha = build_muspel_line_model(atmosphere, pops[:H], atoms[:H], 5, 2, 3; include_continuum=false)
ca = build_muspel_line_model(atmosphere, pops[:H], atoms[:CA], 5, 3, 5; include_continuum=false)
c = zeros(nz, 4); e = zeros(nz, 4); add_opacity_emissivity!(c, e, model, vac, atmosphere, 3, 3)
cl = copy(c); el = copy(e)
add_opacity_emissivity!(@view(cl[:, 1:2]), @view(el[:, 1:2]), ha, vac[1:2], atmosphere, 3, 3, pops[:H])
add_opacity_emissivity!(@view(cl[:, 3:4]), @view(el[:, 3:4]), ca, vac[3:4], atmosphere, 3, 3, pops[:CA])
s = continuum_state(eos, temperature[:, 3, 3], pgas[:, 3, 3]; threads=1)
h5open(joinpath(R, "falc_continuum_compare.h5"), "w") do f
    f["air_nm"] = air; f["vacuum_m"] = vac; f["chi_continuum"] = c; f["eta_continuum"] = e; f["chi_total"] = cl; f["eta_total"] = el
    f["xne_cm3"] = s.xne_cm3; f["rho_eos"] = s.rho_kg_m3; f["pgas"] = pgas[:, 3, 3]; f["z"] = z; f["ne"] = ne[:, 3, 3]
end
println("done")
