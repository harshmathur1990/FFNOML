# Kurucz LTE lines inside FFNO line regions (FFNOInversion build_synthesis_setup): the FFNO line (Muspel) and the Kurucz
# lines (Wittmann) add opacity and emissivity in one unit (kW m^-3 sr^-1 nm^-1) before one formal solution, and a Kurucz
# line that is the FFNO transition itself is dropped. On a stored forward atmosphere with H and Ca populations (every
# `stride`th column), for H-alpha with Fe I 656.92 nm and for Ca II 854.2 nm with Si I 853.62 nm:
#   A  FFNO only (Muspel continuum)               B  Kurucz blend only (Wittmann continuum)
#   C  FFNO + Kurucz (Muspel continuum)           D  Kurucz + FFNO (Wittmann continuum)
# The Kurucz list of C and D also holds the FFNO line itself, which must be dropped. Prints the continuum of A against B
# (two continuum packages, one unit), C against A at the FFNO line core (an undropped LTE copy would change it), the
# blend's depth in B and C, and C against D.
# usage (hyades, after ~/loadgcc.sh):
#   julia --project=FFNOInversion.jl response_run/mixed_region/check_mixed_region.jl [atmosphere.h5] [stride]
using FFNOInversion, HDF5, Libdl, Printf
import Muspel
include(joinpath(@__DIR__, "..", "..", "FFNOInversion.jl", "scripts", "forward_atmosphere_io.jl"))

const R = "/mn/stornext/d9/data/harshm/FFNOML/response_run"
path = length(ARGS) >= 1 ? ARGS[1] : joinpath(R, "outputs", "population_tangents_s4_atmosphere.h5")
stride = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 16
atmosphere, populations = read_forward_atmosphere(path)
atmosphere = subsample_atmosphere(atmosphere, stride)
populations = Dict(k => v[:, 1:stride:end, 1:stride:end, :] for (k, v) in populations)
eos = WittmannEOS(joinpath(R, "inputs", "wittmann", "libwitt_ffno.$(Libdl.dlext)"), joinpath(R, "inputs", "pf_Kurucz.input"))
atom_files = Dict(:H => joinpath(R, "inputs", "atoms", "atom.h6_tiago2.yaml"), :CA => joinpath(R, "inputs", "atoms", "atom.ca2.yaml"))
nz, nx, ny = size(atmosphere.temperature)

# K94 records (fixed columns). Blends: Fe I 656.9214 nm and Si I 853.6164 nm (zero damping constants: RH's defaults).
# FFNO lines: H-alpha (H I n = 2-3 at 82259.158 and 97492.304 cm^-1, log gf 0.710) and Ca II 3d 2D5/2 - 4p 2P3/2.
const K94 = (
    fe = "  656.9214 -0.420 26.00   53393.669  4.0 (4F)4d g5F   38175.352  3.0 (4F)4p y3D  8.36 -4.38 -7.70FMW  0 0  0 0.000  0 0.000    0    0           1287 1324",
    si = "  853.6164 -0.910 14.00   49850.830  2.0 p3d  3F      61562.477  3.0 p5f'[7]     0.00  0.00  0.00KP  9999  0 0.000  0 0.000    0    0              0    0",
    h  = "  656.2797  0.710  1.00   97492.304  1.5 3p 2P        82259.158  0.5 2s 2S       8.79 -4.50 -7.60K94  0 0  0 0.000  0 0.000    0    0              0 2487",
    ca = "  854.2091 -0.463 20.01   25414.400  1.5 4p 2P        13710.880  2.5 3d 2D       8.21 -5.53 -7.66K94  0 0  0 0.000  0 0.000    0    0              0    0")
regions = (
    (name="H-alpha", species=:H, line=:halpha, ffno=:h, blend=:fe, start=656.00e-9, count=601, blend_window=(656.88e-9, 656.96e-9)),
    (name="Ca II 854.2", species=:CA, line=:ca_ii_8542, ffno=:ca, blend=:si, start=853.40e-9, count=601, blend_window=(853.58e-9, 853.66e-9)))

dir = mktempdir()
rel(a, b) = sqrt(sum(abs2, a ./ b .- 1) / length(a))
mean_profile(I) = vec(sum(I; dims=(2, 3))) ./ (nx * ny)
h5open(joinpath(R, "mixed_region", "mixed_region.h5"), "w") do out
for r in regions
    blend_only = joinpath(dir, "$(r.blend).list"); write(blend_only, K94[r.blend] * "\n")
    with_ffno = joinpath(dir, "$(r.blend)_$(r.ffno).list"); write(with_ffno, K94[r.ffno] * "\n" * K94[r.blend] * "\n")
    ffno = SpectralSourceConfig(:ffno, r.species, r.line, nothing)
    kurucz(file) = SpectralSourceConfig(:kurucz_lte, nothing, nothing, file)
    cases = (A=[ffno], B=[kurucz(blend_only)], C=[ffno, kurucz(with_ffno)], D=[kurucz(with_ffno), ffno])
    println("\n== ", r.name)
    I = Dict{Symbol,Array{Float64,3}}(); wave = Float64[]
    for (name, sources) in pairs(cases)
        setup = build_synthesis_setup((regions=[SpectralRegionConfig(r.start, 0.002e-9, r.count, 1.0, :none, nothing, sources)],),
            atmosphere, populations, eos; atom_files)
        wave = setup[1].wavelength_air_m
        cube = SpectralCube(zeros(Float64, length(wave), 1, nx, ny), wave, StokesSet(:I))
        synthesize!(cube, setup, NonPRD(), atmosphere, populations, ThreadedSynthesisCache(Float64, nz, length(wave)))
        I[name] = cube.data[:, 1, :, :]
        println("  ", name, ": ", join([c.model isa KuruczLTEModel ?
            "Kurucz $(join(round.([l.wavelength0_m * 1e9 for l in c.model.lines]; digits=3), ", ")) nm (vacuum)" :
            string(typeof(c.model).name.name) for c in setup[1].synthesizer.contributors], " + "))
    end
    at(λ) = argmin(abs.(wave .- λ))
    edge = 1
    @printf("  units %s, %d columns\n", FFNOInversion.SYNTHESIS_INTENSITY_UNITS, nx * ny)
    @printf("  continuum %.2f nm: A (Muspel) %.3f  B (Wittmann) %.3f  rms A/B-1 %.4f\n", wave[edge] * 1e9,
        mean_profile(I[:A])[edge], mean_profile(I[:B])[edge], rel(I[:A][edge, :, :], I[:B][edge, :, :]))
    core = argmin(mean_profile(I[:A]))
    @printf("  FFNO core %.4f nm: A %.3f  C %.3f  rms C/A-1 %.2e\n", wave[core] * 1e9, mean_profile(I[:A])[core],
        mean_profile(I[:C])[core], rel(I[:C][core, :, :], I[:A][core, :, :]))
    window = at(r.blend_window[1]):at(r.blend_window[2])
    bmin(name) = window[argmin(mean_profile(I[name])[window])]
    @printf("  blend minimum %.4f nm: depth B %.3f (against B's continuum)  depth C %.3f (against A)\n", wave[bmin(:C)] * 1e9,
        1 - mean_profile(I[:B])[bmin(:B)] / mean_profile(I[:B])[edge], 1 - mean_profile(I[:C])[bmin(:C)] / mean_profile(I[:A])[bmin(:C)])
    @printf("  C vs D (continuum from Muspel vs Wittmann): rms %.4f   max |C/D-1| %.4f\n", rel(I[:C], I[:D]), maximum(abs.(I[:C] ./ I[:D] .- 1)))
    g = create_group(out, string(r.line)); g["wavelength_air_m"] = wave
    for (k, v) in I; g[string(k)] = v; attributes(g[string(k)])["units"] = FFNOInversion.SYNTHESIS_INTENSITY_UNITS; end
end
end
