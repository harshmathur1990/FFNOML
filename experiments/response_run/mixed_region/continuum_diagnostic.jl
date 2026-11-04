# Why the continuum depends on which source of a region comes first (only the first supplies it). Each package's pure
# continuum on a stored forward atmosphere: Muspel (the FFNO line with zero level populations, so no line opacity) and
# Wittmann (a Kurucz line with log gf = -20). Prints, at each region's continuum wavelengths, the emergent intensity and
# the ratio of extinction and source function by depth, plus the H-alpha wing that case A of check_mixed_region.jl had.
# usage (hyades, after ~/loadgcc.sh): julia --project=../FFNOInversion.jl response_run/mixed_region/continuum_diagnostic.jl [atmosphere.h5] [stride]
using FFNOInversion, HDF5, Libdl, Printf, Statistics
import Muspel
include(joinpath(@__DIR__, "..", "..", "FFNOInversion.jl", "scripts", "forward_atmosphere_io.jl"))
const R = "/mn/stornext/d9/data/harshm/FFNOML/response_run"
path = length(ARGS) >= 1 ? ARGS[1] : joinpath(R, "outputs", "population_tangents_s4_atmosphere.h5")
stride = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 16
atmosphere, populations = read_forward_atmosphere(path)
atmosphere = subsample_atmosphere(atmosphere, stride)
populations = Dict(k => v[:, 1:stride:end, 1:stride:end, :] for (k, v) in populations)
eos = WittmannEOS(joinpath(R, "inputs", "wittmann", "libwitt_ffno.$(Libdl.dlext)"), joinpath(R, "inputs", "pf_Kurucz.input"))
atoms = Dict(:H => joinpath(R, "inputs", "atoms", "atom.h6_tiago2.yaml"), :CA => joinpath(R, "inputs", "atoms", "atom.ca2.yaml"))
nz, nx, ny = size(atmosphere.temperature)
weak = joinpath(mktempdir(), "weak.list")
write(weak, "  630.2494-20.000 26.00   45595.083  0.0 s6D)5s e5D   29732.734  1.0 5Dsp3P z5P  7.93 -5.43 -7.59K94  0 0  0 0.000  0 0.000    0    0              0 2487\n")
lines = load_kurucz_linelist(weak).lines
logtau = atmosphere.grid.vertical.values
for (name, species, index, lo, up, air_nm) in (("H-alpha", :H, 5, 2, 3, (656.00, 656.28, 657.5)), ("Ca II", :CA, 5, 3, 5, (853.40, 854.21, 855.5)))
    println("\n== ", name)
    air = collect(air_nm) .* 1e-9
    vac = FFNOInversion._air_to_vacuum_nm.(air .* 1e9) .* 1e-9
    model = build_muspel_line_model(atmosphere, populations[:H], atoms[species], index, lo, up)
    mcont = OpacityContributor(model, zero(populations[species]))
    mfull = OpacityContributor(model, populations[species])
    wcont = OpacityContributor(KuruczLTEModel(lines, eos; include_continuum=true))
    function emergent(c)
        cube = SpectralCube(zeros(length(vac), 1, nx, ny), vac, StokesSet(:I))
        synthesize!(cube, MixedIntensitySynthesizer([c]), NonPRD(), atmosphere, populations)
        cube.data[:, 1, :, :]
    end
    Im, If, Iw = emergent(mcont), emergent(mfull), emergent(wcont)
    for l in eachindex(air)
        @printf("  %.2f nm  Muspel continuum %.3f  Muspel continuum+line %.3f  Wittmann continuum %.3f  Muspel/Wittmann mean %.4f rms %.4f\n",
            air_nm[l], mean(Im[l, :, :]), mean(If[l, :, :]), mean(Iw[l, :, :]), mean(Im[l, :, :]) / mean(Iw[l, :, :]), sqrt(mean(abs2, Im[l, :, :] ./ Iw[l, :, :] .- 1)))
    end
    chi = zeros(nz, nx, ny, 2); eta = zeros(nz, nx, ny, 2)
    for y in 1:ny, x in 1:nx
        χ = zeros(nz, 1); η = zeros(nz, 1)
        add_opacity_emissivity!(χ, η, model, vac[1:1], atmosphere, x, y, mcont.populations); chi[:, x, y, 1] = χ; eta[:, x, y, 1] = η
        χ .= 0; η .= 0
        add_opacity_emissivity!(χ, η, wcont.model, vac[1:1], atmosphere, x, y, nothing); chi[:, x, y, 2] = χ; eta[:, x, y, 2] = η
    end
    state = FFNOInversion.WittmannKuruczState(eos); fn = Libdl.dlsym(state.library_handle, :witt_kurucz_populations)
    println("  log tau500   FFNO neutral H / Wittmann neutral H (median, p5, p95)   FFNO proton / ne")
    for target in (-4, -3, -2, -1, 0, 1)
        k = argmin(abs.(logtau .- target)); ratio = Float64[]; proton = Float64[]
        for y in 1:ny, x in 1:nx
            t = [atmosphere.temperature[k, x, y]]; p = [atmosphere.pgas[k, x, y]]; lower = [0.0]; nh = [0.0]
            ccall(fn, Cint, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Cdouble, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t),
                state.backend, t, p, 26, 0, 0.0, 1.0, lower, nh, 1)
            push!(ratio, sum(populations[:H][k, x, y, 1:5]) / nh[1]); push!(proton, populations[:H][k, x, y, 6] / atmosphere.ne[k, x, y])
        end
        @printf("  %6.2f        %.3f  (%.3f, %.3f)     %.3f\n", logtau[k], median(ratio), quantile(ratio, 0.05), quantile(ratio, 0.95), median(proton))
    end
    println("  log tau500   median chi Muspel/Wittmann  (p5, p95)   median S Muspel/Wittmann   median T   ne/(Wittmann ne)")
    for target in (-4, -3, -2, -1, 0, 1)
        k = argmin(abs.(logtau .- target))
        rc = vec(chi[k, :, :, 1] ./ chi[k, :, :, 2]); rs = vec((eta[k, :, :, 1] ./ chi[k, :, :, 1]) ./ (eta[k, :, :, 2] ./ chi[k, :, :, 2]))
        wne = [continuum_state(eos, [atmosphere.temperature[k, x, y]], [atmosphere.pgas[k, x, y]]; threads=1).xne_cm3[1] * 1e6 for x in 1:nx, y in 1:ny]
        @printf("  %6.2f        %.3f  (%.3f, %.3f)            %.4f                 %6.0f   %.3f\n", logtau[k], median(rc), quantile(rc, 0.05), quantile(rc, 0.95),
            median(rs), median(atmosphere.temperature[k, :, :]), median(vec(atmosphere.ne[k, :, :] ./ wne)))
    end
end
