# FALC with its tabulated ne as the truth: option A (P, rho from HSE, ne from FALC), option B (P, rho inverted from T and ne through the
# EOS, no HSE) and A on the log tau route. The FFNO on the population mesh; neutral H in line broadening from the EOS; continuum with
# the given ne (H- and Thomson), no FFNO populations. The atmospheres are built here: no change to HSEMHS. Writes falc_ne<case>_*.h5.
# usage (eagle7, run_falc_ne_options.sh)
using FFNOInversion, HDF5, Libdl, Printf
import Muspel, HSEMHS
const F = FFNOInversion; const R = @__DIR__; const A = joinpath(R, "..", "inputs"); const TRAIN = joinpath(R, "..", "..", "training_FFNO3D_zscale_expand_lognlte")
falc(name) = h5read(joinpath(R, "FALC_82_5x5.hdf5"), name)
neF = Float64.(falc("electron_density")[:, 1, 1, 1]); nF = 82
table = HSEMHS.load_table("/mn/stornext/d9/data/harshm/hsemhs_env112/work/e2e/inputs/eos_table_fullz.h5")
function load_state(run)
    h5open(joinpath(R, "falc_$(run)_atmosphere.h5")) do f
        d(n) = permutedims(read(f[n])[1, :, :, :], (1, 3, 2))
        (T=d("temperature"), P=d("pgas"), rho=d("rho"), ne=d("ne"), z=d("z"), vturb=d("vturb"), x=read(f["x_m"]), y=read(f["y_m"]))
    end
end
function option_A(s)
    ne = copy(s.ne); nz = size(ne, 1); ne[nz-nF+1:nz, :, :] .= reshape(neF, nF, 1, 1); (; s..., ne)
end
function option_B(s)
    P, rho, ne = copy(s.P), copy(s.rho), copy(s.ne); nz, nx, ny = size(P)
    for y in 1:ny, x in 1:nx
        for (i, k) in enumerate(nz-nF+1:nz)
            lnT, target = log(s.T[k, x, y]), log(neF[i]); lo, hi = -14.0, 16.0
            for _ in 1:90; m = (lo + hi) / 2; n = HSEMHS.state(table, lnT, m)[2]; (isfinite(n) && n < target) ? (lo = m) : (hi = m); end
            lp = (lo + hi) / 2; P[k, x, y] = exp(lp); rho[k, x, y] = exp(HSEMHS.state(table, lnT, lp)[1]); ne[k, x, y] = neF[i]
        end
        top = nz - nF + 1                                              # the corona above keeps its shape, joined at FALC's top
        for k in 1:top-1; f = P[top, x, y] / s.P[top, x, y]; P[k, x, y] = s.P[k, x, y] * f; rho[k, x, y] = s.rho[k, x, y] * f; ne[k, x, y] = s.ne[k, x, y] * f; end
    end
    (; s..., P, rho, ne)
end
atmosphere_of(s) = (nz = size(s.T, 1); FFNOInversion.Atmosphere3D(Grid3D(VerticalAxis(:z_m, s.z[:, 1, 1]), s.x, s.y), s.T, zero(s.T), zero(s.T), zero(s.T), s.vturb; pgas=s.P, rho=s.rho, ne=s.ne, z=s.z))
struct GivenNeContinuum{S} <: AbstractLineOpacityModel; state::S; end
function FFNOInversion.add_opacity_emissivity!(chi, eta, m::GivenNeContinuum, wavelength, a, x, y, populations=nothing)
    T = Float64.(a.temperature[:, x, y]); P = Float64.(a.pgas[:, x, y]); ne = Float64.(a.ne[:, x, y]) .* 1e-6
    s = continuum_state(m.state, T, P; threads=1)
    for k in eachindex(T), l in eachindex(wavelength)
        partials = copy(s.partials[:, k]); partials[3] *= ne[k] / s.xne_cm3[k]                     # H- with the given ne
        c = 100 * first(continuum_extinction_cm(T[k], wavelength[l] * 1e10, s.xna_cm3[k], ne[k], partials))
        chi[k, l] += c; eta[k, l] += c * F.source_planck(wavelength[l], T[k])
    end
end
function neutral_h(s, eos)
    st = F.WittmannKuruczState(eos); fn = Libdl.dlsym(st.library_handle, :witt_kurucz_populations); hp = fill(1.0, size(s.T)..., 6); low = [0.0]; nh = [0.0]
    for I in CartesianIndices(s.T)
        ccall(fn, Cint, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Cdouble, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t), st.backend, [s.T[I]], [s.P[I]], 26, 0, 0.0, 1.0, low, nh, 1)
        hp[I, 1] = nh[1]
    end
    hp
end
metadata(sp) = PopulationMetadata(FFNO_INPUT_CHANNELS, Tuple("$sp level $i" for i in 1:6), "unrecorded-example-checkpoint")
service = launch_fsdp_service([FSDPModelSpec(sp, joinpath(TRAIN, "3D_sim_train_$sp.pt"), metadata(sp)) for sp in (:H, :CA)]; launcher=ENV["FFNO_FSDP_LAUNCHER"], timeout_seconds=900.0, diagnostics_directory=joinpath(R, "..", "diagnostics"))
eos = WittmannEOS(joinpath(A, "wittmann", "libwitt_ffno.$(Libdl.dlext)"), joinpath(A, "pf_Kurucz.input"))
atoms = Dict(:H => joinpath(A, "atoms", "atom.h6_tiago2.yaml"), :CA => joinpath(A, "atoms", "atom.ca2.yaml"))
config = PopulationMeshConfig()
regions = [(region=SpectralRegionConfig(655.8e-9, 0.002e-9, 701, 1.0, :none, nothing, SpectralSourceConfig[]), sp=:H, a=(5, 2, 3), list="kurucz_halpha_blend.list"),
           (region=SpectralRegionConfig(853.4e-9, 0.002e-9, 601, 1.0, :none, nothing, SpectralSourceConfig[]), sp=:CA, a=(5, 3, 5), list="kurucz_8542_blend.list")]
scale = output_intensity_scale([r.region for r in regions], :si)
try
    for (name, s) in (("A", option_A(load_state("zextc"))), ("B", option_B(load_state("zextc"))), ("Atau", option_A(load_state("ltauc"))))
        a = atmosphere_of(s); nz, nx, ny = size(s.T); pops = Dict{Symbol,Array{Float64,4}}()
        for sp in (:H, :CA)
            model = MeshPopulationModel(FSDPFFNOModel(service, sp, metadata(sp), sp === :H ? 0 : 1), config, table, 274.0; native_z=true)
            p = zeros(nz, nx, ny, 6); predict_populations!(p, model, a); pops[sp] = p
        end
        # synthesis on FALC's levels
        i = nz-nF+1:nz; sub = (; s..., T=s.T[i, :, :], P=s.P[i, :, :], rho=s.rho[i, :, :], ne=s.ne[i, :, :], z=s.z[i, :, :], vturb=s.vturb[i, :, :])
        b = atmosphere_of(sub); hp = neutral_h(sub, eos); level = Dict(k => v[i, :, :, :] for (k, v) in pops)
        spectra = Float64[]; wave = Float64[]
        for r in regions
            air = wavelengths(r.region); vac = F._air_to_vacuum_nm.(air .* 1e9) .* 1e-9
            line = build_muspel_line_model(b, hp, atoms[r.sp], r.a...; include_continuum=false)
            lines = load_kurucz_linelist(joinpath(R, r.list)).lines
            synth = MixedIntensitySynthesizer([OpacityContributor(GivenNeContinuum(F.WittmannKuruczState(eos))), OpacityContributor(line, level[r.sp]), OpacityContributor(KuruczLTEModel(lines, eos; include_continuum=false))])
            c = SpectralCube(zeros(length(air), 1, nx, ny), vac, StokesSet(:I)); synthesize!(c, synth, NonPRD(), b, level)
            append!(spectra, vec(c.data[:, 1, 3, 3])); append!(wave, air)
        end
        h5open(joinpath(R, "falc_ne$(name)_synthesis.h5"), "w") do f
            f["wavelength_m"] = wave; f["intensity_center"] = spectra .* scale
            f["P"] = sub.P[:, 3, 3]; f["rho"] = sub.rho[:, 3, 3]; f["ne"] = sub.ne[:, 3, 3]
            f["H"] = level[:H][:, 3, 3, :]; f["CA"] = level[:CA][:, 3, 3, :]
        end
        println("done ", name); flush(stdout)
    end
finally
    close_fsdp_service!(service)
end
