# Our continuum by depth at RH's solveray wavelengths on RH's heights (FALC_82, given nₑ and H populations): Muspel's terms separately, and Wittmann's.
using FFNOInversion, HDF5, Libdl, Printf, Unitful
import Muspel
const F = FFNOInversion; const R = @__DIR__
air = parse.(Float64, split(ARGS[1], ","))
config = load_config(joinpath(R, "forward_falc_givenz.toml")); a0 = read_inversion_inputs(config).atmosphere
context = serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
d = distribute_atmosphere(Float64, a0, context); reconstruct_equilibrium!(prepare_equilibrium(config.equilibrium, d, context), d, context); a = d.local_atmosphere
eos = WittmannEOS("/mn/stornext/d9/data/harshm/hsemhs_env112/work/e2e/inputs/wittmann/libwitt_ffno.$(Libdl.dlext)", "/mn/stornext/d9/data/harshm/hsemhs_env112/work/e2e/inputs/pf_Kurucz.input")
nz = nlevels(a.grid); T = a.temperature[:, 3, 3]; ne = a.ne[:, 3, 3]; hi = a.h_neutral[:, 3, 3]; hp = a.h_proton[:, 3, 3]
out = Dict{String,Any}("air_nm" => air, "T" => T, "z" => a.z[:, 3, 3], "ne" => ne, "h_neutral" => hi, "h_proton" => hp, "pgas" => a.pgas[:, 3, 3])
terms = zeros(nz, length(air), 6); given = zeros(nz, length(air)); witt = zeros(nz, length(air)); vac = F._air_to_vacuum_nm.(air)
wmodel = KuruczLTEModel(KuruczLine{Float64}[], eos; include_continuum=true)
for (j, v) in enumerate(vac)
    itp = F.build_muspel_continuum_model(v).continuum
    for k in 1:nz
        lt = Float32(log10(T[k])); ln = Float32(log10(ne[k])); H = Float32(hi[k] + hp[k])
        terms[k, j, 1] = itp.σ_atoms(lt, ln) * H; terms[k, j, 2] = itp.σ_hminus(lt) * Float32(ne[k]) * Float32(hi[k])
        terms[k, j, 3] = itp.σ_h_ff(lt) * Float32(ne[k]) * Float32(hp[k]); terms[k, j, 4] = itp.σ_h2plus(lt) * Float32(hp[k]) * Float32(hi[k])
        terms[k, j, 5] = ustrip(Muspel.σ_rayleigh_h(itp.λ * u"nm")) * Float32(hi[k]); terms[k, j, 6] = Muspel.σ_THOMSON * Float32(ne[k])
        given[k, j] = Muspel.α_cont(itp, Float32(T[k]), Float32(ne[k]), Float32(hi[k]), Float32(hp[k]))
    end
    c = zeros(nz, 1); e = zeros(nz, 1); add_opacity_emissivity!(c, e, wmodel, [v * 1e-9], a, 3, 3); witt[:, j] = c[:, 1]
end
out["vac_nm"] = vac; out["terms"] = terms; out["given"] = given; out["witt"] = witt
h5open(joinpath(R, "falc_cont_ours.h5"), "w") do f; for (k, v) in out; f[k] = v; end; end
println("written")
