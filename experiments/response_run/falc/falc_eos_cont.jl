# The continuum as an ordinary run has it, Muspel's with hydrogen from the EOS at (T, P), against the old Wittmann continuum and RH, on FALC's heights with its P.
using FFNOInversion, HDF5, Libdl, Printf
import Muspel
const F = FFNOInversion; const R = @__DIR__
d = h5open(joinpath(R, "falc_rh_chi.h5"), "r") do f; (air=read(f["air_nm"]), I=read(f["I_rh"])); end
config = load_config(joinpath(R, "forward_falc_givenz.toml")); a0 = read_inversion_inputs(config).atmosphere
context = serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
g = distribute_atmosphere(Float64, a0, context); reconstruct_equilibrium!(prepare_equilibrium(config.equilibrium, g, context), g, context); a = g.local_atmosphere
eos = WittmannEOS("/mn/stornext/d9/data/harshm/hsemhs_env112/work/e2e/inputs/wittmann/libwitt_ffno.$(Libdl.dlext)", "/mn/stornext/d9/data/harshm/hsemhs_env112/work/e2e/inputs/pf_Kurucz.input"); s = WittmannKuruczState(eos)
plain = F.Atmosphere3D(a.grid, a.temperature, a.vx, a.vy, a.vz, a.vturb; pgas=a.pgas, rho=a.rho, ne=a.ne, z=a.z)       # no given hydrogen: the EOS's
T = a.temperature[:, 3, 3]; P = a.pgas[:, 3, 3]; hi = F.eos_neutral_hydrogen!(zeros(length(T)), s, T, P); hp = F.eos_protons(s, T, P)
@printf("n_H0 EOS / given: %.3f..%.3f; protons EOS / given at tau~1 region k=60,70,78: %s\n", extrema(hi ./ a.h_neutral[:, 3, 3])..., round.((hp ./ a.h_proton[:, 3, 3])[[60, 70, 78]]; digits=3))
@printf("%9s %10s | %9s %9s %9s\n", "air nm", "I_RH 1e-8", "M+EOS/RH", "M+given", "Witt/RH")
for (j, λa) in enumerate(d.air)
    v = F._air_to_vacuum_nm(λa) * 1e-9; scale = 1e12 * v^2 / 299792458.0
    run(m, at) = (c = SpectralCube(zeros(1, 1, 5, 5), [v], StokesSet(:I)); synthesize!(c, MixedIntensitySynthesizer([OpacityContributor(m)]), NonPRD(), at); c.data[1, 1, 3, 3] * scale)
    @printf("%9.2f %10.4f | %9.4f %9.4f %9.4f\n", λa, d.I[j] * 1e8, run(build_muspel_continuum_model(v * 1e9; eos=s), plain) / d.I[j], run(build_muspel_continuum_model(v * 1e9), a) / d.I[j], run(KuruczLTEModel(KuruczLine{Float64}[], eos; include_continuum=true), plain) / d.I[j])
end
