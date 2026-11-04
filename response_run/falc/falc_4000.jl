# Absolute continuum intensity near 4000 Å, FALC_82 given (nₑ, H populations), no FFNO: our synthesis at RH's wavelengths, against RH (falc_rh_air.h5).
# usage (hyades, from response_run): julia --project=../FFNOInversion.jl falc/falc_4000.jl
using FFNOInversion, HDF5, Libdl, Printf
import Muspel
const F = FFNOInversion; const R = @__DIR__
rl = h5read(joinpath(R, "falc_rh_air.h5"), "wavelength_air_nm"); rI = h5read(joinpath(R, "falc_rh_air.h5"), "intensity"); q = sortperm(rl); rl, rI = rl[q], rI[q]
eos = WittmannEOS("/mn/stornext/d9/data/harshm/hsemhs_env112/work/e2e/inputs/wittmann/libwitt_ffno.$(Libdl.dlext)", "/mn/stornext/d9/data/harshm/hsemhs_env112/work/e2e/inputs/pf_Kurucz.input")
targets = [minimum(rl[rl .> 399.8]), minimum(rl[rl .> 401.0]), minimum(rl[rl .> 402.7]), minimum(rl[rl .> 422.0])]
context = serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
for (axis, file) in (("log column mass", "forward_falc_givenm.toml"), ("log tau500", "forward_falc_giventau.toml"), ("height", "forward_falc_givenz.toml"))
    config = load_config(joinpath(R, file)); a0 = read_inversion_inputs(config).atmosphere
    d = distribute_atmosphere(Float64, a0, context); b = prepare_equilibrium(config.equilibrium, d, context); reconstruct_equilibrium!(b, d, context); a = d.local_atmosphere
    println("--- input grid: ", axis)
    @printf("%10s %12s %12s %8s %12s %8s\n", "air nm", "RH [SI/Hz]", "given H,ne", "/RH-1", "EOS (T,P)", "/RH-1")
    for λa in targets
        v = F._air_to_vacuum_nm(λa) * 1e-9; scale = 1e12 * v^2 / 299792458.0
        function run(model)
            cube = SpectralCube(zeros(1, 1, 5, 5), [v], StokesSet(:I))
            synthesize!(cube, MixedIntensitySynthesizer([OpacityContributor(model)]), NonPRD(), a); cube.data[1, 1, 3, 3] * scale
        end
        Rv = rI[argmin(abs.(rl .- λa))]; Ig = run(F.build_muspel_continuum_model(v * 1e9)); Ie = run(KuruczLTEModel(KuruczLine{Float64}[], eos; include_continuum=true))
        @printf("%10.4f %12.4e %12.4e %+7.2f%% %12.4e %+7.2f%%\n", λa, Rv, Ig, 100 * (Ig / Rv - 1), Ie, 100 * (Ie / Rv - 1))
    end
end
