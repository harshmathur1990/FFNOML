# What the EOS gives for P and rho from FALC's T and tabulated nₑ, against FALC's own values and against the HSE z route.
using HSEMHS, HDF5, Printf
R = @__DIR__; falc(n) = h5read(joinpath(R, "FALC_82_5x5.hdf5"), n)
T = Float64.(falc("temperature")[:, 3, 3, 1]); ne = Float64.(falc("electron_density")[:, 3, 3, 1]); z = Float64.(falc("z")[:, 1]) ./ 1e3
H = permutedims(h5read(joinpath(R, "falc_rh.h5"), "H_n"), (2, 1)); Pf = (vec(sum(H; dims=2)) .* 1.1 .+ ne) .* 1.380649e-23 .* T
eos = HSEMHS.load_table("/mn/stornext/d9/data/harshm/hsemhs_env112/work/e2e/inputs/eos_table_fullz.h5")
println(" z_km     T | P(T,ne)/P_FALC | nH(rho)/nH_RH | ne/ne_LTE(P_FALC)")
for k in (20, 35, 40, 45, 50, 55, 60, 65, 70, 75, 81)
    lnT, target = log(T[k]), log(ne[k]); lo, hi = -12.0, 16.0
    state(lp) = HSEMHS.state(eos, lnT, lp)
    for _ in 1:80; m = (lo + hi) / 2; n = state(m)[2]; (isfinite(n) && n < target) ? (lo = m) : (hi = m); end
    lp = (lo + hi) / 2; r, n = state(lp)
    @printf("%6.0f %6.0f | %10.3f | %10.3f | %8.3f\n", z[k], T[k], exp(lp) / Pf[k], exp(r) / eos.mass_per_H / sum(H[k, :]), ne[k] / exp(state(log(Pf[k]))[2]))
end
