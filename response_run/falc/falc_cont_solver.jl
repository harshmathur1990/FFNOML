# The same RH chi and S through our formal solver, on the 82 levels and refined; our continuum through it, with and without Muspel's H2+ term, against RH's emergent intensity.
using FFNOInversion, HDF5, Printf
const F = FFNOInversion; const R = @__DIR__
d = h5open(joinpath(R, "falc_rh_chi.h5"), "r") do f; (air=read(f["air_nm"]), I=read(f["I_rh"]), chi=permutedims(read(f["chi"])), S=permutedims(read(f["S"]))); end
o = h5open(joinpath(R, "falc_cont_ours.h5"), "r") do f; (z=read(f["z"]), T=read(f["T"]), given=read(f["given"]), witt=read(f["witt"]), terms=read(f["terms"]), vac=read(f["vac_nm"])); end
sz = size(d.chi); @assert sz[1] == 82 "chi $(sz)"; given = size(o.given,1)==82 ? o.given : permutedims(o.given); witt = size(o.witt,1)==82 ? o.witt : permutedims(o.witt)
terms = size(o.terms,1)==82 ? o.terms : permutedims(o.terms,(3,2,1))
solver = ScalarFormalSolver(); C = 299792458.0
planck(v, t) = 2 * 6.62607015e-34 * C / v^3 / expm1(6.62607015e-34 * C / (v * 1.380649e-23 * t))
function solve(chi, S, z)
    out = zeros(1); F.formal_solve!(out, solver, reshape(copy(chi), :, 1), reshape(chi .* S, :, 1), z); out[1]
end
function refine(chi, S, z, m)
    zf = Float64[]; cf = Float64[]; sf = Float64[]
    for k in 1:length(z)-1, q in 0:m-1
        t = q / m; push!(zf, z[k] + t * (z[k+1] - z[k])); push!(cf, exp((1-t)*log(chi[k]) + t*log(chi[k+1]))); push!(sf, exp((1-t)*log(S[k]) + t*log(S[k+1])))
    end
    push!(zf, z[end]); push!(cf, chi[end]); push!(sf, S[end]); solve(cf, sf, zf)
end
@printf("%8s %9s | RHχ,S: %6s %6s %6s | %8s %8s %8s %8s\n", "air nm", "I_RH 1e-8", "82 lev", "x4", "x32", "gχ,B", "g-H2+,B", "wχ,B", "RHχ,B")
for j in eachindex(d.air)
    z = o.z; c = d.chi[:, j]; S = d.S[:, j]; B = planck.(o.vac[j] * 1e-9, o.T); r = d.I[j]
    @printf("%8.2f %9.4f |        %6.3f %6.3f %6.3f | %8.3f %8.3f %8.3f %8.3f\n", d.air[j], r * 1e8, solve(c, S, z)/r, refine(c, S, z, 4)/r, refine(c, S, z, 32)/r,
        solve(given[:, j], B, z)/r, solve(given[:, j] .- terms[:, j, 4], B, z)/r, solve(witt[:, j], B, z)/r, solve(c, B, z)/r)
end

# H2+: Muspel's free-free and bound-free separately (RH: free-free only, Bates 1952), with RH's table at the same depth
using Unitful; import Muspel
println("\nair nm  T(k at tau_RH=1) | H2+ ff, bf [1e-49 m^-1 per n_H n_p] | RH ff table | I/I_RH: Muspel ff+bf, ff only, no H2+")
for j in eachindex(d.air)
    d.air[j] < 390 || d.air[j] > 800 && continue
    z = o.z; c = d.chi[:, j]; dt = 0.5 .* (c[2:end] .+ c[1:end-1]) .* abs.(diff(z)); tau = vcat(1e-12, 1e-12 .+ cumsum(dt)); k = argmin(abs.(tau .- 1)); r = d.I[j]
    λ = o.vac[j] * u"nm"; ffbf(t) = (ustrip(u"m^5", Muspel.σ_h2plus_ff(λ, t * u"K")), ustrip(u"m^5", Muspel.σ_h2plus_bf(λ, t * u"K")))
    ff, bf = ffbf(o.T[k]); B = planck.(o.vac[j] * 1e-9, o.T)
    fraction = [(p = ffbf(t); p[1] / (p[1] + p[2])) for t in o.T]
    gff = given[:, j] .- terms[:, j, 4] .* (1 .- fraction)
    rhtab = d.air[j] < 470 ? 0.325 : d.air[j] < 700 ? 0.42 : 0.61
    @printf("%7.2f %6.0f | %6.3f %6.3f | ~%.2f | %6.3f %6.3f %6.3f\n", d.air[j], o.T[k], ff * 1e49, bf * 1e49, rhtab, solve(given[:, j], B, z)/r, solve(gff, B, z)/r, solve(given[:, j] .- terms[:, j, 4], B, z)/r)
end
