# Cost of the Wittmann continuum per column (forward and adjoint) on 95 levels and 88 wavelengths.
using FFNOInversion, Libdl, Printf
R = joinpath(@__DIR__, ".."); eos = WittmannEOS(joinpath(R, "inputs", "wittmann", "libwitt_ffno.$(Libdl.dlext)"), joinpath(R, "inputs", "pf_Kurucz.input"))
nz, nl = 95, 88; shape = (nz, 1, 1); f(v) = reshape(collect(Float64, v), shape); z = collect(range(2e6, -1e5; length=nz))
Temp = range(5000.0, 9000.0; length=nz); P = 10 .^ range(-1.0, 4.3; length=nz)
a = FFNOInversion.Atmosphere3D(Grid3D(VerticalAxis(:z_m, z), [0.0], [0.0]), f(Temp), zeros(shape), zeros(shape), zeros(shape), zeros(shape); pgas=f(P), rho=f(3e-8 .* P), ne=f(1e17 .* P .^ 0.6), z=f(z))
model = KuruczLTEModel(KuruczLine{Float64}[], eos; include_continuum=true); wave = collect(range(656e-9, 657e-9; length=nl))
chi = zeros(nz, nl); eta = zeros(nz, nl); FFNOInversion._add_wittmann_kurucz!(chi, eta, model, wave, a, 1, 1)
forward = @elapsed for _ in 1:5; fill!(chi, 0); fill!(eta, 0); FFNOInversion._add_wittmann_kurucz!(chi, eta, model, wave, a, 1, 1); end
bar = FFNOInversion.AtmosphereCotangent(a); cb = ones(nz, nl); eb = ones(nz, nl)
FFNOInversion.add_opacity_emissivity_vjp!(bar, nothing, nothing, cb, eb, model, wave, a, 1, 1, nothing)
adjoint = @elapsed FFNOInversion.add_opacity_emissivity_vjp!(bar, nothing, nothing, cb, eb, model, wave, a, 1, 1, nothing)
@printf("per column, 95 levels, 88 wavelengths: forward %.4f s, adjoint %.3f s\n", forward / 5, adjoint)
