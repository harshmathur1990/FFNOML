# H-alpha temperature response from finite differences with a Gaussian bump in log tau500
# (response_run/fd_bump_response.jl, MULTI3D populations on the Bifrost atmosphere): departure coefficients fixed
# (solid) against populations fixed (dashed), per pixel, core integrated over 6562.96 ± 0.15 Å (top) and the wing at
# the bluest grid point (bottom). Units: intensity per K of bump amplitude (core: × Å).
import sys, h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
departure = sys.argv[1] if len(sys.argv) > 1 else "response_run/outputs/fd_bump_departure_4pixels.h5"
fixed = sys.argv[2] if len(sys.argv) > 2 else "response_run/outputs/fd_bump_4pixels_small.h5"
out = sys.argv[3] if len(sys.argv) > 3 else "figures/column_response/fd_bump_departure_halpha.png"
CENTRE, HALF = 6562.96, 0.15
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # categorical slots 1-4

def load(path):
    with h5py.File(path) as f:
        return (f["R_temperature"][:], f["wavelength_air_m"][:] * 1e10, f["logtau500"][:], f["pixels"][:],
                f.attrs["sigma_dex"], f.attrs["fraction"])
Rd, w, lt, pixels, sigma, fraction = load(departure); Rf = load(fixed)[0]
core = np.where(abs(w - CENTRE) <= HALF + 1e-6)[0]; wing = np.argmin(w)

fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True)
for p, (x, y) in enumerate(pixels):
    for r in range(2):
        ax = axes[r, p]
        for R, style, label in ((Rd, "-", "departure coefficients fixed"), (Rf, "--", "populations fixed")):
            v = np.trapezoid(R[p][:, core], w[core], axis=1) if r == 0 else R[p][:, wing]
            ax.plot(lt, v, style, color=colors[p], lw=2 if style == "-" else 1.4, label=label)
        ax.axhline(0, color="0.6", lw=0.8, zorder=0); ax.grid(alpha=0.25, lw=0.6)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.set_title(f"x={x}, y={y}: " + (f"core {CENTRE:.2f} ± {HALF:.2f} Å" if r == 0 else f"wing {w[wing]:.2f} Å"),
                     fontsize=10)
        if r == 1: ax.set_xlabel("log$_{10}$ τ$_{500}$ of bump centre")
axes[0, 0].set_ylabel("∫(I$_+$ − I$_-$)/2A dλ  [I·Å / K]", fontsize=9)
axes[1, 0].set_ylabel("(I$_+$ − I$_-$)/2A  [I / K]", fontsize=9)
axes[0, 0].legend(fontsize=9, frameon=False)
fig.suptitle(f"H-alpha temperature response, Gaussian bump σ = {sigma} dex, A = {fraction:.1%} T, "
             "MULTI3D populations on the Bifrost atmosphere (ne, pgas fixed)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=110); print("wrote", out)
for p, (x, y) in enumerate(pixels):
    d = np.trapezoid(Rd[p][:, core], w[core], axis=1); f = np.trapezoid(Rf[p][:, core], w[core], axis=1)
    deep = abs(d[lt > -4.5]).sum() / abs(d).sum(); deepf = abs(f[lt > -4.5]).sum() / abs(f).sum()
    print((int(x), int(y)), "core: peak logτ departure %.1f fixed %.1f | |R| below -4.5: departure %.2f fixed %.2f | "
          "max|R| ratio departure/fixed %.1f" % (lt[np.argmax(abs(d))], lt[np.argmax(abs(f))], deep, deepf,
                                                abs(d).max() / abs(f).max()))
