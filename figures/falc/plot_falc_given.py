# Line cores at air 656.28 and 854.209 nm in both codes (RH: run_falc_air). FALC_82 forward synthesis with its own nₑ and hydrogen populations given, no HSEMHS, on each input grid
# (response_run/falc/forward_falc_given{m,tau,z}.toml), against RH on the same model with the same nₑ (run_falc, read_rh_output.py):
# spectra with FFNO line populations on the population mesh, and the geometry against RH's convertScales (geometry.out).
import h5py, numpy as np, struct, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "response_run/falc/"
runs = [("Input grid: log column mass", "givenm", "#2a78d6", "-", 2.6), ("Input grid: log τ₅₀₀", "giventau", "#8a5cd1", "-", 1.6),
        ("Input grid: height", "givenz", "#1baf7a", "--", 1.6)]
def spec(n):
    with h5py.File(D + f"falc_{n}_synthesis.h5") as f:
        return f["wavelength_m"][:] * 1e9, f["intensity"][2, 2, :, 0, 0] * 1e8
def field(n, k):
    with h5py.File(D + f"falc_{n}_atmosphere.h5") as f:
        return f[k][2, 2, -82:, 0]
with h5py.File(D + "falc_rh_air.h5") as f:
    lr = f["wavelength_air_nm"][:]; Ir = f["intensity"][:] * 1e8
q = np.argsort(lr); lr, Ir = lr[q], Ir[q]
b = open("/mn/stornext/d9/data/harshm/rh_falc/rhf1d/run_falc/geometry.out", "rb").read()
_, nr, n = struct.unpack(">iii", b[:12]); g = np.frombuffer(b, ">f8", 3 * n, 12 + 16 * nr); zr, mr, tr = g[:n].copy(), g[n:2 * n].copy(), g[2 * n:3 * n].copy()
def clean(ax):
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharex="col", height_ratios=[2.2, 1], constrained_layout=True)
for c, (lo, hi, title) in enumerate(((655.8, 657.2, "Hα 656.28 nm, with the Fe I 656.92 nm blend"), (853.4, 854.6, "Ca II 854.2 nm, with the Si I 853.62 nm blend"))):
    w = (lr > lo) & (lr < hi); axes[0, c].plot(lr[w], Ir[w], color="k", lw=2.0, label="RH (reference): NLTE H and Ca II, cores set to 656.280 and 854.209 nm")
    for label, name, color, ls, lw in runs:
        l, I = spec(name); m = (l > lo) & (l < hi)
        axes[0, c].plot(l[m], I[m], color=color, ls=ls, lw=lw, label=label)
        axes[1, c].plot(l[m], 100 * (I[m] / np.interp(l[m], lr, Ir) - 1), color=color, ls=ls, lw=lw)
    axes[0, c].set_title(title, fontsize=11); axes[0, c].set_ylim(bottom=0); axes[1, c].axhline(0, color="k", lw=0.8)
    axes[1, c].set_xlabel("Air wavelength [nm]"); clean(axes[0, c]); clean(axes[1, c])
    axes[0, c].set_ylabel("Intensity [10$^{-8}$ J s$^{-1}$ m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]"); axes[1, c].set_ylabel("Difference from RH [%]")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9.5, frameon=False)
fig.suptitle("Spectra: FALC_82 with its nₑ and hydrogen populations given (no HSEMHS), FFNO line populations, against RH. Centre pixel, μ = 1", fontsize=12)
fig.savefig("figures/falc/falc_given_spectra.png", dpi=110); print("wrote spectra")

fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), constrained_layout=True)
lm = np.log10(mr); zk = zr / 1e3
ax = axes[0, 0]; ax.plot(lm, zk, color="k", lw=2.0, label="RH (reference): geometry.out")
for label, name, color, ls, lw in runs:
    ax.plot(lm, field(name, "z") / 1e3, color=color, ls=ls, lw=lw, label=label)
ax.set_xlabel("log$_{10}$ column mass [kg m$^{-2}$]"); ax.set_ylabel("Height [km]"); ax.set_title("Height built by each run", fontsize=11); clean(ax)
ax = axes[0, 1]; ax.axhline(0, color="k", lw=0.8)
for label, name, color, ls, lw in runs:
    ax.plot(lm, (field(name, "z") - zr) / 1e3, color=color, ls=ls, lw=lw)
ax.set_xlabel("log$_{10}$ column mass [kg m$^{-2}$]"); ax.set_ylabel("Height − RH height [km]"); ax.set_title("Height error against RH (height-grid run is 0 by construction)", fontsize=11); clean(ax)
ax = axes[1, 0]; ax.axhline(0, color="k", lw=0.8)
for label, name, color, ls, lw in runs:
    ax.plot(zk, field(name, "logtau500") - np.log10(tr), color=color, ls=ls, lw=lw)
ax.set_ylim(-1, 1); ax.set_xlabel("RH height [km]"); ax.set_ylabel("log$_{10}$ τ₅₀₀ − RH [dex], axis clipped at ±1")
ax.set_title("τ₅₀₀ error against RH (log τ run is 0 by construction)", fontsize=11); clean(ax)
ax = axes[1, 1]; ax.axhline(1, color="k", lw=0.8)
P = field("givenm", "pgas"); rho = field("givenm", "rho"); v = field("givenm", "vturb")
ax.plot(zk, P / (274.0 * mr), color="#d2691e", lw=1.8, label="Gas pressure only: P / (g m)")
ax.plot(zk, (P + 0.5 * rho * v ** 2) / (274.0 * mr), color="#d2691e", ls="--", lw=1.8, label="With turbulent pressure: (P + ρ v$_{turb}^2$/2) / (g m)")
ax.set_xlabel("RH height [km]"); ax.set_ylabel("Ratio to g m (1 = hydrostatic)"); ax.set_title("Pressure from the given nₑ and hydrogen, against g m", fontsize=11); ax.legend(fontsize=9, frameon=False, loc="lower left"); clean(ax)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9.5, frameon=False)
fig.suptitle("Geometry: FALC_82 built from its nₑ and hydrogen populations (no HSEMHS), against RH. μ = 1", fontsize=12)
fig.savefig("figures/falc/falc_given_geometry.png", dpi=110); print("wrote geometry")
for label, name, *_ in runs:
    l, I = spec(name); R = np.interp(l, lr, Ir); e = I / R - 1
    for lab, lo, hi, a0, b0 in (("Ha", 655.8, 657.2, 656.15, 656.42), ("Ca", 853.4, 854.6, 854.0, 854.4)):
        m = (l > lo) & (l < hi); near = m & (l > a0 + 0.08) & (l < b0 - 0.08); wing = m & ((l < a0) | (l > b0))
        print("%-28s %s rms %.3f wings %+.3f core %.3f RH %.3f near-core residual min %+.0f%% max %+.0f%%" % (label, lab, np.sqrt(np.mean(e[m] ** 2)), e[wing].mean(), I[m].min(), R[m].min(), 100 * e[near].min(), 100 * e[near].max()))
