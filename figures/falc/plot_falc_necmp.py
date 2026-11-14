# FALC_82 with three electron densities, on the log column mass and height axes (response_run/falc/forward_falc_necmp_*.toml,
# run_necmp_eagle7.sh): FALC's own nₑ and hydrogen populations given (no HSEMHS), and the LTE and NLTE EOS (HSEMHS column HSE
# from the temperature alone: P = g m on log m, P at the bottom on z). Against RH on FALC_82 with FALC's nₑ (run_falc_air).
import h5py, numpy as np, struct, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "response_run/falc/"
models = [("FALC nₑ (given, with its H populations)", "falc", "#2a78d6"), ("LTE EOS nₑ", "lte", "#eb6834"), ("NLTE EOS nₑ", "nlte", "#1baf7a")]
axes_ = [("log column mass grid", "m", "-"), ("height grid", "z", "--")]
def spec(n):
    with h5py.File(D + f"falc_necmp_{n}_synthesis.h5") as f:
        return f["wavelength_m"][:] * 1e9, f["intensity"][2, 2, :, 0, 0] * 1e8
def field(n, k):
    with h5py.File(D + f"falc_necmp_{n}_atmosphere.h5") as f:
        if k in f: return f[k][2, 2, -82:, 0]
    with h5py.File(D + f"falc_necmp_{n}_hydrogen.h5") as f:                   # EOS runs: necmp_hydrogen.jl
        return f[k][2, 2, -82:, 0]
with h5py.File(D + "falc_rh_air.h5") as f:
    lr = f["wavelength_air_nm"][:]; Ir = f["intensity"][:] * 1e8
q = np.argsort(lr); lr, Ir = lr[q], Ir[q]
with h5py.File(D + "falc_logm_z.h5") as f:
    ne0 = f["ne"][:, 0, 0]; hp = f["hydrogen_populations"][:, :, 0, 0]
nhi0, np0 = hp[:-1].sum(0), hp[-1]
b = open("/mn/stornext/d9/data/harshm/rh_falc/rhf1d/run_falc/geometry.out", "rb").read()
_, nr, n = struct.unpack(">iii", b[:12]); g = np.frombuffer(b, ">f8", 3 * n, 12 + 16 * nr); zr, mr, tr = g[:n].copy(), g[n:2 * n].copy(), g[2 * n:3 * n].copy()
zk = zr / 1e3
def clean(ax):
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

fig, ax = plt.subplots(4, 2, figsize=(13, 13), sharex="col", height_ratios=[2.2, 1, 2.2, 1], constrained_layout=True)
for c, (lo, hi, title) in enumerate(((655.8, 657.2, "Hα 656.28 nm, with Fe I 656.92 nm"), (853.4, 854.6, "Ca II 854.2 nm, with Si I 853.62 nm"))):
    w = (lr > lo) & (lr < hi)
    for r, (alabel, a, ls) in enumerate(axes_):
        top, bot = ax[2 * r, c], ax[2 * r + 1, c]
        top.plot(lr[w], Ir[w], color="k", lw=2.0, label="RH (reference), FALC nₑ, NLTE H and Ca II")
        for label, m, color in models:
            l, I = spec(m + a); s = (l > lo) & (l < hi)
            top.plot(l[s], I[s], color=color, ls=ls, lw=1.6, label=label)
            bot.plot(l[s], 100 * (I[s] / np.interp(l[s], lr, Ir) - 1), color=color, ls=ls, lw=1.5)
        top.set_title(f"{title}: {alabel}", fontsize=11); top.set_ylim(bottom=0); bot.axhline(0, color="k", lw=0.8)
        top.set_ylabel("I [10$^{-8}$ J s$^{-1}$ m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]"); bot.set_ylabel("Difference from RH [%]")
        clean(top); clean(bot)
    ax[3, c].set_xlabel("Air wavelength [nm]")
h, lab = ax[0, 0].get_legend_handles_labels()
fig.legend(h, lab, loc="outside lower center", ncol=4, fontsize=9.5, frameon=False)
fig.suptitle("FALC_82, centre pixel, μ = 1: FFNO line populations with three electron densities, against RH", fontsize=12)
fig.savefig("figures/falc/falc_necmp_spectra.png", dpi=105); print("wrote spectra")

fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
panels = [(ax[0, 0], "ne", None, "nₑ [m$^{-3}$]", "Electron density", ne0),
          (ax[0, 1], "ne", ne0, "nₑ / FALC nₑ", "Electron density against FALC's", None),
          (ax[0, 2], "h_neutral", nhi0, "n$_{HI}$ / FALC n$_{HI}$", "Neutral hydrogen against FALC's", None),
          (ax[1, 0], "h_proton", np0, "n$_p$ / FALC n$_p$", "Protons against FALC's", None)]
for p, key, ref, ylabel, title, truth in panels:
    if truth is not None: p.plot(zk, truth, color="k", lw=2.0, label="FALC_82 table")
    for label, m, color in models:
        for alabel, a, ls in axes_:
            v = field(m + a, key); p.plot(zk, v if ref is None else v / ref, color=color, ls=ls, lw=1.5, label=f"{label}, {alabel}")
    if ref is None: p.set_yscale("log")
    else: p.axhline(1, color="k", lw=0.8); p.set_yscale("log")
    p.set_xlabel("RH height [km]"); p.set_ylabel(ylabel); p.set_title(title, fontsize=11); clean(p)
p = ax[1, 1]; p.axhline(0, color="k", lw=0.8)
for label, m, color in models:
    p.plot(np.log10(mr), (field(m + "m", "z") - zr) / 1e3, color=color, lw=1.5)
p.set_xlabel("log$_{10}$ column mass [kg m$^{-2}$]"); p.set_ylabel("Height − RH height [km]"); p.set_title("Height built on the log column mass grid", fontsize=11); clean(p)
p = ax[1, 2]; p.axhline(0, color="k", lw=0.8)
for label, m, color in models:
    for alabel, a, ls in axes_:
        p.plot(zk, field(m + a, "logtau500") - np.log10(tr), color=color, ls=ls, lw=1.5)
p.set_ylim(-1, 1); p.set_xlabel("RH height [km]"); p.set_ylabel("log$_{10}$ τ₅₀₀ − RH [dex], clipped at ±1"); p.set_title("τ₅₀₀ against RH", fontsize=11); clean(p)
h, lab = ax[0, 0].get_legend_handles_labels()
fig.legend(h, lab, loc="outside lower center", ncol=4, fontsize=9.5, frameon=False)
fig.suptitle("FALC_82 with three electron densities: solid log column mass grid, dashed height grid", fontsize=12)
fig.savefig("figures/falc/falc_necmp_atmosphere.png", dpi=105); print("wrote atmosphere")

for alabel, a, _ in axes_:
    print(f"== {alabel}")
    for label, m, _ in models:
        l, I = spec(m + a); R = np.interp(l, lr, Ir); e = I / R - 1
        out = []
        for lab, lo, hi in (("Ha", 655.8, 657.2), ("Ca", 853.4, 854.6)):
            s = (l > lo) & (l < hi); rw = (lr > lo) & (lr < hi)
            out.append("%s rms %.3f core %.3f (RH %.3f, %+.0f%%) far wing %+.1f%%" % (lab, np.sqrt(np.mean(e[s] ** 2)), I[s].min(), Ir[rw].min(), 100 * (I[s].min() / Ir[rw].min() - 1), 100 * e[s][0]))
        print(f"  {label:42s}", " | ".join(out))
print("nₑ / FALC nₑ at RH heights [km]")
hs = (2000, 1500, 1000, 700, 500, 250, 0)
ks = [int(np.argmin(abs(zk - h))) for h in hs]
print("  " + " ".join(f"{int(zk[k]):>6d}" for k in ks))
for label, m, _ in models:
    for alabel, a, _ in axes_:
        v = field(m + a, "ne") / ne0; print("  " + " ".join(f"{v[k]:6.2f}" for k in ks), f"  {m} {a}")
