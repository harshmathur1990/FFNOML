# FALC_82 H-alpha, centre pixel, mu = 1, on FALC's own grid ([population.mesh] enabled = false): the zaug H model against
# l004p, both run with the same code (falc_l004p_native_rerun), for the six necmp runs (FALC, LTE EOS and NLTE EOS nₑ on the
# log column mass and height axes), against RH on FALC_82 with FALC's nₑ (falc_rh_air.h5); and, for falcz (FALC's nₑ given,
# height axis), the H-alpha level populations n2, n3 and n3/n2 against RH's own (falc_rh.h5).
import h5py, numpy as np, struct, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "/mn/stornext/d9/data/harshm/"; OLD = R + "FFNOML/response_run/falc/"; OUT = R + "ffnoml_runs/falc_zaug_native/"
runs = [("l004p", R + "ffnoml_runs/falc_l004p_native_rerun/falc/", ":", 1.6), ("zaug", OUT + "falc/", "-", 1.8)]
models = [("FALC nₑ (given)", "falc", "#2a78d6"), ("LTE EOS nₑ", "lte", "#eb6834"), ("NLTE EOS nₑ", "nlte", "#1baf7a")]
grids = [("log column mass grid", "m"), ("height grid", "z")]
lo, hi = 655.8, 657.2
def spec(d, n):
    with h5py.File(d + f"falc_necmp_{n}_synthesis.h5") as f:
        return f["wavelength_m"][:] * 1e9, f["intensity"][2, 2, :, 0, 0] * 1e8
with h5py.File(OLD + "falc_rh_air.h5") as f:
    lr = f["wavelength_air_nm"][:]; Ir = f["intensity"][:] * 1e8
q = np.argsort(lr); lr, Ir = lr[q], Ir[q]; rw = (lr > lo) & (lr < hi); core = (lr > 656.18) & (lr < 656.38)
fig, ax = plt.subplots(4, 1, figsize=(10, 14), sharex=True, height_ratios=[2.2, 1, 2.2, 1], constrained_layout=True)
print(f"RH core {Ir[rw].min():.3f}")
for r, (glabel, g) in enumerate(grids):
    top, bot = ax[2 * r], ax[2 * r + 1]
    top.plot(lr[rw], Ir[rw], color="k", lw=2.2, label="RH (reference), FALC nₑ")
    print(f"== {glabel}")
    for label, m, color in models:
        for mlabel, d, ls, lw in runs:
            l, I = spec(d, m + g); s = (l > lo) & (l < hi); e = I[s] / np.interp(l[s], lr, Ir) - 1
            c = (l[s] > 656.18) & (l[s] < 656.38)
            top.plot(l[s], I[s], color=color, ls=ls, lw=lw, label=f"{label}: {mlabel}")
            bot.plot(l[s], 100 * e, color=color, ls=ls, lw=lw)
            print(f"  {label:18s} {mlabel:6s} core {I[s].min():.3f} ({100 * (I[s].min() / Ir[rw].min() - 1):+5.1f}% vs RH), "
                  f"rms {100 * np.sqrt(np.mean(e ** 2)):4.1f}%, rms within ±0.1 nm {100 * np.sqrt(np.mean(e[c] ** 2)):4.1f}%, max |err| {100 * np.abs(e).max():4.1f}%")
    top.set_title(f"Hα 656.28 nm (with Fe I 656.92 nm), {glabel}", fontsize=11); top.set_ylim(bottom=0)
    top.set_ylabel("I [10$^{-8}$ J s$^{-1}$ m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]"); bot.set_ylabel("Difference from RH [%]"); bot.axhline(0, color="k", lw=0.8)
    for a in (top, bot):
        a.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
ax[3].set_xlabel("Air wavelength [nm]"); ax[0].legend(fontsize=8.5, ncol=2, frameon=False)
fig.suptitle("FALC_82, centre pixel, μ = 1, FALC's own grid: l004p (dotted) vs zaug (solid) against RH", fontsize=11)
fig.savefig(OUT + "falc_halpha_zaug_vs_l004p.png", dpi=105); print("wrote falc_halpha_zaug_vs_l004p.png")

b = open(R + "rh_falc/rhf1d/run_falc/geometry.out", "rb").read()
_, nr, n = struct.unpack(">iii", b[:12]); g = np.frombuffer(b, ">f8", 3 * n, 12 + 16 * nr); zr = g[:n].copy() / 1e3
with h5py.File(OLD + "falc_rh.h5") as f:
    rh = np.array(f["H_n"]); rh = rh if rh.shape[0] == 6 else rh.T
def pops(d):
    with h5py.File(d + "falc_necmp_falcz_atmosphere.h5") as f:
        return f["populations/H"][2, 2, -82:, :].T, f["temperature"][2, 2, -82:, 0]
fig, ax = plt.subplots(1, 4, figsize=(17, 5.2), constrained_layout=True)
cols = {"l004p": "#eb6834", "zaug": "#2a78d6"}
for mlabel, d, ls, lw in runs:
    p, T = pops(d)
    for a, y in zip(ax[:3], (p[1] / rh[1], p[2] / rh[2], (p[2] / p[1]) / (rh[2] / rh[1]))):
        a.plot(zr, y, color=cols[mlabel], lw=1.8, label=mlabel)
ax[3].plot(zr, T, "k", lw=1.6); ax[3].set_yscale("log"); ax[3].set_title("FALC_82 temperature", fontsize=11); ax[3].set_ylabel("T [K]")
for a, t in zip(ax[:3], ("n$_2$ (Hα lower level) / RH", "n$_3$ (Hα upper level) / RH", "(n$_3$/n$_2$) / RH  (Hα source function)")):
    a.axhline(1, color="k", lw=0.8); a.set_yscale("log"); a.set_title(t, fontsize=11); a.set_ylabel("ratio to RH")
for a in ax:
    a.set_xlabel("height [km]"); a.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"): a.spines[s].set_visible(False)
ax[0].legend(frameon=False)
fig.suptitle("FALC_82, centre pixel, FALC's nₑ given, height axis, FALC's own grid: FFNO H populations against RH's", fontsize=12)
fig.savefig(OUT + "falc_populations_zaug_vs_l004p.png", dpi=100); print("wrote falc_populations_zaug_vs_l004p.png")
hs = (2100, 2000, 1800, 1500, 1200, 1000, 700, 400, 100, 0)
ks = [int(np.argmin(abs(zr - h))) for h in hs]
print("height km      " + " ".join(f"{int(zr[k]):>6d}" for k in ks))
print("T [kK]         " + " ".join(f"{pops(runs[0][1])[1][k] / 1e3:6.1f}" for k in ks))
for mlabel, d, _, _ in runs:
    p, _ = pops(d)
    for name, y in (("n2/RH", p[1] / rh[1]), ("n3/RH", p[2] / rh[2]), ("S ratio", (p[2] / p[1]) / (rh[2] / rh[1]))):
        print(f"{mlabel:6s} {name:8s}" + " ".join(f"{y[k]:6.2f}" for k in ks))
