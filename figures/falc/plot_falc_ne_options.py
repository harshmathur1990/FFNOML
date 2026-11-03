# FALC centre pixel with FALC's nₑ as the truth, against RH with the same nₑ. A: P, rho from HSE. B: P, rho from T and nₑ through the EOS.
# Top: profiles; bottom: this code / RH. Prints the numbers and the populations against RH.
import h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "response_run/falc/"
with h5py.File(D + "falc_rh.h5") as f:
    lr = f["wavelength_air_nm"][:]; Ir = f["intensity"][:] * 1e8; H = f["H_n"][...]; Ca = f["CA_n"][...]
o = np.argsort(lr); lr, Ir = lr[o], Ir[o]
z = h5py.File(D + "FALC_82_5x5.hdf5")["z"][0] / 1e3
series = [("A: nₑ given, P and ρ from HSE (z axis)", "A", "#2a78d6"), ("B: nₑ given, P and ρ from the EOS (z axis)", "B", "#eb6834"), ("A on the tau axis", "Atau", "#1baf7a")]
fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True, height_ratios=[3, 1.2], sharex="col")
at = lambda l, I, x: np.interp(x, l, I)
for j, (lims, title, pts) in enumerate((((655.8, 657.2), "H-alpha + Fe I 656.92 nm", (655.8, 657.2)), ((853.4, 854.6), "Ca II 854.2 nm + Si I 853.62 nm", (853.4, 854.0, 854.6)))):
    w = (lr > lims[0]) & (lr < lims[1]); ax = axes[0, j]; ax.plot(lr[w], Ir[w], "k", lw=2.1, label="RH, FALC's nₑ")
    print(title[:8], "RH", " ".join(f"{at(lr, Ir, x):.3f}" for x in pts), "core %.3f" % Ir[w].min())
    for label, name, color in series:
        with h5py.File(D + f"falc_ne{name}_synthesis.h5") as f:
            l = f["wavelength_m"][:] * 1e9; I = f["intensity_center"][:] * 1e8
        m = (l > lims[0]) & (l < lims[1]); r = I[m] / np.interp(l[m], lr, Ir)
        ax.plot(l[m], I[m], color=color, lw=1.7, label=label); axes[1, j].plot(l[m], r, color=color, lw=1.5)
        print(f"  {name:5s}", " ".join(f"{at(l[m], I[m], x):.3f} ({100*(at(l[m], I[m], x)/at(lr, Ir, x)-1):+.1f}%)" for x in pts), "core %.3f rms %.3f" % (I[m].min(), np.sqrt(np.mean((r - 1) ** 2))))
    ax.set_title(title, fontsize=11); ax.set_ylabel("I$_ν$ [10$^{-8}$ J s$^{-1}$ m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]"); ax.set_ylim(bottom=0); ax.grid(alpha=.25, lw=.5); ax.legend(fontsize=9, frameon=False, loc="lower right" if j == 0 else "lower left")
    b = axes[1, j]; b.axhline(1, color="k", lw=1); b.set_ylabel("this code / RH"); b.set_xlabel("air wavelength [nm]"); b.set_ylim(0.5, 1.5); b.grid(alpha=.25, lw=.5)
    for a in (ax, b):
        for s in ("top", "right"): a.spines[s].set_visible(False)
fig.suptitle("FALC_82_5x5, centre pixel, μ = 1, FALC's nₑ", fontsize=12); fig.savefig("figures/falc/falc_ne_options.png", dpi=105); print("wrote")
print("populations / RH: z_km | total H A B Atau | H n2 | H n3 | Ca n3 | Ca n5")
for h in (1877, 1349, 978, 702, 490, 252, 102, 11):
    k = int(np.argmin(abs(z - h))); t = {n: h5py.File(D + f"falc_ne{n}_synthesis.h5") for _, n, _ in series}
    g = lambda s, i: " ".join(f"{(t[n][s][i, k] if t[n][s].shape[1] == 82 else t[n][s][k, i]) / (H[k, i] if s == 'H' else Ca[k, i]):.2f}" for _, n, _ in series)
    tot = " ".join(f"{(t[n]['H'][:, k].sum() if t[n]['H'].shape[1] == 82 else t[n]['H'][k].sum()) / H[k].sum():.2f}" for _, n, _ in series)
    print(int(z[k]), "|", tot, "|", g("H", 1), "|", g("H", 2), "|", g("CA", 2), "|", g("CA", 4))
