# FALC centre pixel through the real code path with its nₑ as an input dataset (HSEMHS derives P and rho from T and nₑ), against RH with the
# same nₑ: the z axis with a corona, and the log tau axis with and without it. Top: profiles; bottom: this code / RH.
import h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "response_run/falc/"
with h5py.File(D + "falc_rh.h5") as f:
    lr = f["wavelength_air_nm"][:]; Ir = f["intensity"][:] * 1e8
o = np.argsort(lr); lr, Ir = lr[o], Ir[o]
series = [("z axis + corona", "nez", "#2a78d6"), ("tau axis + corona", "netauext", "#1baf7a"), ("tau axis, FALC's levels only", "netau", "#eb6834")]
fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True, height_ratios=[3, 1.2], sharex="col")
for j, (lims, title) in enumerate((((655.8, 657.2), "H-alpha + Fe I 656.92 nm"), ((853.4, 854.6), "Ca II 854.2 nm + Si I 853.62 nm"))):
    w = (lr > lims[0]) & (lr < lims[1]); ax = axes[0, j]; ax.plot(lr[w], Ir[w], "k", lw=2.1, label="RH, FALC's nₑ")
    for label, name, color in series:
        with h5py.File(D + f"falc_{name}_synthesis.h5") as f:
            l = f["wavelength_m"][:] * 1e9; I = f["intensity"][2, 2, :, 0, 0] * 1e8
        m = (l > lims[0]) & (l < lims[1]); ax.plot(l[m], I[m], color=color, lw=1.7, label=label); axes[1, j].plot(l[m], I[m] / np.interp(l[m], lr, Ir), color=color, lw=1.5)
    ax.set_title(title, fontsize=11); ax.set_ylabel("I$_ν$ [10$^{-8}$ J s$^{-1}$ m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]"); ax.set_ylim(bottom=0); ax.grid(alpha=.25, lw=.5); ax.legend(fontsize=9, frameon=False, loc="lower right" if j == 0 else "lower left")
    b = axes[1, j]; b.axhline(1, color="k", lw=1); b.set_ylabel("this code / RH"); b.set_xlabel("air wavelength [nm]"); b.set_ylim(0.4, 1.4); b.grid(alpha=.25, lw=.5)
    for a in (ax, b):
        for s in ("top", "right"): a.spines[s].set_visible(False)
fig.suptitle("FALC_82_5x5, centre pixel, μ = 1, nₑ given", fontsize=12); fig.savefig("figures/falc/falc_ne_given.png", dpi=105); print("wrote")
