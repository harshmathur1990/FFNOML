# FALC centre pixel: this code (default EOS continuum, FFNO populations, nₑ from the EOS) against RH run with the same nₑ
# (rh_ne_eos_spectrum.npy, FALC_82 with the EOS nₑ) and RH with FALC's tabulated nₑ. Top: profiles; bottom: this code / RH with the same nₑ.
import h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "response_run/falc/"
def run(name):
    with h5py.File(D + name) as f:
        return f["wavelength_m"][:] * 1e9, f["intensity"][2, 2, :, 0, 0] * 1e8
lE, IE = np.load(D + "rh_ne_eos_spectrum.npy")
with h5py.File(D + "falc_rh.h5") as f:
    lF = f["wavelength_air_nm"][:]; IF = f["intensity"][:] * 1e8
o = np.argsort(lF); lF, IF = lF[o], IF[o]
series = [("this code, FFNO, tau axis", "falc_ltauc_synthesis.h5", "#1baf7a"), ("this code, FFNO, z axis + corona", "falc_zextc_synthesis.h5", "#2a78d6")]
fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True, height_ratios=[3, 1.2], sharex="col")
for j, (lims, title) in enumerate((((655.8, 657.2), "H-alpha + Fe I 656.92 nm"), ((853.4, 854.6), "Ca II 854.2 nm + Si I 853.62 nm"))):
    ax = axes[0, j]; w = (lE > lims[0]) & (lE < lims[1]); v = (lF > lims[0]) & (lF < lims[1])
    ax.plot(lE[w], IE[w], "k", lw=2.1, label="RH, same nₑ as this code (EOS)")
    ax.plot(lF[v], IF[v], color="#888888", ls="--", lw=1.4, label="RH, FALC's tabulated nₑ")
    for label, name, color in series:
        l, I = run(name); m = (l > lims[0]) & (l < lims[1]); ax.plot(l[m], I[m], color=color, lw=1.7, label=label)
        axes[1, j].plot(l[m], I[m] / np.interp(l[m], lE, IE), color=color, lw=1.5)
    ax.set_title(title, fontsize=11); ax.set_ylabel("I$_ν$ [10$^{-8}$ J s$^{-1}$ m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]"); ax.set_ylim(bottom=0); ax.grid(alpha=.25, lw=.5); ax.legend(fontsize=9, frameon=False, loc="lower right" if j == 0 else "lower left")
    b = axes[1, j]; b.axhline(1, color="k", lw=1); b.set_ylabel("this code / RH (same nₑ)"); b.set_xlabel("air wavelength [nm]"); b.set_ylim(0.6, 1.4); b.grid(alpha=.25, lw=.5)
    for a in (ax, b):
        for s in ("top", "right"): a.spines[s].set_visible(False)
fig.suptitle("FALC_82_5x5, centre pixel, μ = 1", fontsize=12); fig.savefig("figures/falc/falc_same_ne.png", dpi=105); print("wrote")
