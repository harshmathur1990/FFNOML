# FALC centre pixel: RH's spectrum, this code's solver with RH's own H and Ca II populations (falc_rh_populations.jl), and the FFNO's
# populations on the log tau axis. Top row: profiles. Bottom row: this code / RH.
import h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "response_run/falc/"
def run(name):
    with h5py.File(D + name) as f:
        return f["wavelength_m"][:] * 1e9, f["intensity"][2, 2, :, 0, 0] * 1e8
with h5py.File(D + "falc_rh.h5") as f:
    lr = f["wavelength_air_nm"][:]; Ir = f["intensity"][:] * 1e8
series = [("this code, RH's populations", "falc_rhpops_synthesis.h5", "#eb6834"), ("this code, FFNO populations (tau axis)", "falc_ltauc_synthesis.h5", "#1baf7a")]
fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True, height_ratios=[3, 1.2], sharex="col")
for j, (lims, title) in enumerate((((655.8, 657.2), "H-alpha + Fe I 656.92 nm"), ((853.4, 854.6), "Ca II 854.2 nm + Si I 853.62 nm"))):
    w = (lr > lims[0]) & (lr < lims[1]); ax = axes[0, j]; ax.plot(lr[w], Ir[w], "k", lw=2, label="RH: H and Ca II NLTE, FALC_82")
    for label, name, color in series:
        l, I = run(name); m = (l > lims[0]) & (l < lims[1])
        ax.plot(l[m], I[m], color=color, lw=1.7, label=label); axes[1, j].plot(l[m], I[m] / np.interp(l[m], lr, Ir), color=color, lw=1.5)
        r = I[m] / np.interp(l[m], lr, Ir); print(f"{title[:8]:8s} {label:40s} edges {I[m][0]:.3f} {I[m][-1]:.3f} core {I[m].min():.3f} at {l[m][I[m].argmin()]:.3f} rms/RH {np.sqrt(np.mean((r-1)**2)):.3f} max|r-1| {abs(r-1).max():.3f}")
    print(f"{title[:8]:8s} RH edges {Ir[w][0]:.3f} {Ir[w][-1]:.3f} core {Ir[w].min():.3f} at {lr[w][Ir[w].argmin()]:.3f}")
    ax.set_title(title, fontsize=11); ax.set_ylabel("I$_ν$ [10$^{-8}$ J s$^{-1}$ m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]"); ax.set_ylim(bottom=0); ax.grid(alpha=.25, lw=.5); ax.legend(fontsize=9, frameon=False, loc="lower right" if j == 0 else "lower left")
    b = axes[1, j]; b.axhline(1, color="k", lw=1); b.set_ylabel("this code / RH"); b.set_xlabel("air wavelength [nm]"); b.set_ylim(0.6, 1.6); b.grid(alpha=.25, lw=.5)
    for a in (ax, b):
        for s in ("top", "right"): a.spines[s].set_visible(False)
fig.suptitle("FALC_82_5x5, centre pixel, μ = 1", fontsize=12); fig.savefig("figures/falc/falc_rh_populations.png", dpi=105); print("wrote")
