# Centre pixel of FALC_82_5x5 through FFNOInversion (response_run/falc), all with microturbulence in the FFNO lines, against RH on
# FALC_82 (read_rh_output.py): H-alpha with Fe I 656.92 nm and Ca II 854.2 nm with Si I 853.62 nm. The FFNO on the population mesh
# with FALC on z (top values held to 14 Mm, or FALC extended with a 1 MK corona, build_falc_extended.py) and through the log tau route.
import h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "response_run/falc/"; C = 299792458.0
def run(name):
    with h5py.File(D + name) as f:
        return f["wavelength_m"][:] * 1e9, f["intensity"][2, 2, :, 0, 0] * 1e8
series = [("FFNO on z mesh, top values held to 14 Mm", "falc_zmesh_synthesis.h5", "#2a78d6", "-"),
          ("FFNO via log tau route", "falc_ltau_vturb_synthesis.h5", "#8a5cd1", "-"),
          ("FFNO on z mesh, FALC extended with a 1 MK corona", "falc_zext_synthesis.h5", "#1baf7a", "-")]
with h5py.File(D + "falc_rh.h5") as f:
    lr = f["wavelength_air_nm"][:]; Ir = f["intensity"][:] * 1e8
fig, axes = plt.subplots(2, 1, figsize=(10.5, 9.5), constrained_layout=True)
for ax, sel, title in ((axes[0], lambda l: l < 700, "H-alpha + Fe I 656.92 nm"), (axes[1], lambda l: l > 700, "Ca II 854.2 nm + Si I 853.62 nm")):
    for label, name, color, ls in series:
        l, I = run(name); m = sel(l)
        ax.plot(l[m], I[m], color=color, ls=ls, lw=1.8, label=label)
        print(f"{title[:8]:8s} {label:50s} edges {I[m][0]:.3f} {I[m][-1]:.3f}  min {I[m].min():.3f} at {l[m][I[m].argmin()]:.3f} nm")
    window = (lr > 655.8) & (lr < 657.2) if ax is axes[0] else (lr > 853.4) & (lr < 854.6)
    ax.plot(lr[window], Ir[window], color="k", lw=2.0, label="RH: H and Ca II NLTE, FALC_82")
    ax.set_title(title, fontsize=11); ax.set_xlabel("air wavelength [nm]"); ax.set_ylabel("I$_ν$ [10$^{-8}$ J s$^{-1}$ m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]")
    ax.set_ylim(bottom=0); ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=8.5, frameon=False, loc="lower right" if ax is axes[0] else "lower left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.suptitle("FALC_82_5x5, centre pixel, μ = 1", fontsize=12)
fig.savefig("figures/falc/falc_center_pixel.png", dpi=110); print("wrote figures/falc/falc_center_pixel.png")
