# FALC centre pixel against RH (read_rh_output.py): the FFNO with the corona on the z route and on the log tau grid, with
# populations rescaled to abundance x n_H(rho) (response_run/falc/falc_mass_norm.jl).
import h5py, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "response_run/falc/"
def run(name):
    with h5py.File(D + name) as f:
        return f["wavelength_m"][:] * 1e9, f["intensity"][2, 2, :, 0, 0] * 1e8
series = [("z axis, corona, FFNO totals as predicted", "falc_zext_synthesis.h5", "#2a78d6", "--"),
          ("z axis, corona, conservation enforced", "falc_zext_norm_synthesis.h5", "#2a78d6", "-"),
          ("tau axis, corona, conservation enforced", "falc_ltauext_norm_synthesis.h5", "#1baf7a", "-")]
with h5py.File(D + "falc_rh.h5") as f:
    lr = f["wavelength_air_nm"][:]; Ir = f["intensity"][:] * 1e8
fig, axes = plt.subplots(2, 1, figsize=(10.5, 9.5), constrained_layout=True)
for ax, sel, lims, title, loc in ((axes[0], lambda l: l < 700, (655.8, 657.2), "H-alpha + Fe I 656.92 nm", "lower right"),
                                  (axes[1], lambda l: l > 700, (853.4, 854.6), "Ca II 854.2 nm + Si I 853.62 nm", "lower left")):
    for label, name, color, ls in series:
        l, I = run(name); m = sel(l)
        ax.plot(l[m], I[m], color=color, ls=ls, lw=1.6 if ls == "--" else 1.9, label=label)
    w = (lr > lims[0]) & (lr < lims[1])
    ax.plot(lr[w], Ir[w], color="k", lw=2.0, label="RH: H and Ca II NLTE, FALC_82")
    ax.set_title(title, fontsize=11); ax.set_xlabel("air wavelength [nm]"); ax.set_ylabel("I$_ν$ [10$^{-8}$ J s$^{-1}$ m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]")
    ax.set_ylim(bottom=0); ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=9, frameon=False, loc=loc)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.suptitle("FALC_82_5x5, centre pixel, μ = 1", fontsize=12)
fig.savefig("figures/falc/falc_conservation.png", dpi=110); print("wrote")
