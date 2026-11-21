# Paper appendix figure: FALC_82 H-alpha (left) and Ca II 8542 (right), centre pixel, mu = 1, FALC's own 82-point height grid (-0.10 to 2.16 Mm, no
# population mesh) with FALC's n_e given: the zaug H model's populations (falc/falc_necmp_falcz_*) and the zaug H and Ca II
# models' (falc/falc_ca_falcz_*, run_ca.sh) against RH on FALC_82 with the same n_e (falc_rh_air.h5). Doppler velocity from
# the RH line-core wavelength. usage: python paper_fig_falc.py
import h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "/mn/stornext/d9/data/harshm/"
OUT = R + "FNOML-Paper/images/falc_halpha_ca8542_zaug.pdf"
c = 2.99792458e5
with h5py.File(R + "FFNOML/response_run/falc/falc_rh_air.h5") as f:
    lr = f["wavelength_air_nm"][:]; Ir = f["intensity"][:] * 1e8
q = np.argsort(lr); lr, Ir = lr[q], Ir[q]
lines = [("H$\\alpha$", "falc_necmp_falcz", 656.0, 656.6, 50.0), ("Ca II 8542 Å", "falc_ca_falcz", 854.0, 854.4, 25.0)]
matplotlib.rc("font", size=8)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)
for ax, (name, run, lo, hi, vmax) in zip(axs, lines):
    with h5py.File(R + f"ffnoml_runs/falc_zaug_native/falc/{run}_synthesis.h5") as f:
        lf = f["wavelength_m"][:] * 1e9; If = f["intensity"][2, 2, :, 0, 0] * 1e8
    w = (lr > lo) & (lr < hi); l0 = lr[w][np.argmin(Ir[w])]
    vr, vf = c * (lr / l0 - 1), c * (lf / l0 - 1)
    sr, sf = np.abs(vr) <= vmax, np.abs(vf) <= vmax
    e = If[sf] / np.interp(lf[sf], lr, Ir) - 1
    print(f"{name}: l0 {l0:.4f} nm; FFNO core {If[sf].min():.3f}, RH core {Ir[sr].min():.3f} ({100 * (If[sf].min() / Ir[sr].min() - 1):+.1f}%); "
          f"|v| <= {vmax:.0f} km/s: rms {100 * np.sqrt(np.mean(e ** 2)):.1f}%, max {100 * np.abs(e).max():.1f}%")
    ax.plot(vr[sr], Ir[sr], color="k", lw=1.4, label="RH")
    ax.plot(vf[sf], If[sf], color="#d62728", ls="--", lw=1.2, label="FFNO")
    ax.set_xlim(-vmax, vmax); ax.set_ylim(bottom=0); ax.set_title(name)
    ax.set_xlabel("Doppler velocity [km s$^{-1}$]")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
axs[0].set_ylabel("$I$ [10$^{-8}$ W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]")
axs[0].legend(frameon=False, loc="lower left")
fig.savefig(OUT); print("wrote", OUT)
