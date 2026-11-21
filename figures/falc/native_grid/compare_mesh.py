# FALC_82, FALC's n_e, height axis, zaug H and Ca II networks: H-alpha and Ca II 8542 with the population mesh on
# (falc_mesh_*) and off (FALC's own grid), against RH (falc_rh_air.h5), and mesh against own grid. Writes
# falc_mesh_vs_native.png. usage: python compare_mesh.py
import h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "/mn/stornext/d9/data/harshm/"; D = R + "ffnoml_runs/falc_zaug_native/"
c = 2.99792458e5
with h5py.File(R + "FFNOML/response_run/falc/falc_rh_air.h5") as f:
    lr = f["wavelength_air_nm"][:]; Ir = f["intensity"][:] * 1e8
q = np.argsort(lr); lr, Ir = lr[q], Ir[q]
def spec(run):
    with h5py.File(D + f"falc/falc_{run}_synthesis.h5") as f:
        return f["wavelength_m"][:] * 1e9, f["intensity"][2, 2, :, 0, 0] * 1e8
lines = [("H$\\alpha$", "necmp_falcz", 656.0, 656.6, 50.0), ("Ca II 8542 Å", "ca_falcz", 854.0, 854.4, 25.0)]
matplotlib.rc("font", size=8)
fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex="col", height_ratios=[2, 1], constrained_layout=True)
for k, (name, run, lo, hi, vmax) in enumerate(lines):
    w = (lr > lo) & (lr < hi); l0 = lr[w][np.argmin(Ir[w])]
    sr = np.abs(c * (lr / l0 - 1)) <= vmax
    axs[0, k].plot(c * (lr[sr] / l0 - 1), Ir[sr], "k", lw=1.4, label="RH")
    print(f"{name}: RH core {Ir[sr].min():.3f}")
    nat = None
    for label, r, col, ls in (("FFNO, own grid", run, "#d62728", "--"), ("FFNO, population mesh", "mesh_" + run, "#1f77b4", ":")):
        l, I = spec(r); v = c * (l / l0 - 1); s = np.abs(v) <= vmax
        e = I[s] / np.interp(l[s], lr, Ir) - 1
        axs[0, k].plot(v[s], I[s], color=col, ls=ls, lw=1.3, label=label)
        axs[1, k].plot(v[s], 100 * e, color=col, ls=ls, lw=1.3)
        msg = (f"  {label:22s} core {I[s].min():.3f} ({100 * (I[s].min() / Ir[sr].min() - 1):+5.1f}% vs RH), "
               f"rms {100 * np.sqrt(np.mean(e ** 2)):4.1f}%, max |err| {100 * np.abs(e).max():4.1f}%")
        if nat is None: nat = (l, I)
        else:
            d = I[s] / np.interp(l[s], *nat) - 1
            msg += f"; mesh vs own grid: core {100 * (I[s].min() / nat[1][s].min() - 1):+.1f}%, rms {100 * np.sqrt(np.mean(d ** 2)):.1f}%, max {100 * np.abs(d).max():.1f}%"
        print(msg)
    axs[0, k].set_title(name); axs[0, k].set_ylim(bottom=0); axs[1, k].axhline(0, color="k", lw=0.7)
    axs[1, k].set_xlabel("Doppler velocity [km s$^{-1}$]"); axs[1, k].set_xlim(-vmax, vmax)
    for a in axs[:, k]:
        for s_ in ("top", "right"): a.spines[s_].set_visible(False)
axs[0, 0].set_ylabel("$I$ [10$^{-8}$ W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]"); axs[1, 0].set_ylabel("FFNO / RH − 1 [%]")
axs[0, 0].legend(frameon=False, loc="lower left")
fig.savefig(D + "falc_mesh_vs_native.png", dpi=150); print("wrote", D + "falc_mesh_vs_native.png")
