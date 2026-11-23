# FNOML-Paper Appendix D numbers relative to the far-wing intensity: FALC_82, centre pixel, mu = 1, zaug (falc_zaug_native)
# and l004p (falc_l004p_native_rerun) against RH with the same n_e (falc_rh_air.h5). I_wing is RH's intensity at the
# wavelength of the FFNO grid farthest from the RH line core; the line-core difference and, within |v| <= 50 km/s
# (H-alpha) or 25 km/s (Ca II 8542), the rms and largest |I_FFNO - I_RH| are given in % of it, as in paper_fig_falc.py.
import h5py, numpy as np
R = "/mn/stornext/d9/data/harshm/"; c = 2.99792458e5
with h5py.File(R + "FFNOML/response_run/falc/falc_rh_air.h5") as f:
    lr = f["wavelength_air_nm"][:]; Ir = f["intensity"][:] * 1e8
q = np.argsort(lr); lr, Ir = lr[q], Ir[q]
for name, run, lo, hi, vmax in (("H-alpha", "falc_necmp_falcz", 656.0, 656.6, 50.0), ("Ca II 8542", "falc_ca_falcz", 854.0, 854.4, 25.0)):
    w = (lr > lo) & (lr < hi); l0 = lr[w][np.argmin(Ir[w])]
    for model, d in (("zaug", "falc_zaug_native"), ("l004p", "falc_l004p_native_rerun")):
        with h5py.File(R + f"ffnoml_runs/{d}/falc/{run}_synthesis.h5") as f:
            lf = f["wavelength_m"][:] * 1e9; If = f["intensity"][2, 2, :, 0, 0] * 1e8
        k = np.where((lf > lo - 0.6) & (lf < hi + 0.6))[0]; lf, If = lf[k], If[k]
        far = lf[np.argmax(np.abs(lf - l0))]; Iw = np.interp(far, lr, Ir)
        vf = c * (lf / l0 - 1); s = np.abs(vf) <= vmax
        d_ = If[s] - np.interp(lf[s], lr, Ir); core_f, core_r = If[s].min(), Ir[w].min()
        print(f"{name:10s} {model:5s} I_wing (RH at {c * (far / l0 - 1):+.0f} km/s) {Iw:.3f}; core {core_f:.3f} vs RH {core_r:.3f}: "
              f"{100 * (core_f - core_r) / Iw:+.1f}% of I_wing ({core_f / core_r:.2f} x RH); |v| <= {vmax:.0f} km/s: "
              f"rms {100 * np.sqrt(np.mean(d_ ** 2)) / Iw:.1f}%, max {100 * np.abs(d_).max() / Iw:.1f}% of I_wing")
