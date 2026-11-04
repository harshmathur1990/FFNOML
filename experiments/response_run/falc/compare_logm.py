# FALC on its column-mass axis (given nₑ and H) against RH: heights and profiles.
import h5py, numpy as np, os, struct
D = os.path.dirname(os.path.abspath(__file__)) + "/"
rh = "/mn/stornext/d9/data/harshm/rh_falc/rhf1d/run_falc/"
b = open(rh + "geometry.out", "rb").read()
_, nr, nd = struct.unpack(">iii", b[:12]); off = 12 + 16 * nr
g = np.frombuffer(b, ">f8", 3 * nd, off); zrh, mrh, trh = g[:nd], g[nd:2 * nd], g[2 * nd:3 * nd]
print("RH geometry: Nrays", nr, "Ndep", nd, "z[km]", zrh[0] / 1e3, zrh[-1] / 1e3, "cmass[kg/m2]", mrh[0], mrh[-1], "tau", trh[0], trh[-1])
r = h5py.File(D + "falc_rh.h5"); lr = r["wavelength_air_nm"][:]; Ir = r["intensity"][:] * 1e8
print("RH file keys", list(r), "attrs", dict(r.attrs))
for n in ("givenm", "giventau", "givenz"):
    if not os.path.exists(D + f"falc_{n}_atmosphere.h5"): continue
    a = h5py.File(D + f"falc_{n}_atmosphere.h5"); s = h5py.File(D + f"falc_{n}_synthesis.h5")
    z = a["z"][2, 2, :, 0]; rho = a["rho"][2, 2, :, 0]; P = a["pgas"][2, 2, :, 0]; lt = a["logtau500"][2, 2, :, 0]; m = mrh
    print(f"--- {n}: P/RH-style g m below")
    tp = (trh > 1e-5); print("  z-RH [km]: rms %.2f, max %.2f; above tau=1e-5 excluded: rms %.2f max %.2f" % (np.sqrt(np.mean((z - zrh) ** 2)) / 1e3, abs(z - zrh).max() / 1e3, np.sqrt(np.mean((z - zrh)[tp] ** 2)) / 1e3, abs(z - zrh)[tp].max() / 1e3))
    print("  log tau-RH: rms %.3f max %.3f; P/(g m) range %.3f %.3f" % (np.sqrt(np.mean((lt - np.log10(trh)) ** 2)), abs(lt - np.log10(trh)).max(), (P / (274 * m)).min(), (P / (274 * m)).max()))
    w = s["wavelength_m"][:] * 1e9; I = s["intensity"][2, 2, :, 0, 0] * 1e8
    Ri = np.interp(w, lr, Ir); e = I / Ri - 1
    at = lambda x: np.argmin(abs(w - x))
    for lab, sel in (("Halpha", (w > 655.8) & (w < 657.2)), ("Ca II", (w > 853.4) & (w < 854.6))):
        c = np.where(sel)[0][np.argmin(I[sel])]; cr = np.argmin(np.where((lr > w[sel][0]) & (lr < w[sel][-1]), Ir, np.inf))
        print("  %s: rms %.3f; first point %.3f RH %.3f; core %.3f RH %.3f (%+.1f%%)" % (lab, np.sqrt(np.mean(e[sel] ** 2)), I[sel][0], Ri[sel][0], I[c], Ir[cr], 100 * (I[c] / Ir[cr] - 1)))
