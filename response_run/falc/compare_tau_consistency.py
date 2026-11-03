# Does the tau route recover the z route's column when the tau grid comes from the same given-nₑ atmosphere? P, rho, chi500, z by level; profiles against RH.
import h5py, numpy as np
D = "/mn/stornext/d9/data/harshm/FFNOML/response_run/falc/"
r = h5py.File(D + "falc_rh.h5"); lr = r["wavelength_air_nm"][:]; Ir = r["intensity"][:] * 1e8; q = np.argsort(lr); lr, Ir = lr[q], Ir[q]; H = r["H_n"][...]; Ca = r["CA_n"][...]
zF = h5py.File(D + "FALC_82_5x5.hdf5")["z"][0] / 1e3; pts = (655.8, 657.2, 853.4, 854.0, 854.6)
g = lambda a, n: a[n][2, 2, :, 0]
for zr, tr in (("nez", "netaucext"), ("nez82", "netauc")):
    a = h5py.File(D + f"falc_{zr}_atmosphere.h5"); b = h5py.File(D + f"falc_{tr}_atmosphere.h5")
    print(f"\n{zr} vs {tr}: max |ratio-1| P {np.max(abs(g(b,'pgas')/g(a,'pgas')-1)):.2e} rho {np.max(abs(g(b,'rho')/g(a,'rho')-1)):.2e} chi500 {np.max(abs(g(b,'chi500')/g(a,'chi500')-1)):.2e} ne {np.max(abs(g(b,'ne')/g(a,'ne')-1)):.2e}")
    dz = g(b, 'z') - g(a, 'z'); print("  z(tau route) - z(z route) [km]: top %.1f, FALC top %.1f, 1349 km %.1f, 500 km %.1f, 11 km %.1f, bottom %.1f" % (dz[0] / 1e3, dz[-82] / 1e3, dz[-82 + int(np.argmin(abs(zF - 1349)))] / 1e3, dz[-82 + int(np.argmin(abs(zF - 490)))] / 1e3, dz[-82 + int(np.argmin(abs(zF - 11)))] / 1e3, dz[-1] / 1e3))
    print("  z of levels on tau route, km: top %.0f, FALC top %.0f, 1349 km level %.0f" % (g(b, 'z')[0] / 1e3, g(b, 'z')[-82] / 1e3, g(b, 'z')[-82 + int(np.argmin(abs(zF - 1349)))] / 1e3))
    for name in (zr, tr):
        s = h5py.File(D + f"falc_{name}_synthesis.h5"); l = s["wavelength_m"][:] * 1e9; I = s["intensity"][2, 2, :, 0, 0] * 1e8; h = l < 700; c = l > 700
        P = h5py.File(D + f"falc_{name}_atmosphere.h5")["populations/H"][2, 2, -82:, :]; k = lambda x: int(np.argmin(abs(zF - x)))
        rms = lambda m: np.sqrt(np.mean((I[m] / np.interp(l[m], lr, Ir) - 1) ** 2))
        print(f"  {name:10s}", " ".join(f"{np.interp(x, l, I):.3f}" for x in pts), "| Ha core %.3f Ca core %.3f | rms %.3f %.3f | total H/RH 252, 11 km %.2f %.2f" % (I[h].min(), I[c].min(), rms(h), rms(c), P[k(252)].sum() / H[k(252)].sum(), P[k(11)].sum() / H[k(11)].sum()))
print("\nRH           ", " ".join(f"{np.interp(x, lr, Ir):.3f}" for x in pts))
