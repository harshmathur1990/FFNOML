# FALC through the real code path with nₑ given, against RH with the same nₑ and against the offline test of option B.
import h5py, numpy as np
D = "/mn/stornext/d9/data/harshm/FFNOML/response_run/falc/"
r = h5py.File(D + "falc_rh.h5"); lr = r["wavelength_air_nm"][:]; Ir = r["intensity"][:] * 1e8; q = np.argsort(lr); lr, Ir = lr[q], Ir[q]; H = r["H_n"][...]; Ca = r["CA_n"][...]
z = h5py.File(D + "FALC_82_5x5.hdf5")["z"][0] / 1e3; pts = (655.8, 657.2, 853.4, 854.0, 854.6)
def show(name, l, I):
    h = l < 700; c = l > 700; rms = lambda m: np.sqrt(np.mean((I[m] / np.interp(l[m], lr, Ir) - 1) ** 2))
    print(f"{name:13s}", " ".join(f"{np.interp(x, l, I):.3f}" for x in pts), "| Ha core %.3f Ca core %.3f | rms %.3f %.3f" % (I[h].min(), I[c].min(), rms(h), rms(c)))
m = (lr > 655.8) & (lr < 657.2) | (lr > 853.4) & (lr < 854.6); show("RH", lr[m], Ir[m])
with h5py.File(D + "falc_neB_synthesis.h5") as f: show("offline B", f["wavelength_m"][:] * 1e9, f["intensity_center"][:] * 1e8)
for n in ("nez", "netauext", "netau"):
    s = h5py.File(D + f"falc_{n}_synthesis.h5"); A = s["intensity"][...]; show(n, s["wavelength_m"][:] * 1e9, A[2, 2, :, 0, 0] * 1e8)
    a = h5py.File(D + f"falc_{n}_atmosphere.h5"); P = a["populations/H"][2, 2, :, :]; C = a["populations/CA"][2, 2, :, :]; o = P.shape[0] - 82
    k = lambda h: int(np.argmin(abs(z - h))); ne = a["ne"][2, 2, -82:, 0]; zz = a["z"][2, 2, -82:, 0] / 1e3
    print("   column spread %.3f | total H/RH 1349, 252, 11 km: %.2f %.2f %.2f | z at FALC 1349 km, top: %.0f %.0f km" % ((np.ptp(A, axis=(0, 1)) / A.mean(axis=(0, 1))).max(), P[o + k(1349)].sum() / H[k(1349)].sum(), P[o + k(252)].sum() / H[k(252)].sum(), P[o + k(11)].sum() / H[k(11)].sum(), zz[k(1349)], zz[0]))
