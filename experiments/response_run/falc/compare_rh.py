# All FALC runs against RH: profile numbers and populations (FALC's 82 levels are the last 82 of each column).
import h5py, numpy as np, sys, os
D = os.path.dirname(os.path.abspath(__file__)) + "/"
r = h5py.File(D + "falc_rh.h5"); lr = r["wavelength_air_nm"][:]; Ir = r["intensity"][:] * 1e8; H = r["H_n"][...]; Ca = r["CA_n"][...]
z = h5py.File(D + "FALC_82_5x5.hdf5")["z"][0] / 1e3
def nums(l, I):
    at = lambda x: np.argmin(abs(l - x)); h = (l > 655.8) & (l < 657.2); c = (l > 853.4) & (l < 854.6)
    def depth(lo, hi):
        b = (l > lo) & (l < hi); n = I[(l > lo - 0.05) & (l < lo)].mean(); return 1 - I[b].min() / n
    return [I[at(655.8)], I[at(657.2)], I[h].min(), depth(656.85, 656.97), I[at(853.4)], I[at(854.6)], I[at(854.0)], I[c].min(), depth(853.58, 853.66)]
print("%-14s Ha_b  Ha_r  Ha_c  Fe    Ca_b  Ca_r  C854.0 Ca_c  Si   | rms/RH Ha Ca | spread | sumH/RH 252km 11km | sumCa/sumH_RH 252km | n3Ca/RH 1349km S54/RH 1349km" % "run")
print("%-14s " % "RH" + " ".join("%.3f" % v for v in nums(lr, Ir)))
for n in sys.argv[1:]:
    s = h5py.File(D + f"falc_{n}_synthesis.h5"); w = s["wavelength_m"][:] * 1e9; Ia = s["intensity"][...]; I = Ia[2, 2, :, 0, 0] * 1e8
    rms = lambda sel: np.sqrt(np.mean((I[sel] / np.interp(w[sel], lr, Ir) - 1) ** 2))
    a = h5py.File(D + f"falc_{n}_atmosphere.h5"); A = a["populations/H"][2, 2, :, :]; C = a["populations/CA"][2, 2, :, :]; o = A.shape[0] - 82
    k = lambda h: int(np.argmin(abs(z - h))); p, q, m = k(252), k(11), k(1349)
    print("%-14s " % n + " ".join("%.3f" % v for v in nums(w, I)), "| %.3f %.3f | %.3f | %.2f %.2f | %.2e | %.2f %.2f" % (rms(w < 700), rms(w > 700), (np.ptp(Ia, axis=(0, 1)) / Ia.mean(axis=(0, 1))).max(),
        A[o + p].sum() / H[p].sum(), A[o + q].sum() / H[q].sum(), C[o + p].sum() / H[p].sum(), C[o + m, 2] / Ca[m, 2], (C[o + m, 4] / C[o + m, 2]) / (Ca[m, 4] / Ca[m, 2])))
