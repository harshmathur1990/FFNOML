# FALC on 5 x 5 and 96 x 96 columns (centre column) against RH: profiles, populations, spread over identical columns.
import h5py, numpy as np
D = "/mn/stornext/d9/data/harshm/FFNOML/response_run/falc/"
r = h5py.File(D + "falc_rh.h5"); lr = r["wavelength_air_nm"][:]; Ir = r["intensity"][:] * 1e8; H = r["H_n"][...]; Ca = r["CA_n"][...]
z = h5py.File(D + "FALC_82_5x5.hdf5")["z"][0] / 1e3
print("RH      Ha edges %.3f %.3f core %.3f | Ca edges %.3f %.3f core %.3f" % (Ir[np.argmin(abs(lr-655.8))], Ir[np.argmin(abs(lr-657.2))], Ir[(lr>655.8)&(lr<657.2)].min(), Ir[np.argmin(abs(lr-853.4))], Ir[np.argmin(abs(lr-854.6))], Ir[(lr>853.4)&(lr<854.6)].min()))
for n, cx in (("ltauc", 2), ("ltau96", 48), ("zextc", 2), ("zext96", 48)):
    s = h5py.File(D + f"falc_{n}_synthesis.h5"); w = s["wavelength_m"][:] * 1e9; A = s["intensity"][...]; I = A[cx, cx, :, 0, 0] * 1e8
    h = w < 700; c = w > 700; rms = lambda m: np.sqrt(np.mean((I[m] / np.interp(w[m], lr, Ir) - 1) ** 2))
    spread = (np.ptp(A, axis=(0, 1)) / A.mean(axis=(0, 1))).max()
    a = h5py.File(D + f"falc_{n}_atmosphere.h5"); P = a["populations/H"][cx, cx, :, :]; C = a["populations/CA"][cx, cx, :, :]; o = P.shape[0] - 82
    k = lambda q: int(np.argmin(abs(z - q))); p, m, t = k(252), k(1349), k(978)
    print(f"{n:7s} Ha edges {I[h][0]:.3f} {I[h][-1]:.3f} core {I[h].min():.3f} | Ca edges {I[c][0]:.3f} {I[c][-1]:.3f} core {I[c].min():.3f} | rms {rms(h):.3f} {rms(c):.3f} | spread {spread:.4f} | sumH/RH 252km {P[o+p].sum()/H[p].sum():.2f} | Ca n5/n3 /RH 1349km {(C[o+m,4]/C[o+m,2])/(Ca[m,4]/Ca[m,2]):.2f} n3 {C[o+m,2]/Ca[m,2]:.2f} | Ha n2/RH 978km {P[o+t,1]/H[t,1]:.2f} n3/n2 {(P[o+t,2]/P[o+t,1])/(H[t,2]/H[t,1]):.2f}")
