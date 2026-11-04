# Where the FALC differences from RH come from: RH's abundances in rho, its chi500 against ours, and the turbulent pressure in g m.
import h5py, numpy as np, struct
rh = "/mn/stornext/d9/data/harshm/rh_falc/"; b = open(rh + "rhf1d/run_falc/geometry.out", "rb").read()
_, nr, n = struct.unpack(">iii", b[:12]); g = np.frombuffer(b, ">f8", 3 * n, 12 + 16 * nr); zr, mr, tr = g[:n].copy(), g[n:2 * n].copy(), g[2 * n:3 * n].copy()
W = dict(H=1.008, HE=4.003, C=12.011, N=14.007, O=15.999, NE=20.179, NA=22.99, MG=24.305, AL=26.982, SI=28.086, P=30.974, S=32.06, CL=35.453, AR=39.948, K=39.102, CA=40.08, CR=51.996, MN=54.938, FE=55.847, NI=58.71)
ab = [l.split() for l in open(rh + "Atoms/abundance.input") if l.strip() and not l.startswith("#")]
wr = sum(10 ** (float(v) - 12) * W[e] for e, v in ab if e in W); At = sum(10 ** (float(v) - 12) for e, v in ab)
print("RH wght_per_H (main elements) %.4f amu, total abundance %.4f; Wittmann 1.4090, 1.0950" % (wr, At))
a = h5py.File("falc_givenm_atmosphere.h5"); rho = a["rho"][2, 2, :, 0]; z = a["z"][2, 2, :, 0]; lt = a["logtau500"][2, 2, :, 0]; P = a["pgas"][2, 2, :, 0]; chi = a["chi500"][2, 2, :, 0]; v = a["vturb"][2, 2, :, 0]
def geo(rho, chi):
    Z = np.zeros(n); t = np.zeros(n); t[0] = chi[0] / rho[0] * mr[0]
    for k in range(1, n):
        Z[k] = Z[k - 1] - 2 * (mr[k] - mr[k - 1]) / (rho[k - 1] + rho[k]); t[k] = t[k - 1] + .5 * (chi[k - 1] + chi[k]) * (Z[k - 1] - Z[k])
    return Z - np.interp(1., t, Z), t
Z, t = geo(rho, chi); print("repro of the run: max|dz| %.3f km" % (abs(Z - z).max() / 1e3))
Z2, t2 = geo(rho * wr / 1.4090, chi); print("z-RH with our rho: rms %.2f max %.2f km; with RH's wght_per_H: rms %.2f max %.2f km" % (np.sqrt(np.mean((z - zr) ** 2)) / 1e3, abs(z - zr).max() / 1e3, np.sqrt(np.mean((Z2 - zr) ** 2)) / 1e3, abs(Z2 - zr).max() / 1e3))
for lo, hi in ((-6, -4), (-4, -2), (-2, 0), (0, 2)):
    s = (np.log10(tr) >= lo) & (np.log10(tr) < hi); print("log tau %d..%d: mean log tau diff (ours-RH) %+.3f, n=%d" % (lo, hi, (lt - np.log10(tr))[s].mean(), s.sum()))
tb = P + .5 * rho * v ** 2; print("P/(g m) %.3f..%.3f; (P + rho v^2/2)/(g m) %.3f..%.3f" % ((P / (274.0 * mr)).min(), (P / (274.0 * mr)).max(), (tb / (274.0 * mr)).min(), (tb / (274.0 * mr)).max()))
zz = h5py.File("falc_givenz_atmosphere.h5")["logtau500"][2, 2, :, 0]; print("z axis log tau-RH by level 1,2,5,10,30,60,82:", np.round((zz - np.log10(tr))[[0, 1, 4, 9, 29, 59, 81]], 3))
ta = h5py.File("falc_giventau_atmosphere.h5")["z"][2, 2, :, 0]; print("tau axis z-RH km at levels 1,5,10,20,30,40,60,82:", np.round((ta - zr)[[0, 4, 9, 19, 29, 39, 59, 81]] / 1e3, 1))
