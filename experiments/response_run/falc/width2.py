# The same width measure on the controls: the FFNOInversion synthesis with RH's or the FFNO's populations, against RH.
import h5py, numpy as np
exec(open("width.py").read().split("for name, lo, hi, levels")[0])
f = h5py.File("falc_ca_controls.h5"); lc = f["wavelength_air_nm"][:]
lo, hi = 853.9, 854.5; wr = (lr > lo) & (lr < hi); lR, IR = lr[wr], Ir[wr]; cR = int(np.argmin(IR)); print("scale check: control max", f["RH H, RH Ca, ABO"][:].max(), "RH max", Ir[(lr>853.4)&(lr<854.6)].max())
for k in ("RH H, RH Ca, ABO", "RH H, RH Ca, Unsold", "RH H, FFNO Ca, ABO", "FFNO H, RH Ca, ABO", "FFNO H, FFNO Ca, ABO"):
    I = f[k][:]; I = I * (1e8 if I.max() < 1e-3 else 1); m = (lc > lo) & (lc < hi); l, J = lc[m], I[m]; c = int(np.argmin(J)); out = []
    for v in (1.0, 1.5, 2.0, 2.5):
        a, b = cross(lR, IR, cR, v); A, B = cross(l, J, c, v); out.append("%.2f:%+.0f%%" % (v, 100 * ((B - A) / (b - a) - 1)))
    print("%-24s core %.3f (RH %.3f) shift %+.1f pm  width vs RH %s" % (k, J[c], IR[cR], (l[c] - lR[cR]) * 1e3, " ".join(out)))
g = h5py.File("falc_h_controls.h5"); l = g["wavelength_air_nm"][:]; I = g["intensity"][:]; I = I * (1e8 if I.max() < 1e-3 else 1); m = (l > 656.0) & (l < 656.6); l, J = l[m], I[m]; c = int(np.argmin(J))
wr = (lr > 656.0) & (lr < 656.6); lR, IR = lr[wr], Ir[wr]; cR = int(np.argmin(IR)); out = []
for v in (1.0, 1.5, 2.0, 2.5):
    a, b = cross(lR, IR, cR, v); A, B = cross(l, J, c, v); out.append("%.2f:%+.0f%%" % (v, 100 * ((B - A) / (b - a) - 1)))
print("Halpha RH populations: core %.3f (RH %.3f) shift %+.1f pm  width vs RH %s" % (J[c], IR[cR], (l[c] - lR[cR]) * 1e3, " ".join(out)))
# damping wing: Doppler width with vturb at the Ca II formation region
a = h5py.File("falc_givenm_atmosphere.h5"); T = a["temperature"][2, 2, :, 0]; v = a["vturb"][2, 2, :, 0]; z = a["z"][2, 2, :, 0]
d = lambda lam, M, t, vt: lam * np.sqrt(2 * 1.380649e-23 * t / (M * 1.66054e-27) + vt ** 2) / 299792458 * 1e3
for zk in (200, 500, 1000, 1500):
    k = int(np.argmin(abs(z - zk * 1e3))); print("z %4d km T %6.0f vturb %.1f km/s: Doppler width Ca 854.2 %.1f pm (without vturb %.1f), Halpha %.1f pm (without %.1f)" % (zk, T[k], v[k] / 1e3, d(854.2, 40.08, T[k], v[k]), d(854.2, 40.08, T[k], 0), d(656.28, 1.008, T[k], v[k]), d(656.28, 1.008, T[k], 0)))
