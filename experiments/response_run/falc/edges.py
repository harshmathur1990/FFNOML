import h5py, numpy as np
r = h5py.File("falc_rh_air.h5"); lr = r["wavelength_air_nm"][:]; Ir = r["intensity"][:] * 1e8; q = np.argsort(lr); lr, Ir = lr[q], Ir[q]
print("RH covers", lr.min(), lr.max(), "nm; n", len(lr))
for x in (655.8, 657.2, 853.4, 854.6):
    out = []
    for n in ("givenm", "giventau", "givenz"):
        s = h5py.File(f"falc_{n}_synthesis.h5"); l = s["wavelength_m"][:] * 1e9; I = s["intensity"][2, 2, :, 0, 0] * 1e8; k = int(np.argmin(abs(l - x))); out.append(100 * (I[k] / np.interp(l[k], lr, Ir) - 1))
    print("%.1f nm RH %.3f  log m %+.2f%%  log tau %+.2f%%  height %+.2f%%" % (x, np.interp(x, lr, Ir), *out))
# how far RH itself is from its own continuum, if it has points beyond the line
for lo, hi in ((650, 655.5), (657.5, 662), (845, 853), (855, 866)):
    m = (lr > lo) & (lr < hi); print(lo, hi, "RH points", int(m.sum()), "max I %.3f" % (Ir[m].max() if m.any() else float("nan")))
