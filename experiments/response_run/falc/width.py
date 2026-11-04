# Line width and position against RH: full width and bisector at fixed intensities, and the core position.
import h5py, numpy as np
r = h5py.File("falc_rh_air.h5"); lr = r["wavelength_air_nm"][:]; Ir = r["intensity"][:] * 1e8; q = np.argsort(lr); lr, Ir = lr[q], Ir[q]
def sp(n):
    s = h5py.File(f"falc_{n}_synthesis.h5"); return s["wavelength_m"][:] * 1e9, s["intensity"][2, 2, :, 0, 0] * 1e8
def cross(l, I, core, level):
    k = core
    while k > 0 and I[k] < level: k -= 1
    a = l[k] + (level - I[k]) * (l[k + 1] - l[k]) / (I[k + 1] - I[k])
    k = core
    while k < len(l) - 1 and I[k] < level: k += 1
    b = l[k - 1] + (level - I[k - 1]) * (l[k] - l[k - 1]) / (I[k] - I[k - 1])
    return a, b
for name, lo, hi, levels in (("Halpha", 656.0, 656.6, (0.8, 1.0, 1.5, 2.0, 2.5)), ("Ca II", 853.9, 854.5, (1.0, 1.25, 1.5, 2.0, 2.5))):
    wr = (lr > lo) & (lr < hi); lR, IR = lr[wr], Ir[wr]; cR = int(np.argmin(IR))
    print(f"--- {name}: RH core at {lR[cR]:.4f} nm, I {IR[cR]:.3f}; RH grid step {np.median(np.diff(lR)) * 1e3:.1f} pm")
    for n in ("givenm", "giventau", "givenz"):
        l, I = sp(n); m = (l > lo) & (l < hi); l, I = l[m], I[m]; c = int(np.argmin(I))
        print(f"  {n}: core {l[c]:.4f} nm ({(l[c] - lR[cR]) * 1e3:+.1f} pm), I {I[c]:.3f}")
        for v in levels:
            a, b = cross(lR, IR, cR, v); A, B = cross(l, I, c, v)
            print(f"    I={v:.2f}: width RH {(b - a) * 1e3:6.1f} code {(B - A) * 1e3:6.1f} pm ({100 * ((B - A) / (b - a) - 1):+5.1f}%); bisector shift {((A + B) - (a + b)) / 2 * 1e3:+6.1f} pm")
