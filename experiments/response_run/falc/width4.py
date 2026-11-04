# Line width with each profile's own levels: width at a fraction of its own depth (core to the reference at the window edges). 50% is the FWHM.
import h5py, numpy as np
exec(open("width.py").read().split("for name, lo, hi, levels")[0])
fr = (0.1, 0.25, 0.5, 0.75)
def widths(l, I, lo, hi):
    m = (l >= lo) & (l <= hi); l, I = l[m], I[m]; c = int(np.argmin(I)); ref = min(I[0], I[-1]); out = []
    for f in fr: a, b = cross(l, I, c, I[c] + f * (ref - I[c])); out.append((b - a) * 1e3)
    return np.array(out), I[c], ref
for name, lo, hi in (("Halpha", 655.8, 656.8), ("Ca II", 853.7, 854.6)):
    R, cr, rr = widths(lr, Ir, lo, hi)
    print(f"--- {name} window {lo}-{hi} nm; RH core {cr:.3f} reference {rr:.3f}")
    print("%-22s" % "fraction of depth" + "".join("%18s" % ("%d%%" % (100 * f)) for f in fr))
    print("%-22s" % "RH [pm]" + "".join("%18.1f" % v for v in R))
    for lab, n in (("log column mass", "givenm"), ("log tau", "giventau"), ("height", "givenz")):
        W, c, r0 = widths(*sp(n), lo, hi)
        print("%-22s" % lab + "".join("%10.1f (%+4.0f%%)" % (v, 100 * (v / w - 1)) for v, w in zip(W, R)))
for name, lo, hi, ref in (("Halpha", 656.0, 656.6, 656.28), ("Ca II", 853.9, 854.5, 854.209)):
    m = (lr > lo) & (lr < hi); a = lr[m]; b = Ir[m]; c = int(np.argmin(b)); A, B = cross(a, b, c, b[c] + 0.5 * (min(b[0], b[-1]) - b[c])); print(name, "RH core %.4f nm, half-depth bisector %.4f, target %.3f" % (a[c], (A + B) / 2, ref))
    for n in ("givenm", "giventau", "givenz"):
        l, I = sp(n); q = (l > lo) & (l < hi); a, b = l[q], I[q]; c = int(np.argmin(b)); A, B = cross(a, b, c, b[c] + 0.5 * (min(b[0], b[-1]) - b[c])); print("  ", n, "core %.4f nm I %.3f, half-depth bisector %.4f" % (a[c], b[c], (A + B) / 2))
