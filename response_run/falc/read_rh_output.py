# RH (rhf1d, solveray) on FALC_82: H and Ca II NLTE, no field, Fe I 656.92 and Si I 853.62 nm as Kurucz lines, on our wave grid.
# Reads its XDR outputs (run directory rh_falc/rhf1d/run_falc) into response_run/falc/falc_rh.h5.
import struct, h5py, numpy as np, os
R = "/mn/stornext/d9/data/harshm/rh_falc/rhf1d/run_falc/"
here = os.path.dirname(os.path.abspath(__file__))
class X:
    def __init__(s, path): s.b = open(path, "rb").read(); s.p = 0
    def i(s): v = struct.unpack(">i", s.b[s.p:s.p+4])[0]; s.p += 4; return v
    def d(s, n=1):
        v = np.frombuffer(s.b, ">f8", n, s.p).astype(float); s.p += 8 * n; return v
    def text(s):
        n = s.i(); v = s.b[s.p:s.p+n].decode(); s.p += n + (-n % 4); return v
def pops(path, nl=6, ns=82):
    x = X(path); x.p = len(x.b) - 2 * nl * ns * 8      # n and nstar follow a short header
    return "", path[-10:], x.d(nl*ns).reshape(nl, ns).T, x.d(nl*ns).reshape(nl, ns).T     # (depth, level)
sp = X(R + "spectrum.out"); nspect = sp.i(); lam = sp.d(nspect)
ray = X(R + "spectrum_1.00"); mu = ray.d(1)[0]; I = ray.d(nspect)
ga = pops(R + "pops.H.out"); ca = pops(R + "pops.CA.out")
print("atmos", ga[0].strip(), "atoms", ga[1].strip(), ca[1].strip(), "mu", mu, "Nspect", nspect, "H", ga[2].shape)
with h5py.File(os.path.join(here, "falc_rh.h5"), "w") as f:
    f["wavelength_air_nm"] = lam; f["intensity"] = I; f["intensity"].attrs["units"] = "J s^-1 m^-2 Hz^-1 sr^-1"
    f["H_n"] = ga[2]; f["H_nstar"] = ga[3]; f["CA_n"] = ca[2]; f["CA_nstar"] = ca[3]
