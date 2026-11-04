# RH against our continuum on FALC_82 at line-free wavelengths: split the emergent intensity into opacity and source function, same integral for all.
import struct, h5py, numpy as np
R = "/mn/stornext/d9/data/harshm/rh_falc/rhf1d/run_falc_cont/"; C = 299792458.0
sp = open(R + "spectrum.out", "rb").read(); nspect = struct.unpack(">i", sp[:4])[0]
b = open(R + "spectrum_1.00", "rb").read(); p = 8 + 8 * nspect; I_rh = np.frombuffer(b, ">f8", nspect, 8)
n = struct.unpack(">i", b[p:p + 4])[0]; p += 4; nz = 82; chi = {}; S = {}
for _ in range(n):
    i = struct.unpack(">i", b[p:p + 4])[0]; p += 4
    chi[i] = np.frombuffer(b, ">f8", nz, p).astype(float); p += 8 * nz; S[i] = np.frombuffer(b, ">f8", nz, p).astype(float); p += 8 * nz
ids = sorted(chi)
f = h5py.File("falc_cont_ours.h5"); air = f["air_nm"][:]; vac = f["vac_nm"][:] * 1e-9; T = f["T"][:]; z = f["z"][:]
given = f["given"][...]; witt = f["witt"][...]; terms = f["terms"][...]
if given.shape[0] != nz: given = given.T; witt = witt.T
print("terms shape", terms.shape); 
if terms.shape[0] != nz: terms = terms.transpose(2, 1, 0)
planck = lambda l, t: 2 * 6.62607015e-34 * C / l ** 3 / np.expm1(6.62607015e-34 * C / (l * 1.380649e-23 * t))
def em(ch, s):
    dt = 0.5 * (ch[1:] + ch[:-1]) * np.abs(np.diff(z)); tau = np.concatenate([[1e-12], 1e-12 + np.cumsum(dt)])
    return np.trapezoid(s * np.exp(-tau), tau) + s[-1] * np.exp(-tau[-1]), tau
print("%8s %9s | %7s | %7s %7s %7s | %7s %7s | %6s %6s %6s" % ("air nm", "I_RH 1e-8", "formal", "RHχ,B", "gχ,RHS", "gχ,B", "wχ,RHS", "wχ,B", "χg/RH", "χw/RH", "S/B"))
for j, i in enumerate(ids):
    B = planck(vac[j], T); a, tr = em(chi[i], S[i]); k = int(np.argmin(abs(tr - 1)))
    print("%8.2f %9.4f | %7.4f | %7.4f %7.4f %7.4f | %7.4f %7.4f | %6.3f %6.3f %6.3f" % (air[j], I_rh[i] * 1e8, a / I_rh[i], em(chi[i], B)[0] / a, em(given[:, j], S[i])[0] / a, em(given[:, j], B)[0] / a, em(witt[:, j], S[i])[0] / a, em(witt[:, j], B)[0] / a, given[k, j] / chi[i][k], witt[k, j] / chi[i][k], S[i][k] / B[k]))
print("\nratios are to the RH chi, S emergent intensity; columns at tau_RH = 1; g = Muspel given, w = Wittmann\n")
names = ("atoms bf", "H- bf", "H ff", "H2+", "Rayleigh H", "Thomson")
for w in (399.88, 422.17, 499.86, 652.45, 853.4):
    j = int(np.argmin(abs(air - w))); i = ids[j]; print("%.2f nm: depth: z km, T, tau_RH | chi g/RH, w/RH | S/B | term shares of g" % air[j]); a, tr = em(chi[i], S[i])
    for t in (0.01, 0.1, 0.3, 1, 3):
        k = int(np.argmin(abs(tr - t))); sh = terms[k, j] / terms[k, j].sum()
        print("  %6.0f %6.0f %8.3f | %6.3f %6.3f | %6.3f | " % (z[k] / 1e3, T[k], tr[k], given[k, j] / chi[i][k], witt[k, j] / chi[i][k], S[i][k] / planck(vac[j], T[k])) + " ".join("%s %.3f" % (nm, v) for nm, v in zip(names, sh) if v > 0.005))
