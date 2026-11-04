# RH's chi and S (solveray output after the mu = 1 intensities) against this code's, by depth, at near-continuum wavelengths; and the
# emergent intensity for each combination of the two codes' chi and S, to split the difference between opacity and source function.
import struct, h5py, numpy as np
R = "/mn/stornext/d9/data/harshm/rh_falc/rhf1d/run_falc/"; D = "/mn/stornext/d9/data/harshm/FFNOML/response_run/falc/"
sp = open(R + "spectrum.out", "rb").read(); nspect = struct.unpack(">i", sp[:4])[0]
b = open(R + "spectrum_1.00", "rb").read(); p = 8 + 8 * nspect; I_rh = np.frombuffer(b, ">f8", nspect, 8)
n = struct.unpack(">i", b[p:p+4])[0]; p += 4; nz = 82; chi = {}; S = {}
for _ in range(n):
    i = struct.unpack(">i", b[p:p+4])[0]; p += 4
    chi[i] = np.frombuffer(b, ">f8", nz, p).astype(float); p += 8 * nz; S[i] = np.frombuffer(b, ">f8", nz, p).astype(float); p += 8 * nz
f = h5py.File(D + "falc_continuum_compare.h5"); air = f["air_nm"][:]; vac = f["vacuum_m"][:]; z = f["z"][:]
c = f["chi_total"][...].T if f["chi_total"].shape[0] == 4 else f["chi_total"][...]; e = f["eta_total"][...].T if f["eta_total"].shape[0] == 4 else f["eta_total"][...]
cc = f["chi_continuum"][...].T if f["chi_continuum"].shape[0] == 4 else f["chi_continuum"][...]
C = 299792458.0; T = h5py.File(D + "FALC_82_5x5.hdf5")["temperature"][0, 2, 2, :].astype(float)
def planck_nu(l, t):
    h = 6.62607015e-34; k = 1.380649e-23; return 2 * h * C / l**3 / np.expm1(h * C / (l * k * t))
def emergent(ch, s):
    dtau = 0.5 * (ch[1:] + ch[:-1]) * np.abs(np.diff(z)); tau = np.concatenate([[1e-12], 1e-12 + np.cumsum(dtau)])
    return np.trapezoid(s * np.exp(-tau), tau) + s[-1] * np.exp(-tau[-1]), tau
for j, i in enumerate((1229, 570, 1288, 1955)):
    if i not in chi: continue
    conv = 1e12 * vac[j]**2 / C                                 # this code's eta/chi in kW m^-2 sr^-1 nm^-1 -> SI per Hz
    so = e[:, j] / c[:, j] * conv; B = planck_nu(vac[j], T)
    a = emergent(chi[i], S[i])[0]; o = emergent(c[:, j], so)[0]; m1 = emergent(chi[i], so)[0]; m2 = emergent(c[:, j], S[i])[0]
    print(f"\n{air[j]:.2f} nm: I_RH(file) {I_rh[i]*1e8:.3f}  formal(RH chi,S) {a*1e8:.3f}  formal(our chi,S) {o*1e8:.3f}  our/RH {o/a:.4f} | our chi+RH S {m2*1e8:.3f} ({m2/a:.4f})  RH chi+our S {m1*1e8:.3f} ({m1/a:.4f})")
    tr = emergent(chi[i], S[i])[1]; to = emergent(c[:, j], so)[1]
    print("  z_km T tau_RH tau_ours | chi ours/RH | S_RH/B  S_ours/B")
    for k in (40, 50, 55, 60, 63, 66, 68, 70, 72, 74, 78, 81):
        print(f"  {z[k]/1e3:6.0f} {T[k]:6.0f} {tr[k]:.3e} {to[k]:.3e} | {c[k,j]/chi[i][k]:.3f} | {S[i][k]/B[k]:.3f} {so[k]/B[k]:.3f}")
