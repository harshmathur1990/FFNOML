# FALC's own 82 levels with the automatic corona against RH with the same nₑ: z and log tau with FALC's nₑ, and log tau with HSE nₑ against RH with the EOS nₑ.
import h5py, numpy as np
D = "/mn/stornext/d9/data/harshm/FFNOML/response_run/falc/"
r = h5py.File(D + "falc_rh.h5"); lr = r["wavelength_air_nm"][:]; Ir = r["intensity"][:] * 1e8; q = np.argsort(lr); lr, Ir = lr[q], Ir[q]; H = r["H_n"][...]; Ca = r["CA_n"][...]
lE, IE = np.load(D + "rh_ne_eos_spectrum.npy"); zF = h5py.File(D + "FALC_82_5x5.hdf5")["z"][0] / 1e3; pts = (655.8, 657.2, 853.4, 854.0, 854.6)
print("RH FALC ne ", " ".join(f"{np.interp(x, lr, Ir):.3f}" for x in pts), "| Ha core %.3f Ca core %.3f" % (Ir[(lr>655.8)&(lr<657.2)].min(), Ir[(lr>853.4)&(lr<854.6)].min()))
print("RH EOS ne  ", " ".join(f"{np.interp(x, lE, IE):.3f}" for x in pts), "| Ha core %.3f Ca core %.3f" % (IE[(lE>655.8)&(lE<657.2)].min(), IE[(lE>853.4)&(lE<854.6)].min()))
for n, (lq, Iq) in (("nez", (lr, Ir)), ("autoz", (lr, Ir)), ("autotau", (lr, Ir)), ("autotauhse", (lE, IE))):
    s = h5py.File(D + f"falc_{n}_synthesis.h5"); l = s["wavelength_m"][:] * 1e9; A = s["intensity"][...]; I = A[2, 2, :, 0, 0] * 1e8; h = l < 700; c = l > 700
    rms = lambda m: np.sqrt(np.mean((I[m] / np.interp(l[m], lq, Iq) - 1) ** 2))
    a = h5py.File(D + f"falc_{n}_atmosphere.h5"); P = a["populations/H"][2, 2, -82:, :]; C = a["populations/CA"][2, 2, -82:, :]; k = lambda x: int(np.argmin(abs(zF - x)))
    print(f"{n:11s}", " ".join(f"{np.interp(x, l, I):.3f}" for x in pts), "| Ha core %.3f Ca core %.3f | rms %.3f %.3f | spread %.3f | total H/RH 1349 252 11 km %.2f %.2f %.2f | Ca/H_RH 252 %.2e" % (I[h].min(), I[c].min(), rms(h), rms(c), (np.ptp(A, axis=(0,1))/A.mean(axis=(0,1))).max(), P[k(1349)].sum()/H[k(1349)].sum(), P[k(252)].sum()/H[k(252)].sum(), P[k(11)].sum()/H[k(11)].sum(), C[k(252)].sum()/H[k(252)].sum()))
