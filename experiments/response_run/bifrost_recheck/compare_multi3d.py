# Bifrost en024048_hion 385, every 4th column, against MULTI3D: H-alpha (output_ray, 504 x 504) and Ca II 854.2 nm (252 x 252, nearest column), with the current code
# through the log tau route (HSE nₑ, FFNO on the mesh) and native z (hydrostatic, or Bifrost nₑ given). Mean profile, and per-column map correlation and relative rms.
import h5py, numpy as np, os, sys
sys.path.insert(0, "")
from helita.sim.multi3d import Multi3dOut
H = os.path.dirname(os.path.abspath(__file__)) + "/"; C = 299792458.0
with h5py.File("/mn/stornext/d9/data/harshm/halpha_output/Multi-3D-H_6_level_populations/output_ray.hdf5") as f:
    wh = f["wavelength"][:]; m = (wh > 655.8) & (wh < 657.2); i = np.where(m)[0]
    Ih = f["intensity"][:, :, i[0]:i[-1] + 1][::4][:, 503 - np.arange(0, 504, 4)]; lh = wh[m]              # (x, y, l) on our columns
d = "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/CA/"; o = Multi3dOut(directory=d); o.readall()
nu = np.array(o.sp.nu, dtype=float)[np.array(o.outff) - 1]; nn = len(nu); print("nu range", nu.min(), nu.max())
ie = np.memmap(d + "ie_+0.00_+0.00_+1.00_allnu", dtype=np.float32, mode="r", order="F", shape=(252, 252, nn))
la = C / nu * 1e9; sq = (1e3 / la * 10) ** 2 * 0 + (1e7 / la) ** 2; lv = la; lair = la / (1.0000834213 + 2.406030e6 / (1.30e10 - sq) + 1.5997e4 / (3.89e9 - sq))
k = np.where((lair > 853.4) & (lair < 854.6))[0]; k = k[np.argsort(lair[k])]; lc = lair[k]
jx = np.arange(0, 504, 4) // 2; jy = (503 - np.arange(0, 504, 4)) // 2
Ic = np.stack([np.array(ie[:, :, q])[np.ix_(jx, jy)] for q in k], axis=2).astype(float)
print("MULTI3D raw units check: H continuum %.3e  Ca edge %.3e; wavelengths Ca %d points %.3f..%.3f" % (Ih[:, :, 0].mean(), Ic[:, :, 0].mean(), len(lc), lc[0], lc[-1]))
if Ic[:, :, 0].mean() > 1e-6: Ic = Ic * 1e-3                                                                    # cgs -> SI
Ih = Ih * 1e8; Ic = Ic * 1e8
def rows(name):
    with h5py.File(H + name) as f:
        w = f["wavelength_m"][:] * 1e9; I = f["intensity"][...][:, :, :, 0, 0] * 1e8      # (x, y, l)
    return w, I
pts = {"Ha": (655.8, 657.2), "Ca": (853.4, 854.6)}
for label, name in (("log tau route", "tau_synthesis.h5"), ("z, hydrostatic P", "z_hse_synthesis.h5"), ("z, nₑ given", "z_ne_synthesis.h5")):
    w, I = rows(name); print("\n" + label)
    for line, (lr, Ir, low, high, bandc) in {"Ha": (lh, Ih, 655.8, 657.2, 0.03), "Ca": (lc, Ic, 853.4, 854.6, 0.03)}.items():
        s = (w > low) & (w < high); wl = w[s]; X = I[:, :, s]
        mean_r = Ir.mean((0, 1)); mean_x = X.mean((0, 1)); c_r = lr[np.argmin(mean_r)]; c_x = wl[np.argmin(mean_x)]
        # common grid relative to each line minimum
        g = np.arange(-0.5, 0.5, 0.005) if line == "Ha" else np.arange(-0.55, 0.38, 0.005)
        R = np.stack([np.interp(g + c_r, lr, Ir[a, b]) for a in range(Ir.shape[0]) for b in range(Ir.shape[1])]).reshape(Ir.shape[0], Ir.shape[1], -1)
        Y = np.stack([np.interp(g + c_x, wl, X[a, b]) for a in range(X.shape[0]) for b in range(X.shape[1])]).reshape(X.shape[0], X.shape[1], -1)
        core = abs(g) <= bandc; wing = (abs(g) >= 0.15) & (abs(g) <= 0.3); far = (abs(g) >= 0.4)
        st = lambda m: (np.corrcoef(Y[:, :, m].mean(2).ravel(), R[:, :, m].mean(2).ravel())[0, 1], np.sqrt(np.mean((Y[:, :, m].mean(2) / R[:, :, m].mean(2) - 1) ** 2)), Y[:, :, m].mean() / R[:, :, m].mean())
        print(f"  {line}: centre M3D {c_r:.3f} ours {c_x:.3f} nm | " + " | ".join("%s r=%.3f rms=%.3f mean ratio=%.3f" % ((n,) + st(q)) for n, q in (("core", core), ("wing", wing), ("far", far))))
