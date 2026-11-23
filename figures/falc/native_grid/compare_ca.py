# FALC_82, FALC's n_e, height axis, FALC's own grid: Ca II 8542 core of the l004p and zaug H + Ca II runs (run_ca.sh in each
# run directory) against RH (falc_rh_air.h5), and their Ca II populations against RH's (falc_rh.h5) at selected heights.
import h5py, numpy as np, struct
R = "/mn/stornext/d9/data/harshm/"
runs = [("l004p", R + "ffnoml_runs/falc_l004p_native_rerun/falc/"), ("zaug", R + "ffnoml_runs/falc_zaug_native/falc/")]
with h5py.File(R + "FFNOML/response_run/falc/falc_rh_air.h5") as f:
    lr = f["wavelength_air_nm"][:]; Ir = f["intensity"][:] * 1e8
q = np.argsort(lr); lr, Ir = lr[q], Ir[q]; w = (lr > 854.0) & (lr < 854.4)
print(f"RH 8542 core {Ir[w].min():.3f}")
b = open(R + "rh_falc/rhf1d/run_falc/geometry.out", "rb").read()
_, nr, n = struct.unpack(">iii", b[:12]); zr = np.frombuffer(b, ">f8", n, 12 + 16 * nr).copy() / 1e3
with h5py.File(R + "FFNOML/response_run/falc/falc_rh.h5") as f:
    rh = np.array(f["CA_n"]); rh = rh if rh.shape[0] == 6 else rh.T
hs = (2100, 1800, 1500, 1200, 1000, 700, 400, 100, 0)
ks = [int(np.argmin(abs(zr - h))) for h in hs]
print("height km        " + " ".join(f"{int(zr[k]):>6d}" for k in ks))
for name, d in runs:
    with h5py.File(d + "falc_ca_falcz_synthesis.h5") as f:
        l = f["wavelength_m"][:] * 1e9; I = f["intensity"][2, 2, :, 0, 0] * 1e8
    s = (l > 854.0) & (l < 854.4)
    print(f"{name}: 8542 core {I[s].min():.3f} ({100 * (I[s].min() / Ir[w].min() - 1):+.1f}% vs RH)")
    with h5py.File(d + "falc_ca_falcz_atmosphere.h5") as f:
        p = f["populations/CA"][2, 2, -82:, :].T
    for lab, y in (("Ca II tot/RH", p[:5].sum(0) / rh[:5].sum(0)), ("Ca III/RH", p[5] / rh[5]), ("n(3d5/2)/RH", p[2] / rh[2]),
                   ("n(4p3/2)/RH", p[4] / rh[4]), ("S ratio", (p[4] / p[2]) / (rh[4] / rh[2]))):
        print(f"  {lab:14s}" + " ".join(f"{y[k]:6.2f}" for k in ks))
