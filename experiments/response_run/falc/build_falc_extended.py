# FALC (falc_input.h5) extended above its top for the z route: T rises log-linearly from FALC's 100,000 K at 2.158 Mm to
# 1 MK at 2.4 Mm (20 km spacing), then stays at 1 MK to 14 Mm (100 km spacing). FALC's 82 levels are the last 82 of the column.
import h5py, numpy as np, os
here = os.path.dirname(os.path.abspath(__file__))
with h5py.File(os.path.join(here, "falc_input.h5")) as g, h5py.File(os.path.join(here, "falc_input_extended.h5"), "w") as o:
    z = g["z_m"][...]; T = g["temperature"][:, 2, 2]; top = z[0]
    tr = np.arange(top + 2e4, 2.4e6, 2e4); corona = np.arange(2.4e6, 14.0001e6, 1e5)
    T_tr = 10 ** (np.log10(T[0]) + (np.log10(1e6) - np.log10(T[0])) * (tr - top) / (tr[-1] - top))
    znew = np.concatenate([corona[::-1], tr[::-1], z]); Tnew = np.concatenate([np.full(corona.size, 1e6), T_tr[::-1], T])
    n = znew.size; o["z_m"] = znew; o["x_m"] = g["x_m"][...]; o["y_m"] = g["y_m"][...]
    o["temperature"] = np.broadcast_to(Tnew[:, None, None], (n, 5, 5)).copy()
    vt = g["vturb"][:, 2, 2]; vtn = np.concatenate([np.full(n - 82, vt[0]), vt])
    o["vturb"] = np.broadcast_to(vtn[:, None, None], (n, 5, 5)).copy()
    for name in ("vx", "vy", "vz"): o[name] = np.zeros((n, 5, 5))
    print("levels", n, "added", n - 82, "z", znew[0] / 1e3, znew[-1] / 1e3, "T", Tnew[0], Tnew[n - 83], Tnew[n - 82])
