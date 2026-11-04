# RH's FALC_82.atmos as a native log column-mass input: 5 identical columns, its nₑ and the sum of its hydrogen populations
# (top first; SI, except the axis, which keeps the file's g cm^-2 through a units attribute). Heights to compare with RH.
import h5py, numpy as np, os, struct, sys
here = os.path.dirname(os.path.abspath(__file__))
rh = "/mn/stornext/d9/data/harshm/rh_falc"
lines = [l.split() for l in open(rh + "/Atmos/FALC_82.atmos") if l.strip() and not l.startswith("*")]
nd = int(lines[3][0]); a = np.array(lines[4:4 + nd], float); h = np.array(lines[4 + nd:4 + 2 * nd], float)
rep = lambda v: np.repeat(np.repeat(v[:, None, None], 5, axis=1), 5, axis=2)                     # python order (z, y, x)
with h5py.File(os.path.join(here, "falc_logm.h5"), "w") as o:
    o["log10m"] = a[:, 0]; o["log10m"].attrs["units"] = "g cm-2"
    o["x_m"] = 48e3 * np.arange(5.0); o["y_m"] = 48e3 * np.arange(5.0)
    o["temperature"] = rep(a[:, 1]); o["ne"] = rep(a[:, 2] * 1e6); o["vz"] = rep(a[:, 3] * 1e3); o["vturb"] = rep(a[:, 4] * 1e3)
    o["vx"] = rep(0 * a[:, 3]); o["vy"] = rep(0 * a[:, 3])
    o["hydrogen_populations"] = np.repeat(np.repeat(h.T[:, :, None, None] * 1e6, 5, axis=2), 5, axis=3)   # python order (level, z, y, x)
# RH's own height and log tau500 scales, for the same model on the other two axes
import struct
b = open(rh + "/rhf1d/run_falc/geometry.out", "rb").read(); _, nr, n = struct.unpack(">iii", b[:12]); g = np.frombuffer(b, ">f8", 3 * n, 12 + 16 * nr)
assert n == nd
for name, key, axis in (("falc_logm_tau.h5", "logtau_500", np.log10(g[2 * n:3 * n])), ("falc_logm_z.h5", "z_m", g[:n].copy())):
    with h5py.File(os.path.join(here, "falc_logm.h5")) as o, h5py.File(os.path.join(here, name), "w") as t:
        t[key] = axis
        for k in o:
            if k != "log10m": t[k] = o[k][...]
print("levels", nd, "log m", a[0, 0], a[-1, 0], "nH total top, bottom [m-3]", h[0].sum() * 1e6, h[-1].sum() * 1e6)
