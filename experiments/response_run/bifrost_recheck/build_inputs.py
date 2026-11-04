# Bifrost en024048_hion 385, every 4th column: native z inputs for the recheck (T, v and, for the given-nₑ case, Bifrost's nₑ) on the file's own heights.
import h5py, numpy as np, os
here = os.path.dirname(os.path.abspath(__file__))
f = h5py.File("/mn/stornext/d9/data/harshm/halpha_output/BIFROST_en024048_hion_0_504_0_504.nc")
s = (0, slice(None, None, 4), slice(None, None, 4), slice(None))
get = lambda n: np.transpose(f[n][s], (2, 1, 0)).astype(float)         # (x, y, z) -> python order (z, y, x)
with h5py.File(os.path.join(here, "bifrost_z.h5"), "w") as o:
    o["z_m"] = f["z"][0].astype(float); o["x_m"] = f["x"][::4].astype(float) - f["x"][0]; o["y_m"] = f["y"][::4].astype(float) - f["y"][0]
    for name, src in (("temperature", "temperature"), ("vx", "velocity_x"), ("vy", "velocity_y"), ("vz", "velocity_z"), ("ne", "electron_density")): o[name] = get(src)
    o["vturb"] = np.zeros_like(o["vx"]); P = get("gas_pressure")
    print("shape", o["temperature"].shape, "bottom P median", np.median(P[-1]), "z ends km", o["z_m"][0] / 1e3, o["z_m"][-1] / 1e3)
