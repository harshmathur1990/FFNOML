# Log tau inputs consistent with the given nₑ: first (no z82 run yet) FALC's own 82 levels on z with nₑ; then log tau500 from the z runs that
# used the given nₑ (falc_nez, falc_nez82), so the tau route should recover the same column.
import h5py, numpy as np, os, sys
here = os.path.dirname(os.path.abspath(__file__)); j = lambda n: os.path.join(here, n)
with h5py.File(j("FALC_82_5x5.hdf5")) as f: ne = f["electron_density"][0, 2, 2, :].astype(float)
with h5py.File(j("falc_input.h5")) as g, h5py.File(j("falc_ne_z82.h5"), "w") as o:
    for n in g: o[n] = g[n][...]
    o["ne"] = np.broadcast_to(ne[:, None, None], g["temperature"].shape).copy()
for run, src, dst in (("nez", "falc_ne_z.h5", "falc_ne_tauc_ext.h5"), ("nez82", "falc_ne_z82.h5", "falc_ne_tauc.h5")):
    p = j(f"falc_{run}_atmosphere.h5")
    if not os.path.exists(p): continue
    with h5py.File(p) as a, h5py.File(j(src)) as g, h5py.File(j(dst), "w") as o:
        lt = a["logtau500"][2, 2, :, 0]; o["logtau_500"] = lt
        for n in ("x_m", "y_m", "temperature", "vx", "vy", "vz", "vturb", "ne"): o[n] = g[n][...]
        print(dst, "log tau", lt[0], lt[-1], "monotonic", bool(np.all(np.diff(lt) > 0)))
