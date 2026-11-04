# FALC inputs with its tabulated nₑ as a dataset (the real code path): the z column with the corona (corona nₑ from the falc_zextc run),
# the same column on log tau, and FALC's own 82 levels on log tau.
import h5py, numpy as np, os
here = os.path.dirname(os.path.abspath(__file__)); j = lambda n: os.path.join(here, n)
with h5py.File(j("FALC_82_5x5.hdf5")) as f: ne_falc = f["electron_density"][0, 2, 2, :].astype(float)
with h5py.File(j("falc_zextc_atmosphere.h5")) as a: ne_ext = a["ne"][2, 2, :, 0].astype(float)
ne_ext[-82:] = ne_falc
def build(src, dst, ne):
    with h5py.File(j(src)) as g, h5py.File(j(dst), "w") as o:
        for n in g: o[n] = g[n][...]
        o["ne"] = np.broadcast_to(ne[:, None, None], g["temperature"].shape).copy(); print(dst, o["ne"].shape)
build("falc_input_extended.h5", "falc_ne_z.h5", ne_ext)
build("falc_input_ltau_extended.h5", "falc_ne_tau_ext.h5", ne_ext)
build("falc_input_ltau.h5", "falc_ne_tau.h5", ne_falc)
