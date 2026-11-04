# FALC_82_5x5.hdf5 (RH's Atmos, ITA-Solar/rh) as a native-z FFNOInversion input: heights, T, vturb and vz from the file,
# vx = vy = 0 (it has none), 48 km column spacing (the file's axes are placeholder 0..4 m; the columns are identical).
import h5py, numpy as np, os
here = os.path.dirname(os.path.abspath(__file__))
with h5py.File(os.path.join(here, "FALC_82_5x5.hdf5")) as f, h5py.File(os.path.join(here, "falc_input.h5"), "w") as g:
    g["z_m"] = f["z"][0].astype(float)
    g["x_m"] = 48e3 * np.arange(5.0); g["y_m"] = 48e3 * np.arange(5.0)
    for out, name in (("temperature", "temperature"), ("vz", "velocity_z"), ("vturb", "velocity_turbulent")):
        g[out] = np.transpose(f[name][0].astype(float), (2, 1, 0))          # python order (z, y, x)
    g["vx"] = np.zeros_like(g["vz"]); g["vy"] = np.zeros_like(g["vz"])
    print({k: g[k].shape for k in g}, "h5py order: first index", "z" if g["temperature"].shape[0] == 82 else "other")

# The same model on native log tau500 (the production route: the FFNO predicts on a height mesh like its training grids).
# log tau500 from the native-z run's HSEMHS atmosphere (falc_atmosphere.h5, h5py order x, y, z, 1).
atmosphere = os.path.join(here, "falc_atmosphere.h5")
if os.path.exists(atmosphere):
    with h5py.File(atmosphere) as a, h5py.File(os.path.join(here, "falc_input.h5")) as g, h5py.File(os.path.join(here, "falc_input_ltau.h5"), "w") as o:
        o["logtau_500"] = a["logtau500"][2, 2, :, 0]
        for name in ("x_m", "y_m", "temperature", "vx", "vy", "vz", "vturb"):
            o[name] = g[name][...]
        print("log tau500 top, bottom", o["logtau_500"][0], o["logtau_500"][-1], "monotonic", np.all(np.diff(o["logtau_500"][...]) > 0))
